# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division

import numpy as np

from .bayesian_ridge_regression import BayesianRidgeRegression
from .solver import Solver


class MICE(Solver):
    """
    Basic implementation of MICE from R.
    This version assumes all of the columns are continuous,
    and uses linear regression.

    Parameters
    ----------
    visit_sequence : str
        Possible values: "monotone" (default), "roman", "arabic", "revmonotone".

    n_imputations : int
        Defaults to 100

    n_burn_in : int
        Defaults to 10

    impute_type : str
        "row" means classic PMM, "col" (default) means fill in linear preds.

    n_neighbors : int
        Number of nearest neighbors for PMM, defaults to 5.

    model : predictor function
        A model that has fit, predict, and predict_dist methods.
        Defaults to BayesianRidgeRegression(lambda_reg=1e-5)

    add_ones : boolean
        Whether to add a constant column of ones. Defaults to True.

    approximate_but_fast_mode : Boolean
        If True, uses linear algebra trickery to update the inverse covariance.
        Defaults to False as it is not currently faster than brute force.

    verbose : boolean
    """

    def __init__(
            self,
            visit_sequence='monotone',  # order in which we visit the columns
            n_imputations=100,
            n_burn_in=10,  # this many replicates will be thrown away
            n_neighbors=5,  # number of nearest neighbors in PMM
            impute_type='col',
            model=BayesianRidgeRegression(lambda_reg=1e-5),
            add_ones=True,
            approximate_but_fast_mode=False,
            verbose=True):
        """
        Parameters
        ----------
        visit_sequence : str
            Possible values: "monotone" (default), "roman", "arabic", "revmonotone".

        n_imputations : int
            Defaults to 100

        n_burn_in : int
            Defaults to 10

        impute_type : str
            "row" means classic PMM, "col" (default) means fill in linear preds.

        n_neighbors : int
            Number of nearest neighbors for PMM, defaults to 5.

        model : predictor function
            A model that has fit, predict, and predict_dist methods.
            Defaults to BayesianRidgeRegression(lambda_reg=1e-5)

        add_ones : boolean
            Whether to add a constant column of ones. Defaults to True.

        approximate_but_fast_mode : Boolean
            If True, uses linear algebra trickery to update the inverse covariance.
            Defaults to False as it is not currently faster than brute force.

        verbose : boolean

        """
        self.visit_sequence = visit_sequence
        self.n_imputations = n_imputations
        self.n_burn_in = n_burn_in
        self.n_neighbors = n_neighbors
        self.impute_type = impute_type
        self.model = model
        self.add_ones = add_ones
        self.approximate_but_fast_mode = approximate_but_fast_mode
        self.verbose = verbose

    def _sub_inverse_covariance(self, S_inv, col_idx):
        """
        Takes as input a d by d inverse covariance matrix S_inv
        and returns a d-1 dimensional sub-covariance matrix
        that is equivalent to having removed that dimension
        from the original data X before taking the inverse.

        See: http://www.cs.ubc.ca/~emtiyaz/Writings/OneColInv.pdf

        Parameters
        ----------
        S_inv : np.ndarray
            Inverse covariance matrix
        col_idx : int
            Which column to remove

        """
        n_cols = len(S_inv)
        other_cols = np.array(
            list(range(0, col_idx)) + list(range(col_idx + 1, n_cols)))
        F = S_inv[np.ix_(other_cols, other_cols)]
        sigma_squared = S_inv[col_idx, col_idx]
        u = -S_inv[col_idx, other_cols] / sigma_squared
        S_inv_sub = F - sigma_squared * np.outer(u, u)
        return S_inv_sub

    def _update_inverse_covariance(
            self,
            X_filled,
            S,
            S_inv,
            col_idx):
        """
        Iterative update to inverse covariance matrix when only
        one column is changed in X

        See:
        http://www.ini.ruhr-uni-bochum.de/uploads/document/attachment/107/SalmenEtAl_EfficientUpdateLda_PRL2010.pdf
        Page 6

        Parameters
        ----------
        X_filled : np.ndarray
            Partially imputed data matrix
        S : np.ndarray
            Current estimate of the covariance matrix
        S_inv : np.ndarray
            Current estimate of the inverse covariance matrix
        col_idx : int
            The index of the updated column
        """
        n_rows, n_cols = X_filled.shape
        U = np.zeros((n_cols, 2))
        updated_cov = np.dot(X_filled.T, X_filled[:, col_idx])
        U[:, 0] = updated_cov - S[:, col_idx]
        U[col_idx, 1] = 1
        V = np.zeros((2, n_cols))
        V[0, col_idx] = 1
        V[1, :] = U[:, 0]
        V[1, col_idx] = 0
        prod_1 = np.dot(S_inv, U)
        prod_2 = np.dot(V, S_inv)
        inner_inv = np.linalg.inv(np.eye(2) + np.dot(prod_2, U))
        S_inv -= np.dot(np.dot(prod_1, inner_inv), prod_2)
        S[:, col_idx] = updated_cov
        S[col_idx, :] = updated_cov
        return S, S_inv

    def perform_imputation_round(
            self,
            X_filled,
            missing_mask,
            visit_indices,
            S=None,
            S_inv=None):
        """
        Does one entire round-robin set of updates.
        """
        n_rows, n_cols = X_filled.shape
        observed_mask = ~missing_mask
        for col_idx in visit_indices:
            missing_mask_col = missing_mask[:, col_idx]  # missing mask for this column
            n_missing_for_this_col = missing_mask_col.sum()
            if n_missing_for_this_col > 0:  # if we have any missing data at all
                n_observed_for_this_col = n_rows - n_missing_for_this_col

                observed_row_mask_for_col = observed_mask[:, col_idx]
                # The other columns we will use to predict the current one
                other_cols = np.array(list(range(0, col_idx)) + list(range(col_idx + 1, n_cols)))
                # only take rows for which we have observed vals for the current column
                inputs = X_filled[np.ix_(observed_row_mask_for_col, other_cols)]
                output = X_filled[observed_row_mask_for_col, col_idx]
                brr = self.model
                # now we either use an approximate inverse
                # or an exact one (slow updates)
                if self.approximate_but_fast_mode:
                    scaling_for_S_inv = n_rows / n_observed_for_this_col
                    S_inv_slice = self._sub_inverse_covariance(S_inv, col_idx)
                    S_inv_sub_est = scaling_for_S_inv * S_inv_slice
                    brr.fit(inputs, output, inverse_covariance=S_inv_sub_est)
                else:
                    brr.fit(inputs, output, inverse_covariance=None)

                # Now we choose the row method (PMM) or the column method.
                if self.impute_type == 'row':  # this is the PMM procedure
                    # predict values for missing values using random beta draw
                    X_missing = X_filled[np.ix_(missing_mask_col, other_cols)]
                    col_preds_missing = brr.predict(X_missing, random_draw=True)
                    # predict values for observed values using best estimated beta
                    X_observed = X_filled[np.ix_(observed_row_mask_for_col, other_cols)]
                    col_preds_observed = brr.predict(X_observed, random_draw=False)
                    # for each missing value, find its nearest neighbors in the observed values
                    D = np.abs(col_preds_missing[:, np.newaxis] - col_preds_observed)  # distances
                    # take top k neighbors
                    k = np.minimum(self.n_neighbors, len(col_preds_observed) - 1)
                    # NN = np.argsort(D,1)[:,:k] too slooooow
                    NN = np.argpartition(D, k, 1)[:, :k]  # <- bottleneck!
                    # pick one of the 5 nearest neighbors at random! that's right!
                    # not even an average
                    NN_sampled = [np.random.choice(NN_row) for NN_row in NN]
                    # set the missing values to be the values of the  nearest
                    # neighbor in the output space
                    X_filled[missing_mask_col, col_idx] = \
                        X_filled[observed_row_mask_for_col, col_idx][NN_sampled]
                elif self.impute_type == 'col':
                    X_missing = X_filled[np.ix_(missing_mask_col, other_cols)]
                    # predict values for missing values using posterior predictive draws
                    # see the end of this:
                    # https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
                    # X_filled[missing_mask_col,col] = \
                    #   brr.posterior_predictive_draw(X_missing)
                    mus, sigmas_squared = brr.predict_dist(X_missing)
                    X_filled[missing_mask_col, col_idx] = \
                        np.random.normal(mus, np.sqrt(sigmas_squared))
                # now we update the covariance and inverse covariance matrices
                if self.approximate_but_fast_mode:
                    S, S_inv = self._update_inverse_covariance(
                        X_filled=X_filled,
                        S=S,
                        S_inv=S_inv,
                        col_idx=col_idx)
        return X_filled, S, S_inv

    def initialize(self, X, missing_mask, visit_indices):
        """
        Initialize the missing values by simple sampling from the same column.
        """
        X_filled = X.copy()
        observed_mask = ~missing_mask
        for col_idx in visit_indices:
            missing_mask_col = missing_mask[:, col_idx]
            if np.sum(missing_mask_col) > 0:
                observed_row_mask_for_col = observed_mask[:, col_idx]
                observed_col = X_filled[observed_row_mask_for_col, col_idx]
                n_missing = np.sum(missing_mask_col)
                random_values = np.random.choice(observed_col, n_missing)
                X_filled[missing_mask_col, col_idx] = random_values
        return X_filled

    def get_visit_indices(self, missing_mask):
        """
        Decide what order we will update the columns.
        As a homage to the MICE package, we will have 4 options of
        how to order the updates.
        """
        n_rows, n_cols = missing_mask.shape
        if self.visit_sequence == 'roman':
            return np.arange(n_cols)
        elif self.visit_sequence == 'arabic':
            return np.arange(n_cols - 1, -1, -1)  # same as np.arange(d)[::-1]
        elif self.visit_sequence == 'monotone':
            return np.argsort(missing_mask.sum(0))[::-1]
        elif self.visit_sequence == 'revmonotone':
            return np.argsort(missing_mask.sum(0))
        else:
            raise ValueError("Invalid choice for visit order: %s" % self.visit_sequence)

    def multiple_imputations(self, X):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns a sequence of arrays of the imputed missing values
        of length self.n_imputations, and a mask that specifies where these values
        belong in X.
        """

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        visit_indices = self.get_visit_indices(missing_mask)
        n_rows = len(X)
        if self.add_ones:
            X = np.column_stack((X, np.ones(n_rows)))
            missing_mask = np.column_stack([
                missing_mask,
                np.zeros(n_rows, dtype=missing_mask.dtype)
            ])
        n_cols = X.shape[1]

        X_filled = self.initialize(
            X,
            missing_mask=missing_mask,
            visit_indices=visit_indices)

        # compute S and S_inv if required
        if self.approximate_but_fast_mode:
            S = np.dot(X_filled.T, X_filled)
            regularization_matrix = self.model.lambda_reg * np.eye(n_cols)
            if self.add_ones:  # then don't regularize the offset
                regularization_matrix[-1, -1] = 0
            S_inv = np.linalg.inv(S + regularization_matrix)
        else:
            S = S_inv = None

        # now we jam up in the usual fashion for n_burn_in + n_imputations iterations
        results_list = []  # all of the imputed values, in a flattened format
        total_rounds = self.n_burn_in + self.n_imputations

        for m in range(total_rounds):
            if self.verbose:
                print("[MICE] Imputation round %d/%d:" % (
                    m + 1, total_rounds))
            X_filled, S, S_inv = self.perform_imputation_round(
                X_filled=X_filled,
                missing_mask=missing_mask,
                visit_indices=visit_indices,
                S=S,
                S_inv=S_inv)
            if m >= self.n_burn_in:
                results_list.append(X_filled[missing_mask])
        if self.add_ones:
            # chop off the missing mask corresponding to the constant ones
            missing_mask = missing_mask[:, :-1]
        return np.array(results_list), missing_mask

    def complete(self, X):
        X_completed = X.copy()
        imputed_arrays, missing_mask = self.multiple_imputations(X)
        # average the imputed values for each feature
        average_imputated_values = imputed_arrays.mean(axis=0)
        X_completed[missing_mask] = average_imputated_values
        return X_completed
