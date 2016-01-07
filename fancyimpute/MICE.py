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
    """

    def __init__(
            self,
            visit_sequence='monotone',  # order in which we visit the columns
            n_imputations=100,
            n_burn_in=10,  # this many replicates will be thrown away
            n_neighbors=5,  # number of nearest neighbors in PMM
            impute_type='row',
            model=BayesianRidgeRegression(lambda_reg=1e-5),
            add_ones=True,
            approximate_but_fast_mode=True):
        """
        Parameters
        ----------
        visit_sequence : str
            Possible values: "monotone", "roman", "arabic", "revmonotone"

        n_imputations : int

        n_burn_in : int

        impute_type : str
            "row" means classic pmm, "column" means fill in linear preds

        model : predictor
        """
        self.visit_sequence = visit_sequence
        self.n_imputations = n_imputations
        self.n_burn_in = n_burn_in
        self.n_neighbors = n_neighbors
        self.impute_type = impute_type
        self.model = model
        self.add_ones = add_ones
        self.approximate_but_fast_mode = approximate_but_fast_mode
        self.S = None # covariance matrix
        self.S_inv = None # inverse covariance matrix

    def _inverse_covariance(self,add_ones=True):

        regularization_matrix = self.model.lambda_reg * np.eye(self.d)
        regularization_matrix[-1, -1] = 0  # don't need to regularize the intercept
        # the big expensive inverse that we use over and over
        X_ones = np.column_stack((self.X_filled, np.ones(self.X_filled.shape[0])))
        S_inv_full = np.linalg.inv(np.dot(X_ones.T, X_ones) + regularization_matrix)
        
    def _sub_inverse_covariance(self,col):
        """
        Takes as input a d by d inverse covariance matrix S_inv
        and returns a d-1 dimensional sub-covariance matrix
        that is equivalent to having removed that dimension 
        from the original data X before taking the inverse.
        
        See: http://www.cs.ubc.ca/~emtiyaz/Writings/OneColInv.pdf
        
        Parameters
        ----------
        col : int
            which column to remove
        
        """
        other_cols = np.array(list(range(0, col)) + list(range(col + 1, self.d)))
        F = self.S_inv[other_cols,:][:,other_cols]
        sigma_squared = self.S_inv[col,col]
        u = -self.S_inv[col,other_cols] / sigma_squared
        S_inv_sub = F - sigma_squared*np.outer(u,u)
        return S_inv_sub

    def _update_inverse_covariance(self,col):
        """
        Iterative update to inverse covariance matrix when only
        one column is changed in X
        
        See: http://www.ini.ruhr-uni-bochum.de/uploads/document/attachment/107/SalmenEtAl_EfficientUpdateLda_PRL2010.pdf
        Page 6
        
        Parameters
        ----------
        col : int
            The index of the updated column
        """
        U = np.zeros((self.d,2))
        updated_cov = np.dot(self.X_filled.T,self.X_filled[:,col])
        U[:,0] = updated_cov - self.S[:,col]
        U[col,1] = 1
        V = np.zeros((2,self.d))
        V[0,col] = 1
        V[1,:] = U[:,0]
        V[1,col] = 0 
        prod_1 = np.dot(self.S_inv,U)
        prod_2 = np.dot(V,self.S_inv)
        inner_inv = np.linalg.inv(np.eye(2) + np.dot(prod_2,U))
        self.S_inv -= np.dot(np.dot(prod_1,inner_inv),prod_2)
        self.S[:,col] = updated_cov
        self.S[col,:] = updated_cov

        
    def _perform_imputation_round(self):
        """
        Does one entire round-robin set of updates.
        """
        for col in self.visit_indices:
            missing_mask_col = self.missing_mask[:, col]  # missing mask for this column
            if np.sum(missing_mask_col) > 0:  # if we have any missing data at all
                observed_row_mask_for_col = self.observed_mask[:, col]  # observed mask for this column
                # The other columns we will use to predict the current one
                other_cols = np.array(list(range(0, col)) + list(range(col + 1, self.d)))
                # only take rows for which we have observed vals for the current column
                inputs = self.X_filled[observed_row_mask_for_col][:, other_cols]
                output = self.X_filled[observed_row_mask_for_col, col]
                brr = self.model
                # now we either use an approximate inverse (fast updates)
                # or an exact one (slow updates)
                if self.approximate_but_fast_mode:
                    scaling_for_S_inv = len(observed_row_mask_for_col)/np.sum(observed_row_mask_for_col)
                    S_inv_sub_est = scaling_for_S_inv * self._sub_inverse_covariance(col)
                    brr.fit(inputs, output, S_inv_sub_est)
                else:
                    brr.fit(inputs, output)
                    
                # Now we choose the row method (PMM) or the column method.
                if self.impute_type == 'row':  # this is the PMM procedure
                    # predict values for missing values using random beta draw
                    X_missing = self.X_filled[missing_mask_col][:, other_cols]
                    col_preds_missing = brr.predict(X_missing, random_draw=True)
                    # predict values for observed values using best estimated beta
                    X_observed = self.X_filled[observed_row_mask_for_col][:, other_cols]
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
                    self.X_filled[missing_mask_col, col] = \
                        self.X_filled[observed_row_mask_for_col, col][NN_sampled]
                elif self.impute_type == 'col':
                    X_missing = self.X_filled[missing_mask_col][:, other_cols]
                    # predict values for missing values using posterior predictive draws
                    # see the end of this:
                    # https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
                    # self.X_filled[missing_mask_col,col] = \
                    #   brr.posterior_predictive_draw(X_missing)
                    mus, sigmas_squared = brr.predict_dist(X_missing)
                    self.X_filled[missing_mask_col, col] = \
                        np.random.normal(mus, np.sqrt(sigmas_squared))
                # now we update the covariance and inverse covariance matrices
                if self.approximate_but_fast_mode:
                    self._update_inverse_covariance(col)


    def multiple_imputations(self, X, verbose=True):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns a sequence of arrays of the imputed missing values
        of length self.n_imputations, and a mask that specifies where these values
        belong in X.
        """
        
        self._check_input(X)
        if self.add_ones:
            X = np.column_stack((X, np.ones(X.shape[0])))
        self.missing_mask = np.isnan(X)
        self._check_missing_value_mask(self.missing_mask)
        self.observed_mask = ~self.missing_mask
        self.n, self.d = X.shape

        # Decide what order we will update the columns.
        # As a homage to the MICE package, we will have 4 options of how to order the updates.
        if self.visit_sequence == 'roman':
            self.visit_indices = np.arange(self.d)
        elif self.visit_sequence == 'arabic':
            self.visit_indices = np.arange(self.d - 1, -1, -1)  # same as np.arange(d)[::-1]
        elif self.visit_sequence == 'monotone':
            self.visit_indices = np.argsort(self.missing_mask.sum(0))[::-1]
        elif self.visit_sequence == 'revmonotone':
            self.visit_indices = np.argsort(self.missing_mask.sum(0))
        else:
            self.visit_indices = np.arange(self.d)
        
        # Initialize the missing values by simple sampling from the same column.
        # It's what Stef what do
        self.X_filled = X.copy()
        for col in self.visit_indices:
            missing_mask_col = self.missing_mask[:, col]
            if np.sum(missing_mask_col) > 0:
                observed_row_mask_for_col = self.observed_mask[:, col]
                observed_col = self.X_filled[observed_row_mask_for_col, col]
                n_missing = np.sum(missing_mask_col)
                self.X_filled[missing_mask_col, col] = np.random.choice(observed_col, n_missing)

        # compute S and S_inv if not already computed
        if self.S is None and self.approximate_but_fast_mode:
            self.S = np.dot(self.X_filled.T, self.X_filled)
            regularization_matrix = self.model.lambda_reg * np.eye(self.d)
            if self.add_ones: # then don't regularize the offset
                regularization_matrix[-1,-1] = 0 
            self.S_inv = np.linalg.inv(self.S + regularization_matrix)
            
        # now we jam up in the usual fashion for n_burn_in + n_imputations iterations
        self.X_filled_storage = []  # all of the imputed values, in a flattened format
        for m in range(self.n_burn_in + self.n_imputations):
            if verbose:
                print("Run:", m)
            self._perform_imputation_round()
            if m >= self.n_burn_in:
                self.X_filled_storage.append(self.X_filled[self.missing_mask])

        return np.array(self.X_filled_storage), self.missing_mask

    def complete(self, X, verbose=True):
        X_multiple_imputations, missing_mask = self.multiple_imputations(
            X, verbose)
        X_completed = X.copy()
        # average the imputed values for each feature
        X_completed[missing_mask] = X_multiple_imputations.mean(axis=0)
        return X_completed
