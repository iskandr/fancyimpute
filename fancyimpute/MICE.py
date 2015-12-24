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

from __future__ import absolute_import
import numpy as np
from .bayesian_ridge_regression import BayesianRidgeRegression
from .bayesian_regression import BayesianRegression

class MICE():
    """
    Basic implementation of MICE from R.
    This version assumes all of the columns are continuous,
    and uses linear regression.
    """

    def __init__(self,
             visit_sequence = 'monotone', # order in which we visit the columns
             n_imputations=100,
             n_burn_in=10, # this many replicates will be thrown away
             n_neighbors=5, # number of nearest neighbors in PMM
             impute_type='row',
             model=BayesianRidgeRegression(lam=1e-5)): # row means classic pmm, column means fill in linear preds :
        self.visit_sequence = visit_sequence
        self.n_imputations = n_imputations
        self.n_burn_in = n_burn_in
        self.n_neighbors = n_neighbors
        self.impute_type = impute_type
        self.model = model

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _perform_imputation_round(self):
        """
        Does one entire round-robin set of updates.
        """
        for col in self.visit_indices:
            missing_mask_col = self.missing_mask[:,col] # missing mask for this column
            if sum(missing_mask_col) > 0: # if we have any missing data at all
                observed_mask_col = self.observed_mask[:,col] # observed mask for this column
                # the columns we will use to predict the current one
                other_cols = np.array(range(0,col) + range(col+1,self.d))
                # only take rows for which we have observed vals for the current column
                inputs = self.X_filled[observed_mask_col][:,other_cols]
                output = self.X_filled[observed_mask_col,col]
                # fit a ridge model
                brr = self.model
                brr.fit(inputs,output)
                # now we split between the row method (PMM) and the column method
                # note: for the column method, we could use other regressors
                # but I am not sure how to do a posterior predictive draw in the arbitrary case
                # and without this draw, the column algorithm doesn't work
                if self.impute_type == 'row': # this is the PMM procedure
                    # predict values for missing values using random beta draw
                    col_preds_missing = brr.predict(self.X_filled[missing_mask_col][:,other_cols],random_draw=True)
                    # predict values for observed values using best estimated beta
                    col_preds_observed = brr.predict(self.X_filled[observed_mask_col][:,other_cols],random_draw=False)
                    # for each missing value, find its nearest neighbors in the observed values
                    D = np.abs(col_preds_missing[:,np.newaxis] - col_preds_observed) # distances
                    # take top k neighbors
                    k = np.minimum(self.n_neighbors, len(col_preds_observed)-1)
                    # NN = np.argsort(D,1)[:,:k] too slooooow
                    NN = np.argpartition(D,k,1)[:,:k] # <- bottleneck!
                    # pick one of the 5 nearest neighbors at random! that's right! not even an average
                    NN_sampled = [np.random.choice(NN_row) for NN_row in NN]
                    # set the missing values to be the values of the  nearest neighbor in the output space
                    self.X_filled[missing_mask_col,col] = self.X_filled[observed_mask_col,col][NN_sampled]
                elif self.impute_type == 'col':
                    # predict values for missing values using posterior predictive draws
                    # see the end of this: https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
                    #self.X_filled[missing_mask_col,col] = brr.posterior_predictive_draw(self.X_filled[missing_mask_col][:,other_cols]) 
                    mus,sigmas_squared = brr.predict_dist(self.X_filled[missing_mask_col][:,other_cols])
                    self.X_filled[missing_mask_col,col] = np.random.normal(mus,np.sqrt(sigmas_squared))
            
    def complete(
        self,
        X,
        verbose=True):
        
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns a sequence of arrays of the imputed missing values
        of length self.n_imputations, and a mask that specifies where these values
        belong in X.
        """
        
        self._check_input(X)
        self.missing_mask = np.isnan(X) 
        self._check_missing_value_mask(self.missing_mask)
        self.observed_mask = ~self.missing_mask 
        self.d = X.shape[1]
        
        # Decide what order we will update the columns.
        # As a homage to the MICE package, we will have 4 options of how to order the updates.
        if self.visit_sequence == 'roman':
            self.visit_indices = np.arange(self.d)
        elif self.visit_sequence == 'arabic':
            self.visit_indices = np.arange(self.d-1,-1,-1) # same as np.arange(d)[::-1]
        elif self.visit_sequence == 'monotone':
            self.visit_indices = np.argsort(self.missing_mask.sum(0))[::-1]
        elif self.visit_sequence == 'revmonotone':
            self.visit_indices = np.argsort(self.missing_mask.sum(0))
        else:
            self.visit_indices = np.arange(self.d)
            
        # Initialize the missing values by simple samling from the same column.
        # It's what Stef what do
        self.X_filled = X.copy()
        for col in self.visit_indices:
            missing_mask_col = self.missing_mask[:,col]
            observed_mask_col = self.observed_mask[:,col]
            self.X_filled[missing_mask_col,col] = np.random.choice(self.X_filled[observed_mask_col,col],sum(missing_mask_col))
            
        # now we jam up in the usual fashion for n_burn_in + n_imputations iterations
        self.X_filled_storage = [] # all of the imputed values, in a flattened format
        for m in range(self.n_burn_in+self.n_imputations):
            if verbose:
                print "Run:", m
            self._perform_imputation_round()
            if m >= self.n_burn_in:
                self.X_filled_storage.append(self.X_filled[self.missing_mask])
    
        return np.array(self.X_filled_storage), self.missing_mask
    