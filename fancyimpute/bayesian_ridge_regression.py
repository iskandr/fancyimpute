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

import numpy as np

class BayesianRidgeRegression():
    
    def __init__(self,lam=1e-5):
        self.lam = lam
        
    def fit(self,X,y):
        X_ones = self.add_column_of_ones(X)
        # first add a column of all ones to X
        n,d = X_ones.shape
        # regularization matrix
        regularization_matrix = self.lam*np.eye(d)
        regularization_matrix[-1,-1] = 0 # don't need to regularize the intercept
        # the big expensive inverse that we use over and over
        self.inv_matrix = np.linalg.inv( np.dot(X_ones.T,X_ones) + regularization_matrix ) 
        # estimate of the parameters 
        self.beta_estimate = np.dot(np.dot(self.inv_matrix,X_ones.T),y)
        # now we need the estimate of the noise variance
        # reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
        residuals_sqr = (y - self.predict(X))**2
        self.sigma_squared_estimate = np.sum(residuals_sqr) / np.maximum((n-d),1)
    
    def predict(self,X,beta=None):
        X_ones = self.add_column_of_ones(X)
        if beta is not None:
            return np.dot(X_ones,self.random_beta_draw(1)[0])
        else:
            return np.dot(X_ones,self.beta_estimate)
        
    def add_column_of_ones(self,X):
        if len(X.shape) == 1:
            return np.append(X,1)
        else:
            return np.column_stack((X,np.ones(X.shape[0])))
        
    # random draws work as follows
    # reference: https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
    # bottom of page 1
    # note that the pros do wackier stuff: https://github.com/jwb133/smcfcs/blob/master/R/smcfcs.r
    # see lines 363 to 365. not sure what exactly is happening here
    def random_beta_draw(self,num_draws=1):
        covar = self.sigma_squared_estimate*self.inv_matrix
        return np.random.multivariate_normal(self.beta_estimate,covar,num_draws)
        
    # posterior predictive draw
    def posterior_predictive_draw(self,X):
        X_ones = self.add_column_of_ones(X) # adding constant offset
        mu = self.predict(X)
        covar = self.sigma_squared_estimate*self.inv_matrix
        sigma_sq = self.sigma_squared_estimate + np.sum(np.dot(X_ones,covar)*X_ones, axis = 1)
        draws = np.random.normal(mu,np.sqrt(sigma_sq))
        return np.array(draws)