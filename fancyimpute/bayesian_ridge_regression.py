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


class BayesianRidgeRegression(object):
    """
    Bayesian Ridge Regression
    
    """
    def __init__(self, lambda_reg=1e-5,add_ones=False,normalize_lambda=True):
        self.lambda_reg = lambda_reg
        self.add_ones = add_ones
        self.normalize_lambda = normalize_lambda
    
    def fit(self, X, y, inverse_covariance=None):
        if self.add_ones:
            X_ones = self.add_column_of_ones(X)
        else:
            X_ones = X
        # first add a column of all ones to X
        n, d = X_ones.shape
        # the big expensive step when d is large
        if inverse_covariance is None:
            outer_product = np.dot(X_ones.T, X_ones)
            if self.normalize_lambda:
                lambda_reg = self.lambda_reg * np.linalg.norm(outer_product)
    
            else:
                lambda_reg = self.lambda_reg
            regularization_matrix = lambda_reg * np.eye(d)
            regularization_matrix[-1, -1] = 0  # don't need to regularize the intercept
            self.inverse_covariance = np.linalg.inv(
               outer_product + regularization_matrix)
        else:
            self.inverse_covariance = inverse_covariance
        # estimate of the parameters
        self.beta_estimate = np.dot(np.dot(self.inverse_covariance, X_ones.T), y)        
        # now we need the estimate of the noise variance
        # reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
        residuals_sqr = (y - self.predict(X)) ** 2
        self.sigma_squared_estimate = np.sum(residuals_sqr) / np.maximum((n - d), 1)

    def predict(self, X, random_draw=False):
        if self.add_ones:
            X_ones = self.add_column_of_ones(X)
        else:
            X_ones = X
        if random_draw:
            return np.dot(X_ones, self.random_beta_draw(num_draws=1)[0])
        else:
            return np.dot(X_ones, self.beta_estimate)

    def add_column_of_ones(self, X):
        if len(X.shape) == 1:
            return np.append(X, 1)
        else:
            return np.column_stack((X, np.ones(X.shape[0])))

    # Random draws from the posterior over beta coefficients
    # reference: https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
    # page 1
    # note that the pros something different: 
    # https://github.com/stefvanbuuren/mice/blob/master/R/mice.impute.norm.r
    def random_beta_draw(self, num_draws=1):
        covar = self.sigma_squared_estimate * self.inverse_covariance
        return np.random.multivariate_normal(self.beta_estimate, covar, num_draws)

    # Returns the mean and variance of the posterior predictive distribution
    # Reference: https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
    # page 2
    def predict_dist(self, X):
        if self.add_ones:
            X_ones = self.add_column_of_ones(X) 
        else:
            X_ones = X
        mus = self.predict(X)
        covar = self.sigma_squared_estimate * self.inverse_covariance
        sigmas_squared = self.sigma_squared_estimate + np.sum(
            np.dot(X_ones, covar) * X_ones, axis=1)
        return mus, sigmas_squared
