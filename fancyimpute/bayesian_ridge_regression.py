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

from numpy import dot, append, column_stack, ones
from numpy.linalg import norm, inv, multi_dot
from numpy.random import multivariate_normal


class BayesianRidgeRegression(object):
    """
    Bayesian Ridge Regression
    """
    def __init__(self, lambda_reg=0.001, add_ones=False, normalize_lambda=True):
        '''
        Parameters
        ----------
        lambda_reg : float
            Ridge regularization parameter.
            Default is 0.001.

        add_ones : boolean
            Whether to add a constant column of ones.
            Default is False.

        normalize_lambda : boolean
            Default is True.
            This variant multiplies lambda_reg by
            np.linalg.norm(np.dot(X.T,X))
        '''
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
            outer_product = dot(X_ones.T, X_ones)
            if self.normalize_lambda:
                lambda_reg = self.lambda_reg * norm(outer_product)
            else:
                lambda_reg = self.lambda_reg

            for i in range(d - 1):
                # Replacing `outer_product + lambda_reg * eye(d)` with
                # a direct modification of the outer_product matrix
                #
                # We're trading a little more time spent in the Python
                # interpreter with a savings of allocated arrays.
                outer_product[i, i] += lambda_reg

            self.inverse_covariance = inv(outer_product)
        else:
            self.inverse_covariance = inverse_covariance
        # estimate of the parameters
        self.beta_estimate = multi_dot([self.inverse_covariance, X_ones.T, y])
        # now we need the estimate of the noise variance
        # reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
        residuals = (y - self.predict(X))
        # square all the residuals
        residuals **= 2
        self.sigma_squared_estimate = residuals.sum() / max((n - d), 1)
        self.covar = self.sigma_squared_estimate * self.inverse_covariance

    def predict(self, X, random_draw=False):
        if self.add_ones:
            X_ones = self.add_column_of_ones(X)
        else:
            X_ones = X
        if random_draw:
            return dot(X_ones, self.random_beta_draw(num_draws=1)[0])
        else:
            return dot(X_ones, self.beta_estimate)

    def add_column_of_ones(self, X):
        if len(X.shape) == 1:
            return append(X, 1)
        else:
            return column_stack((X, ones(X.shape[0])))

    # Random draws from the posterior over beta coefficients
    # reference: https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
    # page 1
    # note that the pros something different:
    # https://github.com/stefvanbuuren/mice/blob/master/R/mice.impute.norm.r
    def random_beta_draw(self, num_draws=1):
        return multivariate_normal(self.beta_estimate, self.covar, num_draws)

    # Returns the mean and variance of the posterior predictive distribution
    # Reference: https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf
    # page 2
    def predict_dist(self, X):
        if self.add_ones:
            X_ones = self.add_column_of_ones(X)
        else:
            X_ones = X
        mus = dot(X_ones, self.beta_estimate)

        rescaled_X = dot(X_ones, self.covar)
        rescaled_X *= X_ones
        sigmas_squared = rescaled_X.sum(axis=1)
        sigmas_squared += self.sigma_squared_estimate
        return mus, sigmas_squared
