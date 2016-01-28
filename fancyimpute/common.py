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
import logging

import numpy as np
from six.moves import range


def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


def masked_mse(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(masked_diff ** 2)


def generate_random_column_samples(column):
    col_mask = np.isnan(column)
    n_missing = np.sum(col_mask)
    if n_missing == len(column):
        logging.warn("No observed values in column")
        return np.zeros_like(column)

    mean = np.nanmean(column)
    std = np.nanstd(column)

    if np.isclose(std, 0):
        return np.array([mean] * n_missing)
    else:
        return np.random.randn(n_missing) * std + mean


def choose_solution_using_percentiles(
        X_original,
        solutions,
        parameters=None,
        verbose=False,
        percentiles=list(range(10, 100, 10))):
    """
    It's tricky to pick a single matrix out of all the candidate
    solutions with differing shrinkage thresholds.
    Our heuristic is to pick the matrix whose percentiles match best
    between the missing and observed data.
    """
    missing_mask = np.isnan(X_original)
    min_mse = np.inf
    best_solution = None
    for i, candidate in enumerate(solutions):
        for col_idx in range(X_original.shape[1]):
            col_data = candidate[:, col_idx]
            col_missing = missing_mask[:, col_idx]
            col_observed = ~col_missing
            if col_missing.sum() < 2:
                continue
            elif col_observed.sum() < 2:
                continue
            missing_data = col_data[col_missing]
            observed_data = col_data[col_observed]

            missing_percentiles = np.array([
                np.percentile(missing_data, p)
                for p in percentiles])

            observed_percentiles = np.array([
                np.percentile(observed_data, p)
                for p in percentiles])

            mse = np.mean((missing_percentiles - observed_percentiles) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_solution = candidate
        if verbose:
            print("Candidate #%d/%d%s: %f" % (
                i + 1,
                len(solutions),
                (" (parameter=%s) " % parameters[i]
                    if parameters is not None
                    else ""),
                mse))
    return best_solution
