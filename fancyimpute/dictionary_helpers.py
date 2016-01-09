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

"""
Helper functions for incomplete matrices represented using dictionaries.
"""

from collections import defaultdict

from scipy.sparse import dok_matrix
import numpy as np


def collect_nested_keys(nested_dict):
    outer_key_list = list(sorted(nested_dict.keys()))
    inner_key_set = set([])
    for k in outer_key_list:
        inner_dict = nested_dict[k]
        inner_key_set = inner_key_set.union(inner_dict.keys())
    inner_key_list = list(sorted(inner_key_set))
    return outer_key_list, inner_key_list


def nested_key_indices(nested_dict):
    """
    Give an ordering to the outer and inner keys used in a dictionary that
    maps to dictionaries.
    """
    outer_keys, inner_keys = collect_nested_keys(nested_dict)
    outer_key_indices = {k: i for (i, k) in enumerate(outer_keys)}
    inner_key_indices = {k: i for (i, k) in enumerate(inner_keys)}
    return outer_key_indices, inner_key_indices


def flattened_nested_key_indices(nested_dict):
    """
    Combine the outer and inner keys of nested dictionaries into a single
    ordering.
    """
    outer_keys, inner_keys = collect_nested_keys(nested_dict)
    combined_keys = list(sorted(set(outer_keys + inner_keys)))
    return {k: i for (i, k) in enumerate(combined_keys)}


def array_from_nested_dictionary(
        nested_dict,
        array_fn,
        dtype="float32",
        square_result=False):
    """
    Parameters
    ----------
    nested_dict : dict
        Dictionary which contains dictionaries

    array_fn : function
        Takes shape and dtype as arguments, returns empty array.

    dtype : dtype
        NumPy dtype of result array

    square_result : bool
        Combine keys from outer and inner dictionaries.

    Returns array and two dictionaries mapping keys from the outer/inner arrays
    to indices in the returned array.
    """
    if square_result:
        outer_key_indices = inner_key_indices = flattened_nested_key_indices(
            nested_dict)
    else:
        outer_key_indices, inner_key_indices = nested_key_indices(
            nested_dict)

    n_rows = len(outer_key_indices)
    n_cols = len(inner_key_indices)
    shape = (n_rows, n_cols)
    result = array_fn(shape, dtype)
    for outer_key, sub_dictionary in nested_dict.items():
        i = outer_key_indices[outer_key]
        for inner_key, value in sub_dictionary.items():
            j = inner_key_indices[inner_key]
            result[i, j] = value
    return result, outer_key_indices, inner_key_indices


def sparse_dok_matrix_from_nested_dictionary(
        nested_dict,
        dtype="float32",
        square_result=False):
    return array_from_nested_dictionary(
        nested_dict,
        array_fn=dok_matrix,
        dtype=dtype,
        square_result=square_result)


def dense_nan_matrix(shape, dtype):
    return np.ones(shape, dtype=dtype) * np.nan


def dense_matrix_from_nested_dictionary(
        nested_dict,
        dtype="float32",
        square_result=False):
    return array_from_nested_dictionary(
        nested_dict,
        array_fn=dense_nan_matrix,
        dtype=dtype,
        square_result=square_result)


def matrix_to_pair_dictionary(X, row_keys, column_keys=None):
    if column_keys is None:
        column_keys = row_keys

    n_rows, n_cols = X.shape

    if len(row_keys) != n_rows:
        raise ValueError("Need %d row keys but got list of length %d" % (
            n_rows,
            len(row_keys)))
    if len(column_keys) != n_cols:
        raise ValueError("Need %d column keys but got list of length %d" % (
            n_cols,
            len(column_keys)))
    result_dict = {}
    for i, X_i in enumerate(X):
        row_key = row_keys[i]
        for j, X_ij in enumerate(X_i):
            column_key = column_keys[j]
            key_pair = (row_key, column_key)
            result_dict[key_pair] = X_ij
    return result_dict


def curry_pair_dictionary(key_pair_dict, default_value=0.0):
    """
    Transform dictionary from pairs of keys to dict -> dict -> float
    """
    result = defaultdict(dict)
    for (a, b), value in key_pair_dict.items():
        result[a][b] = value
    return result


def uncurry_nested_dictionary(curried_dict):
    """
    Transform dictionary from (key_a -> key_b -> float) to
    (key_a, key_b) -> float
    """
    result = {}
    for a, a_dict in curried_dict.items():
        for b, value in a_dict.items():
            result[(a, b)] = value
    return result


def matrix_to_nested_dictionary(X, row_keys, column_keys=None):
    pair_dict = matrix_to_pair_dictionary(
        X,
        row_keys=row_keys,
        column_keys=column_keys)
    return curry_pair_dictionary(pair_dict)


def pair_dict_key_sets(pair_dict):
    row_keys = set([])
    column_keys = set([])
    for (row_key, column_key) in pair_dict.keys():
        row_keys.add(row_key)
        column_keys.add(column_keys)
    return row_keys, column_keys


def array_from_pair_dict(
        pair_dict,
        array_fn,
        dtype="float32",
        square_result=False):
    """
    Convert a dictionary whose keys are pairs (k1, k2) into a sparse
    or incomplete array.

    Parameters
    ----------
    pair_dict : dict
        Dictionary from pairs of keys to values.

    array_fn : function
        Takes shape and dtype as arguments, returns empty array.

    dtype : dtype
        NumPy dtype of result array

    square_result : bool
        Combine keys from rows and columns

    Returns array and dictionaries mapping row/column keys to indices.
    """
    row_keys, column_keys = pair_dict_key_sets(pair_dict)

    if square_result:
        keys = row_keys.union(column_keys)
        key_list = list(sorted(keys))
        row_key_indices = column_key_indices = {k: i for (i, k) in key_list}
    else:
        row_keys = list(sorted(row_keys))
        row_key_indices = {k: i for (i, k) in row_keys}
        column_key_indices = {k: i for (i, k) in column_keys}

    n_rows = len(row_key_indices)
    n_cols = len(column_key_indices)
    shape = (n_rows, n_cols)
    result = array_fn(shape, dtype)
    for (row_key, column_key), value in pair_dict.items():
        i = row_key_indices[row_key]
        j = column_key_indices[column_keys]
        result[i, j] = value
    return result, row_key_indices, column_key_indices
