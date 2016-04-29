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

from __future__ import absolute_import, print_function, division
from collections import defaultdict

from six.moves import range
import numpy as np

from scipy.sparse import dok_matrix

def dense_nan_matrix(shape, dtype):
    return np.ones(shape, dtype=dtype) * np.nan


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


def index_dict_to_sorted_list(key_indices):
    sorted_list = [None] * len(key_indices)
    for (key, index) in key_indices.items():
        sorted_list[index] = key
    return sorted_list


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

    Returns array and sorted lists of the outer and inner keys.
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
    outer_key_list = index_dict_to_sorted_list(outer_key_indices)
    inner_key_list = index_dict_to_sorted_list(inner_key_indices)
    return result, outer_key_list, inner_key_list


def sparse_dok_matrix_from_nested_dictionary(
        nested_dict,
        dtype="float32",
        square_result=False):
    return array_from_nested_dictionary(
        nested_dict,
        array_fn=dok_matrix,
        dtype=dtype,
        square_result=square_result)


def dense_matrix_from_nested_dictionary(
        nested_dict,
        dtype="float32",
        square_result=False):
    return array_from_nested_dictionary(
        nested_dict,
        array_fn=dense_nan_matrix,
        dtype=dtype,
        square_result=square_result)


def matrix_to_pair_dictionary(
        X, row_keys=None, column_keys=None, filter_fn=None):
    """
    X : numpy.ndarray

    row_keys : dict
        Dictionary mapping indices to row names. If omitted then maps each
        number to its string representation, such as 1 -> "1".

    column_keys : dict
        If omitted and matrix is square, then use the same dictionary
        as the rows. Otherwise map each column index to its string form.

    filter_fn : function
        If given then only add elements for which this function returns True.
    """
    n_rows, n_cols = X.shape

    if row_keys is None:
        row_keys = {i: i for i in range(n_rows)}

    if column_keys is None:
        if n_rows == n_cols:
            column_keys = row_keys
        else:
            column_keys = {j: j for j in range(n_cols)}

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
            if filter_fn and not filter_fn(X_ij):
                continue
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


def matrix_to_nested_dictionary(
        X,
        row_keys=None,
        column_keys=None,
        filter_fn=None):
    pair_dict = matrix_to_pair_dictionary(
        X,
        row_keys=row_keys,
        column_keys=column_keys,
        filter_fn=filter_fn)
    return curry_pair_dictionary(pair_dict)


def pair_dict_key_sets(pair_dict):
    row_keys = set([])
    column_keys = set([])
    for (row_key, column_key) in pair_dict.keys():
        row_keys.add(row_key)
        column_keys.add(column_key)
    return row_keys, column_keys


def array_from_pair_dictionary(
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

    Returns array and sorted lists of the row and column keys.
    """
    row_key_set, column_key_set = pair_dict_key_sets(pair_dict)

    if square_result:
        combined_key_set = row_key_set.union(column_key_set)
        row_key_list = column_key_list = list(sorted(combined_key_set))
        row_key_indices = column_key_indices = {
            k: i for (i, k) in enumerate(row_key_list)
        }
    else:
        row_key_list = list(sorted(row_key_set))
        column_key_list = list(sorted(column_key_set))
        row_key_indices = {k: i for (i, k) in enumerate(row_key_list)}
        column_key_indices = {k: i for (i, k) in enumerate(column_key_list)}

    n_rows = len(row_key_indices)
    n_cols = len(column_key_indices)
    shape = (n_rows, n_cols)
    result = array_fn(shape, dtype)
    for (row_key, column_key), value in pair_dict.items():
        i = row_key_indices[row_key]
        j = column_key_indices[column_key]
        result[i, j] = value
    return result, row_key_list, column_key_list


def sparse_dok_matrix_from_pair_dictionary(
        pair_dict,
        dtype="float32",
        square_result=False):
    return array_from_pair_dictionary(
        pair_dict,
        array_fn=dok_matrix,
        dtype=dtype,
        square_result=square_result)


def dense_matrix_from_pair_dictionary(
        pair_dict,
        dtype="float32",
        square_result=False):
    return array_from_pair_dictionary(
        pair_dict,
        array_fn=dense_nan_matrix,
        dtype=dtype,
        square_result=square_result)


def transpose_nested_dictionary(nested_dict):
    """
    Given a nested dictionary from k1 -> k2 > value
    transpose its outer and inner keys so it maps
    k2 -> k1 -> value.
    """
    result = defaultdict(dict)
    for k1, d in nested_dict.items():
        for k2, v in d.items():
            result[k2][k1] = v
    return result


def reverse_lookup_from_nested_dict(values_dict):
    """
    Create reverse-lookup dictionary mapping each row key to a list of triplets:
    [(column key, value), ...]

    Parameters
    ----------
    nested_values_dict : dict
        column_key -> row_key -> value

    weights_dict : dict
        column_key -> row_key -> sample weight

    Returns dictionary mapping row_key -> [(column key, value)]
    """
    reverse_lookup = defaultdict(list)
    for column_key, column_dict in values_dict.items():
        for row_key, value in column_dict.items():
            entry = (column_key, value)
            reverse_lookup[row_key].append(entry)
    return reverse_lookup
