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
from collections import defaultdict

import numpy as np

from .dictionary_helpers import (
    collect_nested_keys,
    reverse_lookup_from_nested_dict,
    matrix_to_nested_dictionary,
    transpose_nested_dictionary,
)


class SimilarityWeightedAveraging(object):
    """
    Fill in missing each missing row/column value by averaging across the
    k-nearest neighbors columns (taking into account missing data when
    computing column similarities and choosing which neighbors to inspect).

    Currently does not inherit from Solver since it expects sparse inputs in
    the form of nested dictionaries.
    """

    def __init__(
            self,
            min_weight_for_similarity=0.1,
            min_count_for_similarity=2,
            similarity_exponent=4.0,
            shrinkage_coef=0.0001,
            orientation="rows",
            verbose=False):
        """
        Parameters
        ----------
        min_weight_for_similarity : float
            If sum of values in shared rows between two columns falls below this
            threhold then similarity can't be computed between those columns.

        min_count_for_similarity : int
            If number of overlapping rows between two columns falls below this
            threhold then similarity can't be computed between those columns.

        similarity_exponent : float
            Exponent for turning similarities into weights on values of other
            columns.

        shrinkage_coef : float
            Shrinks reconstructed values toward 0

        orientation : str
            Whether to compute similarities along rows or columns

        verbose : bool
        """
        self.min_weight_for_similarity = min_weight_for_similarity
        self.min_count_for_similarity = min_count_for_similarity
        self.similarity_exponent = similarity_exponent
        self.shrinkage_coef = shrinkage_coef
        self.orientation = orientation
        self.verbose = verbose

    def jacard_similarity_from_nested_dicts(self, nested_dictionaries):
        """
        Compute the continuous Jacard similarity between all pairs
        of keys in dictionary-of-dictionaries given as an input.

        Returns three element tuple:
            - similarity dictionary: (key, key) -> float
            - overlap count dictionary: key -> key -> int
            - weight dictionary: key -> key -> float
        """
        sims = {}
        overlaps = {}
        weights = {}
        for a, column_dict_a in nested_dictionaries.items():
            row_set_a = set(column_dict_a.keys())
            for b, column_dict_b in nested_dictionaries.items():
                row_set_b = set(column_dict_b.keys())
                common_rows = row_set_a.intersection(row_set_b)
                n_overlap = len(common_rows)
                overlaps[(a, b)] = n_overlap
                total = 0.0
                weight = 0.0
                for row_name in common_rows:
                    value_a = column_dict_a[row_name]
                    value_b = column_dict_b[row_name]
                    minval = min(value_a, value_b)
                    maxval = max(value_a, value_b)
                    total += minval
                    weight += maxval
                weights[(a, b)] = weight
                if weight < self.min_weight_for_similarity:
                    continue
                if n_overlap < self.min_count_for_similarity:
                    continue
                sims[(a, b)] = total / weight
        return sims, overlaps, weights

    def complete_dict(
            self,
            values_dict):
        """
        Keys of nested dictionaries can be arbitrary objects.
        """
        if self.orientation != "rows":
            values_dict = transpose_nested_dictionary(values_dict)

        row_keys, column_keys = collect_nested_keys(values_dict)
        if self.verbose:
            print("[SimilarityWeightedAveraging] # rows = %d" % (len(row_keys)))
            print("[SimilarityWeightedAveraging] # columns = %d" % (len(column_keys)))
        similarities, overlaps, weights = \
            self.jacard_similarity_from_nested_dicts(values_dict)
        if self.verbose:
            print(
                "[SimilarityWeightedAveraging] Computed %d similarities between rows" % (
                    len(similarities),))
        column_to_row_values = reverse_lookup_from_nested_dict(values_dict)

        result = defaultdict(dict)

        exponent = self.similarity_exponent
        shrinkage_coef = self.shrinkage_coef
        for i, row_key in enumerate(row_keys):
            for column_key, value_triplets in column_to_row_values.items():
                total = 0
                denom = shrinkage_coef
                for (other_row_key, y) in value_triplets:
                    sample_weight = 1.0
                    sim = similarities.get((row_key, other_row_key), 0)
                    combined_weight = sim ** exponent
                    combined_weight *= sample_weight
                    total += combined_weight * y
                    denom += combined_weight
                if denom > shrinkage_coef:
                    result[row_key][column_key] = total / denom
        if self.orientation != "rows":
            result = transpose_nested_dictionary(result)
        return result

    def complete(self, X):
        if self.verbose:
            print(
                "[SimilarityWeightedAveraging] Creating dictionary from matrix with shape %s" % (
                    X.shape,))
        missing_mask = np.isnan(X)
        observed_mask = ~missing_mask
        sparse_dict = matrix_to_nested_dictionary(
            X,
            filter_fn=np.isfinite)

        completed_dict = self.complete_dict(
            sparse_dict)
        array_result = np.zeros_like(X)
        for row_idx, row_dict in completed_dict.items():
            for col_idx, value in row_dict.items():
                array_result[row_idx, col_idx] = value
        array_result[observed_mask] = X[observed_mask]
        return array_result
