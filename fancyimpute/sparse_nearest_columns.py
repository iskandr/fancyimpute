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

from scipy.sparse import dok_matrix


class SparseNearestColumns(object):
    """
    Fill in missing each missing row/column value by averaging across the
    k-nearest neighbors columns (taking into account missing data when
    computing column similarities and choosing which neighbors to inspect).

    Currently does not inherit from Solver since it expects sparse inputs in
    the form of nested dictionaries.
    """

    def __init__(
            self,
            k=5,
            min_weight_for_similarity=0.5,
            min_count_for_similarity=2):
        """
        Parameters
        ----------
        k : int, optional
            If omitted, then defaults to 5

        min_weight_for_similarity : float
            If sum of values in shared rows between two columns falls below this
            threhold then similarity can't be computed between those columns.

        min_count_for_similarity : int
            If number of overlapping rows between two columns falls below this
            threhold then similarity can't be computed between those columns.

        """
        self.k = k
        self.min_weight_for_similarity = min_weight_for_similarity
        self.min_count_for_similarity = min_count_for_similarity

    @classmethod
    def _collect_nested_keys(cls, nested_dictionaries):
        outer_key_list = list(sorted(nested_dictionaries.keys()))
        inner_key_set = set([])
        for k in outer_key_list:
            inner_key_set = inner_key_set.union(nested_dictionaries[k].keys())
        inner_key_list = list(sorted(inner_key_set))
        return outer_key_list, inner_key_list

    @classmethod
    def sparse_array_from_nested_dicts(cls, d, dtype=float):
        """
        TODO: Actually use sparse arrays instead of dictionaries!
        """
        outer_keys, inner_keys = cls._collect_nested_keys(d)
        outer_key_indices = {k: i for (i, k) in enumerate(outer_keys)}
        inner_key_indices = {k: i for (i, k) in enumerate(inner_keys)}

        n_rows = len(outer_keys)
        n_cols = len(inner_keys)
        shape = (n_rows, n_cols)
        result = dok_matrix(shape, dtype)
        for outer_key, sub_dictionary in d.items():
            i = outer_key_indices[outer_key]
            for inner_key, value in sub_dictionary.items():
                j = inner_key_indices[inner_key]
                result[i, j] = value
        return result

    def jacard_similarity_columns(self, nested_dictionaries):
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

    def complete(self, nested_dictionaries):
        """
        Keys of nested dictionaries can be arbitrary objects.
        """
        outer_keys, inner_keys = self._collect_nested_keys(nested_dictionaries)
        similarities, overlaps, weights = self.jacard_similarity_columns(
            nested_dictionaries)
        raise ValueError("Not yet implemented!")

