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

from .solver import Solver


class SimpleFill(Solver):
    def __init__(self, fill_method="mean", min_value=None, max_value=None):
        """
        Possible values for fill_method:
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian noise according to mean/std of column
        """
        Solver.__init__(
            self,
            fill_method=fill_method,
            min_value=None,
            max_value=None)

    def solve(self, X, missing_mask):
        """
        Since X is given to us already filled, just return it.
        """
        return X
