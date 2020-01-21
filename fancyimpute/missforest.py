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

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_array
import numpy as np

from .solver import Solver

class MissForest(Solver):
    def __init__(
            self,
            min_value=None,
            max_value=None,
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='auto',
            max_samples=None):
        """
        Parameters
        ----------
        n_estimators: integer, optional (default=10)

        max_depth: integer or None, optional (default=None)
            The maximum depth of the tree.
            If None, then nodes are expanded until all leaves are pure
            or until all leaves contain less than min_samples_split samples.

        min_samples_split: int, float, optional (default=2)
            The minimum number of samples required to split an internal node

        min_samples_leaf: int, float, optional (default=1)
             The minimum number of samples required to be at a leaf node.
             A split point at any depth will only be considered if it leaves
             at least min_samples_leaf training samples in each of the left and right branches.
             This may have the effect of smoothing the model, especially in regression.
        max_features: int, float, string or None, optional (default=”auto”)
            The number of features to consider when looking for the best split
            if int, then consider max_features features at each split.
            If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
            If “auto”, then max_features=n_features.
            If “sqrt”, then max_features=sqrt(n_features).
            If “log2”, then max_features=log2(n_features).
            If None, then max_features=n_features.

        max_samples: int or float, default=None
            If bootstrap is True, the number of samples to draw from X to train each base estimator.
            If None (default), then draw X.shape[0] samples.
            If int, then draw max_samples samples.
            If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0, 1)
        """
        self.coltype_dict = None
        self.mask_memo_dict = None
        self.sorted_col = None
        self.stop = False
        self.rf_reg = RandomForestRegressor(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            max_samples=max_samples,
                                            min_samples_split=min_samples_split)
        self.rf_cla = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            max_samples=max_samples,
                                            min_samples_split=min_samples_split)
        self.imp_continuous_index = None
        self.imp_categorical_index = None

        Solver.__init__(self,
            min_value=min_value,
            max_value=max_value)

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)
        self.sorted_col = self.sort_col(missing_mask)
        self.coltype_dict = self._judge_type(X)

        self.imp_continuous_index, self.imp_categorical_index = \
                self.get_type_index(missing_mask, self.coltype_dict)

        differ_categorical = float('inf')
        differ_continuous = float('inf')

        init_fill = X

        while self.stop is False:

            differ_categorical_old = differ_categorical
            differ_continuous_old = differ_continuous

            x_old_imp = init_fill

            x_new_imp = []

            for col in self.sorted_col:
                tmp = []
                if self.coltype_dict[col] is 'categorical':
                    model = self.rf_cla
                else:
                    model = self.rf_reg

                x_obs, y_obs, x_mis = self.split(init_fill, col, missing_mask)
                model.fit(x_obs, y_obs)
                y_mis = model.predict(x_mis)
                for ele in y_mis:
                    tmp.append(ele)
                    x_new_imp.append(ele)
                init_fill[:, col][missing_mask[:,col]] = tmp
            x_new_imp = np.asarray(x_new_imp)

            differ_continuous, differ_categorical = self._lose_func(x_new_imp, x_old_imp)
            if differ_continuous >= differ_continuous_old and differ_categorical >= differ_categorical_old:
                self.stop = True
        return init_fill

    def _lose_func(self, imp_new, imp_old):
        """
        Evaluation Method, mathematical concept are available at 'https://www.stu-zhouyc.com/iterForest/metrics'

        :param imputed_data_old: a dict like {'col name':[predicted value1,...],...}
                                        the dict contains original missing index which is part of the original data
                                        its the last estimated data
                                        accompany with brand-new imputed data, they are going to be evaluate.
        :return:
        """

        continuous_imp_new = imp_new[self.imp_continuous_index]
        continuous_imp_old = imp_old[self.imp_continuous_index]
        categorical_imp_new = imp_new[self.imp_categorical_index]
        categorical_imp_old = imp_old[self.imp_categorical_index]

        try:
            continuous_div = continuous_imp_new - continuous_imp_old
            continuous_div = continuous_div.dot(continuous_div)
            continuous_sum = continuous_imp_new.dot(continuous_imp_new)

            categorical_count = np.sum(categorical_imp_new == categorical_imp_old)
            categorical_var_len = len(categorical_imp_new)

        except:
            categorical_var_len = 0.01
            categorical_count = 0

            continuous_div = 0
            continuous_sum = 0.001

        if categorical_var_len is 0:
            categorical_differ = 0
        else:
            categorical_differ = categorical_count / categorical_var_len

        if continuous_sum is 0:
            continuous_differ = 0
        else:
            continuous_differ = continuous_div / continuous_sum
        return continuous_differ, categorical_differ

    def split(self, X, target_col, mask):
        col_mask = mask[:,target_col]
        nan_index = np.where(col_mask == True)
        not_nan_index = np.where(col_mask == False)

        contain_nan_rows = np.delete(X, not_nan_index, 0)
        no_contain_nan_rows = np.delete(X, nan_index, 0)

        train_X = np.delete(no_contain_nan_rows, target_col, 1)
        train_y = no_contain_nan_rows[:, target_col]
        test_X = np.delete(contain_nan_rows, target_col, 1)

        return train_X, train_y, test_X

    def _judge_type(self, X):
        coltype_dic = {}
        for col in range(X.shape[1]):
            col_val = X[:, col]
            nan_index = np.where(np.isnan(col_val))
            col_val = np.delete(col_val, nan_index)
            if len(np.unique(col_val)) <= 0.5*len(col_val):
                coltype_dic[col] = 'categotical'
            else:
                coltype_dic[col] = 'continuous'
        return coltype_dic

    def get_type_index(self, mask_all, col_type_dict):
        """
        get the index of every missing value, because the imputed array is 1D
        where the continuous and categorical index are needed.

        :param mask_all:
        :param col_type_dict:
        :return: double list
        """
        where_target = np.argwhere(mask_all == True)
        imp_categorical_index = []
        imp_continuous_index = []
        for index in where_target:
            col_type = col_type_dict[index[1]]
            if col_type is 'categotical':
                imp_categorical_index.append(index)
            elif col_type is 'continuous':
                imp_continuous_index.append(index)

        return imp_continuous_index, imp_categorical_index

    def sort_col(self, mask):
        """
        count various cols, the missing value wages,
        :param X: the original data matrix which is waiting to be imputed
        :return: col1, col2,.... colx, those cols has been sorted according its status of missing values
        """
        nan_index = np.where(mask == True)[1]
        unique = np.unique(nan_index)
        nan_index = list(nan_index)
        dict = {}
        for item in unique:
            count = nan_index.count(item)
            dict[item] = count
        tmp = sorted(dict.items(), key=lambda e: e[1], reverse=True)
        sort_index = []
        for item in tmp:
            sort_index.append(item[0])
        return sort_index











        

