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
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.models import Model
from sklearn.utils import shuffle, check_array

from .common import import_from
from .scaler import Scaler
from .keras_models import KerasMatrixFactorizer
from .solver import Solver


class MatrixFactorization(Solver):
    """
    Given an incomplete (m,n) matrix X, factorize it into
    U, V where U.shape = (m, k) and V.shape = (k, n).

    The U, V are found by minimizing the difference between U.dot.V and
    X at the observed entries along with a sparsity penalty for U and an
    L2 penalty for V.
    """

    def __init__(
            self,
            rank=10,
            learning_rate=0.001,
            patience=5,
            l2_penalty=1e-5,
            use_bias=True,
            min_improvement=0.001,
            optimization_algorithm="nadam",
            loss='mse',
            validation_frac=0.1,
            min_value=None,
            max_value=None,
            normalizer=Scaler(),
            verbose=True):
        Solver.__init__(
            self,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer)
        self.rank = rank
        self.learning_rate = learning_rate
        self.patience = patience
        self.l2_penalty = l2_penalty
        self.use_bias = use_bias
        self.optimization_algorithm = optimization_algorithm
        self.loss = loss
        self.validation_frac = validation_frac
        self.min_improvement = min_improvement
        self.normalizer = normalizer
        self.verbose = verbose

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)

        # shape data to fit into keras model
        (n_samples, n_features) = X.shape
        observed_mask = ~missing_mask
        missing_mask_flat = missing_mask.flatten()
        observed_mask_flat = observed_mask.flatten()

        columns, rows = np.meshgrid(np.arange(n_features), np.arange(n_samples))

        # training data
        i_tr = rows.flatten()[observed_mask_flat]
        j_tr = columns.flatten()[observed_mask_flat]
        ij_tr = np.vstack([i_tr, j_tr]).T  # input to factorizer
        y_tr = X.flatten()[observed_mask_flat]  # output of factorizer
        ij_tr, y_tr = shuffle(ij_tr, y_tr)

        # make a keras model
        main_input = Input(shape=(2,), dtype='int32')
        embed = KerasMatrixFactorizer(
            rank=self.rank,
            input_dim_i=n_samples,
            input_dim_j=n_features,
            embeddings_regularizer=regularizers.l2(self.l2_penalty)
        )(main_input)
        model = Model(inputs=main_input, outputs=embed)
        optimizer = import_from(
            'keras.optimizers', self.optimization_algorithm
        )(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.loss)
        callbacks = [EarlyStopping(patience=self.patience, min_delta=self.min_improvement)]
        model.fit(
            ij_tr,
            y_tr,
            batch_size=int(len(y_tr) * (1 - self.validation_frac)),
            epochs=10000,
            validation_split=self.validation_frac,
            callbacks=callbacks,
            shuffle=True,
            verbose=self.verbose
        )

        # reassemble the original X
        i_ts = rows.flatten()[missing_mask_flat]
        j_ts = columns.flatten()[missing_mask_flat]
        ij_ts = np.vstack([i_ts, j_ts]).T  # input to factorizer
        X[i_ts, j_ts] = model.predict(ij_ts).T[0]

        return X
