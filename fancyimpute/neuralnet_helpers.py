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
from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.regularizers import l1l2


def make_reconstruction_loss(n_features, mask_indicates_missing_values=False):

    def reconstruction_loss(input_and_mask, y_pred):
        X_values = input_and_mask[:, :n_features]
        X_values.name = "$X_values"

        if mask_indicates_missing_values:
            missing_mask = input_and_mask[:, n_features:]
            missing_mask.name = "$missing_mask"
            observed_mask = 1 - missing_mask
        else:
            observed_mask = input_and_mask[:, n_features:]
        observed_mask.name = "$observed_mask"

        X_values_observed = X_values * observed_mask
        X_values_observed.name = "$X_values_observed"

        pred_observed = y_pred * observed_mask
        pred_observed.name = "$y_pred_observed"

        return mse(y_true=X_values_observed, y_pred=pred_observed)
    return reconstruction_loss


def make_network(
        n_dims,
        output_activation="linear",
        hidden_activation="relu",
        hidden_layer_sizes=None,
        dropout_probability=0,
        optimizer="adam",
        init="glorot_normal",
        l1_penalty=0,
        l2_penalty=0):
    if not hidden_layer_sizes:
        # start with a layer larger than the input vector and its
        # mask of missing values and then transform down to a layer
        # which is smaller than the input -- a bottleneck to force
        # generalization
        hidden_layer_sizes = [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]
        print("Hidden layer sizes: %s" % (hidden_layer_sizes,))

    nn = Sequential()
    first_layer_size = hidden_layer_sizes[0]
    nn.add(Dense(
        first_layer_size,
        input_dim=2 * n_dims,
        activation=hidden_activation,
        W_regularizer=l1l2(l1_penalty, l2_penalty),
        init=init))
    nn.add(Dropout(dropout_probability))

    for layer_size in hidden_layer_sizes[1:]:
        nn.add(Dense(
            layer_size,
            activation=hidden_activation,
            W_regularizer=l1l2(l1_penalty, l2_penalty),
            init=init))
        nn.add(Dropout(dropout_probability))
    nn.add(
        Dense(
            n_dims,
            activation=output_activation,
            W_regularizer=l1l2(l1_penalty, l2_penalty),
            init=init))
    loss_function = make_reconstruction_loss(
        n_dims,
        mask_indicates_missing_values=True)
    nn.compile(optimizer=optimizer, loss=loss_function)
    return nn
