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
        optimizer="rmsprop",
        l1_penalty=0,
        l2_penalty=0):
    if not hidden_layer_sizes:
        # start with a layer larger than the input vector and its
        # mask of missing values and then transform down to a layer
        # which is smaller than the input -- a bottleneck to force
        # generalization
        hidden_layer_sizes = [
            8 * n_dims,
            2 * n_dims,
            int(np.ceil(0.5 * n_dims)),
        ]
        print("Hidden layer sizes: %s" % (hidden_layer_sizes,))

    nn = Sequential()
    first_layer_size = hidden_layer_sizes[0]
    nn.add(Dense(
        first_layer_size,
        input_dim=2 * n_dims,
        activation=hidden_activation,
        W_regularizer=l1l2(l1_penalty, l2_penalty)))
    nn.add(Dropout(dropout_probability))

    for layer_size in hidden_layer_sizes[1:]:
        nn.add(Dense(
            layer_size,
            activation=hidden_activation,
            W_regularizer=l1l2(l1_penalty, l2_penalty)))
        nn.add(Dropout(dropout_probability))
    nn.add(
        Dense(
            n_dims,
            activation=output_activation,
            W_regularizer=l1l2(l1_penalty, l2_penalty)))
    loss_function = make_reconstruction_loss(
        n_dims,
        mask_indicates_missing_values=True)
    nn.compile(optimizer=optimizer, loss=loss_function)
    return nn


def train_network(X, missing_mask, network, n_training_epochs, batch_size):
    n_samples, n_features = X.shape
    print("Training network %s with %d samples and %d features" % (
        network,
        n_samples,
        n_features))
    indices = np.arange(n_samples)
    combined_data = np.hstack([X, missing_mask])
    for epoch in range(n_training_epochs):
        np.random.shuffle(indices)
        n_batches = n_samples // batch_size
        for batch_idx in range(n_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = (batch_idx + 1) * batch_size
            batch_indices = indices[batch_start_idx:batch_end_idx]
            X_batch = X[batch_indices, :]
            combined_batch = combined_data[batch_indices, :]
            print("X_combined.shape = %s, X_value.shapes=%s" % (
                combined_batch.shape,
                X_batch.shape))
            network.train_on_batch(
                X=combined_batch,
                y=combined_batch)


class AutoEncoder(object):
    """
    Neural network which takes as an input a vector of feature values and a
    binary mask of indicating which features are missing. It's trained
    on reconstructing the non-missing values and hopefully achieves
    generalization due to a "bottleneck" hidden layer that is smaller than
    the input size.
    """

    def __init__(
            self,
            hidden_activation="tanh",
            output_activation="linear",
            hidden_layer_sizes=None,
            optimizer="rmsprop",
            dropout_probability=0,
            batch_size=32,
            l1_penalty=0,
            l2_penalty=0,
            n_training_epochs=None,
            verbose=False):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.batch_size = batch_size
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.n_training_epochs = n_training_epochs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.verbose = verbose

        # network and its input size get set on first call to complete()
        self.network = None
        self.network_input_size = None

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def complete(self, X, verbose=True):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """
        self._check_input(X)
        (n_samples, n_features) = X.shape

        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)

        X = X.copy()
        # replace NaN's with 0
        X[missing_mask] = 0

        if self.network_input_size != n_features:
            # create a network for each distinct input size
            self.network = make_network(
                n_dims=n_features,
                output_activation=self.output_activation,
                hidden_activation=self.hidden_activation,
                hidden_layer_sizes=self.hidden_layer_sizes,
                dropout_probability=self.dropout_probability,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                optimizer=self.optimizer)
        assert self.network is not None, \
            "Network should have been constructed but was found to be None"
        if not self.n_training_epochs:
            actual_batch_size = min(self.batch_size, n_samples)
            n_updates_per_epoch = int(np.ceil(n_samples / actual_batch_size))
            # heuristic of ~1M updates for each model
            epochs = int(np.ceil(0.5 * 10 ** 6 / n_updates_per_epoch))
            print("Epochs: %d" % epochs)
        else:
            epochs = self.n_training_epochs
        X_with_observed_mask = np.hstack([X, missing_mask])

        self.network.fit(
            X=X_with_observed_mask,
            y=X_with_observed_mask,
            nb_epoch=epochs,
            verbose=self.verbose)
        return self.network.predict(X_with_observed_mask)
