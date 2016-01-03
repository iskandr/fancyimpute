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
from collections import deque

import numpy as np

from keras.objectives import mse
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.regularizers import l1l2

from .common import masked_mae


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
            optimizer="adam",
            dropout_probability=0,
            batch_size=32,
            l1_penalty=0,
            l2_penalty=0,
            verbose=True):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.batch_size = batch_size
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
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

    def _create_fresh_network(self, n_features):
        return make_network(
            n_dims=n_features,
            output_activation=self.output_activation,
            hidden_activation=self.hidden_activation,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout_probability=self.dropout_probability,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            optimizer=self.optimizer)

    def _train_epoch(self, X, missing_mask, X_with_missing_mask=None):
        """
        Trains the network for one pass over the data,
        returns the network's predictions on the training data.
        """
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / self.batch_size))
        if X_with_missing_mask is None:
            X_with_missing_mask = np.hstack([X, missing_mask])
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X_with_missing_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            self.network.train_on_batch(
                X=batch_data,
                y=batch_data)
        return self.network.predict(X_with_missing_mask)

    def multiple_imputations(
            self,
            X,
            X_complete=None,
            hallucination_weight=0.5,
            patience_epochs=None,
            min_improvement=0.995,
            max_training_epochs=None,
            n_predictions=100,
            missing_mask=None):
        self._check_input(X)
        (n_samples, n_features) = X.shape

        if missing_mask is None:
            missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)

        if self.network_input_size != n_features:
            # create a network for each distinct input size
            self.network = self._create_fresh_network(n_features)
            self.network_input_size = n_features

        assert self.network is not None, \
            "Network should have been constructed but was found to be None"

        if not max_training_epochs:
            actual_batch_size = min(self.batch_size, n_samples)
            n_updates_per_epoch = int(np.ceil(n_samples / actual_batch_size))
            # heuristic of ~1M updates for each model
            max_training_epochs = int(np.ceil(10 ** 6 / n_updates_per_epoch))
            if self.verbose:
                print("Max Epochs: %d" % max_training_epochs)

        if not patience_epochs:
            patience_epochs = int(np.ceil(max_training_epochs / 100))
            if self.verbose:
                print("Default patience (# epochs before improvement): %d" % (
                    patience_epochs,))

        n_predictions = min(max_training_epochs, n_predictions)

        X = X.copy()
        # replace NaN's with 0
        X[missing_mask] = 0

        observed_mask = ~missing_mask

        recent_errors = []
        best_error_seen = np.inf
        best_error_seen_median = np.inf
        epochs_since_best_error = 0
        recent_predictions = deque([], maxlen=n_predictions)

        for epoch in range(max_training_epochs):
            X_pred = self._train_epoch(X=X, missing_mask=missing_mask)
            recent_predictions.append(X_pred)
            observed_mae = masked_mae(
                X_true=X,
                X_pred=X_pred,
                mask=observed_mask)

            if X_complete is not None:
                missing_mae = masked_mae(
                    X_true=X_complete,
                    X_pred=X_pred,
                    mask=missing_mask)
                print(
                    ("Epoch %d/%d "
                     "Training MAE=%0.4f "
                     "Test MAE=%0.4f") % (
                        epoch + 1,
                        max_training_epochs,
                        observed_mae,
                        missing_mae))
            if epoch == 0:
                best_error_seen = observed_mae
                recent_errors = [observed_mae]
            elif observed_mae / best_error_seen_median < min_improvement:
                best_error_seen = observed_mae
                best_error_seen_median = np.median(recent_errors)
                recent_errors = [observed_mae]
                epochs_since_best_error = 0
            else:
                epochs_since_best_error += 1
                recent_errors.append(observed_mae)

            if patience_epochs and epochs_since_best_error > patience_epochs:
                if self.verbose:
                    print(
                        "Patience exceeded at epoch %d (best MAE=%0.4f)" % (
                            epoch + 1,
                            best_error_seen))
                break

            if hallucination_weight:
                X[missing_mask] = (1.0 - hallucination_weight) * X[missing_mask]
                X[missing_mask] += hallucination_weight * X_pred[missing_mask]
        return recent_predictions

    def complete(
            self,
            X,
            X_complete=None,
            hallucination_weight=0.05,
            patience_epochs=None,
            min_improvement=0.995,
            max_training_epochs=None,
            n_averaged_predictions=100):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """

        missing_mask = np.isnan(X)
        X_result = X.copy()

        recent_predictions = self.multiple_imputations(
            X=X,
            X_complete=X_complete,
            hallucination_weight=hallucination_weight,
            patience_epochs=patience_epochs,
            min_improvement=min_improvement,
            max_training_epochs=max_training_epochs,
            n_predictions=n_averaged_predictions,
            missing_mask=missing_mask)

        X_mean_pred = np.mean(recent_predictions, axis=0)
        X_result[missing_mask] = X_mean_pred[missing_mask]
        return X_result
