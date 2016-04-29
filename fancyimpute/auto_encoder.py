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
from six.moves import range

from .neuralnet_helpers import make_network
from .common import masked_mae
from .solver import Solver


class AutoEncoder(Solver):
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
            recurrent_weight=0.5,
            n_burn_in_epochs=1,
            missing_input_noise_weight=0,
            output_history_size=25,
            patience_epochs=100,
            min_improvement=0.999,
            max_training_epochs=None,
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            verbose=True):
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.batch_size = batch_size
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.hidden_layer_sizes = hidden_layer_sizes
        self.recurrent_weight = recurrent_weight
        self.n_burn_in_epochs = n_burn_in_epochs
        self.missing_input_noise_weight = missing_input_noise_weight
        self.output_history_size = output_history_size
        self.patience_epochs = patience_epochs
        self.min_improvement = min_improvement
        self.max_training_epochs = max_training_epochs
        self.verbose = verbose

        # network and its input size get set on first call to complete()
        self.network = None
        self.network_input_size = None

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

    def _train_epoch(self, X, missing_mask):
        """
        Trains the network for one pass over the data,
        returns the network's predictions on the training data.
        """
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_with_missing_mask = np.hstack([X, missing_mask])
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X_with_missing_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            self.network.train_on_batch(batch_data, batch_data)
        return self.network.predict(X_with_missing_mask)

    def _get_training_params(self, n_samples):
        if not self.max_training_epochs:
            actual_batch_size = min(self.batch_size, n_samples)
            n_updates_per_epoch = int(np.ceil(n_samples / actual_batch_size))
            # heuristic of ~1M updates for each model
            max_training_epochs = int(
                np.ceil(0.5 * 10 ** 6 / n_updates_per_epoch))
            if self.verbose:
                print("[AutoEncoder] Max Epochs: %d" % max_training_epochs)
        else:
            max_training_epochs = self.max_training_epochs

        if not self.patience_epochs:
            patience_epochs = int(np.ceil(max_training_epochs / 100))
            if self.verbose:
                print(
                    ("[AutoEncoder] Default patience"
                     "(# epochs before improvement): %d") % (patience_epochs,))
        else:
            patience_epochs = self.patience_epochs

        return max_training_epochs, patience_epochs

    def solve(self, X, missing_mask):
        n_samples, n_features = X.shape

        if self.network_input_size != n_features:
            # create a network for each distinct input size
            self.network = self._create_fresh_network(n_features)
            self.network_input_size = n_features

        assert self.network is not None, \
            "Network should have been constructed but was found to be None"

        max_training_epochs, patience_epochs = self._get_training_params(
            n_samples)

        observed_mask = ~missing_mask

        best_error_seen = np.inf
        epochs_since_best_error = 0
        recent_predictions = deque([], maxlen=self.output_history_size)

        for epoch in range(max_training_epochs):
            X_pred = self._train_epoch(X=X, missing_mask=missing_mask)
            recent_predictions.append(X_pred)
            observed_mae = masked_mae(
                X_true=X,
                X_pred=X_pred,
                mask=observed_mask)

            if epoch == 0:
                best_error_seen = observed_mae
            elif observed_mae / best_error_seen < self.min_improvement:
                best_error_seen = observed_mae
                epochs_since_best_error = 0
            else:
                epochs_since_best_error += 1

            if self.verbose:
                print("[AutoEncoder] Epoch %d/%d Observed MAE=%f %s" % (
                    epoch + 1,
                    max_training_epochs,
                    observed_mae,
                    " *" if epochs_since_best_error == 0 else ""))
            if patience_epochs and epochs_since_best_error > patience_epochs:
                if self.verbose:
                    print(
                        "Patience exceeded at epoch %d (best MAE=%0.4f)" % (
                            epoch + 1,
                            best_error_seen))
                break

            # start updating the inputs with imputed values after
            # pre-specified number of epochs exceeded
            if epoch >= self.n_burn_in_epochs:
                old_weight = (1.0 - self.recurrent_weight)
                X[missing_mask] *= old_weight
                pred_missing = X_pred[missing_mask]
                X[missing_mask] += self.recurrent_weight * pred_missing
                if self.missing_input_noise_weight:
                    noise = np.random.randn(*pred_missing.shape)
                    X[missing_mask] += (
                        self.missing_input_noise_weight * noise)
        return np.mean(recent_predictions, axis=0)
