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

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal


class KerasMatrixFactorizer(Layer):
    def __init__(self, rank, input_dim_i, input_dim_j, embeddings_regularizer=None, use_bias=True, **kwargs):
        self.rank = rank
        self.input_dim_i = input_dim_i
        self.input_dim_j = input_dim_j
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.use_bias = use_bias
        super(KerasMatrixFactorizer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.i_embedding = self.add_weight(
            shape=(self.input_dim_i, self.rank),
            initializer=RandomNormal(mean=0.0, stddev=1 / np.sqrt(self.rank)),
            name="i_embedding",
            regularizer=self.embeddings_regularizer,
        )
        self.j_embedding = self.add_weight(
            shape=(self.input_dim_j, self.rank),
            initializer=RandomNormal(mean=0.0, stddev=1 / np.sqrt(self.rank)),
            name="j_embedding",
            regularizer=self.embeddings_regularizer,
        )
        if self.use_bias:
            self.i_bias = self.add_weight(shape=(self.input_dim_i, 1), initializer="zeros", name="i_bias")
            self.j_bias = self.add_weight(shape=(self.input_dim_j, 1), initializer="zeros", name="j_bias")
            self.constant = self.add_weight(
                shape=(1, 1),
                initializer="zeros",
                name="constant",
            )

        self.built = True
        super(KerasMatrixFactorizer, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != "int32":
            inputs = K.cast(inputs, "int32")
        # get the embeddings
        i = inputs[:, 0]  # by convention
        j = inputs[:, 1]
        i_embedding = K.gather(self.i_embedding, i)
        j_embedding = K.gather(self.j_embedding, j)
        # <i_embed, j_embed> + i_bias + j_bias + constant
        out = K.batch_dot(i_embedding, j_embedding, axes=[1, 1])
        if self.use_bias:
            i_bias = K.gather(self.i_bias, i)
            j_bias = K.gather(self.j_bias, j)
            out += i_bias + j_bias + self.constant
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
