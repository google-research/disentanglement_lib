# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
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

"""Library of Keras layers used to build relational neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class RelationalLayer(tf.keras.layers.Layer):
  """Implements a single relational layer.

  This layer is based on equation (1) of https://arxiv.org/pdf/1706.01427.pdf
  and applies (in order) three layers to the input: PairwiseEdgeEmbeddings,
  the specified edge_layer and the reduce_layer. The input tensor and output
  tensor of this layer are of shape (batch_size, num_nodes, num_dims).
  """

  def __init__(self, edge_layer, reduce_layer):
    """Constructs a RelationalLayer.

    Args:
      edge_layer: tf.keras.layers.Layer that is applied to the edge embeddings.
        Should accept a (batch_size, num_nodes, num_nodes, 2*num_dims)-sized
        tensor (where the node embeddings are stacked along the last axis)
        and should return a (batch_size, num_nodes, num_nodes, num_dims)-sized
        tensor.
      reduce_layer: tf.keras.layers.Layer that is used to reduce the edge
        embeddings back to node embeddings. Should accept a
        (batch_size, num_nodes, num_nodes, num_dims)-sized tensor and return a
        (batch_size, num_nodes, num_dims)-sized tensor.
    """
    self._pairwise_edge_embeddings_layer = PairwiseEdgeEmbeddings()
    self.edge_layer = edge_layer
    self.reduce_layer = reduce_layer
    super(RelationalLayer, self).__init__()

  def call(self, inputs, **kwargs):
    # Create edge embedding by concatenating the node embeddings.
    x = self._pairwise_edge_embeddings_layer(inputs, **kwargs)
    # Update edge embeddings using edge layer.
    x = self.edge_layer(x, **kwargs)
    # Reduce back to node embeddings.
    outputs = self.reduce_layer(x, **kwargs)
    return outputs


class PairwiseEdgeEmbeddings(tf.keras.layers.Layer):
  """Creates pairwise edge embeddings from node embeddings."""

  def call(self, inputs, **kwargs):
    num_nodes = inputs.get_shape().as_list()[-2]
    return tf.concat(
        (repeat(tf.expand_dims(inputs, -2), num_nodes, -2),
         repeat(tf.expand_dims(inputs, -3), num_nodes, -3)),
        axis=-1)


def repeat(tensor, num, axis):
  """Repeats tensor num times along the specified axis."""
  multiples = [1] * tensor.get_shape().ndims
  multiples[axis] = num
  return tf.tile(tensor, multiples)


def positional_encoding_like(tensor, positional_encoding_axis=-2,
                             value_axis=-1):
  """Creates positional encoding matching the provided tensor.

  Let each slice along the last axis of the tensor be a row. This function
  computes the index of each row with respect to the specified
  positional_encoding_axis and returns this index using a one-hot embedding.

  The resulting tensor has the same shape as the provided tensor except for the
  value_axis dimension. That dimension contains a one-hot encoding of the
  positional_encoding_axis, i.e., each slice along value_axis and the
  positional_encoding_axis corresponds to the identity matrix.

  Args:
    tensor: Input tensor.
    positional_encoding_axis: Integer with the axis to encode.
    value_axis: Integer with axis where to one-hot encode the
      positional_encoding_axis.

  Returns:
    Positional encoding tensor of the same dtype as tensor.
  """
  # First, create identity matrix for one-hot embedding.
  shape = tensor.get_shape()
  num_values = shape.as_list()[positional_encoding_axis]
  result = tf.eye(num_values, dtype=tensor.dtype)
  # Second, reshape to proper dimensionality.
  new_shape = [1] * shape.ndims
  new_shape[positional_encoding_axis] = num_values
  new_shape[value_axis] = num_values
  result = tf.reshape(result, new_shape)
  # Third, broadcast to final shape.
  multiplier = shape.as_list()
  multiplier[positional_encoding_axis] = 1
  multiplier[value_axis] = 1
  if not shape.is_fully_defined():
    dynamic_shape = tf.unstack(tf.shape(tensor))
    multiplier = [
        (dynamic_shape[i] if n is None else n) for i, n in enumerate(multiplier)
    ]
  return tf.tile(result, multiplier)


class AddPositionalEncoding(tf.keras.layers.Layer):
  """Adds positional encoding to embedding based on position along axis.

  Consider a tensor of shape (batch_size, num_pos, num_dim). This layer will
  return a tensor of shape (batch_size, num_pos, num_dim + num_pos) where
  the slice [:, :, :num_dim] contains the original data while each slice
  [i, :, num_dim:] is the identity matrix. The idea is that this adds extra
  dimensions along axis -1 that encode the position of the element along
  axis -2.
  """

  def __init__(self, positional_encoding_axis=-2, embedding_axis=-1):
    self.positional_encoding_axis = positional_encoding_axis
    self.embedding_axis = embedding_axis
    super(AddPositionalEncoding, self).__init__()

  def call(self, inputs, **kwargs):
    onehot_embedding = positional_encoding_like(
        inputs,
        positional_encoding_axis=self.positional_encoding_axis,
        value_axis=self.embedding_axis)
    return tf.concat([inputs, onehot_embedding], axis=self.embedding_axis)


class StackAnswers(tf.keras.layers.Layer):
  """Concatenates each each answer with the context and stacks them."""

  def __init__(self, answer_axis=-2, stack_axis=-3):
    super(StackAnswers, self).__init__()
    self.answer_axis = answer_axis
    self.stack_axis = stack_axis

  def call(self, inputs, **kwargs):
    context, answers = inputs
    num_answers = answers.get_shape().as_list()[self.answer_axis]
    answer_blocks = []
    for i in range(num_answers):
      ith_answer = tf.gather(answers, [i], axis=self.answer_axis)
      ith_answer_block = tf.concat([context, ith_answer], axis=self.answer_axis)
      answer_blocks.append(ith_answer_block)
    return tf.stack(answer_blocks, axis=self.stack_axis)


class MultiDimBatchApply(tf.keras.layers.Layer):
  """Applies layer for each element in a multi-dimensional batch."""

  def __init__(self, layer, num_dims_to_keep=3):
    """Constructs a MultiDimBatchApply.

    Args:
      layer: tf.keras.layers.Layer to apply.
      num_dims_to_keep: Integer with number of dimensions to provide to the
        layer. The dimensions 0:-num_dims_to_keep correspond to the multi
          dimensional mini batch, i.e., the layer is applied independently for
          each of these elements.
    """
    self.layer = layer
    self.num_dims_to_keep = num_dims_to_keep
    super(MultiDimBatchApply, self).__init__()

  def call(self, inputs, **kwargs):
    shape = inputs.get_shape().as_list()
    collapsed_shape = [-1] + shape[-self.num_dims_to_keep:]
    inputs = tf.reshape(inputs, collapsed_shape)
    # Apply the layer.
    output = self.layer(inputs, **kwargs)
    active_shape = output.get_shape().as_list()[1:]
    output_shape = [-1] + shape[1:-self.num_dims_to_keep] + active_shape
    return tf.reshape(output, output_shape)
