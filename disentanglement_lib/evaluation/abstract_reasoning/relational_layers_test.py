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

"""Tests for relational_layers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.evaluation.abstract_reasoning import relational_layers
import numpy as np
import tensorflow.compat.v1 as tf


def _create_positional_encoding_matrices():
  """Shared input/output pair for the positional encoding tests."""
  input_array = np.arange(24, dtype=np.float64).reshape((1, 4, 3, 2))
  output_array = np.eye(4)
  output_array = np.repeat(np.expand_dims(output_array, -1), 2, axis=-1)
  output_array = np.expand_dims(output_array, 0)
  return input_array, output_array


class RelationalLayersTest(tf.test.TestCase):

  def test_repeat_for_tensor(self):
    a = np.arange(24).reshape((1, 4, 3, 2))
    shouldbe = np.concatenate([a] * 3, axis=-2)
    result = self.evaluate(relational_layers.repeat(tf.constant(a), 3, axis=-2))
    self.assertAllClose(shouldbe, result)

  def test_pairwise_edge_embeddings_for_tensor(self):
    a = np.array([[[1], [2]]])
    shouldbe = np.array([[[[1, 1], [1, 2]], [[2, 1], [2, 2]]]])
    layer = relational_layers.PairwiseEdgeEmbeddings()
    result = self.evaluate(layer(tf.constant(a)))
    self.assertAllClose(shouldbe, result)

  def test_relational_layer_for_tensor(self):
    a = np.array([[[1], [2]]])
    shouldbe = np.array([[[2, 3], [4, 3]]])
    layer = relational_layers.RelationalLayer(
        tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2)))
    result = self.evaluate(layer(tf.constant(a)))
    self.assertAllClose(shouldbe, result)

  def test_positional_encoding_like_for_static_shape_tensor(self):
    value, shouldbe = _create_positional_encoding_matrices()
    a = tf.constant(value)
    output_tensor = relational_layers.positional_encoding_like(a, -3, -2)
    result = self.evaluate(output_tensor)
    self.assertEqual((1, 4, 4, 2), result.shape)
    self.assertAllClose(shouldbe, result)

  def test_positional_encoding_like_for_dynamic_shape_tensor(self):
    value, shouldbe = _create_positional_encoding_matrices()
    a = tf.placeholder(tf.float32, shape=(None, 4, 3, 2))
    output_tensor = relational_layers.positional_encoding_like(a, -3, -2)
    # Check the static shape.
    self.assertEqual([None, 4, 4, 2], output_tensor.get_shape().as_list())
    # Check the solution.
    with self.session() as sess:
      result = sess.run(output_tensor, feed_dict={a: value})
    self.assertAllClose(shouldbe, result)

  def test_add_positional_encoding_layer_for_tensor(self):
    value, shouldbe_positional = _create_positional_encoding_matrices()
    shouldbe = np.concatenate([value, shouldbe_positional], axis=-2)
    a = tf.constant(value)
    output_tensor = relational_layers.AddPositionalEncoding(-3, -2)(a)
    result = self.evaluate(output_tensor)
    self.assertAllClose(shouldbe, result)

  def test_stack_answers_for_tensors(self):
    # Tensors used for testing.
    context = np.arange(24).reshape((2, 3, 4))
    answers = np.arange(24, 48).reshape((2, 3, 4))
    # Compute the correct solutions.
    results = []
    for i in range(answers.shape[-1]):
      results.append(
          np.concatenate([context, answers[:, :, i:(i + 1)]], axis=-1))
    shouldbe = np.stack(results, axis=-2)
    # Compute the solution based on the layer.
    layer = relational_layers.StackAnswers(answer_axis=-1, stack_axis=-2)
    result = self.evaluate(layer([tf.constant(context), tf.constant(answers)]))
    # Check that they are the same.
    self.assertAllClose(shouldbe, result)

  def test_multi_dim_batch_apply_for_tensors(self):
    # Tensors used for testing.
    input_tensor = np.arange(24).reshape((2, 3, 4))
    kernel = np.arange(24, 36).reshape((4, 3))
    # Compute the correct solutions.
    shouldbe = np.matmul(input_tensor, kernel)
    # Compute the solution based on the layer.
    layer = relational_layers.MultiDimBatchApply(
        tf.keras.layers.Lambda(lambda x: tf.matmul(x, tf.constant(kernel))),
        num_dims_to_keep=1)
    result = self.evaluate(layer(tf.constant(input_tensor)))
    # Check that they are the same.
    self.assertAllClose(shouldbe, result)


if __name__ == '__main__':
  tf.test.main()
