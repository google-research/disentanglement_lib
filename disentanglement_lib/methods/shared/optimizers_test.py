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

"""Tests for optimizer.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from disentanglement_lib.methods.shared import optimizers
from six.moves import range
import tensorflow.compat.v1 as tf

import gin.tf.external_configurables  # pylint: disable=unused-import
import gin.tf


def _make_vae_optimizer_configs():
  """Yield different vae_optimizer test configurations.

  Yields:
    A tuple containing a list of gin bindings, and the expected learning rate
    after 10 steps.
  """
  # Constant learning rate specified in the optimizer.
  bindings = [
      "vae_optimizer.optimizer_fn = @GradientDescentOptimizer",
      "GradientDescentOptimizer.learning_rate = 0.1",
  ]
  yield (bindings, 0.1)

  # Constant learning rate specified in vae_optimizer.
  bindings = [
      "vae_optimizer.optimizer_fn = @GradientDescentOptimizer",
      "vae_optimizer.learning_rate = 0.1",
  ]
  yield (bindings, 0.1)

  # Piecewise constant learning rate.
  bindings = [
      "vae_optimizer.optimizer_fn = @GradientDescentOptimizer",
      "vae_optimizer.learning_rate = @piecewise_constant",
      "piecewise_constant.boundaries = (3, 5)",
      "piecewise_constant.values = (0.2, 0.1, 0.01)",
  ]
  yield (bindings, 0.01)

  # Exponential decay learning rate.
  bindings = [
      "vae_optimizer.optimizer_fn = @GradientDescentOptimizer",
      "vae_optimizer.learning_rate = @exponential_decay",
      "exponential_decay.learning_rate = 0.1",
      "exponential_decay.decay_steps = 1",
      "exponential_decay.decay_rate = 0.9",
  ]
  yield (bindings, 0.03486784401)


class OptimizerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(list(_make_vae_optimizer_configs()))
  def test_vae_optimizer(self, gin_bindings, expected_learning_rate):
    gin.parse_config_files_and_bindings([], gin_bindings)

    with self.test_session():
      x = tf.Variable(0.0)
      y = tf.pow(x + 2.0, 2.0)
      global_step = tf.train.get_or_create_global_step()
      optimizer = optimizers.make_vae_optimizer()
      train_op = optimizer.minimize(loss=y, global_step=global_step)
      tf.global_variables_initializer().run()
      for it in range(10):
        self.evaluate(train_op)
        self.assertEqual(it + 1, self.evaluate(global_step))
      current_learning_rate = self.evaluate(optimizer._learning_rate_tensor)
      self.assertAlmostEqual(expected_learning_rate, current_learning_rate)

    gin.clear_config()


if __name__ == "__main__":
  tf.test.main()
