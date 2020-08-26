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

"""Tests for the weakly-supervised methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from disentanglement_lib.methods.weak import weak_vae  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf


class WeakVaeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (np.zeros([64, 10]),
       np.zeros([64, 10]),
       np.ones([64, 10]),
       np.ones([64, 10]),
       np.concatenate((np.zeros([64, 5]), np.ones([64, 5])), axis=1),
       np.concatenate((np.ones([64, 5]), np.zeros([64, 5])), axis=1)),
      (np.array([[1, 1]]),
       np.array([[1, 1]]),
       np.array([[0, 0]]),
       np.array([[0, 0]]),
       np.array([[0, 0.1]]),
       np.array([[0, 1]]))
      )
  def test_aggregate_argmax(self, z_mean, z_logvar, new_mean, new_log_var,
                            kl_per_point, target):

    mean_tf = tf.convert_to_tensor(z_mean, dtype=np.float32)
    logvar_tf = tf.convert_to_tensor(z_logvar, dtype=np.float32)
    new_mean_tf = tf.convert_to_tensor(new_mean, dtype=np.float32)
    new_log_var_tf = tf.convert_to_tensor(new_log_var, dtype=np.float32)
    kl_per_point_tf = tf.convert_to_tensor(kl_per_point, dtype=np.float32)
    with self.session() as sess:
      test_value = sess.run(weak_vae.aggregate_argmax(
          mean_tf, logvar_tf, new_mean_tf, new_log_var_tf, None,
          kl_per_point_tf))
      self.assertEqual((test_value[0] == target).all(), True)
      self.assertEqual((test_value[1] == target).all(), True)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()

