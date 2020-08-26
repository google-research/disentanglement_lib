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

"""Tests for util.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Dependency imports
from absl.testing import parameterized
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf


class UtilTest(parameterized.TestCase, tf.test.TestCase):

  def test_tfdata(self):
    ground_truth_data = dummy_data.DummyData()
    dataset = util.tf_data_set_from_ground_truth_data(ground_truth_data, 0)
    one_shot_iterator = dataset.make_one_shot_iterator()
    next_element = one_shot_iterator.get_next()
    with self.test_session() as sess:
      for _ in range(10):
        sess.run(next_element)


class StateSpaceAtomIndexTest(parameterized.TestCase, tf.test.TestCase):

  def test(self):
    features = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int64)
    state_space_atom_index = util.StateSpaceAtomIndex([2, 2], features)
    self.assertAllEqual(
        state_space_atom_index.features_to_index(features), list(range(4)))


if __name__ == '__main__':
  tf.test.main()
