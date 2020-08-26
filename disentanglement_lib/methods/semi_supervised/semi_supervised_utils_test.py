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

"""Tests for the semi supervised utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.methods.semi_supervised import semi_supervised_utils  # pylint: disable=unused-import
from disentanglement_lib.methods.semi_supervised import semi_supervised_vae  # pylint: disable=unused-import
from disentanglement_lib.methods.semi_supervised import train_semi_supervised_lib as train_s2_lib
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

import gin.tf


class SemiSupervisedDataTest(parameterized.TestCase, tf.test.TestCase):

  def test_semi_supervised_data(self):
    num_labels = 1000
    gin.clear_config()
    gin_bindings = ["labeller.labeller_fn = @perfect_labeller"]
    gin.parse_config_files_and_bindings([], gin_bindings)
    ground_truth_data = dummy_data.DummyData()
    (sampled_observations, sampled_factors,
     _) = semi_supervised_utils.sample_supervised_data(
         0, ground_truth_data, num_labels)
    dataset = train_s2_lib.semi_supervised_dataset_from_ground_truth_data(
        ground_truth_data, num_labels, 0, sampled_observations,
        sampled_factors)
    one_shot_iterator = dataset.make_one_shot_iterator()
    next_element = one_shot_iterator.get_next()
    with self.test_session() as sess:
      for _ in range(1):
        elem = sess.run(next_element)
        self.assertEqual(elem[0].shape, (64, 64, 1))
        self.assertEqual(elem[1][0].shape, (64, 64, 1))
        self.assertLen(elem[1][1], 10)


class LabellerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((np.random.randint(0, 5, size=(100, 10)), 0.))
  def test_perfect_labeller(self, labels, target):

    ground_truth_data = dummy_data.DummyData()
    processed_labels, _ = semi_supervised_utils.perfect_labeller(
        labels, ground_truth_data, np.random.RandomState(0))
    test_value = np.sum(np.abs(processed_labels - labels))
    self.assertEqual(test_value, target)

  @parameterized.parameters(
      (np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64),
       np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64), 10),
      (np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64),
       np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64), 5),
      (np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64),
       np.array([0, 0, 1, 1, 2, 0, 0, 1, 1, 2], dtype=np.int64), 3),
      (np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64),
       np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int64), 2))
  def test_bin_labeller(self, labels, target, num_bins):
    labels = labels.reshape((1, 10))
    target = target.reshape((1, 10))
    ground_truth_data = dummy_data.DummyData()
    processed_labels, _ = semi_supervised_utils.bin_labeller(
        labels,
        ground_truth_data,
        np.random.RandomState(0),
        num_bins=num_bins)
    test_value = np.all(processed_labels == target)
    self.assertEqual(test_value, True)

  @parameterized.parameters((np.random.randint(0, 5, size=(10000, 10)), 7000.,
                             12000))
  def test_noisy_labeller(self, labels, target_low, target_high):

    ground_truth_data = dummy_data.DummyData()
    old_labels = labels.copy()
    processed_labels, _ = semi_supervised_utils.noisy_labeller(
        labels, ground_truth_data, np.random.RandomState(0), 0.1)
    index_equal = (processed_labels - old_labels).flatten()
    test_value = np.count_nonzero(index_equal)
    self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters((np.random.randint(0, 2, size=(10)), 2))
  def test_permuted_labeller(self, labels, num_factors):
    permuted = semi_supervised_utils.permute(labels, num_factors,
                                             np.random.RandomState(0))
    result = np.all(labels == permuted) or np.all(
        labels == np.logical_not(permuted))

    self.assertEqual(result, True)

  @parameterized.parameters((np.random.randint(0, 5, size=(10000, 10)), 3))
  def test_filter_factors(self, labels, target):
    new_labels, _ = semi_supervised_utils.filter_factors(
        labels, target, np.random.RandomState(0))
    self.assertEqual(new_labels.shape[0], labels.shape[0])
    self.assertEqual(new_labels.shape[1], target)


if __name__ == "__main__":
  tf.test.main()
