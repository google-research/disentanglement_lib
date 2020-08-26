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

"""Tests for unified_scores.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import strong_downstream_task
from disentanglement_lib.evaluation.metrics import utils
import numpy as np

import gin.tf


def _identity_discretizer(target, num_bins):
  del num_bins
  return target


class StrongDownstreamTaskTest(absltest.TestCase):

  def test_intervene(self):
    ground_truth_data = dummy_data.DummyData()

    random_state = np.random.RandomState(0)

    ys_train = ground_truth_data.sample_factors(1000, random_state)
    ys_test = ground_truth_data.sample_factors(1000, random_state)
    num_factors = ys_train.shape[1]
    for i in range(num_factors):
      (y_train_int, y_test_int, interv_factor,
       factor_interv_train) = strong_downstream_task.intervene(
           ys_train.copy(), ys_test.copy(), i, num_factors, ground_truth_data)
      assert interv_factor != i, "Wrong factor interevened on."
      assert (y_train_int[:, interv_factor] == factor_interv_train
             ).all(), "Training set not intervened on."
      assert (y_test_int[:, interv_factor] != factor_interv_train
             ).all(), "Training set not intervened on."

  def test_task(self):
    gin.bind_parameter("predictor.predictor_fn",
                       utils.gradient_boosting_classifier)
    gin.bind_parameter("strong_downstream_task.num_train",
                       [1000])
    gin.bind_parameter("strong_downstream_task.num_test",
                       1000)
    gin.bind_parameter("strong_downstream_task.n_experiment",
                       2)
    ground_truth_data = dummy_data.DummyData()
    def representation_function(x):
      return np.array(x, dtype=np.float64)[:, :, 0, 0]
    random_state = np.random.RandomState(0)
    scores = strong_downstream_task.compute_strong_downstream_task(
        ground_truth_data, representation_function, random_state,
        artifact_dir=None)
    self.assertBetween(scores["1000:mean_strong_test_accuracy"], 0.0, 0.3)

if __name__ == "__main__":
  absltest.main()
