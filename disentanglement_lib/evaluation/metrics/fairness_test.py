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

"""Tests for fairness.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import fairness
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf


class FairnessTest(absltest.TestCase):

  def test_metric(self):
    gin.bind_parameter("predictor.predictor_fn",
                       utils.gradient_boosting_classifier)
    ground_truth_data = dummy_data.DummyData()
    def representation_function(x):
      return np.array(x, dtype=np.float64)[:, :, 0, 0]
    random_state = np.random.RandomState(0)
    _ = fairness.compute_fairness(ground_truth_data, representation_function,
                                  random_state, None, 1000, 1000)

  def test_inter_group_fairness(self):
    counts = np.array([[0, 100], [100, 100]])
    mean_fairness, max_fairness = fairness.inter_group_fairness(counts)
    # The overall distribution is 1/3 to 2/3.
    # The first sensitive class has a distribution of 0 to 1.
    # The second sensitive class has a distribution of 1/2 to 1/2.
    # The total variation distances are hence 1/3 and 1/6 respectively.
    # The mean fairness is hence 1/3*1/3 + 2/3*1/6 = 2/9.
    self.assertAlmostEqual(mean_fairness, 2. / 9.)
    self.assertAlmostEqual(max_fairness, 1 / 3.)

  def test_compute_scores_dict(self):
    scores = np.array([[0., 2., 4.], [4., 0., 8.], [2., 10., 0.]])
    results = fairness.compute_scores_dict(scores, "test")
    shouldbe = {
        # Single scores.
        "test:pred0:sens1": 2.,
        "test:pred0:sens2": 4.,
        "test:pred1:sens0": 4.,
        "test:pred1:sens2": 8.,
        "test:pred2:sens0": 2.,
        "test:pred2:sens1": 10.,
        # Column scores.
        "test:pred0:mean_sens": 3.,
        "test:pred0:max_sens": 4.,
        "test:pred1:mean_sens": 6.,
        "test:pred1:max_sens": 8.,
        "test:pred2:mean_sens": 6.,
        "test:pred2:max_sens": 10.,
        # Column scores.
        "test:sens0:mean_pred": 3.,
        "test:sens0:max_pred": 4.,
        "test:sens1:mean_pred": 6.,
        "test:sens1:max_pred": 10.,
        "test:sens2:mean_pred": 6.,
        "test:sens2:max_pred": 8.,
        # All scores.
        "test:mean_sens:mean_pred": 5.,
        "test:mean_sens:max_pred": 22. / 3.,
        "test:max_sens:mean_pred": 6.,
        "test:max_sens:max_pred": 10.,
        "test:mean_pred:mean_sens": 5.,
        "test:mean_pred:max_sens": 22. / 3.,
        "test:max_pred:mean_sens": 6.,
        "test:max_pred:max_sens": 10.,
    }
    self.assertDictEqual(shouldbe, results)


if __name__ == "__main__":
  absltest.main()
