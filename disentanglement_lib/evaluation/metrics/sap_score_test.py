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

"""Tests for sap_score.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import sap_score
import numpy as np


class SapScoreTest(absltest.TestCase):

  def test_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = sap_score.compute_sap(
        ground_truth_data, representation_function, random_state, None, 3000,
        3000, continuous_factors=True)
    self.assertBetween(scores["SAP_score"], 0.9, 1.0)

  def test_bad_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.zeros_like(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = sap_score.compute_sap(
        ground_truth_data, representation_function, random_state, None, 3000,
        3000, continuous_factors=True)
    self.assertBetween(scores["SAP_score"], 0.0, 0.2)

  def test_duplicated_latent_space(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = sap_score.compute_sap(
        ground_truth_data, representation_function, random_state, None, 3000,
        3000, continuous_factors=True)
    self.assertBetween(scores["SAP_score"], 0.0, 0.2)

if __name__ == "__main__":
  absltest.main()
