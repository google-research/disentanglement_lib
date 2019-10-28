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

"""Tests for factor_vae.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import factor_vae
import numpy as np


class FactorVaeTest(absltest.TestCase):

  def test_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    scores = factor_vae.compute_factor_vae(
        ground_truth_data, representation_function, random_state, None, 5, 3000,
        2000, 2500)
    self.assertBetween(scores["train_accuracy"], 0.9, 1.0)
    self.assertBetween(scores["eval_accuracy"], 0.9, 1.0)

  def test_bad_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = np.zeros_like
    random_state = np.random.RandomState(0)
    scores = factor_vae.compute_factor_vae(
        ground_truth_data, representation_function, random_state, None, 5, 3000,
        2000, 2500)
    self.assertBetween(scores["train_accuracy"], 0.0, 0.2)
    self.assertBetween(scores["eval_accuracy"], 0.0, 0.2)


if __name__ == "__main__":
  absltest.main()
