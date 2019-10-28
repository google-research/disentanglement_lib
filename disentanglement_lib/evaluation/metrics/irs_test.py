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

"""Tests for irs.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import irs
import numpy as np
import gin.tf


def _identity_discretizer(target, num_bins):
  del num_bins
  return target


class IrsTest(absltest.TestCase):

  def test_metric(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)

    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = irs.compute_irs(ground_truth_data, representation_function,
                             random_state, None, 0.99, 3000, 3000)
    self.assertBetween(scores["IRS"], 0.9, 1.0)

  def test_bad_metric(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)

    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.zeros_like(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = irs.compute_irs(ground_truth_data, representation_function,
                             random_state, None, 0.99, 3000, 3000)
    self.assertBetween(scores["IRS"], 0.0, 0.1)

  def test_drop_constant_dims(self):
    random_state = np.random.RandomState(0)
    ys = random_state.normal(0.0, 1.0, (100, 100))
    ys[0, :] = 1.
    ys[-1, :] = 0.
    active_ys = irs._drop_constant_dims(ys)
    np.testing.assert_array_equal(active_ys, ys[1:-1])


if __name__ == "__main__":
  absltest.main()
