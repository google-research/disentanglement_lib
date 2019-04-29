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

"""Tests for utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.evaluation.metrics import utils
import numpy as np


class UtilsTest(absltest.TestCase):

  def test_histogram_discretizer(self):
    # Input of 2D samples.
    target = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                       [0.6, .5, .4, .3, .2, .1]])
    result = utils._histogram_discretize(target, num_bins=3)
    shouldbe = np.array([[1, 1, 2, 2, 3, 3], [3, 3, 2, 2, 1, 1]])
    np.testing.assert_array_equal(result, shouldbe)

  def test_discrete_entropy(self):
    target = np.array([[1, 1, 2, 2, 3, 3], [3, 3, 2, 2, 1, 1]])
    result = utils.discrete_entropy(target)
    shouldbe = np.log(3)
    np.testing.assert_allclose(result, [shouldbe, shouldbe])

  def test_discrete_mutual_info(self):
    xs = np.array([[1, 2, 1, 2], [1, 1, 2, 2]])
    ys = np.array([[1, 2, 1, 2], [2, 2, 1, 1]])
    result = utils.discrete_mutual_info(xs, ys)
    shouldbe = np.array([[np.log(2), 0.], [0., np.log(2)]])
    np.testing.assert_allclose(result, shouldbe)

  def test_split_train_test(self):
    xs = np.zeros([10, 100])
    xs_train, xs_test = utils.split_train_test(xs, 0.9)
    shouldbe_train = np.zeros([10, 90])
    shouldbe_test = np.zeros([10, 10])
    np.testing.assert_allclose(xs_train, shouldbe_train)
    np.testing.assert_allclose(xs_test, shouldbe_test)

if __name__ == '__main__':
  absltest.main()
