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

"""Tests for the visualize_scores.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from disentanglement_lib.visualize import visualize_scores
import numpy as np


class VisualizeScoresTest(parameterized.TestCase):

  @parameterized.parameters((np.zeros((5, 10), dtype=np.float32),
                             np.zeros((2, 10), dtype=np.float32),
                             np.zeros((10, 10), dtype=np.float32)))
  def plot_recovery_vs_independent(self, matrix):
    visualize_scores.plot_recovery_vs_independent(
        matrix, self.create_tempdir().full_path, "save_image.png")


class ComputeMatrixStatsTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.eye(5), [(0, 0)], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], 2),
      (np.triu(np.ones(5)), [(0, 0)], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 10),
      (np.ones((5, 5)), [(0, 0)], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 10),
      (np.array([[1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1]]),
       [(0, 0)], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], 3),
      (np.array([[1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 1]]),
       [(0, 0)], [1, 0, 0, 1, 0], [1, 1, 0, 0, 0], 4)
  )
  def test_bfs(self, matrix, to_visit, visited_factors, visited_codes,
               visited_size):
    factors = np.zeros(matrix.shape[0])
    codes = np.zeros(matrix.shape[1])
    factors, codes, size = visualize_scores.bfs(
        matrix, to_visit, factors, codes, 1)

    self.assertEqual((factors == visited_factors).all(), True)
    self.assertEqual((codes == visited_codes).all(), True)
    self.assertEqual(size, visited_size)

  @parameterized.parameters(
      (np.zeros((5, 5)), 0),
      (np.eye(5), 5),
      (np.triu(np.ones(5)), 1),
      (np.ones((5, 5)), 1)
  )
  def test_precision(self, matrix, cc):
    cc_count = visualize_scores.precision(matrix, 1)
    self.assertEqual(cc, cc_count)

  @parameterized.parameters(
      (np.zeros((5, 5)), 0),
      (np.eye(5), 5),
      (np.triu(np.ones(5)), 5),
      (np.ones((5, 5)), 5)
  )
  def test_recall(self, matrix, cc):
    cc_count = visualize_scores.recall(matrix, 1)
    self.assertEqual(cc, cc_count)

if __name__ == "__main__":
  absltest.main()
