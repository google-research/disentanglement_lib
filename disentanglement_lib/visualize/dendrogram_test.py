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

"""Tests for the dendrogram.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from disentanglement_lib.visualize import dendrogram
import numpy as np


class DendrogramTest(parameterized.TestCase):

  @parameterized.parameters((np.zeros((5, 10), dtype=np.float32),))
  def test_dendrogram_plot(self, matrix):
    dendrogram.dendrogram_plot(
        matrix, self.create_tempdir().full_path, ["Factor"] * matrix.shape[0])

  @parameterized.parameters(
      (
          # Step by step simulation of the union-find algorithm on the event
          # matrix [[1, 3, 4], [1, 2, 3], [0, 3, 2], [0, 2, 1]], we're now
          # considering the first step.
          np.array([0, 1, 2, 3]),  # Roots for nodes in the graph.
          1,  # Merge node 1 ...
          3,  # With node 3
          0,  # Threshold value of the considered edge: this is the first edge.
          np.array([[0., 0., 0., 0.]]),  # Current dendrogram matrix.
          {0: 0, 1: 1},  # Dictionary factors of variation to cluster IDs.
          {0: 1, 1: 1},  # Size of each cluster.
          np.array([[1, 2], [3, 4]]),  # Original edge-weight matrix.
          1,  # Max cluster ID we have so far.
          0,  # How many events have been found.
          np.array([[0., 0., 0., 0.]]),  # A factor and a code are merged.
          {0: 0, 1: 1},
          {0: 1, 1: 1},
          1, 0.
      ),
      (
          # We're now considering the second step. Node 1 and 3 were merged in
          # the previous step.
          np.array([0, 1, 2, 1]),  # Roots for nodes in the graph.
          1,  # Merge node 1 ...
          2,  # With node 2
          1,  # Threshold value of the considered edge: this is the first edge.
          np.array([[0., 0., 0., 0.]]),  # Current dendrogram matrix.
          {0: 0, 1: 1},  # Dictionary factors of variation to cluster IDs.
          {0: 1, 1: 1},  # Size of each cluster.
          np.array([[1, 2], [3, 4]]),  # Original edge-weight matrix.
          1,  # Max cluster ID we have so far.
          0,  # How many events have been found.
          np.array([[0., 0., 0., 0.]]),  # A factor and a code are merged.
          {0: 0, 1: 1},
          {0: 1, 1: 1},
          1, 0.
      ),
      (
          # We're now considering the third step. Cluster 1 and node 2 were
          # merged in the previous step. Cluster 0 and 1 merged into cluster 2
          # which has now size of 2.
          np.array([0, 1, 1, 1]),  # Nodes in the graph.
          0,  # Merge node 0 ...
          3,  # With node 3
          2,  # Threshold value of the considered edge: this is the second edge.
          np.array([[0., 0., 0., 0.]]),  # Current dendrogram matrix.
          {0: 0, 1: 1},  # Dictionary factors of variation to cluster IDs.
          {0: 1, 1: 1},  # Size of each cluster.
          np.array([[1, 2], [3, 4]]),  # Original edge-weight matrix.
          1,  # Max cluster ID we have so far.
          0,  # How many events have been found.
          np.array([[0., 1., 2., 2.]]),  # Two clusters merged.
          {0: 2, 1: 2},  # Both factors are now mapped to the new cluster 2.
          {0: 1, 1: 1, 2: 2},  # Size of the new cluster is added.
          2,  # Our new max cluster id is 2.
          1  # We have had our first cluster merge.
      ),
      (
          # We're now considering the last step. The last code gets merged into
          # the now only cluster. Since cluster 0 and 1 were merged, 0 is the
          # root of cluster 2.
          np.array([0, 0, 1, 1]),  # Nodes in the graph. Smallest node is root.
          0,  # Merge node 0 ...
          2,  # With node 2
          3,  # Threshold value of the considered edge: this is the first edge.
          np.array([[0., 1., 2., 2.]]),  # Current dendrogram matrix.
          {0: 2, 1: 2},  # Dictionary factors of variation to cluster IDs.
          {0: 1, 1: 1, 2: 2},  # Size of each cluster.
          np.array([[1, 2], [3, 4]]),  # Original edge-weight matrix.
          2,  # Max cluster ID we have so far.
          1,  # How many events have been found.
          np.array([[0., 1., 2., 2.]]),  # A code is merged into a cluster.
          {0: 2, 1: 2},  # Both factors are now mapped to the new cluster 2.
          {0: 1, 1: 1, 2: 2},  # Size of the new cluster is added.
          2,  # Our new max cluster id is 2.
          1  # We have had one cluster merge.
      ))
  def test_union(
      self, nodes, idx, idy, val, z, cluster_id, size, matrix, n_clusters,
      idx_found, z_target, cluster_id_target, size_target, n_clusters_target,
      idx_found_target):

    z, cluster_id, size, n_clusters, idx_found = dendrogram._union(
        nodes, idx, idy, val, z, cluster_id, size, matrix, n_clusters,
        idx_found)
    self.assertEqual((z_target == z).all(), True)
    self.assertEqual((cluster_id_target == cluster_id), True)
    self.assertEqual((size_target == size), True)
    self.assertEqual((n_clusters_target == n_clusters), True)
    self.assertEqual((idx_found_target == idx_found), True)


if __name__ == "__main__":
  absltest.main()

