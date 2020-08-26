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

"""Utilities to make dendrogram plots based on the importance matrices.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use("Agg")  # Set headless-friendly backend.
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import tensorflow.compat.v1 as tf


def dendrogram_plot(matrix, output_dir, factor_names):
  """Make dendrogram plot recording at which threshold factors and codes merge.

  This plotting function produce a dendrogram plot recording at which factors of
  variation and latent codes are most related by running the union-find
  algorithm https://en.wikipedia.org/wiki/Disjoint-set_data_structure on the
  matrix relating factors of variation and latent codes.

  Args:
    matrix: Input matrix of shape [num_factors, num_codes] encoding the
      statistical relation between factors and codes.
    output_dir: Directory to save the plot in.
    factor_names: Lables for the factors of variation to be used in the plot.

  Returns:
    Dictionary containing the threshold ID of each merging events and which
    factors were merged.
  """
  tmp = pd.melt(pd.DataFrame(matrix).reset_index(), id_vars="index")
  # The columns of the dataframe are: index, variable and value.
  tmp = tmp.to_numpy()
  # Sort the matrix by threshold
  tmp = tmp[tmp[:, -1].argsort()[::-1]]
  # The codes have index code + num_factors.
  tmp[:, 1] += matrix.shape[0]

  # Initialize dictionaries for cluster IDs and size.
  size = {}
  cluster_id = {}
  for i in range(matrix.shape[0]):
    size[i] = 1
    cluster_id[i] = i
  # Initialize dendrogram matrix. Each row is an event, each event is composed
  # by [cluster_id_1, cluster_id_2, threshold, size of the new cluster]
  z = np.zeros([matrix.shape[0]-1, 4])
  # Each factor of variation is in its own tree. So the maximum cluster ID we
  # have is matrix.shape[0]-1.
  n_clusters = matrix.shape[0]-1
  nodes = list(range(matrix.shape[0] + matrix.shape[1]))
  idx_found = 0
  discovered = {}
  # Run the Union-Find Algorithm
  for id_i, i in enumerate(tmp):
    # Record if we just discovered a new factor of variation.
    if i[0] not in discovered:
      discovered[i[0]] = id_i
    # Merge trees.
    z, cluster_id, size, n_clusters, idx_found = _union(
        nodes, i[0], i[1], id_i, z, cluster_id, size, matrix, n_clusters,
        idx_found)
  # Obtain the dendrogram plot data structure from the matrix z
  fig, ax = plt.subplots()
  dn = hierarchy.dendrogram(z, ax=ax, orientation="left", no_plot=True)
  # Create a dictionary to map the location on the plot to the leaf
  id_to_leaf = {}
  id_conv = 5
  for l in dn["leaves"]:
    id_to_leaf[id_conv] = l
    id_conv += 10
  # Update the dcoord to when the cluster was actually discovered.
  for d, i in zip(dn["dcoord"], dn["icoord"]):
    if d[0] == 0:
      idx = id_to_leaf[i[0]]
      d[0] = discovered[idx]
    if d[-1] == 0:
      idx = id_to_leaf[i[-1]]
      d[-1] = discovered[idx]
  # Set colors to be all the same.
  dn["color_list"] = ["b"]*len(dn["color_list"])
  dn["ivl"] = np.array(factor_names)[dn["leaves"]]

  hierarchy._plot_dendrogram(dn["icoord"], dn["dcoord"], dn["ivl"], p=30,  # pylint: disable=protected-access
                             n=z.shape[0] + 1, mh=max(z[:, 2]),
                             orientation="right", no_labels=False,
                             color_list=dn["color_list"],
                             leaf_font_size=None,
                             leaf_rotation=None,
                             contraction_marks=None,
                             ax=ax,
                             above_threshold_color="b")
  plt.xlabel("Threshold")
  plt.ylabel("Factor")
  thresholds = tmp[:, 2]
  thresholds_ids = range(0, thresholds.shape[0], 10)
  plt.xticks(
      thresholds_ids,
      np.around(np.array(thresholds, dtype="float32")[thresholds_ids], 2))
  output_path = output_dir+".png"
  with tf.gfile.Open(output_path, "wb") as path:
    fig.savefig(path, bbox_inches="tight")
  return report_merges(z, matrix.shape[0])


def report_merges(z, num_factors):
  """Saves which factors of variations are merged and at which threshold.

  Args:
    z: Dendrogram matrix. Each row is an event, each event is composed by
      [cluster_id_1, cluster_id_2, threshold, size of the new cluster].
    num_factors: Number of factors of Variations.

  Returns:
    Dictionary containing the threshold ID of each merging events and which
    factors were merged.
  """
  scores = {}
  id_to_node_list = {}
  n_clusters = num_factors-1
  for i in range(num_factors):
    id_to_node_list[i] = [i,]
  for i in range(z.shape[0]):
    cluster_id_1 = z[i, 0]
    cluster_id_2 = z[i, 1]
    threshold_id = z[i, 2]
    list_nodes_1 = id_to_node_list[cluster_id_1]
    list_nodes_2 = id_to_node_list[cluster_id_2]

    for node_1 in list_nodes_1:
      for node_2 in list_nodes_2:
        scores["merge_{}_{}".format(node_1, node_2)] = threshold_id

    del id_to_node_list[cluster_id_1]
    del id_to_node_list[cluster_id_2]
    n_clusters += 1

    id_to_node_list[n_clusters] = list_nodes_1 + list_nodes_2

  return scores


def _find(nodes, i):
  """Find function for the Union-Find algorithm."""
  if nodes[i] != i:
    nodes[i] = _find(nodes, nodes[i])
  return nodes[i]


def _union(nodes, idx, idy, val, z, cluster_id, size, matrix, n_clusters,
           idx_found):
  """Implements the a modification to the union of the Union-Find algorithm.

  In this function we first perform the standard union of the Union-Find
  algorithm. We mantain the root of the trees with more than 1 element to
  factors of variation. If two trees rooted at factors of variation gets merged
  we record the event in the dendrogram matrix.

  Args:
    nodes: Array with the nodes of the graph. The first num_factors nodes
      correspond to factors of variation. The rest to codes.
    idx: First node to eventually merge.
    idy: Second node to eventually merge.
    val: Threshold value of the considered edge.
    z: Dendrogram matrix. Each row is an event, each event is composed by
      [cluster_id_1, cluster_id_2, threshold, size of the new cluster].
    cluster_id: Dictionary mapping factors of variation to cluster IDs.
    size: Dictionary mapping cluster ID to its size.
    matrix: Matrix of shape  [num_factors, num_codes] on which we compute the
      dendrogram.
    n_clusters: How many clusters have been discovered.
    idx_found: How many evenys have been found.

  Returns:
    z, cluster_id, size, n_clusters, idx_found
  """
  parent_idx = _find(nodes, idx)
  parent_idy = _find(nodes, idy)
  # Set the parent node the one with smalles id. This ensures that the trees
  # roots are on the factors of variations as they have smaller ID. The two
  # nodes cause a merge if they are not already in the same tree.
  if parent_idx != parent_idy:
    if parent_idy > parent_idx:
      nodes[parent_idy] = parent_idx
    else:
      nodes[parent_idx] = parent_idy

    if parent_idx < matrix.shape[0] and parent_idy < matrix.shape[0]:
      # There is a merge happening on two trees rooted at a factor of variation.
      # These are the events we need to record. First, we get the cluster id of
      # the trees:
      cc_idx = cluster_id[parent_idx]
      cc_idy = cluster_id[parent_idy]
      # Now we create a new cluster by merging the two trees. The size will be
      # the sum of the two subtrees as they are disjoint.
      n_clusters += 1
      size[n_clusters] = size[cc_idx] + size[cc_idy]
      # Update the dendrogram matrix with the new event.
      z[idx_found, :] = [cc_idx, cc_idy, val, size[n_clusters]]
      idx_found += 1
      # Set the new cluster ID for the parent nodes.
      cluster_id[parent_idy] = n_clusters
      cluster_id[parent_idx] = n_clusters
  return z, cluster_id, size, n_clusters, idx_found
