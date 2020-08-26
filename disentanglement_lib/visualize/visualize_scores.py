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

"""Utilities to visualize the unified scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import matplotlib
matplotlib.use("Agg")  # Set headless-friendly backend.
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.compat.v1 as tf


def heat_square(matrix, output_dir, name, xlabel, ylabel, max_val=None,
                factor_names=None):
  """Plot values of a matrix.

  Each entry is represented as a square of increasing size and different color.

  Args:
    matrix: Matrix of values to plot. Values should be in range [0, max_val].
    output_dir: Where to save the image.
    name: File name.
    xlabel: Name of the x axis of the matrix.
    ylabel: Name of the y axis of the matrix.
    max_val: Maximum value acceptable in the matrix. If None, the max_val will
      be set as the maximum value in the matrix.
    factor_names: Names of the factors of variation.
  """
  sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
  sns.set_style("whitegrid")
  fig, _ = plt.subplots()
  plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=1.2)
  ax = plt.subplot(plot_grid[:, :-1])
  if max_val is None:
    max_val = np.max(matrix)
    if max_val == 0:
      max_val = 1.
  else:
    if max_val < np.max(matrix):
      raise ValueError("The matrix has maximum value larger than max_val")
  palette = sns.color_palette("Blues", 256)
  # Estimates the area of the squares: the length of the edge is
  # roughly: length of the grid in inches * how many points per inch - space for
  # the axis names times * 14/15 as the last 1/15 part of the figure is occupied
  # by the colorbar legend.
  size_scale = ((((ax.get_position().xmax - ax.get_position().xmin) *
                  fig.get_size_inches()[0] * fig.get_dpi() - 40) * 14/15*0.8) /
                (matrix.shape[0]))**2
  plot_matrix_squares(matrix, max_val, palette, size_scale, ax)
  plt.xticks(range(matrix.shape[0]))
  if factor_names is not None:
    plt.yticks(range(matrix.shape[1]), factor_names)
  else:
    plt.yticks(range(matrix.shape[1]))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  # Add color legend on the right side of the plot.
  ax = plt.subplot(plot_grid[:, -1])
  plot_bar_palette(palette, max_val, ax)

  if not tf.gfile.IsDirectory(output_dir):
    tf.gfile.MakeDirs(output_dir)
  output_path = os.path.join(output_dir, "{}.png".format(name))
  with tf.gfile.Open(output_path, "wb") as path:
    fig.savefig(path, bbox_inches="tight")


def plot_matrix_squares(matrix, max_val, palette, size_scale, ax):
  """Grid of squares where the size is proportional to the matrix values.

  Args:
    matrix: Matrix of values to plot.
    max_val: Maximum value that is allowed in the matrix.
    palette: Color palette.
    size_scale: Maximum size of the squares.
    ax: Axis of the subplot.
  """
  tmp = pd.melt(pd.DataFrame(matrix).reset_index(), id_vars="index")
  # The columns of the dataframe are: index, variable and value.
  def to_color(val):
    ind = int(val/max_val*255)
    return palette[ind]
  ax.scatter(x=tmp["index"], y=tmp["variable"],
             s=size_scale * tmp["value"]/max_val, marker="s",
             c=tmp["value"].apply(to_color))
  ax.set_xticks([v+0.5 for v in range(matrix.shape[0])], minor=True)
  ax.set_yticks([v+0.5 for v in range(matrix.shape[1])], minor=True)

  ax.grid(False, "major")
  ax.grid(True, "minor")
  ax.set_xlim([-0.5, matrix.shape[0] - 0.5])
  ax.set_ylim([-0.5, matrix.shape[1] - 0.5])
  ax.tick_params(right=False, top=False, left=False, bottom=False)
  ax.set_aspect(aspect=1.)


def plot_bar_palette(palette, max_val, ax):
  """Plot color bar legend."""
  col_x = [0]*len(palette)
  bar_y = np.linspace(0, max_val, 256, ax)

  bar_height = bar_y[1] - bar_y[0]
  ax.barh(bar_y, np.array([5]*len(palette)), height=bar_height, left=col_x,
          align="center", color=palette, linewidth=0)

  ax.set_xlim(1, 2)
  ax.set_ylim(0, max_val)
  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticks(np.linspace(0, max_val, 3))
  ax.yaxis.tick_right()


def plot_recovery_vs_independent(matrix, output_dir, name):
  """Plot how many factors are recovered and in how many independent groups.

  Plot how many factors of variation are independently captured in a
  representation at different thresholds. It takes as input a matrix
  relating factors of variation and latent dimensions, sort the elements and
  then plot for each threshold (1) how many factors are discovered and (2)
  how many factors are encoded independently in the representation.

  Args:
    matrix: Contains statistical relations between factors of variation and
      latent codes.
    output_dir: Output directory where to save the plot.
    name: Filename of the plot.
  """
  thresholds = np.sort(matrix.flatten())[::-1]
  precisions = [precision(matrix, x) for x in thresholds]
  recalls = [recall(matrix, x) for x in thresholds]
  sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
  sns.set_style("whitegrid")
  fig, ax = plt.subplots()
  palette = sns.color_palette()
  plt.plot(range(thresholds.shape[0]), precisions, label="Independent groups",
           color=palette[0], linewidth=3)
  plt.plot(range(thresholds.shape[0]), recalls, "--", label="Discovered",
           color=palette[1], linewidth=3)
  thresholds_ids = range(0, thresholds.shape[0], 10)
  plt.xticks(thresholds_ids, np.around(thresholds[thresholds_ids], 2))
  ax.set_ylim([0, matrix.shape[0]*1.1])
  ax.tick_params(right=False, top=False, left=False, bottom=False)
  ax.set_yticks(np.linspace(0, matrix.shape[0], matrix.shape[0]+1))
  plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
  plt.xlabel("Threshold")
  plt.ylabel("Number of Factors")
  if not tf.gfile.IsDirectory(output_dir):
    tf.gfile.MakeDirs(output_dir)
  output_path = os.path.join(output_dir, name+".png")
  with tf.gfile.Open(output_path, "wb") as path:
    fig.savefig(path, bbox_inches="tight")


def precision(matrix, th):
  """How many independent components are discovered for a given threshold.

  Args:
    matrix: Adjacency matrix  of shape (num_codes, num_factors) encoding the
      statistical relations between factors and codes.
    th: Eliminate all edges smaller than this threshold.

  Returns:
    Number of connected components.
  """
  tmp = matrix.copy()
  tmp[tmp < th] = 0
  factors = np.zeros(tmp.shape[0])
  codes = np.zeros(tmp.shape[1])
  cc = 0

  for i in range(len(factors)):
    if factors[i] == 0:
      to_visit = [(i, 0)]
      factors, codes, size = bfs(tmp, to_visit, factors, codes, 1)
      if size > 1:
        cc += 1
  return cc


def recall(matrix, th):
  """How many factors are discovered for a given threshold.

  Counts as many factors of variation are captured in the representation.
  First, we remove all edges in the adjacency matrix with weight smaller than
  the threshold. Then, we count how many factors are connected to some codes.

  Args:
    matrix: Adjacency matrix for the graph.
    th: Eliminate all edges smaller than this threshold.

  Returns:
    Number of discovered factors of variation for the given threshold.
  """
  tmp = matrix.copy()
  tmp[tmp < th] = 0
  return np.sum(np.sum(tmp, axis=1) != 0)


def bfs(matrix, to_visit, factors, codes, size):
  """Traverse the matrix across connected components.

  Implements breadth first search on an adjacency matrix. In our case, the
  adjacency matrix encodes the statistical relations between factors of
  variation and codes. This is used to traverse the adjacency matrix and
  discover whether a factor is captured in multiple codes and whether there is a
  path in the graph connecting two factors.

  Args:
    matrix: Adjacency matrix for the graph.
    to_visit: Queue with the nodes to visit. We index the factors and codes in
      the adjacency matrix and implement the queue with an array containing the
      nodes that need to  be visited.
    factors: Array of shape (num_factors, ) with flags marking whether factors
      of variation are visited.
    codes: Array of shape (num_codes, ) with flags marking whether codes are
      visited.
    size: Count how many node are in the same connected component.

  Returns:
    factors: Array of shape (num_factors, ) with flags marking whether factors
      of variation are visited.
    codes: Array of shape (num_codes, ) with flags marking whether codes are
      visited.
    size: How many nodes were visited.
  """
  (current_node, flag) = to_visit.pop()
  if flag == 0:
    factors[current_node] = 1
    for i in range(len(matrix[current_node, :])):
      if matrix[current_node, i] != 0:
        if codes[i] == 0:
          to_visit.append((i, 1))
          size += 1
          factors, codes, size = bfs(matrix, to_visit, factors, codes, size)
  else:
    codes[current_node] = 1
    for i in range(len(matrix[:, current_node])):
      if matrix[i, current_node] != 0:
        if factors[i] == 0:
          to_visit.append((i, 0))
          size += 1
          factors, codes, size = bfs(matrix, to_visit, factors, codes, size)
  return factors, codes, size
