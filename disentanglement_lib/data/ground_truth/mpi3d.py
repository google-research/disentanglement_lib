
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

"""MPI3D data set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import glob
import imageio
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import tensorflow as tf


class MPI3D(ground_truth_data.GroundTruthData):
  """MPI3D dataset.

  MPI3D datasets have been introduced as a part of NEURIPS 2019 Disentanglement
  Competition.(https://www.aicrowd.com/challenges/neurips-2019-disentanglement-challenge).
  There are three different datasets
  1. Simplistic rendered images (mpi3d_toy).
  2. Realistic rendered images (mpi3d_realistic).
  3. Real world images (mpi3d_real).
  More details about this dataset can be found in "On the Transfer of Inductive Bias from
  Simulation to the Real World: a New Disentanglement Dataset"(https://arxiv.org/abs/1906.03292).

  The ground-truth factors of variation in the dataset are:
  0 - Object Color (4 different values)
  1 - Object Shape (4 different values)
  2 - Object Size (2 different values)
  3 - Camera Height (3 different values)
  4 - Background Colors (3 different values)
  5 - First DOF (40 different values)
  6 - Second DOF (40 different values)
  """

  def __init__(self, mode="mpi3d_toy"):
    if mode == "mpi3d_toy":
        MPI3D_PATH = os.path.join(
            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_toy")
    elif mode == "mpi3d_realistic":
        MPI3D_PATH = os.path.join(
            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_realistic")
    elif mode == "mpi3d_real":
        MPI3D_PATH = os.path.join(
            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_real")
    else:
        raise Exception("Unknown mode provided.")

    self.factor_sizes = [4, 4, 2, 3, 3, 40, 40]
    self.latent_factor_indices = [0, 1, 2, 3, 4, 5, 6]
    self.num_total_factors = 7
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)
    self.images = self._load_data(path = MPI3D_PATH)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes

  @property
  def observation_shape(self):
    return [64, 64, 3]

  def numericalSort(self, value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    return self.images[indices]/255.

  def _load_data(self, path):
    imgs_list = []
    for infile in sorted(glob.glob(os.path.join(path ,'*.png')), key=self.numericalSort):
        imgs_list.append(imageio.imread(infile))
    return np.asarray(imgs_list, dtype = np.uint8)
