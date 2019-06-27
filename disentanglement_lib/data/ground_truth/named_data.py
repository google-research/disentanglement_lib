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

"""Provides named, gin configurable ground truth data sets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import cars3d
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.data.ground_truth import norb
from disentanglement_lib.data.ground_truth import shapes3d
import gin.tf


@gin.configurable("dataset")
def get_named_ground_truth_data(name):
  """Returns ground truth data set based on name.

  Args:
    name: String with the name of the dataset.

  Raises:
    ValueError: if an invalid data set name is provided.
  """

  if name == "dsprites_full":
    return dsprites.DSprites([1, 2, 3, 4, 5])
  elif name == "dsprites_noshape":
    return dsprites.DSprites([2, 3, 4, 5])
  elif name == "color_dsprites":
    return dsprites.ColorDSprites([1, 2, 3, 4, 5])
  elif name == "noisy_dsprites":
    return dsprites.NoisyDSprites([1, 2, 3, 4, 5])
  elif name == "scream_dsprites":
    return dsprites.ScreamDSprites([1, 2, 3, 4, 5])
  elif name == "smallnorb":
    return norb.SmallNORB()
  elif name == "cars3d":
    return cars3d.Cars3D()
  elif name == "mpi3d_toy":
    return mpi3d.MPI3D(mode="mpi3d_toy")
  elif name == "mpi3d_realistic":
    return mpi3d.MPI3D(mode="mpi3d_realistic")
  elif name == "mpi3d_real":
    return mpi3d.MPI3D(mode="mpi3d_real")
  elif name == "shapes3d":
    return shapes3d.Shapes3D()
  elif name == "dummy_data":
    return dummy_data.DummyData()
  else:
    raise ValueError("Invalid data set name.")
