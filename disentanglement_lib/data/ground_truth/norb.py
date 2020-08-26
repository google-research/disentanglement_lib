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

"""SmallNORB dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import PIL
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf


SMALLNORB_TEMPLATE = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "small_norb",
    "smallnorb-{}-{}.mat")

SMALLNORB_CHUNKS = [
    "5x46789x9x18x6x2x96x96-training",
    "5x01235x9x18x6x2x96x96-testing",
]


class SmallNORB(ground_truth_data.GroundTruthData):
  """SmallNORB dataset.

  The data set can be downloaded from
  https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/. Images are resized to 64x64.

  The ground-truth factors of variation are:
  0 - category (5 different values)
  1 - elevation (9 different values)
  2 - azimuth (18 different values)
  3 - lighting condition (6 different values)

  The instance in each category is randomly sampled when generating the images.
  """

  def __init__(self):
    self.images, features = _load_small_norb_chunks(SMALLNORB_TEMPLATE,
                                                    SMALLNORB_CHUNKS)
    self.factor_sizes = [5, 10, 9, 18, 6]
    # Instances are not part of the latent space.
    self.latent_factor_indices = [0, 2, 3, 4]
    self.num_total_factors = features.shape[1]
    self.index = util.StateSpaceAtomIndex(self.factor_sizes, features)
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return [self.factor_sizes[i] for i in self.latent_factor_indices]

  @property
  def observation_shape(self):
    return [64, 64, 1]


  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = self.index.features_to_index(all_factors)
    return np.expand_dims(self.images[indices].astype(np.float32), axis=3)


def _load_small_norb_chunks(path_template, chunk_names):
  """Loads several chunks of the small norb data set for final use."""
  list_of_images, list_of_features = _load_chunks(path_template, chunk_names)
  features = np.concatenate(list_of_features, axis=0)
  features[:, 3] = features[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
  return np.concatenate(list_of_images, axis=0), features




def _load_chunks(path_template, chunk_names):
  """Loads several chunks of the small norb data set into lists."""
  list_of_images = []
  list_of_features = []
  for chunk_name in chunk_names:
    norb = _read_binary_matrix(path_template.format(chunk_name, "dat"))
    list_of_images.append(_resize_images(norb[:, 0]))
    norb_class = _read_binary_matrix(path_template.format(chunk_name, "cat"))
    norb_info = _read_binary_matrix(path_template.format(chunk_name, "info"))
    list_of_features.append(np.column_stack((norb_class, norb_info)))
  return list_of_images, list_of_features


def _read_binary_matrix(filename):
  """Reads and returns binary formatted matrix stored in filename."""
  with tf.gfile.GFile(filename, "rb") as f:
    s = f.read()
    magic = int(np.frombuffer(s, "int32", 1))
    ndim = int(np.frombuffer(s, "int32", 1, 4))
    eff_dim = max(3, ndim)
    raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
    dims = []
    for i in range(0, ndim):
      dims.append(raw_dims[i])

    dtype_map = {
        507333717: "int8",
        507333716: "int32",
        507333713: "float",
        507333715: "double"
    }
    data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
  data = data.reshape(tuple(dims))
  return data


def _resize_images(integer_images):
  resized_images = np.zeros((integer_images.shape[0], 64, 64))
  for i in range(integer_images.shape[0]):
    image = PIL.Image.fromarray(integer_images[i, :, :])
    image = image.resize((64, 64), PIL.Image.ANTIALIAS)
    resized_images[i, :, :] = image
  return resized_images / 255.
