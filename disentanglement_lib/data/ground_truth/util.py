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

"""Various utilities used in the data set code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf


def tf_data_set_from_ground_truth_data(ground_truth_data, random_seed):
  """Generate a tf.data.DataSet from ground_truth data."""

  def generator():
    # We need to hard code the random seed so that the data set can be reset.
    random_state = np.random.RandomState(random_seed)
    while True:
      yield ground_truth_data.sample_observations(1, random_state)[0]

  return tf.data.Dataset.from_generator(
      generator, tf.float32, output_shapes=ground_truth_data.observation_shape)


class SplitDiscreteStateSpace(object):
  """State space with factors split between latent variable and observations."""

  def __init__(self, factor_sizes, latent_factor_indices):
    self.factor_sizes = factor_sizes
    self.num_factors = len(self.factor_sizes)
    self.latent_factor_indices = latent_factor_indices
    self.observation_factor_indices = [
        i for i in range(self.num_factors)
        if i not in self.latent_factor_indices
    ]

  @property
  def num_latent_factors(self):
    return len(self.latent_factor_indices)

  def sample_latent_factors(self, num, random_state):
    """Sample a batch of the latent factors."""
    factors = np.zeros(
        shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
    for pos, i in enumerate(self.latent_factor_indices):
      factors[:, pos] = self._sample_factor(i, num, random_state)
    return factors

  def sample_all_factors(self, latent_factors, random_state):
    """Samples the remaining factors based on the latent factors."""
    num_samples = latent_factors.shape[0]
    all_factors = np.zeros(
        shape=(num_samples, self.num_factors), dtype=np.int64)
    all_factors[:, self.latent_factor_indices] = latent_factors
    # Complete all the other factors
    for i in self.observation_factor_indices:
      all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
    return all_factors

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.factor_sizes[i], size=num)


class StateSpaceAtomIndex(object):
  """Index mapping from features to positions of state space atoms."""

  def __init__(self, factor_sizes, features):
    """Creates the StateSpaceAtomIndex.

    Args:
      factor_sizes: List of integers with the number of distinct values for each
        of the factors.
      features: Numpy matrix where each row contains a different factor
        configuration. The matrix needs to cover the whole state space.
    """
    self.factor_sizes = factor_sizes
    num_total_atoms = np.prod(self.factor_sizes)
    self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
    feature_state_space_index = self._features_to_state_space_index(features)
    if np.unique(feature_state_space_index).size != num_total_atoms:
      raise ValueError("Features matrix does not cover the whole state space.")
    lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
    lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
    self.state_space_to_save_space_index = lookup_table

  def features_to_index(self, features):
    """Returns the indices in the input space for given factor configurations.

    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the input space should be
        returned.
    """
    state_space_index = self._features_to_state_space_index(features)
    return self.state_space_to_save_space_index[state_space_index]

  def _features_to_state_space_index(self, features):
    """Returns the indices in the atom space for given factor configurations.

    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the atom space should be
        returned.
    """
    if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
        np.any(features < 0)):
      raise ValueError("Feature indices have to be within [0, factor_size-1]!")
    return np.array(np.dot(features, self.factor_bases), dtype=np.int64)
