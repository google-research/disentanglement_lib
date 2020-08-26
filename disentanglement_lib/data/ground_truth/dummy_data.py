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

"""Dummy data sets used for testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import ground_truth_data


class IdentityObservationsData(ground_truth_data.GroundTruthData):
  """Data set where dummy factors are also the observations."""

  @property
  def num_factors(self):
    return 10

  @property
  def observation_shape(self):
    return 10

  @property
  def factors_num_values(self):
    return [1] * 10

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return random_state.random_integers(10, size=(num, self.num_factors))

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    return factors

  @property
  def factor_names(self):
    return ["Factor {}".format(i) for i in range(self.num_factors)]


class DummyData(ground_truth_data.GroundTruthData):
  """Dummy image data set of random noise used for testing."""

  @property
  def num_factors(self):
    return 10

  @property
  def factors_num_values(self):
    return [5] * 10

  @property
  def observation_shape(self):
    return [64, 64, 1]

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return random_state.randint(5, size=(num, self.num_factors))

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    return random_state.random_sample(size=(factors.shape[0], 64, 64, 1))

