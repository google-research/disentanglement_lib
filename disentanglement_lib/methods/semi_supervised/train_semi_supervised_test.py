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

"""Tests for the semi supervised training protocol.

We perform a test for each model so they can be performed in parallel.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from disentanglement_lib.methods.semi_supervised import semi_supervised_utils  # pylint: disable=unused-import
from disentanglement_lib.methods.semi_supervised import semi_supervised_vae  # pylint: disable=unused-import
from disentanglement_lib.methods.semi_supervised import train_semi_supervised_lib
from disentanglement_lib.utils import resources
import tensorflow.compat.v1 as tf
import gin.tf

ANNEAL_LOSS_LIST = [
    [
        "annealer.anneal_fn = @fixed", "supervised_loss.loss_fn = @xent",
        "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fine_tune", "supervised_loss.loss_fn = @xent",
        "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fixed", "supervised_loss.loss_fn = @l2",
        "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fine_tune", "supervised_loss.loss_fn = @l2",
        "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fixed", "supervised_loss.loss_fn = @cov",
        "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fine_tune", "supervised_loss.loss_fn = @cov",
        "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fixed", "supervised_loss.loss_fn = @embed",
        "embed.sigma = 1", "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fixed", "supervised_loss.loss_fn = @embed",
        "embed.sigma = 'learn'", "labeller.labeller_fn = @perfect_labeller"
    ],
    [
        "annealer.anneal_fn = @fixed", "supervised_loss.loss_fn = @xent",
        "labeller.labeller_fn = @partial_labeller",
        "partial_labeller.num_observed_factors=2"
    ],
]


def _s2_config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/semi_supervised/train_test.gin")
  # Test s2_vae.
  s2_vae = [
      "model.model = @s2_vae", "model.num_labelled_samples = 100",
      "model.train_percentage = 0.9", "s2_vae.beta = 4", "s2_vae.gamma_sup = 4",
      "annealer.iteration_threshold = 1",
      "model.model_seed = 0",
      "model.unsupervised_data_seed = 0", "model.supervised_data_seed = 0",
      "model.num_labelled_samples = 100", "model.train_percentage = 0.9"
  ]
  for anneal_loss in ANNEAL_LOSS_LIST:
    yield [model_config_path], s2_vae + anneal_loss


def _supervised_config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/semi_supervised/train_test.gin")
  # Test for s2_vae.
  supervised = [
      "model.model = @supervised", "model.num_labelled_samples = 100",
      "model.train_percentage = 0.9",
      "annealer.iteration_threshold = 1",
      "model.model_seed = 0",
      "model.unsupervised_data_seed = 0", "model.supervised_data_seed = 0",
      "model.num_labelled_samples = 100", "model.train_percentage = 0.9"
  ]
  for anneal_loss in ANNEAL_LOSS_LIST:
    yield [model_config_path], supervised + anneal_loss


def _s2_factor_config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/semi_supervised/train_test.gin")
  # Test for s2_factor_vae.
  s2_factor_vae = [
      "model.model = @s2_factor_vae", "model.num_labelled_samples = 100",
      "model.train_percentage = 0.9", "s2_factor_vae.gamma = 4",
      "s2_factor_vae.gamma_sup = 4", "annealer.iteration_threshold = 1",
      "discriminator.discriminator_fn = @fc_discriminator",
      "discriminator_optimizer.optimizer_fn = @AdamOptimizer",
      "model.model_seed = 0",
      "model.unsupervised_data_seed = 0", "model.supervised_data_seed = 0",
      "model.num_labelled_samples = 100", "model.train_percentage = 0.9"
  ]
  for anneal_loss in ANNEAL_LOSS_LIST:
    yield [model_config_path], s2_factor_vae + anneal_loss


def _s2_dip_config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/semi_supervised/train_test.gin")
  # Test for s2_dip_vae.
  s2_dip_vae_i = [
      "model.model = @s2_dip_vae", "model.num_labelled_samples = 100",
      "model.train_percentage = 0.9", "s2_dip_vae.lambda_d_factor = 10",
      "s2_dip_vae.dip_type = 'i'", "s2_dip_vae.lambda_od = 10.",
      "s2_dip_vae.gamma_sup = 4", "annealer.iteration_threshold = 1",
      "model.model_seed = 0",
      "model.unsupervised_data_seed = 0", "model.supervised_data_seed = 0",
      "model.num_labelled_samples = 100", "model.train_percentage = 0.9"
  ]
  for anneal_loss in ANNEAL_LOSS_LIST:
    yield [model_config_path], s2_dip_vae_i + anneal_loss

  s2_dip_vae_ii = [
      "model.model = @s2_dip_vae", "model.num_labelled_samples = 100",
      "model.train_percentage = 0.9", "s2_dip_vae.lambda_d_factor = 1",
      "s2_dip_vae.dip_type = 'ii'", "s2_dip_vae.lambda_od = 10.",
      "s2_dip_vae.gamma_sup = 4", "annealer.iteration_threshold = 1",
      "model.model_seed = 0",
      "model.unsupervised_data_seed = 0", "model.supervised_data_seed = 0",
      "model.num_labelled_samples = 100", "model.train_percentage = 0.9"
  ]
  for anneal_loss in ANNEAL_LOSS_LIST:
    yield [model_config_path], s2_dip_vae_ii + anneal_loss


def _s2_beta_tc_config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/semi_supervised/train_test.gin")
  # Test for s2_beta_tc_vae.
  s2_beta_tc_vae = [
      "model.model = @s2_beta_tc_vae", "model.num_labelled_samples = 100",
      "model.train_percentage = 0.9", "s2_beta_tc_vae.beta = 10.",
      "s2_beta_tc_vae.gamma_sup = 4", "annealer.iteration_threshold = 1",
      "model.model_seed = 0",
      "model.unsupervised_data_seed = 0", "model.supervised_data_seed = 0",
      "model.num_labelled_samples = 100", "model.train_percentage = 0.9"
  ]
  for anneal_loss in ANNEAL_LOSS_LIST:
    yield [model_config_path], s2_beta_tc_vae + anneal_loss


def _vae_config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/semi_supervised/train_test.gin")
  # Test for vae, both unsupervised and s2 methods runs with the s2
  # training_lib.
  vae = [
      "model.model = @vae", "model.num_labelled_samples = 100",
      "model.train_percentage = 0.9", "vae.beta = 10.",
      "annealer.iteration_threshold = 1",
      "model.model_seed = 0",
      "model.unsupervised_data_seed = 0", "model.supervised_data_seed = 0",
      "model.num_labelled_samples = 100", "model.train_percentage = 0.9"
  ]
  for anneal_loss in ANNEAL_LOSS_LIST:
    yield [model_config_path], vae + anneal_loss


class S2TrainTest(parameterized.TestCase):

  @parameterized.parameters(list(_s2_config_generator()))
  def test_train_model(self, gin_configs, gin_bindings):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    gin.clear_config()
    train_semi_supervised_lib.train_with_gin(self.create_tempdir().full_path,
                                             True, gin_configs, gin_bindings)


class S2FactorTrainTest(parameterized.TestCase):

  @parameterized.parameters(list(_s2_factor_config_generator()))
  def test_train_model(self, gin_configs, gin_bindings):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    gin.clear_config()
    train_semi_supervised_lib.train_with_gin(self.create_tempdir().full_path,
                                             True, gin_configs, gin_bindings)


class S2DipTrainTest(parameterized.TestCase):

  @parameterized.parameters(list(_s2_dip_config_generator()))
  def test_train_model(self, gin_configs, gin_bindings):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    gin.clear_config()
    train_semi_supervised_lib.train_with_gin(self.create_tempdir().full_path,
                                             True, gin_configs, gin_bindings)


class S2BetaTCCTrainTest(parameterized.TestCase):

  @parameterized.parameters(list(_s2_beta_tc_config_generator()))
  def test_train_model(self, gin_configs, gin_bindings):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    train_semi_supervised_lib.train_with_gin(self.create_tempdir().full_path,
                                             True, gin_configs, gin_bindings)


class VAETrainTest(parameterized.TestCase):

  @parameterized.parameters(list(_vae_config_generator()))
  def test_train_model(self, gin_configs, gin_bindings):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    train_semi_supervised_lib.train_with_gin(self.create_tempdir().full_path,
                                             True, gin_configs, gin_bindings)


if __name__ == "__main__":
  tf.test.main()
