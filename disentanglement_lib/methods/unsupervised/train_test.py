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

"""Tests for train.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from absl.testing import parameterized
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.utils import resources
import gin.tf


def _config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/unsupervised/train_test.gin")
  # Test different losses.
  for loss in ["@bernoulli_loss", "@l2_loss"]:
    rec_loss = ["reconstruction_loss.loss_fn = " + loss]
    # Test different activations.
    for act in ["'logits'", "'tanh'"]:
      rec_loss += ["reconstruction_loss.activation = " + act]
      latent_dim = ["encoder.num_latent = 10"]
      # Test different architectures.
      for encoder, decoder in [("@fc_encoder", "@fc_decoder"),
                               ("@conv_encoder", "@deconv_decoder")]:
        architectures = \
            ["encoder.encoder_fn = " + encoder,
             "decoder.decoder_fn = " + decoder]
        structure = rec_loss + architectures + latent_dim
        # Train a BetaVAE with all these settings.
        beta_vae = ["model.model = @vae()", "vae.beta = 10."]
        yield [model_config_path], beta_vae + structure

  # Test all the other different models.
  # Test AnnealedVAE.
  annealed_vae = [
      "model.model = @annealed_vae()", "annealed_vae.c_max = 25",
      "annealed_vae.iteration_threshold = 100000", "annealed_vae.gamma = 1000"
  ]
  yield [model_config_path], annealed_vae

  # Test FactorVAE.
  factor_vae = [
      "model.model = @factor_vae()",
      "discriminator.discriminator_fn = @fc_discriminator",
      "discriminator_optimizer.optimizer_fn = @AdamOptimizer",
      "factor_vae.gamma = 10."
  ]
  yield [model_config_path], factor_vae

  # Test DIP-VAE.
  dip_vae_i = [
      "model.model = @dip_vae()", "dip_vae.lambda_d_factor = 10",
      "dip_vae.dip_type = 'i'", "dip_vae.lambda_od = 10."
  ]
  yield [model_config_path], dip_vae_i

  dip_vae_ii = [
      "model.model = @dip_vae()", "dip_vae.lambda_d_factor = 1",
      "dip_vae.dip_type = 'ii'", "dip_vae.lambda_od = 10."
  ]
  yield [model_config_path], dip_vae_ii

  # Test BetaTCVAE.
  beta_tc_vae = ["model.model = @beta_tc_vae()", "beta_tc_vae.beta = 10."]
  yield [model_config_path], beta_tc_vae


class TrainTest(parameterized.TestCase):

  @parameterized.parameters(list(_config_generator()))
  def test_train_model(self, gin_configs, gin_bindings):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    gin.clear_config()
    train.train_with_gin(self.create_tempdir().full_path, True, gin_configs,
                         gin_bindings)


if __name__ == "__main__":
  absltest.main()
