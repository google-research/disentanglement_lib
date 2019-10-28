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

"""Hyperparameter sweeps and configs for stage 1 of "abstract_reasoning_study".

Are Disentangled Representations Helpful for Abstract Visual Reasoning?
Sjoerd van Steenkiste, Francesco Locatello, Juergen Schmidhuber, Olivier Bachem.
NeurIPS, 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.config import study
from disentanglement_lib.utils import resources
import disentanglement_lib.utils.hyperparams as h
from six.moves import range


def get_datasets():
  """Returns all the data sets."""
  return h.sweep(
      "dataset.name",
      h.categorical(["shapes3d", "abstract_dsprites"]))


def get_num_latent(sweep):
  return h.sweep("encoder.num_latent", h.discrete(sweep))


def get_seeds(num):
  """Returns random seeds."""
  return h.sweep("model.random_seed", h.categorical(list(range(num))))


def get_default_models():
  """Our default set of models (6 model * 6 hyperparameters=36 models)."""
  # BetaVAE config.
  model_name = h.fixed("model.name", "beta_vae")
  model_fn = h.fixed("model.model", "@vae()")
  betas = h.sweep("vae.beta", h.discrete([1., 2., 4., 6., 8., 16.]))
  config_beta_vae = h.zipit([model_name, betas, model_fn])

  # AnnealedVAE config.
  model_name = h.fixed("model.name", "annealed_vae")
  model_fn = h.fixed("model.model", "@annealed_vae()")
  iteration_threshold = h.fixed("annealed_vae.iteration_threshold", 100000)
  c = h.sweep("annealed_vae.c_max", h.discrete([5., 10., 25., 50., 75., 100.]))
  gamma = h.fixed("annealed_vae.gamma", 1000)
  config_annealed_beta_vae = h.zipit(
      [model_name, c, iteration_threshold, gamma, model_fn])

  # FactorVAE config.
  model_name = h.fixed("model.name", "factor_vae")
  model_fn = h.fixed("model.model", "@factor_vae()")
  discr_fn = h.fixed("discriminator.discriminator_fn", "@fc_discriminator")

  gammas = h.sweep("factor_vae.gamma",
                   h.discrete([10., 20., 30., 40., 50., 100.]))
  config_factor_vae = h.zipit([model_name, gammas, model_fn, discr_fn])

  # DIP-VAE-I config.
  model_name = h.fixed("model.name", "dip_vae_i")
  model_fn = h.fixed("model.model", "@dip_vae()")
  lambda_od = h.sweep("dip_vae.lambda_od",
                      h.discrete([1., 2., 5., 10., 20., 50.]))
  lambda_d_factor = h.fixed("dip_vae.lambda_d_factor", 10.)
  dip_type = h.fixed("dip_vae.dip_type", "i")
  config_dip_vae_i = h.zipit(
      [model_name, model_fn, lambda_od, lambda_d_factor, dip_type])

  # DIP-VAE-II config.
  model_name = h.fixed("model.name", "dip_vae_ii")
  model_fn = h.fixed("model.model", "@dip_vae()")
  lambda_od = h.sweep("dip_vae.lambda_od",
                      h.discrete([1., 2., 5., 10., 20., 50.]))
  lambda_d_factor = h.fixed("dip_vae.lambda_d_factor", 1.)
  dip_type = h.fixed("dip_vae.dip_type", "ii")
  config_dip_vae_ii = h.zipit(
      [model_name, model_fn, lambda_od, lambda_d_factor, dip_type])

  # BetaTCVAE config.
  model_name = h.fixed("model.name", "beta_tc_vae")
  model_fn = h.fixed("model.model", "@beta_tc_vae()")
  betas = h.sweep("beta_tc_vae.beta", h.discrete([1., 2., 4., 6., 8., 10.]))
  config_beta_tc_vae = h.zipit([model_name, model_fn, betas])
  all_models = h.chainit([
      config_beta_vae, config_factor_vae, config_dip_vae_i, config_dip_vae_ii,
      config_beta_tc_vae, config_annealed_beta_vae
  ])
  return all_models


def get_config():
  """Returns the hyperparameter configs for different experiments."""
  arch_enc = h.fixed("encoder.encoder_fn", "@conv_encoder", length=1)
  arch_dec = h.fixed("decoder.decoder_fn", "@deconv_decoder", length=1)
  architecture = h.zipit([arch_enc, arch_dec])
  return h.product([
      get_datasets(),
      architecture,
      get_default_models(),
      get_seeds(5),
  ])


class AbstractReasoningStudyV1(study.Study):
  """Defines the study for the paper."""

  def get_model_config(self, model_num=0):
    """Returns model bindings and config file."""
    config = get_config()[model_num]
    model_bindings = h.to_bindings(config)
    model_config_file = resources.get_file(
        "config/abstract_reasoning_study_v1/stage1/model_configs/shared.gin")
    return model_bindings, model_config_file

  def get_postprocess_config_files(self):
    """Returns postprocessing config files."""
    return list(
        resources.get_files_in_folder(
            "config/abstract_reasoning_study_v1/stage1/postprocess_configs/"))

  def get_eval_config_files(self):
    """Returns evaluation config files."""
    return list(
        resources.get_files_in_folder(
            "config/abstract_reasoning_study_v1/stage1/metric_configs/"))
