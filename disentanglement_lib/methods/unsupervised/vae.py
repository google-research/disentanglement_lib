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

"""Library of losses for disentanglement learning.

Implementation of VAE based models for unsupervised learning of disentangled
representations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import gin.tf
from tensorflow.contrib import tpu as contrib_tpu


class BaseVAE(gaussian_encoder_model.GaussianEncoderModel):
  """Abstract base class of a basic Gaussian encoder model."""

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    del labels
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    data_shape = features.get_shape().as_list()[1:]
    z_mean, z_logvar = self.gaussian_encoder(features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
    loss = tf.add(reconstruction_loss, regularizer, name="loss")
    elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = optimizers.make_vae_optimizer()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
      train_op = tf.group([train_op, update_ops])
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)
      tf.summary.scalar("elbo", -elbo)

      logging_hook = tf.train.LoggingTensorHook({
          "loss": loss,
          "reconstruction_loss": reconstruction_loss,
          "elbo": -elbo
      },
                                                every_n_iter=100)
      return contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                       "regularizer", "kl_loss"),
                        [reconstruction_loss, -elbo, regularizer, kl_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")

  def gaussian_encoder(self, input_tensor, is_training):
    """Applies the Gaussian encoder to images.

    Args:
      input_tensor: Tensor with the observations to be encoded.
      is_training: Boolean indicating whether in training mode.

    Returns:
      Tuple of tensors with the mean and log variance of the Gaussian encoder.
    """
    return architectures.make_gaussian_encoder(
        input_tensor, is_training=is_training)

  def decode(self, latent_tensor, observation_shape, is_training):
    """Decodes the latent_tensor to an observation."""
    return architectures.make_decoder(
        latent_tensor, observation_shape, is_training=is_training)


def shuffle_codes(z):
  """Shuffles latent variables across the batch.

  Args:
    z: [batch_size, num_latent] representation.

  Returns:
    shuffled: [batch_size, num_latent] shuffled representation across the batch.
  """
  z_shuffle = []
  for i in range(z.get_shape()[1]):
    z_shuffle.append(tf.random_shuffle(z[:, i]))
  shuffled = tf.stack(z_shuffle, 1, name="latent_shuffled")
  return shuffled


def compute_gaussian_kl(z_mean, z_logvar):
  """Compute KL divergence between input Gaussian and Standard Normal."""
  return tf.reduce_mean(
      0.5 * tf.reduce_sum(
          tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]),
      name="kl_loss")


def make_metric_fn(*names):
  """Utility function to report tf.metrics in model functions."""

  def metric_fn(*args):
    return {name: tf.metrics.mean(vec) for name, vec in zip(names, args)}

  return metric_fn


@gin.configurable("vae")
class BetaVAE(BaseVAE):
  """BetaVAE model."""

  def __init__(self, beta=gin.REQUIRED):
    """Creates a beta-VAE model.

    Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    Args:
      beta: Hyperparameter for the regularizer.

    Returns:
      model_fn: Model function for TPUEstimator.
    """
    self.beta = beta

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    return self.beta * kl_loss


def anneal(c_max, step, iteration_threshold):
  """Anneal function for anneal_vae (https://arxiv.org/abs/1804.03599).

  Args:
    c_max: Maximum capacity.
    step: Current step.
    iteration_threshold: How many iterations to reach c_max.

  Returns:
    Capacity annealed linearly until c_max.
  """
  return tf.math.minimum(c_max * 1.,
                         c_max * 1. * tf.to_float(step) / iteration_threshold)


@gin.configurable("annealed_vae")
class AnnealedVAE(BaseVAE):
  """AnnealedVAE model."""

  def __init__(self,
               gamma=gin.REQUIRED,
               c_max=gin.REQUIRED,
               iteration_threshold=gin.REQUIRED):
    """Creates an AnnealedVAE model.

    Implementing Eq. 8 of "Understanding disentangling in beta-VAE"
    (https://arxiv.org/abs/1804.03599).

    Args:
      gamma: Hyperparameter for the regularizer.
      c_max: Maximum capacity of the bottleneck.
      iteration_threshold: How many iterations to reach c_max.
    """
    self.gamma = gamma
    self.c_max = c_max
    self.iteration_threshold = iteration_threshold

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    c = anneal(self.c_max, tf.train.get_global_step(), self.iteration_threshold)
    return self.gamma * tf.math.abs(kl_loss - c)


@gin.configurable("factor_vae")
class FactorVAE(BaseVAE):
  """FactorVAE model."""

  def __init__(self, gamma=gin.REQUIRED):
    """Creates a FactorVAE model.

    Implementing Eq. 2 of "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983).

    Args:
      gamma: Hyperparameter for the regularizer.
    """
    self.gamma = gamma

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    del labels
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    data_shape = features.get_shape().as_list()[1:]
    z_mean, z_logvar = self.gaussian_encoder(features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    z_shuffle = shuffle_codes(z_sampled)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      logits_z, probs_z = architectures.make_discriminator(
          z_sampled, is_training=is_training)
      _, probs_z_shuffle = architectures.make_discriminator(
          z_shuffle, is_training=is_training)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(
        features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    standard_vae_loss = tf.add(reconstruction_loss, kl_loss, name="VAE_loss")
    # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
    tc_loss_per_sample = logits_z[:, 0] - logits_z[:, 1]
    tc_loss = tf.reduce_mean(tc_loss_per_sample, axis=0)
    regularizer = kl_loss + self.gamma * tc_loss
    factor_vae_loss = tf.add(
        standard_vae_loss, self.gamma * tc_loss, name="factor_VAE_loss")
    discr_loss = tf.add(
        0.5 * tf.reduce_mean(tf.log(probs_z[:, 0])),
        0.5 * tf.reduce_mean(tf.log(probs_z_shuffle[:, 1])),
        name="discriminator_loss")
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer_vae = optimizers.make_vae_optimizer()
      optimizer_discriminator = optimizers.make_discriminator_optimizer()
      all_variables = tf.trainable_variables()
      encoder_vars = [var for var in all_variables if "encoder" in var.name]
      decoder_vars = [var for var in all_variables if "decoder" in var.name]
      discriminator_vars = [var for var in all_variables \
                            if "discriminator" in var.name]
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op_vae = optimizer_vae.minimize(
          loss=factor_vae_loss,
          global_step=tf.train.get_global_step(),
          var_list=encoder_vars + decoder_vars)
      train_op_discr = optimizer_discriminator.minimize(
          loss=-discr_loss,
          global_step=tf.train.get_global_step(),
          var_list=discriminator_vars)
      train_op = tf.group(train_op_vae, train_op_discr, update_ops)
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)
      logging_hook = tf.train.LoggingTensorHook({
          "loss": factor_vae_loss,
          "reconstruction_loss": reconstruction_loss
      },
                                                every_n_iter=50)
      return contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=factor_vae_loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=factor_vae_loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                       "kl_loss"),
                        [reconstruction_loss, regularizer, kl_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")


def compute_covariance_z_mean(z_mean):
  """Computes the covariance of z_mean.

  Uses cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.

  Args:
    z_mean: Encoder mean, tensor of size [batch_size, num_latent].

  Returns:
    cov_z_mean: Covariance of encoder mean, tensor of size [num_latent,
      num_latent].
  """
  expectation_z_mean_z_mean_t = tf.reduce_mean(
      tf.expand_dims(z_mean, 2) * tf.expand_dims(z_mean, 1), axis=0)
  expectation_z_mean = tf.reduce_mean(z_mean, axis=0)
  cov_z_mean = tf.subtract(
      expectation_z_mean_z_mean_t,
      tf.expand_dims(expectation_z_mean, 1) * tf.expand_dims(
          expectation_z_mean, 0))
  return cov_z_mean


def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
  """Compute on and off diagonal regularizers for DIP-VAE models.

  Penalize deviations of covariance_matrix from the identity matrix. Uses
  different weights for the deviations of the diagonal and off diagonal entries.

  Args:
    covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
    lambda_od: Weight of penalty for off diagonal elements.
    lambda_d: Weight of penalty for diagonal elements.

  Returns:
    dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
  """
  covariance_matrix_diagonal = tf.diag_part(covariance_matrix)
  covariance_matrix_off_diagonal = covariance_matrix - tf.diag(
      covariance_matrix_diagonal)
  dip_regularizer = tf.add(
      lambda_od * tf.reduce_sum(covariance_matrix_off_diagonal**2),
      lambda_d * tf.reduce_sum((covariance_matrix_diagonal - 1)**2))
  return dip_regularizer


@gin.configurable("dip_vae")
class DIPVAE(BaseVAE):
  """DIPVAE model."""

  def __init__(self,
               lambda_od=gin.REQUIRED,
               lambda_d_factor=gin.REQUIRED,
               dip_type="i"):
    """Creates a DIP-VAE model.

    Based on Equation 6 and 7 of "Variational Inference of Disentangled Latent
    Concepts from Unlabeled Observations"
    (https://openreview.net/pdf?id=H1kG7GZAW).

    Args:
      lambda_od: Hyperparameter for off diagonal values of covariance matrix.
      lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
        lambda_d = lambda_d_factor*lambda_od.
      dip_type: "i" or "ii".
    """
    self.lambda_od = lambda_od
    self.lambda_d_factor = lambda_d_factor
    self.dip_type = dip_type

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    cov_z_mean = compute_covariance_z_mean(z_mean)
    lambda_d = self.lambda_d_factor * self.lambda_od
    if self.dip_type == "i":  # Eq 6 page 4
      # mu = z_mean is [batch_size, num_latent]
      # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
      cov_dip_regularizer = regularize_diag_off_diag_dip(
          cov_z_mean, self.lambda_od, lambda_d)
    elif self.dip_type == "ii":
      cov_enc = tf.matrix_diag(tf.exp(z_logvar))
      expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
      cov_z = expectation_cov_enc + cov_z_mean
      cov_dip_regularizer = regularize_diag_off_diag_dip(
          cov_z, self.lambda_od, lambda_d)
    else:
      raise NotImplementedError("DIP variant not supported.")
    return kl_loss + cov_dip_regularizer


def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(math.pi)
  normalization = tf.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean)
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def total_correlation(z, z_mean, z_logvar):
  """Estimate of total correlation on a batch.

  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)

  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.

  Returns:
    Total correlation estimated on a batch.
  """
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)


@gin.configurable("beta_tc_vae")
class BetaTCVAE(BaseVAE):
  """BetaTCVAE model."""

  def __init__(self, beta=gin.REQUIRED):
    """Creates a beta-TC-VAE model.

    Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.

    Args:
      beta: Hyperparameter total correlation.
    """
    self.beta = beta

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
    return tc + kl_loss
