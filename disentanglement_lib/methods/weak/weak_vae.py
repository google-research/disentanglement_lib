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

"""Library of losses for weakly-supervised disentanglement learning.

Implementation of weakly-supervised VAE based models from the paper
"Weakly-Supervised Disentanglement Without Compromises"
https://arxiv.org/pdf/2002.02886.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import vae
from six.moves import zip
import tensorflow.compat.v1 as tf

import gin.tf
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import TPUEstimatorSpec


@gin.configurable("weak_loss", blacklist=["z1", "z2", "labels"])
def make_weak_loss(z1, z2, labels, loss_fn=gin.REQUIRED):
  """Wrapper that creates weakly-supervised losses."""

  return loss_fn(z1, z2, labels)


@gin.configurable("group_vae")
class GroupVAEBase(vae.BaseVAE):
  """Beta-VAE with averaging from https://arxiv.org/abs/1809.02383."""

  def __init__(self, beta=gin.REQUIRED):
    """Creates a beta-VAE model with additional averaging for weak supervision.

    Based on https://arxiv.org/abs/1809.02383.

    Args:
      beta: Hyperparameter for KL divergence.
    """
    self.beta = beta

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    return self.beta * kl_loss

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    data_shape = features.get_shape().as_list()[1:]
    data_shape[0] = int(data_shape[0] / 2)
    features_1 = features[:, :data_shape[0], :, :]
    features_2 = features[:, data_shape[0]:, :, :]
    with tf.variable_scope(
        tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      z_mean, z_logvar = self.gaussian_encoder(features_1,
                                               is_training=is_training)
      z_mean_2, z_logvar_2 = self.gaussian_encoder(features_2,
                                                   is_training=is_training)
    labels = tf.squeeze(tf.one_hot(labels, z_mean.get_shape().as_list()[1]))
    kl_per_point = compute_kl(z_mean, z_mean_2, z_logvar, z_logvar_2)

    new_mean = 0.5 * z_mean + 0.5 * z_mean_2
    var_1 = tf.exp(z_logvar)
    var_2 = tf.exp(z_logvar_2)
    new_log_var = tf.math.log(0.5*var_1 + 0.5*var_2)

    mean_sample_1, log_var_sample_1 = self.aggregate(
        z_mean, z_logvar, new_mean, new_log_var, labels, kl_per_point)
    mean_sample_2, log_var_sample_2 = self.aggregate(
        z_mean_2, z_logvar_2, new_mean, new_log_var, labels, kl_per_point)
    z_sampled_1 = self.sample_from_latent_distribution(
        mean_sample_1, log_var_sample_1)
    z_sampled_2 = self.sample_from_latent_distribution(
        mean_sample_2, log_var_sample_2)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      reconstructions_1 = self.decode(z_sampled_1, data_shape, is_training)
      reconstructions_2 = self.decode(z_sampled_2, data_shape, is_training)
    per_sample_loss_1 = losses.make_reconstruction_loss(
        features_1, reconstructions_1)
    per_sample_loss_2 = losses.make_reconstruction_loss(
        features_2, reconstructions_2)
    reconstruction_loss_1 = tf.reduce_mean(per_sample_loss_1)
    reconstruction_loss_2 = tf.reduce_mean(per_sample_loss_2)
    reconstruction_loss = (0.5 * reconstruction_loss_1 +
                           0.5 * reconstruction_loss_2)
    kl_loss_1 = vae.compute_gaussian_kl(mean_sample_1, log_var_sample_1)
    kl_loss_2 = vae.compute_gaussian_kl(mean_sample_2, log_var_sample_2)
    kl_loss = 0.5 * kl_loss_1 + 0.5 * kl_loss_2
    regularizer = self.regularizer(
        kl_loss, None, None, None)

    loss = tf.add(reconstruction_loss,
                  regularizer,
                  name="loss")
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
          "elbo": -elbo,
      },
                                                every_n_iter=100)
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                       "regularizer", "kl_loss"),
                        [reconstruction_loss, -elbo, regularizer, kl_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")


@gin.configurable("group_vae_labels")
class GroupVAELabels(GroupVAEBase):
  """Class implementing the group-VAE with labels on which factor is shared."""

  def aggregate(self, z_mean, z_logvar, new_mean, new_log_var, labels,
                kl_per_point):
    return aggregate_labels(z_mean, z_logvar, new_mean, new_log_var, labels,
                            kl_per_point)


@gin.configurable("group_vae_argmax")
class GroupVAEArgmax(GroupVAEBase):
  """Class implementing the group-VAE without any label."""

  def aggregate(self, z_mean, z_logvar, new_mean, new_log_var, labels,
                kl_per_point):
    return aggregate_argmax(z_mean, z_logvar, new_mean, new_log_var, labels,
                            kl_per_point)


@gin.configurable("mlvae")
class MLVae(vae.BaseVAE):
  """Beta-VAE with averaging from https://arxiv.org/abs/1705.08841."""

  def __init__(self, beta=gin.REQUIRED):
    """Creates a beta-VAE model with additional averaging for weak supervision.

    Based on ML-VAE https://arxiv.org/abs/1705.08841.

    Args:
      beta: Hyperparameter total correlation.
    """
    self.beta = beta

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    return self.beta * kl_loss

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    data_shape = features.get_shape().as_list()[1:]
    data_shape[0] = int(data_shape[0] / 2)
    features_1 = features[:, :data_shape[0], :, :]
    features_2 = features[:, data_shape[0]:, :, :]
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      z_mean, z_logvar = self.gaussian_encoder(features_1,
                                               is_training=is_training)
      z_mean_2, z_logvar_2 = self.gaussian_encoder(features_2,
                                                   is_training=is_training)
    labels = tf.squeeze(tf.one_hot(labels, z_mean.get_shape().as_list()[1]))
    kl_per_point = compute_kl(z_mean, z_mean_2, z_logvar, z_logvar_2)

    var_1 = tf.exp(z_logvar)
    var_2 = tf.exp(z_logvar_2)
    new_var = 2*var_1 * var_2 / (var_1 + var_2)
    new_mean = (z_mean/var_1 +z_mean_2/var_2)*new_var*0.5

    new_log_var = tf.math.log(new_var)

    mean_sample_1, log_var_sample_1 = self.aggregate(
        z_mean, z_logvar, new_mean, new_log_var, labels, kl_per_point)
    mean_sample_2, log_var_sample_2 = self.aggregate(
        z_mean_2, z_logvar_2, new_mean, new_log_var, labels, kl_per_point)

    z_sampled_1 = self.sample_from_latent_distribution(
        mean_sample_1, log_var_sample_1)
    z_sampled_2 = self.sample_from_latent_distribution(
        mean_sample_2, log_var_sample_2)
    with tf.variable_scope(
        tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      reconstructions_1 = self.decode(z_sampled_1, data_shape, is_training)
      reconstructions_2 = self.decode(z_sampled_2, data_shape, is_training)
    per_sample_loss_1 = losses.make_reconstruction_loss(
        features_1, reconstructions_1)
    per_sample_loss_2 = losses.make_reconstruction_loss(
        features_2, reconstructions_2)
    reconstruction_loss_1 = tf.reduce_mean(per_sample_loss_1)
    reconstruction_loss_2 = tf.reduce_mean(per_sample_loss_2)
    reconstruction_loss = (0.5 * reconstruction_loss_1 +
                           0.5 * reconstruction_loss_2)
    kl_loss_1 = vae.compute_gaussian_kl(mean_sample_1, log_var_sample_1)
    kl_loss_2 = vae.compute_gaussian_kl(mean_sample_2, log_var_sample_2)
    kl_loss = 0.5 * kl_loss_1 + 0.5 * kl_loss_2
    regularizer = self.regularizer(
        kl_loss, None, None, None)

    loss = tf.add(reconstruction_loss,
                  regularizer,
                  name="loss")
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
          "elbo": -elbo,
      },
                                                every_n_iter=100)
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                       "regularizer", "kl_loss"),
                        [reconstruction_loss, -elbo, regularizer, kl_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")


@gin.configurable("mlvae_labels")
class MLVaeLabels(MLVae):
  """Class implementing the ML-VAE with labels on which factor is shared."""

  def aggregate(self, z_mean, z_logvar, new_mean, new_log_var, labels,
                kl_per_point):
    return aggregate_labels(z_mean, z_logvar, new_mean, new_log_var, labels,
                            kl_per_point)


@gin.configurable("mlvae_argmax")
class MLVaeArgmax(MLVae):
  """Class implementing the ML-VAE without any label."""

  def aggregate(self, z_mean, z_logvar, new_mean, new_log_var, labels,
                kl_per_point):
    return aggregate_argmax(z_mean, z_logvar, new_mean, new_log_var, labels,
                            kl_per_point)


def aggregate_labels(z_mean, z_logvar, new_mean, new_log_var, labels,
                     kl_per_point):
  """Use labels to aggregate.

  Labels contains a one-hot encoding with a single 1 of a factor shared. We
  enforce which dimension of the latent code learn which factor (dimension 1
  learns factor 1) and we enforce that each factor of variation is encoded in a
  single dimension.

  Args:
    z_mean: Mean of the encoder distribution for the original image.
    z_logvar: Logvar of the encoder distribution for the original image.
    new_mean: Average mean of the encoder distribution of the pair of images.
    new_log_var: Average logvar of the encoder distribution of the pair of
      images.
    labels: One-hot-encoding with the position of the dimension that should not
      be shared.
    kl_per_point: Distance between the two encoder distributions (unused).

  Returns:
    Mean and logvariance for the new observation.
  """
  del kl_per_point
  z_mean_averaged = tf.where(
      tf.math.equal(labels,
                    tf.expand_dims(tf.reduce_max(labels, axis=1), 1)),
      z_mean, new_mean)
  z_logvar_averaged = tf.where(
      tf.math.equal(labels,
                    tf.expand_dims(tf.reduce_max(labels, axis=1), 1)),
      z_logvar, new_log_var)
  return z_mean_averaged, z_logvar_averaged


def aggregate_argmax(z_mean, z_logvar, new_mean, new_log_var, labels,
                     kl_per_point):
  """Argmax aggregation with adaptive k.

  The bottom k dimensions in terms of distance are not averaged. K is
  estimated adaptively by binning the distance into two bins of equal width.

  Args:
    z_mean: Mean of the encoder distribution for the original image.
    z_logvar: Logvar of the encoder distribution for the original image.
    new_mean: Average mean of the encoder distribution of the pair of images.
    new_log_var: Average logvar of the encoder distribution of the pair of
      images.
    labels: One-hot-encoding with the position of the dimension that should not
      be shared.
    kl_per_point: Distance between the two encoder distributions.

  Returns:
    Mean and logvariance for the new observation.
  """
  del labels
  mask = tf.equal(
      tf.map_fn(discretize_in_bins, kl_per_point, tf.int32),
      1)
  z_mean_averaged = tf.where(mask, z_mean, new_mean)
  z_logvar_averaged = tf.where(mask, z_logvar, new_log_var)
  return z_mean_averaged, z_logvar_averaged


def discretize_in_bins(x):
  """Discretize a vector in two bins."""
  return tf.histogram_fixed_width_bins(
      x, [tf.reduce_min(x), tf.reduce_max(x)], nbins=2)


def compute_kl(z_1, z_2, logvar_1, logvar_2):
  var_1 = tf.exp(logvar_1)
  var_2 = tf.exp(logvar_2)
  return var_1/var_2 + tf.square(z_2-z_1)/var_2 - 1 + logvar_2 - logvar_1


def make_metric_fn(*names):
  """Utility function to report tf.metrics in model functions."""

  def metric_fn(*args):
    return {name: tf.metrics.mean(vec) for name, vec in zip(names, args)}

  return metric_fn
