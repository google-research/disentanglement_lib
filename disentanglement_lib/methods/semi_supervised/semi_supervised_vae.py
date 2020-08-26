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

"""Library of losses for semi-supervised disentanglement learning.

Implementation of semi-supervised VAE based models for unsupervised learning of
disentangled representations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import vae
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

import gin.tf
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import TPUEstimatorSpec


class BaseS2VAE(vae.BaseVAE):
  """Abstract base class of a basic semi-supervised Gaussian encoder model."""

  def __init__(self, factor_sizes):
    self.factor_sizes = factor_sizes

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function.

    Args:
      features: Batch of images [batch_size, 64, 64, 3].
      labels: Tuple with batch of features [batch_size, 64, 64, 3] and the
        labels [batch_size, labels_size].
      mode: Mode for the TPUEstimator.
      params: Dict with parameters.

    Returns:
      TPU estimator.
    """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    labelled_features = labels[0]
    labels = tf.to_float(labels[1])
    data_shape = features.get_shape().as_list()[1:]
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    gamma_annealed = make_annealer(self.gamma_sup, tf.train.get_global_step())
    supervised_loss = make_supervised_loss(z_mean_labelled, labels,
                                           self.factor_sizes)
    regularizer = self.unsupervised_regularizer(
        kl_loss, z_mean, z_logvar, z_sampled) + gamma_annealed * supervised_loss
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
          "elbo": -elbo,
          "supervised_loss": supervised_loss
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
                                       "regularizer", "kl_loss",
                                       "supervised_loss"), [
                                           reconstruction_loss, -elbo,
                                           regularizer, kl_loss, supervised_loss
                                       ]))
    else:
      raise NotImplementedError("Eval mode not supported.")


def sample_from_latent_distribution(z_mean, z_logvar):
  """Sample from the encoder distribution with reparametrization trick."""
  return tf.add(
      z_mean,
      tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1),
      name="latent")


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


@gin.configurable("annealer", blacklist=["gamma", "step"])
def make_annealer(gamma,
                  step,
                  iteration_threshold=gin.REQUIRED,
                  anneal_fn=gin.REQUIRED):
  """Wrapper that creates annealing function."""
  return anneal_fn(gamma, step, iteration_threshold)


@gin.configurable("fixed", blacklist=["gamma", "step"])
def fixed_annealer(gamma, step, iteration_threshold):
  """No annealing."""
  del step, iteration_threshold
  return gamma


@gin.configurable("anneal", blacklist=["gamma", "step"])
def annealed_annealer(gamma, step, iteration_threshold):
  """Linear annealing."""
  return tf.math.minimum(gamma * 1.,
                         gamma * 1. * tf.to_float(step) / iteration_threshold)


@gin.configurable("fine_tune", blacklist=["gamma", "step"])
def fine_tune_annealer(gamma, step, iteration_threshold):
  """Fine tuning.

  This annealer returns zero if step < iteration_threshold and gamma otherwise.

  Args:
    gamma: Weight of supervised loss.
    step: Current step of training.
    iteration_threshold: When to return gamma instead of zero.

  Returns:
    Either gamma or zero.
  """
  return gamma * tf.math.minimum(
      tf.to_float(1),
      tf.math.maximum(tf.to_float(0), tf.to_float(step - iteration_threshold)))


@gin.configurable("supervised_loss", blacklist=["representation", "labels"])
def make_supervised_loss(representation, labels,
                         factor_sizes=None, loss_fn=gin.REQUIRED):
  """Wrapper that creates supervised loss."""
  with tf.variable_scope("supervised_loss"):
    loss = loss_fn(representation, labels, factor_sizes)
  return loss


def normalize_labels(labels, factors_num_values):
  """Normalize the labels in [0, 1].

  Args:
    labels: Numpy array of shape (num_labelled_samples, num_factors) of Float32.
    factors_num_values: Numpy array of shape (num_factors,) containing the
      number of distinct values each factor can take.

  Returns:
    labels normalized in [0, 1].
  """
  factors_num_values_reshaped = np.repeat(
      np.expand_dims(np.float32(factors_num_values), axis=0),
      labels.shape[0],
      axis=0)
  return labels / factors_num_values_reshaped


@gin.configurable("l2", blacklist=["representation", "labels"])
def supervised_regularizer_l2(representation, labels,
                              factor_sizes=None,
                              learn_scale=True):
  """Implements a supervised l2 regularizer.

  If the number of latent dimension is greater than the number of factor of
  variations it only uses the first dimensions of the latent code to
  regularize. The number of factors of variation must be smaller or equal to the
  number of latent codes. The representation can be scaled with a learned
  scaling to match the labels or the labels are normalized in [0,1] and the
  representation is projected in the same interval using a sigmoid.

  Args:
    representation: Representation of labelled samples.
    labels: Labels for the labelled samples.
    factor_sizes: Cardinality of each factor of variation (unused).
    learn_scale: Boolean indicating whether the scale should be learned or not.

  Returns:
    L2 loss between the representation and the labels.
  """
  number_latents = representation.shape[1].value
  number_factors_of_variations = labels.shape[1].value
  assert number_latents >= number_factors_of_variations, "Not enough latents."
  if learn_scale:
    b = tf.get_variable("b", initializer=tf.constant(1.))
    return 2. * tf.nn.l2_loss(
        representation[:, :number_factors_of_variations] * b - labels)
  else:
    return 2. * tf.nn.l2_loss(
        tf.sigmoid(
            tf.expand_dims(
                representation[:, :number_factors_of_variations], axis=1)) -
        normalize_labels(labels, factor_sizes))


@gin.configurable("xent", blacklist=["representation", "labels"])
def supervised_regularizer_xent(representation, labels,
                                factor_sizes=None):
  """Implements a supervised cross_entropy regularizer.

  If the number of latent dimension is greater than the number of factor of
  variations it only uses the first dimensions of the latent code to
  regularize. If the number of factors of variation is larger than the latent
  code dimension it raise an exception. Labels are in [0, 1].

  Args:
    representation: Representation of labelled samples.
    labels: Labels for the labelled samples.
    factor_sizes: Cardinality of each factor of variation.

  Returns:
    Xent loss between the representation and the labels.
  """
  number_latents = representation.shape[1].value
  number_factors_of_variations = labels.shape[1].value
  assert number_latents >= number_factors_of_variations, "Not enough latents."
  return tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(
          logits=representation[:, :number_factors_of_variations],
          labels=normalize_labels(labels, factor_sizes)))


@gin.configurable("cov", blacklist=["representation", "labels"])
def supervised_regularizer_cov(representation, labels,
                               factor_sizes=None):
  """Implements a supervised regularizer using a covariance.

  Penalize the deviation from the identity of the covariance between
  representation and factors of varations.
  If the number of latent dimension is greater than the number of factor of
  variations it only uses the first dimensions of the latent code to
  regularize. Labels are in [0, 1].

  Args:
    representation: Representation of labelled samples.
    labels: Labels for the labelled samples.
    factor_sizes: Cardinality of each factor of variation (unused).


  Returns:
    Loss between the representation and the labels.
  """
  del factor_sizes
  number_latents = representation.shape[1].value
  number_factors_of_variations = labels.shape[1].value
  num_diagonals = tf.math.minimum(number_latents, number_factors_of_variations)
  expectation_representation = tf.reduce_mean(representation, axis=0)
  expectation_labels = tf.reduce_mean(labels, axis=0)
  representation_centered = representation - expectation_representation
  labels_centered = labels - expectation_labels
  covariance = tf.reduce_mean(
      tf.expand_dims(representation_centered, 2) * tf.expand_dims(
          labels_centered, 1),
      axis=0)
  return 2. * tf.nn.l2_loss(
      tf.linalg.set_diag(covariance, tf.zeros([num_diagonals])))


@gin.configurable("embed", blacklist=["representation", "labels",
                                      "factor_sizes"])
def supervised_regularizer_embed(representation, labels,
                                 factor_sizes, sigma=gin.REQUIRED,
                                 use_order=False):
  """Embed factors in 1d and compute softmax with the representation.

  Assume a factor of variation indexed by j can take k values. We embed each
  value into k real numbers e_1, ..., e_k. Call e_label(r_j) the embedding of an
  observed label for the factor j. Then, for a dimension r_j of the
  representation, the loss is computed as
  exp(-((r_j - e_label(r_j))*sigma)^2)/sum_{i=1}^k exp(-(r_j - e_i)).
  We compute this term for each factor of variation j and each point. Finally,
  we add these terms into a single number.

  Args:
    representation: Computed representation, tensor of shape (batch_size,
      num_latents)
    labels: Observed values for the factors of variation, tensor of shape
      (batch_size, num_factors).
    factor_sizes: Cardinality of each factor of variation.
    sigma: Temperature for the softmax. Set to "learn" if to be learned.
    use_order: Boolean indicating whether to use the ordering information in the
      factors of variations or not.

  Returns:
    Supervised loss based on the softmax between embedded labels and
    representation.
  """
  number_factors_of_variations = labels.shape[1].value
  supervised_representation = representation[:, :number_factors_of_variations]
  loss = []
  for i in range(number_factors_of_variations):
    with tf.variable_scope(str(i), reuse=tf.AUTO_REUSE):
      if use_order:
        bias = tf.get_variable("bias", [])
        slope = tf.get_variable("slope", [])
        embedding = tf.range(factor_sizes[i], dtype=tf.float32)*slope + bias
      else:
        embedding = tf.get_variable("embedding", [factor_sizes[i]])
      if sigma == "learn":
        sigma_value = tf.get_variable("sigma", [1])
      else:
        sigma_value = sigma
    logits = -tf.square(
        (tf.expand_dims(supervised_representation[:, i], axis=1) - embedding) *
        sigma_value)
    one_hot_labels = tf.one_hot(tf.to_int32(labels[:, i]), factor_sizes[i])
    loss += [tf.losses.softmax_cross_entropy(one_hot_labels, logits)]
  return tf.reduce_sum(tf.add_n(loss))


@gin.configurable("s2_vae")
class S2BetaVAE(BaseS2VAE):
  """Semi-supervised BetaVAE model."""

  def __init__(self, factor_sizes, beta=gin.REQUIRED, gamma_sup=gin.REQUIRED):
    """Creates a semi-supervised beta-VAE model.

    Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl) with additional supervision.

    Args:
      factor_sizes: Size of each factor of variation.
      beta: Hyperparameter for the unsupervised regularizer.
      gamma_sup: Hyperparameter for the supervised regularizer.

    Returns:
      model_fn: Model function for TPUEstimator.
    """
    self.beta = beta
    self.gamma_sup = gamma_sup
    super(S2BetaVAE, self).__init__(factor_sizes)

  def unsupervised_regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    """Standard betaVAE regularizer."""
    del z_mean, z_logvar, z_sampled
    return self.beta * kl_loss


@gin.configurable("supervised")
class SupervisedVAE(BaseS2VAE):
  """Fully supervised method build on top of VAE to have visualizations."""

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function.

    Args:
      features: Batch of images [batch_size, 64, 64, 3].
      labels: Tuple with batch of features [batch_size, 64, 64, 3] and the
        labels [batch_size, labels_size].
      mode: Mode for the TPUEstimator.
      params: Dict with parameters.

    Returns:
      TPU Estimator.
    """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    labelled_features = labels[0]
    labels = tf.to_float(labels[1])
    data_shape = features.get_shape().as_list()[1:]
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(
        tf.stop_gradient(z_sampled), data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    supervised_loss = make_supervised_loss(z_mean_labelled, labels,
                                           self.factor_sizes)
    regularizer = supervised_loss
    loss = tf.add(reconstruction_loss, regularizer, name="loss")
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = optimizers.make_vae_optimizer()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
      train_op = tf.group([train_op, update_ops])
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)

      logging_hook = tf.train.LoggingTensorHook({
          "loss": loss,
          "reconstruction_loss": reconstruction_loss,
          "supervised_loss": supervised_loss
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
          eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                       "supervised_loss"),
                        [reconstruction_loss, regularizer, supervised_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")


def mine(x, z, name_net="estimator_network"):
  """Computes I(X, Z).

  Uses the algorithm in "Mutual Information Neural Estimation"
  (https://arxiv.org/pdf/1801.04062.pdf).

  Args:
    x: Samples from x [batch_size, size_x].
    z: Samples from z [batch_size, size_z].
    name_net: Scope for the variables forming the network.

  Returns:
    Estimate of the mutual information and the update op for the optimizer.
  """
  z_shuffled = vae.shuffle_codes(z)

  concat_x_x = tf.concat([x, x], axis=0)
  concat_z_z_shuffled = tf.stop_gradient(tf.concat([z, z_shuffled], axis=0))

  with tf.variable_scope(name_net, reuse=tf.AUTO_REUSE):
    d1_x = tf.layers.dense(concat_x_x, 20, name="d1_x")
    d1_z = tf.layers.dense(concat_z_z_shuffled, 20, name="d1_z")
    d1 = tf.nn.elu(d1_x + d1_z, name="d1")
    d2 = tf.layers.dense(d1, 1, name="d2")

  batch_size = tf.shape(x)[0]
  pred_x_z = d2[:batch_size]
  pred_x_z_shuffled = d2[batch_size:]
  loss = -(
      tf.reduce_mean(pred_x_z, axis=0) + tf.math.log(tf.to_float(batch_size)) -
      tf.math.reduce_logsumexp(pred_x_z_shuffled))
  all_variables = tf.trainable_variables()
  mine_vars = [var for var in all_variables if "estimator_network" in var.name]
  mine_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(
      loss=loss, var_list=mine_vars)
  return -loss, mine_op


@gin.configurable("s2_mine_vae")
class MineVAE(BaseS2VAE):
  """MineVAE model."""

  def __init__(self, factor_sizes, gamma_sup=gin.REQUIRED, beta=gin.REQUIRED):
    """Creates a semi-supervised MineVAE model.

    Regularize mutual information using mine.

    Args:
      factor_sizes: Size of each factor of variation.
      gamma_sup: Hyperparameter for the supervised regularizer.
      beta: Hyperparameter for the unsupervised regularizer.
    """
    self.gamma_sup = gamma_sup
    self.beta = beta
    super(MineVAE, self).__init__(factor_sizes)

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    labelled_features = labels[0]
    labels = tf.to_float(labels[1])
    data_shape = features.get_shape().as_list()[1:]
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)

    supervised_loss = []
    mine_ops = []

    for l in range(labels.get_shape().as_list()[1]):
      for r in range(z_mean.get_shape().as_list()[1]):
        label_for_mi = tf.layers.flatten(labels[:, l])
        representation_for_mi = tf.layers.flatten(z_mean_labelled[:, r])
        mi_lr, op_lr = mine(representation_for_mi, label_for_mi,
                            "estimator_network_%d_%d" % (l, r))
        if l != r:
          supervised_loss = supervised_loss + [tf.math.square(mi_lr)]
        mine_ops = mine_ops + [op_lr]
    supervised_loss = tf.reshape(tf.add_n(supervised_loss), [])
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    standard_vae_loss = tf.add(
        reconstruction_loss, self.beta * kl_loss, name="VAE_loss")
    gamma_annealed = make_annealer(self.gamma_sup, tf.train.get_global_step())
    s2_mine_vae_loss = tf.add(
        standard_vae_loss, gamma_annealed * supervised_loss,
        name="s2_factor_VAE_loss")
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer_vae = optimizers.make_vae_optimizer()
      all_variables = tf.trainable_variables()
      encoder_vars = [var for var in all_variables if "encoder" in var.name]
      decoder_vars = [var for var in all_variables if "decoder" in var.name]

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op_vae = optimizer_vae.minimize(
          loss=s2_mine_vae_loss,
          global_step=tf.train.get_global_step(),
          var_list=encoder_vars + decoder_vars)
      train_op = tf.group(train_op_vae, mine_ops, update_ops)
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)
      logging_hook = tf.train.LoggingTensorHook({
          "loss": s2_mine_vae_loss,
          "reconstruction_loss": reconstruction_loss,
          "supervised_loss": supervised_loss,
      },
                                                every_n_iter=50)
      return TPUEstimatorSpec(
          mode=mode,
          loss=s2_mine_vae_loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return TPUEstimatorSpec(
          mode=mode,
          loss=s2_mine_vae_loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "supervised_loss",
                                       "kl_loss"),
                        [reconstruction_loss, supervised_loss, kl_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")


@gin.configurable("s2_factor_vae")
class S2FactorVAE(BaseS2VAE):
  """FactorVAE model."""

  def __init__(self, factor_sizes, gamma=gin.REQUIRED, gamma_sup=gin.REQUIRED):
    """Creates a semi-supervised FactorVAE model.

    Implementing Eq. 2 of "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983).

    Args:
      factor_sizes: Size of each factor of variation.
      gamma: Hyperparameter for the unsupervised regularizer.
      gamma_sup: Hyperparameter for the supervised regularizer.
    """
    self.gamma = gamma
    self.gamma_sup = gamma_sup
    super(S2FactorVAE, self).__init__(factor_sizes)

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    labelled_features = labels[0]
    labels = tf.to_float(labels[1])
    data_shape = features.get_shape().as_list()[1:]
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    z_shuffle = vae.shuffle_codes(z_sampled)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      logits_z, probs_z = architectures.make_discriminator(
          z_sampled, is_training=is_training)
      _, probs_z_shuffle = architectures.make_discriminator(
          z_shuffle, is_training=is_training)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    standard_vae_loss = tf.add(reconstruction_loss, kl_loss, name="VAE_loss")
    # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
    tc_loss_per_sample = logits_z[:, 0] - logits_z[:, 1]
    tc_loss = tf.reduce_mean(tc_loss_per_sample, axis=0)
    regularizer = kl_loss + self.gamma * tc_loss
    gamma_annealed = make_annealer(self.gamma_sup, tf.train.get_global_step())
    supervised_loss = make_supervised_loss(z_mean_labelled, labels,
                                           self.factor_sizes)
    s2_factor_vae_loss = tf.add(
        standard_vae_loss,
        self.gamma * tc_loss + gamma_annealed * supervised_loss,
        name="s2_factor_VAE_loss")
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
          loss=s2_factor_vae_loss,
          global_step=tf.train.get_global_step(),
          var_list=encoder_vars + decoder_vars)
      train_op_discr = optimizer_discriminator.minimize(
          loss=-discr_loss,
          global_step=tf.train.get_global_step(),
          var_list=discriminator_vars)
      train_op = tf.group(train_op_vae, train_op_discr, update_ops)
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)
      logging_hook = tf.train.LoggingTensorHook({
          "loss": s2_factor_vae_loss,
          "reconstruction_loss": reconstruction_loss
      },
                                                every_n_iter=50)
      return TPUEstimatorSpec(
          mode=mode,
          loss=s2_factor_vae_loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return TPUEstimatorSpec(
          mode=mode,
          loss=s2_factor_vae_loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                       "kl_loss", "supervised_loss"), [
                                           reconstruction_loss, regularizer,
                                           kl_loss, supervised_loss
                                       ]))
    else:
      raise NotImplementedError("Eval mode not supported.")


@gin.configurable("s2_dip_vae")
class S2DIPVAE(BaseS2VAE):
  """Semi-supervised DIPVAE model."""

  def __init__(self,
               factor_sizes,
               lambda_od=gin.REQUIRED,
               lambda_d_factor=gin.REQUIRED,
               gamma_sup=gin.REQUIRED,
               dip_type="i"):
    """Creates a DIP-VAE model.

    Based on Equation 6 and 7 of "Variational Inference of Disentangled Latent
    Concepts from Unlabeled Observations"
    (https://openreview.net/pdf?id=H1kG7GZAW).

    Args:
      factor_sizes: Size of each factor of variation.
      lambda_od: Hyperparameter for off diagonal values of covariance matrix.
      lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
        lambda_d = lambda_d_factor*lambda_od.
      gamma_sup: Hyperparameter for the supervised regularizer.
      dip_type: "i" or "ii".
    """
    self.lambda_od = lambda_od
    self.lambda_d_factor = lambda_d_factor
    self.dip_type = dip_type
    self.gamma_sup = gamma_sup
    super(S2DIPVAE, self).__init__(factor_sizes)

  def unsupervised_regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    cov_z_mean = vae.compute_covariance_z_mean(z_mean)
    lambda_d = self.lambda_d_factor * self.lambda_od
    if self.dip_type == "i":  # Eq 6 page 4
      # mu = z_mean is [batch_size, num_latent]
      # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
      cov_dip_regularizer = vae.regularize_diag_off_diag_dip(
          cov_z_mean, self.lambda_od, lambda_d)
    elif self.dip_type == "ii":
      cov_enc = tf.matrix_diag(tf.exp(z_logvar))
      expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
      cov_z = expectation_cov_enc + cov_z_mean
      cov_dip_regularizer = vae.regularize_diag_off_diag_dip(
          cov_z, self.lambda_od, lambda_d)
    else:
      raise NotImplementedError("DIP variant not supported.")
    return kl_loss + cov_dip_regularizer


@gin.configurable("s2_beta_tc_vae")
class S2BetaTCVAE(BaseS2VAE):
  """Semi-supervised BetaTCVAE model."""

  def __init__(self, factor_sizes, beta=gin.REQUIRED, gamma_sup=gin.REQUIRED):
    """Creates a beta-TC-VAE model.

    Based on Equation 5 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 5 can be written as ELBO + (1 - beta) * TC.

    Args:
      factor_sizes: Size of each factor of variation.
      beta: Hyperparameter total correlation.
      gamma_sup: Hyperparameter for the supervised regularizer.
    """
    self.beta = beta
    self.gamma_sup = gamma_sup
    super(S2BetaTCVAE, self).__init__(factor_sizes)

  def unsupervised_regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    tc = (self.beta - 1.) * vae.total_correlation(z_sampled, z_mean, z_logvar)
    return tc + kl_loss
