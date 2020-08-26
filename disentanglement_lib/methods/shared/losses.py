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

"""Library of commonly used losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import gin.tf


@gin.configurable("bernoulli_loss", whitelist=["subtract_true_image_entropy"])
def bernoulli_loss(true_images,
                   reconstructed_images,
                   activation,
                   subtract_true_image_entropy=False):
  """Computes the Bernoulli loss."""
  flattened_dim = np.prod(true_images.get_shape().as_list()[1:])
  reconstructed_images = tf.reshape(
      reconstructed_images, shape=[-1, flattened_dim])
  true_images = tf.reshape(true_images, shape=[-1, flattened_dim])

  # Because true images are not binary, the lower bound in the xent is not zero:
  # the lower bound in the xent is the entropy of the true images.
  if subtract_true_image_entropy:
    dist = tfp.distributions.Bernoulli(
        probs=tf.clip_by_value(true_images, 1e-6, 1 - 1e-6))
    loss_lower_bound = tf.reduce_sum(dist.entropy(), axis=1)
  else:
    loss_lower_bound = 0

  if activation == "logits":
    loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=reconstructed_images, labels=true_images),
        axis=1)
  elif activation == "tanh":
    reconstructed_images = tf.clip_by_value(
        tf.nn.tanh(reconstructed_images) / 2 + 0.5, 1e-6, 1 - 1e-6)
    loss = -tf.reduce_sum(
        true_images * tf.log(reconstructed_images) +
        (1 - true_images) * tf.log(1 - reconstructed_images),
        axis=1)
  else:
    raise NotImplementedError("Activation not supported.")

  return loss - loss_lower_bound


@gin.configurable("l2_loss", whitelist=[])
def l2_loss(true_images, reconstructed_images, activation):
  """Computes the l2 loss."""
  if activation == "logits":
    return tf.reduce_sum(
        tf.square(true_images - tf.nn.sigmoid(reconstructed_images)), [1, 2, 3])
  elif activation == "tanh":
    reconstructed_images = tf.nn.tanh(reconstructed_images) / 2 + 0.5
    return tf.reduce_sum(
        tf.square(true_images - reconstructed_images), [1, 2, 3])
  else:
    raise NotImplementedError("Activation not supported.")


@gin.configurable(
    "reconstruction_loss", blacklist=["true_images", "reconstructed_images"])
def make_reconstruction_loss(true_images,
                             reconstructed_images,
                             loss_fn=gin.REQUIRED,
                             activation="logits"):
  """Wrapper that creates reconstruction loss."""
  with tf.variable_scope("reconstruction_loss"):
    per_sample_loss = loss_fn(true_images, reconstructed_images, activation)
  return per_sample_loss
