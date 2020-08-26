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

"""Defines a common interface for Gaussian encoder based models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import gin.tf


class GaussianEncoderModel(object):
  """Abstract base class of a Gaussian encoder model."""

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function used for training/evaluation."""
    raise NotImplementedError()

  def gaussian_encoder(self, input_tensor, is_training):
    """Applies the Gaussian encoder to images.

    Args:
      input_tensor: Tensor with the observations to be encoded.
      is_training: Boolean indicating whether in training mode.

    Returns:
      Tuple of tensors with the mean and log variance of the Gaussian encoder.
    """
    raise NotImplementedError()

  def decode(self, latent_tensor, observation_shape, is_training):
    """Decodes the latent_tensor to an observation."""
    raise NotImplementedError()

  def sample_from_latent_distribution(self, z_mean, z_logvar):
    """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
    return tf.add(
        z_mean,
        tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1),
        name="sampled_latent_variable")


@gin.configurable("export_as_tf_hub", whitelist=[])
def export_as_tf_hub(gaussian_encoder_model,
                     observation_shape,
                     checkpoint_path,
                     export_path,
                     drop_collections=None):
  """Exports the provided GaussianEncoderModel as a TFHub module.

  Args:
    gaussian_encoder_model: GaussianEncoderModel to be exported.
    observation_shape: Tuple with the observations shape.
    checkpoint_path: String with path where to load weights from.
    export_path: String with path where to save the TFHub module to.
    drop_collections: List of collections to drop from the graph.
  """

  def module_fn(is_training):
    """Module function used for TFHub export."""
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      # Add a signature for the Gaussian encoder.
      image_placeholder = tf.placeholder(
          dtype=tf.float32, shape=[None] + observation_shape)
      mean, logvar = gaussian_encoder_model.gaussian_encoder(
          image_placeholder, is_training)
      hub.add_signature(
          name="gaussian_encoder",
          inputs={"images": image_placeholder},
          outputs={
              "mean": mean,
              "logvar": logvar
          })

      # Add a signature for reconstructions.
      latent_vector = gaussian_encoder_model.sample_from_latent_distribution(
          mean, logvar)
      reconstructed_images = gaussian_encoder_model.decode(
          latent_vector, observation_shape, is_training)
      hub.add_signature(
          name="reconstructions",
          inputs={"images": image_placeholder},
          outputs={"images": reconstructed_images})

      # Add a signature for the decoder.
      latent_placeholder = tf.placeholder(
          dtype=tf.float32, shape=[None, mean.get_shape()[1]])
      decoded_images = gaussian_encoder_model.decode(latent_placeholder,
                                                     observation_shape,
                                                     is_training)

      hub.add_signature(
          name="decoder",
          inputs={"latent_vectors": latent_placeholder},
          outputs={"images": decoded_images})

  # Export the module.
  # Two versions of the model are exported:
  #   - one for "test" mode (the default tag)
  #   - one for "training" mode ("is_training" tag)
  # In the case that the encoder/decoder have dropout, or BN layers, these two
  # graphs are different.
  tags_and_args = [
      ({"train"}, {"is_training": True}),
      (set(), {"is_training": False}),
  ]
  spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args,
                                drop_collections=drop_collections)
  spec.export(export_path, checkpoint_path=checkpoint_path)
