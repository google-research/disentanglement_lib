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

"""Tests for the architectures.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from disentanglement_lib.methods.shared import architectures
import numpy as np
import tensorflow.compat.v1 as tf


class ArchitecturesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('fc_encoder', architectures.fc_encoder),
      ('conv_encoder', architectures.conv_encoder),
  )
  def test_encoder(self, encoder_f):
    minibatch = np.ones(shape=(10, 64, 64, 1), dtype=np.float32)
    input_tensor = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
    latent_mean, latent_logvar = encoder_f(input_tensor, 10)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(
          [latent_mean, latent_logvar], feed_dict={input_tensor: minibatch})

  @parameterized.named_parameters(
      ('fc_decoder', architectures.fc_decoder),
      ('deconv_decoder', architectures.deconv_decoder),
  )
  def test_decoder(self, decoder_f):
    latent_variable = np.ones(shape=(10, 15), dtype=np.float32)
    input_tensor = tf.placeholder(tf.float32, shape=(None, 15))
    images = decoder_f(input_tensor, [64, 64, 1])
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(images, feed_dict={input_tensor: latent_variable})

  @parameterized.named_parameters(
      ('fc_discriminator', architectures.fc_discriminator),
  )
  def test_discriminator(self, discriminator_f):
    images = np.ones(shape=(32, 10), dtype=np.float32)
    input_tensor = tf.placeholder(tf.float32, shape=(None, 10))
    logits, probs = discriminator_f(input_tensor)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run([logits, probs], feed_dict={input_tensor: images})

if __name__ == '__main__':
  tf.test.main()
