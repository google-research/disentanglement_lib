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

"""Tests for the weakly-supervised training protocol."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.methods.weak import train_weak_lib
from disentanglement_lib.utils import resources
import tensorflow as tf

import gin.tf

MODELS_TEST = [
    ["model.model = @group_vae_labels()",
     "dynamics.k = 1", "group_vae.beta = 1."],
    ["model.model = @group_vae_argmax()",
     "dynamics.k = 1", "group_vae.beta = 1."],
    ["model.model = @group_vae_labels()",
     "dynamics.k = -1", "group_vae.beta = 1."],
    ["model.model = @group_vae_argmax()",
     "dynamics.k = -1", "group_vae.beta = 1."],
    ["model.model = @group_vae_labels()",
     "dynamics.k = 2", "group_vae.beta = 1."],
    ["model.model = @group_vae_argmax()",
     "dynamics.k = 2", "group_vae.beta = 1."],
    ["model.model = @mlvae_labels()",
     "dynamics.k = 1", "mlvae.beta = 1."],
    ["model.model = @mlvae_argmax()",
     "dynamics.k = 1", "mlvae.beta = 1."],
    ["model.model = @mlvae_labels()",
     "dynamics.k = -1", "mlvae.beta = 1."],
    ["model.model = @mlvae_argmax()",
     "dynamics.k = -1", "mlvae.beta = 1."],
    ["model.model = @mlvae_labels()",
     "dynamics.k = 2", "mlvae.beta = 1."],
    ["model.model = @mlvae_argmax()",
     "dynamics.k = 2", "mlvae.beta = 1."]]


def _config_generator():
  """Yields all model configurations that should be tested."""
  model_config_path = resources.get_file(
      "config/tests/methods/unsupervised/train_test.gin")
  for model in MODELS_TEST:
    yield [model_config_path], model


class TrainTest(parameterized.TestCase):

  @parameterized.parameters(list(_config_generator()))
  def test_train_model(self, gin_configs, gin_bindings):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    gin.clear_config()

    train_weak_lib.train_with_gin(
        self.create_tempdir().full_path, True, gin_configs, gin_bindings)


class WeakDataTest(parameterized.TestCase, tf.test.TestCase):

  def test_weak_data(self):
    ground_truth_data = dummy_data.DummyData()
    binding = ["dynamics.k = 1"]
    gin.parse_config_files_and_bindings([], binding)
    dataset = \
      train_weak_lib.weak_dataset_from_ground_truth_data(
          ground_truth_data, 0)
    one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_element = one_shot_iterator.get_next()
    with self.test_session() as sess:
      elem = sess.run(next_element)
      self.assertEqual(elem[0].shape, (128, 64, 1))
      self.assertEqual(elem[1].shape, (1,))


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
