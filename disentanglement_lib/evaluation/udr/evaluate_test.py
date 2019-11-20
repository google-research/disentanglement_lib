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

"""Tests for evaluate.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from absl.testing import parameterized
from disentanglement_lib.evaluation.udr import evaluate
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.utils import resources
import gin.tf


class EvaluateTest(parameterized.TestCase):

  def setUp(self):
    super(EvaluateTest, self).setUp()
    self.model1_dir = self.create_tempdir(
        "model1/model", cleanup=absltest.TempFileCleanup.OFF).full_path
    self.model2_dir = self.create_tempdir(
        "model2/model", cleanup=absltest.TempFileCleanup.OFF).full_path
    model_config = resources.get_file(
        "config/tests/methods/unsupervised/train_test.gin")
    gin.clear_config()
    train.train_with_gin(self.model1_dir, True, [model_config])
    train.train_with_gin(self.model2_dir, True, [model_config])

    self.output_dir = self.create_tempdir(
        "output", cleanup=absltest.TempFileCleanup.OFF).full_path

  @parameterized.parameters(
      list(resources.get_files_in_folder("config/tests/methods/udr")))
  def test_evaluate(self, gin_config):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    gin.clear_config()
    gin.parse_config_files_and_bindings([gin_config], None)
    evaluate.evaluate([self.model1_dir, self.model2_dir], self.output_dir)


if __name__ == "__main__":
  absltest.main()
