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

"""Tests for validate.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import resources
from disentanglement_lib.validation import validate

import gin.tf


class ValidateTest(parameterized.TestCase):

  def setUp(self):
    super(ValidateTest, self).setUp()
    self.model_dir = self.create_tempdir(
        "model", cleanup=absltest.TempFileCleanup.OFF).full_path
    model_config = resources.get_file(
        "config/tests/methods/unsupervised/train_test.gin")
    train.train_with_gin(self.model_dir, True, [model_config])
    self.output_dir = self.create_tempdir(
        "output", cleanup=absltest.TempFileCleanup.OFF).full_path
    postprocess_config = resources.get_file(
        "config/tests/postprocessing/postprocess_test_configs/mean.gin")
    postprocess.postprocess_with_gin(self.model_dir, self.output_dir, True,
                                     [postprocess_config])

  @parameterized.parameters(
      list(
          resources.get_files_in_folder(
              "config/tests/validation/validation_test_configs")))
  def test_validate(self, gin_config):
    # We clear the gin config before running. Otherwise, if a prior test fails,
    # the gin config is locked and the current test fails.
    gin.clear_config()

    validate.validate_with_gin(self.output_dir,
                               self.create_tempdir().full_path, True,
                               [gin_config])


if __name__ == "__main__":
  absltest.main()
