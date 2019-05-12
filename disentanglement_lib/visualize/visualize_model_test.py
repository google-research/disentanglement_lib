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

"""Test for visualize_model.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from absl.testing import parameterized
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.utils import resources
from disentanglement_lib.visualize import visualize_model

MODEL_CONFIG_PATH = "../config/tests/methods/unsupervised/train_test.gin"


class VisualizeTest(parameterized.TestCase):

  @parameterized.parameters(
      ("logits"),
      ("tanh"),
  )
  def test_visualize_sigmoid(self, activation):
    activation_binding = (
        "reconstruction_loss.activation = '{}'".format(activation))
    self.model_dir = self.create_tempdir(
        "model_{}".format(activation),
        cleanup=absltest.TempFileCleanup.OFF).full_path
    train.train_with_gin(self.model_dir, True, [
        resources.get_file("config/tests/methods/unsupervised/train_test.gin")
    ], [activation_binding])
    visualize_model.visualize(
        self.model_dir,
        self.create_tempdir("visualization_{}".format(activation)).full_path,
        True,
        num_animations=1,
        num_frames=4,
        num_points_irs=100)


if __name__ == "__main__":
  absltest.main()
