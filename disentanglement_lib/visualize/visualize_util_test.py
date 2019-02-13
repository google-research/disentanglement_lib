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

"""Tests for visualize_util.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl.testing import absltest
from disentanglement_lib.visualize import visualize_util
import numpy as np


class VisualizeUtilTest(absltest.TestCase):

  def test_save_image(self):
    image = np.zeros((128, 256, 3), dtype=np.float32)
    path = os.path.join(self.create_tempdir().full_path, "save_image.png")
    visualize_util.save_image(image, path)

  def test_save_image_grayscale(self):
    image = np.ones((128, 256, 1), dtype=np.float32)
    path = os.path.join(self.create_tempdir().full_path,
                        "save_image_grayscale.png")
    visualize_util.save_image(image, path)

  def test_grid_save_images(self):
    images = np.zeros((18, 128, 256, 3), dtype=np.float32)
    path = os.path.join(self.create_tempdir().full_path, "grid_save_images.png")
    visualize_util.grid_save_images(images, path)

  def test_save_animation(self):
    path = os.path.join(self.create_tempdir().full_path, "animation.gif")
    images = np.ones((18, 128, 256, 3), dtype=np.float32)
    visualize_util.save_animation([images, images], path, fps=18)

  def cycle_factor(self):
    result = list(visualize_util.cycle_factor(1, 3, 4))
    shouldbe = [1, 2, 2, 1, 0, 0]
    self.assertAllEqual(result, shouldbe)


if __name__ == "__main__":
  absltest.main()
