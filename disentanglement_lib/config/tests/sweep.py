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

"""Test study."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.config import study
from disentanglement_lib.utils import resources


class TestStudy(study.Study):
  """Defines a study for testing."""

  def get_model_config(self, model_num=0):
    """Returns model bindings and config file."""
    return [], resources.get_file(
        "config/tests/methods/unsupervised/train_test.gin")

  def get_postprocess_config_files(self):
    """Returns postprocessing config files."""
    return list(
        resources.get_files_in_folder(
            "config/tests/postprocessing/postprocess_test_configs"))

  def get_eval_config_files(self):
    """Returns evaluation config files."""
    return list(
        resources.get_files_in_folder(
            "config/tests/evaluation/evaluate_test_configs"))
