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

"""Abstract base class for a study."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Study(object):
  """Abstract base class used for different studies."""

  def get_model_config(self, model_num=0):
    """Returns model bindings and config file."""
    raise NotImplementedError()

  def print_model_config(self, model_num=0):
    """Prints model bindings and config file."""
    model_bindings, model_config_file = self.get_model_config(model_num)
    print("Gin base config for model training:")
    print("--")
    print(model_config_file)
    print()
    print("Gin bindings for model training:")
    print("--")
    for binding in model_bindings:
      print(binding)

  def get_postprocess_config_files(self):
    """Returns postprocessing config files."""
    raise NotImplementedError()

  def print_postprocess_config(self):
    """Prints postprocessing config files."""
    print("Gin config files for postprocessing (random seeds may be set "
          "later):")
    print("--")
    for path in self.get_postprocess_config_files():
      print(path)

  def get_eval_config_files(self):
    """Returns evaluation config files."""
    raise NotImplementedError()

  def print_eval_config(self):
    """Prints evaluation config files."""
    print("Gin config files for evaluation (random seeds may be set later):")
    print("--")
    for path in self.get_eval_config_files():
      print(path)
