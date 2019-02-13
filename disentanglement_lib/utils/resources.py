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

"""Utility to access resources in package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def get_file(path):
  """Returns path relative to file."""
  from pkg_resources import resource_filename  # pylint: disable=g-bad-import-order, g-import-not-at-top
  return resource_filename("disentanglement_lib", path)


def get_files_in_folder(path):
  import pkg_resources  # pylint: disable=g-bad-import-order, g-import-not-at-top
  for name in pkg_resources.resource_listdir("disentanglement_lib", path):
    new_path = path + "/" + name
    if not pkg_resources.resource_isdir("disentanglement_lib", new_path):
      yield pkg_resources.resource_filename("disentanglement_lib", new_path)


