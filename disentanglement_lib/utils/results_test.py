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

"""Tests for results.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.utils import results
import tensorflow.compat.v1 as tf
import gin.tf


@gin.configurable("test")
def test_fn(value=gin.REQUIRED):
  return value


class ResultsTest(tf.test.TestCase):

  def test_namespaced_dict(self):
    """Tests namespacing functionality."""
    base_dict = {"!": "!!"}
    numbers = {"1": "one"}
    chars = {"a": "A"}
    new_dict = results.namespaced_dict(base_dict, numbers=numbers, chars=chars)
    self.assertDictEqual(new_dict, {
        "!": "!!",
        "numbers.1": "one",
        "chars.a": "A"
    })

  def test_dict_to_txt(self):
    """Tests saving functionality to txt file."""
    output_path = os.path.join(self.get_temp_dir(), "export.csv")
    output_dict = {"1": "one"}
    results.save_dict(output_path, output_dict)

  def test_gin_dict_live(self):
    """Tests namespacing functionality based on live gin config."""
    parameter_name = "test.value"
    gin.bind_parameter(parameter_name, 1)
    _ = test_fn()
    self.assertDictEqual(results.gin_dict(), {parameter_name: "1"})

  def test_gin_dict_dir(self):
    """Tests namespacing functionality based on saved gin config."""
    parameter_name = "test.value"
    gin.bind_parameter(parameter_name, 1)
    _ = test_fn()
    config_path = os.path.join(self.get_temp_dir(), "config.gin")
    with tf.gfile.GFile(config_path, "w") as f:
      f.write(gin.operative_config_str())
      f.close()
    self.assertDictEqual(results.gin_dict(config_path), {parameter_name: "1"})

  def test_aggregate_json_results(self):
    """Tests aggregation functionality."""
    tmp_dir = self.get_temp_dir()
    output_path1 = os.path.join(tmp_dir, "export_one.json")
    output_dict1 = {"1": "one"}
    results.save_dict(output_path1, output_dict1)
    output_path2 = os.path.join(tmp_dir, "export_two.json")
    output_dict2 = {"2": "two"}
    results.save_dict(output_path2, output_dict2)
    result_dict = results.aggregate_json_results(tmp_dir)
    self.assertDictEqual(result_dict, {
        "export_one.1": "one",
        "export_two.2": "two"
    })

if __name__ == "__main__":
  tf.test.main()
