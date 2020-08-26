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

"""Utility functions to save results and gin configs in result directory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import os
import re
import uuid
from distutils import dir_util
import numpy as np
import simplejson as json
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
import gin.tf


def update_result_directory(result_directory,
                            step_name,
                            results_dict,
                            old_result_directory=None):
  """One stop solution for updating the result directory.

  1. Copies old_result_directory to result_directory if not None.
  2. Adds a unique id to the result_dict.
  3. Saves the gin config to the gin/{step_name}.gin file.
  4. Saves the gin config dict to json/config_{step_name}.json file.
  5. Saves the results_dict to the json/results_{step_name}.json file.
  6. Aggregates all dicts in json/*.json into a new
     aggregate/aggregate_results_{step_name}.json file.

  Args:
    result_directory: String with path to result directory to update.
    step_name: String with the step name. This will be used as a name space.
    results_dict: Dictionary with results to be persisted.
    old_result_directory: String with path to old directory from which to copy
      results from (if not set to None).
  """
  json_dir = os.path.join(result_directory, "json")

  # Copy the old output directory to the new one if required.
  if old_result_directory is not None:
    copydir(old_result_directory, result_directory)
  else:
    # Creates the output directory if necessary.
    if not tf.gfile.IsDirectory(result_directory):
      tf.gfile.MakeDirs(result_directory)

  # Add unique id to the result dict (useful for obtaining unique runs).
  results_dict["uuid"] = str(uuid.uuid4())

  # Save the gin config in the gin format.
  gin_config_path = os.path.join(result_directory, "gin",
                                 "{}.gin".format(step_name))
  save_gin(gin_config_path)

  # Save gin config in JSON file for aggregation.
  gin_json_path = os.path.join(json_dir, "{}_config.json".format(step_name))
  save_dict(gin_json_path, gin_dict(gin_config_path))

  # Save the results as a dictionary in JSON.
  results_json_path = os.path.join(json_dir,
                                   "{}_results.json".format(step_name))
  save_dict(results_json_path, results_dict)

  # Aggregate all the results present in the result_dir so far.
  aggregate_dict = aggregate_json_results(json_dir)
  aggregate_json_path = results_json_path = os.path.join(
      result_directory, "aggregate", "{}.json".format(step_name))
  save_dict(aggregate_json_path, aggregate_dict)


def _copy_recursively(path_to_old_dir, path_to_new_dir):
  return dir_util.copy_tree(path_to_old_dir, path_to_new_dir)


def copydir(path_to_old_dir, path_to_new_dir):
  """Copies a directory to a new path which is created if necessary.

  Args:
    path_to_old_dir: String with old directory path.
    path_to_new_dir: String with new directory path.
  """
  directory = os.path.dirname(path_to_new_dir)
  if not tf.gfile.IsDirectory(directory):
    tf.gfile.MakeDirs(directory)
  _copy_recursively(path_to_old_dir, path_to_new_dir)


def save_gin(config_path):
  """Saves the operative gin config to a gin config file.

  Args:
    config_path: String with path where to save the gin config.
  """
  # Ensure that the folder exists.
  directory = os.path.dirname(config_path)
  if not tf.gfile.IsDirectory(directory):
    tf.gfile.MakeDirs(directory)
  # Save the actual config.
  with tf.gfile.GFile(config_path, "w") as f:
    f.write(gin.operative_config_str())


class Encoder(json.JSONEncoder):
  """Custom encoder so that we can save special types in JSON."""

  def default(self, obj):
    if isinstance(obj, (np.float_, np.float32, np.float16, np.float64)):
      return float(obj)
    elif isinstance(obj,
                    (np.intc, np.intp, np.int_, np.int8, np.int16, np.int32,
                     np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, np.ndarray):
      obj = obj.tolist()
    return json.JSONEncoder.default(self, obj)


def save_dict(config_path, dict_with_info):
  """Saves a dict to a JSON file.

  Args:
    config_path: String with path where to save the gin config.
    dict_with_info: Dictionary with keys and values which are safed as strings.
  """
  # Ensure that the folder exists.
  directory = os.path.dirname(config_path)
  if not tf.gfile.IsDirectory(directory):
    tf.gfile.MakeDirs(directory)
  # Save the actual config.
  with tf.gfile.GFile(config_path, "w") as f:
    json.dump(dict_with_info, f, cls=Encoder, indent=2)


def gin_dict(config_path=None):
  """Returns dict with gin configs based on active config or config file.

  Args:
    config_path: Path to gin config file. If set to None (default), currently
      active bindings using gin.operative_config_str() are used.

  Returns:
    Dictionary with gin bindings as string keys and string values.
  """
  result = {}
  # Gin does not allow to directly retrieve a dictionary but it allows to
  # obtain a string with all active configs in human readable format.
  if config_path is None:
    operative_str = gin.operative_config_str()
  else:
    with tf.gfile.GFile(config_path, "r") as f:
      operative_str = f.read()
  for line in operative_str.split("\n"):
    # We need to filter out the auto-generated comments and make sure the line
    # contains a valid assignment.
    if not line.startswith("#") and " = " in line:
      key, value = line.split(" = ", 2)
      result[key] = value
  return result


def namespaced_dict(base_dict=None, **named_dicts):
  """Fuses several named dictionaries into one dict by namespacing the keys.

  Example:
  >> base_dict = {"!": "!!"}
  >> numbers = {"1": "one"}
  >> chars = {"a": "A"}
  >> new_dict = namespaced_dict(base_dict, numbers=numbers, chars=chars)
  >> # new_dict = {"!": "!!", "numbers.1": "one", "chars.a": "A"}

  Args:
    base_dict: Base dictionary of which a deepcopy will be use to fuse the named
      dicts into. If set to None, an empty dict will be used.
    **named_dicts: Named dictionary of dictionaries that will be namespaced and
      fused into base_dict. All keys should be string as the new key of any
      value will be outer key + "." + inner key.

  Returns:
    Dictionary with aggregated items.
  """
  result = {} if base_dict is None else copy.deepcopy(base_dict)
  for outer_key, inner_dict in named_dicts.items():
    for inner_key, value in inner_dict.items():
      result["{}.{}".format(outer_key, inner_key)] = value
  return result


def aggregate_json_results(base_path):
  """Aggregates all the result files in a directory into a namespaced dict.

  Args:
    base_path: String with the directory containing JSON files that only contain
      dictionaries.

  Returns:
    Namespaced dictionary with the results.
  """
  result = {}
  compiled_pattern = re.compile(r"(.*)\.json")
  for filename in gfile.ListDirectory(base_path):
    match = compiled_pattern.match(filename)
    if match:
      path = os.path.join(base_path, filename)
      with tf.gfile.GFile(path, "r") as f:
        result[match.group(1)] = json.load(f)
  return namespaced_dict(**result)
