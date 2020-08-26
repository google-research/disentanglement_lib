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

"""Helpers to save and load the results of many experiments in a single JSON."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import multiprocessing
from absl import logging
import pandas as pd
import simplejson as json
from tensorflow.compat.v1 import gfile


def aggregate_results_to_json(result_file_pattern, output_path):
  """Aggregates all the results files in the pattern into a single JSON file.

  Args:
    result_file_pattern: String with glob pattern to all the result files that
      should be aggregated (e.g. /tmp/*/results/aggregate/evaluation.json).
    output_path: String with path to output json file (e.g. /tmp/results.json).
  """
  logging.info("Loading the results.")
  model_results = _get(result_file_pattern)
  logging.info("Saving the aggregated results.")
  with gfile.Open(output_path, "w") as f:
    model_results.to_json(path_or_buf=f)


def load_aggregated_json_results(source_path):
  """Convenience function to load aggregated results from JSON file.

  Args:
    source_path: String with path to aggregated json file (e.g.
      /tmp/results.json).

  Returns:
    pd.DataFrame with aggregated results.
  """
  logging.info("Loading the aggregated results.")
  return pd.read_json(path_or_buf=source_path, orient="columns")


def _load(path):
  with gfile.GFile(path) as f:
    result = json.load(f)
  result["path"] = path
  return result


def _get(pattern):
  files = gfile.Glob(pattern)
  pool = multiprocessing.Pool()
  all_results = pool.map(_load, files)
  return pd.DataFrame(all_results)


