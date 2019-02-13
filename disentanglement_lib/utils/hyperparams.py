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

"""Utility to create hyperparameter configurations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
from six.moves import range
from six.moves import zip


def sweep(name, values):
  """Sweeps the hyperparameter across different values."""
  return [{name: value} for value in values]


def categorical(items):
  """Defines sweep over categorical variable."""
  return items


def discrete(items):
  """Sweeps over discrete variable."""
  return items


def fixed(name, value, length=1):
  """Creates fixed hyperparameter setting."""
  return [{name: value} for _ in range(length)]


def zipit(list_of_items):
  """Zips different hyperparameter settings."""
  if len(list_of_items) == 1:
    return list_of_items[0]
  main_items = list_of_items[0]
  other_items = zipit(list_of_items[1:])
  if len(main_items) != len(other_items):
    if len(main_items) == 1:
      main_items *= len(other_items)
    elif len(other_items) == 1:
      other_items *= len(main_items)
    else:
      raise ValueError("Cannot zip lists of different lengths.")

  result = []
  for main_dict, other_dict in zip(main_items, other_items):
    new_dict = {}
    new_dict.update(main_dict)
    new_dict.update(other_dict)
    result.append(new_dict)
  return result


def chainit(list_of_items):
  """Chains different hyperparameter settings."""
  result = []
  for items in list_of_items:
    result.extend(items)
  return result


def product(list_of_items):
  """Creates outer product of hyperparameter settings."""
  if len(list_of_items) == 1:
    return list_of_items[0]
  result = []
  other_items = product(list_of_items[1:])
  for first_dict in list_of_items[0]:
    for second_dict in other_items:
      new_dict = {}
      new_dict.update(first_dict)
      new_dict.update(second_dict)
      result.append(new_dict)
  return result


def to_bindings(items):
  return [
      "{} = {}".format(key, _escape_value(value))
      for key, value in items.items()
  ]


def _escape_value(value):
  if isinstance(value, (str, six.text_type)) and not value.startswith("@"):
    return "'{}'".format(value)
  return str(value)
