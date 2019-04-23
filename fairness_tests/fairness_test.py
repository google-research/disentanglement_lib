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

"""Test script for the new functionalities of disentanglement_lib for the
fairness project. To be run as:

>> python fairness_test.py <model_num>

The steps are:
1) Download the model specified as argument of the script



"""

# Imports
import urllib.request
import sys
import os
from zipfile import ZipFile


# 1. Model Download
# ------------------------------------------------------------------------------
# Retrieve model number and build the URL
model_num = int(sys.argv[1])
model_url = 'https://storage.googleapis.com/disentanglement_lib/' +\
  'unsupervised_study_v1/' + str(model_num) + '.zip'
# Download model in current directory if not already present
filename = str(model_num) + '.zip'
if not os.path.isfile(filename):
  urllib.request.urlretrieve(model_url, filename)
# Extract zip
zip = ZipFile(filename)
zip.extractall()
# Change directory
os.chdir(str(model_num))

# 2.
# ------------------------------------------------------------------------------







