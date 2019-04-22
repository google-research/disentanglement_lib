#!/usr/bin/env python
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

"""Fair disentanglement

Utils for fair_downstream_task.
"""

from disentanglement_lib.evaluation.metrics import dci
import numpy as np
import gin.tf


@gin.configurable("fair_representation")
def compute_fair_representation(mus_train, ys_train, mus_test, ys_test,
                                sensitive_factor_index,
                                correlation_measure=gin.REQUIRED):
  """Computes a 'fair' representation of the data. In this particular case,
  we take out the factor that is the most correlated to the factor
  corresponding to sensitive_factor_index.
  Args:
    mus_train: latent means of the training batch.
    ys_train: labels of the training batch.
    mus_test: latent means of the test batch.
    ys_test: labels of the test batch.
    sensitive_factor_index: index of the sensitive factor.
    correlation_measure: measure of correlation
  """
  importance_matrix = correlation_measure(mus_train, ys_train, mus_test,
                                          ys_test)
  sensitive_factor_importance = importance_matrix[:, sensitive_factor_index]
  factor_to_remove_index = np.argmax(sensitive_factor_importance)
  fair_representation_train = np.delete(mus_train.copy(),
                                        factor_to_remove_index,
                                        axis=0)
  fair_representation_test = np.delete(mus_test.copy(),
                                       factor_to_remove_index,
                                       axis=0)
  return fair_representation_train, fair_representation_test


@gin.configurable("factorwise_dci",
                  blacklist=["mus_train", "ys_train", "mus_test", "ys_test"])
def compute_factorwise_dci(mus_train, ys_train, mus_test, ys_test):
  """Computes importance of attributes using the importance matrix and returns
  a vector containing the results.
  Args:
    mus_train: latent means of the training batch.
    ys_train: labels of the training batch.
    mus_test: latent means of the test batch.
    ys_test: labels of the test batch.
  """
  importance_matrix, _, _ = dci.compute_importance_gbt(
    mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  return importance_matrix