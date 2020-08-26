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

"""Library of commonly used optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import gin.tf


def make_optimizer(optimizer_fn, learning_rate):
  """Wrapper to create the optimizer with a given learning_rate."""
  if learning_rate is None:
    # Learning rate is specified in the optimizer_fn options, or left to its
    # default value.
    return optimizer_fn()
  else:
    # Learning rate is explicitly specified in vae/discriminator optimizer.
    # If it is callable, we assume it's a LR decay function which needs the
    # current global step.
    if callable(learning_rate):
      learning_rate = learning_rate(global_step=tf.train.get_global_step())

    return optimizer_fn(learning_rate=learning_rate)


@gin.configurable("vae_optimizer")
def make_vae_optimizer(optimizer_fn=gin.REQUIRED, learning_rate=None):
  """Wrapper that uses gin to construct an optimizer for VAEs."""
  return make_optimizer(optimizer_fn, learning_rate)


@gin.configurable("discriminator_optimizer")
def make_discriminator_optimizer(optimizer_fn=gin.REQUIRED, learning_rate=None):
  """Wrapper that uses gin to construct an optimizer for the discriminator."""
  return make_optimizer(optimizer_fn, learning_rate)
