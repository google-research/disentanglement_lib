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

"""Allows to convolute TFHub modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.contrib import framework as contrib_framework


def convolute_and_save(module_path, signature, export_path, transform_fn,
                       transform_checkpoint_path, new_signature=None):
  """Loads TFHub module, convolutes it with transform_fn and saves it again.

  Args:
    module_path: String with path from which the module is constructed.
    signature: String with name of signature to use for loaded module.
    export_path: String with path where to save the final TFHub module.
    transform_fn: Function that creates the graph to be appended to the loaded
      TFHub module. The function should take as keyword arguments the tensors
      returned by the loaded TFHub module. The function should return a
      dictionary of tensor that will be the output of the new TFHub module.
    transform_checkpoint_path: Path to checkpoint from which the transformer_fn
      variables will be read.
    new_signature: String with new name of signature to use for saved module. If
      None, `signature` is used instead.
  """
  if new_signature is None:
    new_signature = signature

  # We create a module_fn that creates the new TFHub module.
  def module_fn():
    module = hub.Module(module_path)
    inputs = _placeholders_from_module(module, signature=signature)
    intermediate_tensor = module(inputs, signature=signature, as_dict=True)
    # We need to scope the variables that are created when the transform_fn is
    # applied.
    with tf.variable_scope("transform"):
      outputs = transform_fn(**intermediate_tensor)
    hub.add_signature(name=new_signature, inputs=inputs, outputs=outputs)

  # We create a new graph where we will build the module for export.
  with tf.Graph().as_default():
    # Create the module_spec for the export.
    spec = hub.create_module_spec(module_fn)
    m = hub.Module(spec, trainable=True)
    # We need to recover the scoped variables and remove the scope when loading
    # from the checkpoint.
    prefix = "transform/"
    transform_variables = {
        k[len(prefix):]: v
        for k, v in m.variable_map.items()
        if k.startswith(prefix)
    }
    if transform_variables:
      init_fn = contrib_framework.assign_from_checkpoint_fn(
          transform_checkpoint_path, transform_variables)

    with tf.Session() as sess:
      # Initialize all variables, this also loads the TFHub parameters.
      sess.run(tf.global_variables_initializer())
      # Load the transformer variables from the checkpoint.
      if transform_variables:
        init_fn(sess)
      # Export the new TFHub module.
      m.export(export_path, sess)


def save_numpy_arrays_to_checkpoint(checkpoint_path, **dict_with_arrays):
  """Saves several NumpyArrays to variables in a TF checkpoint.

  Args:
    checkpoint_path: String with the path to the checkpoint file.
    **dict_with_arrays: Dictionary with keys that signify variable names and
      values that are the corresponding Numpy arrays to be saved.
  """
  with tf.Graph().as_default():
    feed_dict = {}
    assign_ops = []
    nodes_to_save = []
    for array_name, array in dict_with_arrays.items():
      # We will save the numpy array with the corresponding dtype.
      tf_dtype = tf.as_dtype(array.dtype)
      # We create a variable which we would like to persist in the checkpoint.
      node = tf.get_variable(array_name, shape=array.shape, dtype=tf_dtype)
      nodes_to_save.append(node)
      # We feed the numpy arrays into the graph via placeholder which avoids
      # adding the numpy arrays to the graph as constants.
      placeholder = tf.placeholder(tf_dtype, shape=array.shape)
      feed_dict[placeholder] = array
      # We use the placeholder to assign the variable the intended value.
      assign_ops.append(tf.assign(node, placeholder))
    saver = tf.train.Saver(nodes_to_save)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(assign_ops, feed_dict=feed_dict)
      saver.save(sess, checkpoint_path)
  assert saver.last_checkpoints[0] == checkpoint_path


def _placeholders_from_module(tfhub_module, signature):
  """Returns a dictionary with placeholder nodes for a given TFHub module."""
  info_dict = tfhub_module.get_input_info_dict(signature=signature)
  result = {}
  for key, value in info_dict.items():
    result[key] = tf.placeholder(value.dtype, shape=value.get_shape(), name=key)
  return result
