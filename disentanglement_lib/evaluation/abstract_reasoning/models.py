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

"""Keras models to perform abstract reasoning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.abstract_reasoning import relational_layers
import gin
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.contrib import tpu as contrib_tpu


@gin.configurable
class TwoStageModel(object):
  """Two stage model for abstract reasoning tasks.

  This class implements a flexible variation of the Wild Relation Networks model
  of Barrett et al., 2018 (https://arxiv.org/abs/1807.04225). There are two
  stages:

  1. Embedding: This embeds the patches of the PGM each indepently into a lower
    dimensional embedding (e.g., via CNN). It is also supported to take one-hot
    embeddings or integer embeddings of the ground-truth factors (as baselines).

  2. Reasoning: This performs reasoning on the embeddings of the patches of the
    PGM and returns the solution.
  """

  def __init__(self,
               embedding_model_class=gin.REQUIRED,
               reasoning_model_class=gin.REQUIRED,
               optimizer_fn=None):
    """Constructs a TwoStageModel.

    Args:
      embedding_model_class: Either `values`, `onehot`, or a class that has a
        __call__ function that takes as input a two-tuple of
        (batch_size, num_nodes, heigh, width, num_channels) tensors and returns
        two (batch_size, num_nodes, num_embedding_dims) tensors for both the
        context panels and the answer panels.
      reasoning_model_class: Class that has a __call__ function that takes as
        input a two-tuple of (batch_size, num_nodes, num_embedding_dims) tensors
        and returns the solution in a (batch_size,) tensor.
      optimizer_fn: Function that creates a tf.train optimizer.
    """
    if optimizer_fn is None:
      optimizer_fn = tf.train.AdamOptimizer
    self.optimizer_fn = optimizer_fn
    self.embedding_model_class = embedding_model_class
    self.reasoning_model_class = reasoning_model_class

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model_fn."""
    del params
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    update_ops = []

    # First, embed the context and answer panels.
    if self.embedding_model_class == "values":
      # Use the integer values of the ground-truth factors.
      context_embeddings = features["context_factor_values"]
      answer_embeddings = features["answers_factor_values"]
    elif self.embedding_model_class == "onehot":
      # Use one-hot embeddings of the ground-truth factors.
      context_embeddings = features["context_factors_onehot"]
      answer_embeddings = features["answers_factors_onehot"]
    else:
      embedding_model = self.embedding_model_class()
      context_embeddings, answer_embeddings = embedding_model(
          [
              features["context"],
              features["answers"],
          ],
          training=is_training,
      )
      embedding_model.summary(print_fn=tf.logging.info)
      update_ops += embedding_model.updates

    # Apply the reasoning model.
    reasoning_model = self.reasoning_model_class()
    logits = reasoning_model([context_embeddings, answer_embeddings],
                             training=is_training)
    reasoning_model.summary(print_fn=tf.logging.info)
    update_ops += reasoning_model.updates

    loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss_mean = tf.reduce_mean(loss_vec)

    if mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(labels, logits):
        predictions = tf.argmax(logits, 1)
        return {
            "accuracy":
                tf.metrics.accuracy(labels=labels, predictions=predictions),
        }

      return contrib_tpu.TPUEstimatorSpec(
          mode=mode, loss=loss_mean, eval_metrics=(metric_fn, [labels, logits]))

    if mode == tf.estimator.ModeKeys.TRAIN:
      # In case we use batch norm, the following is required.
      with tf.control_dependencies(update_ops):
        optimizer = self.optimizer_fn()
        train_op = optimizer.minimize(
            loss=loss_mean, global_step=tf.train.get_global_step())
      return contrib_tpu.TPUEstimatorSpec(
          mode=mode, loss=loss_mean, train_op=train_op)
    raise NotImplementedError("Unsupported mode.")


@gin.configurable
class BaselineCNNEmbedder(tf.keras.Model):
  """Baseline implementation where a CNN is learned from scratch."""

  def __init__(self,
               num_latent=gin.REQUIRED,
               name="BaselineCNNEmbedder",
               **kwargs):
    """Constructs a BaselineCNNEmbedder.

    Args:
      num_latent: Integer with the number of latent dimensions.
      name: String with the name of the model.
      **kwargs: Other keyword arguments passed to tf.keras.Model.
    """
    super(BaselineCNNEmbedder, self).__init__(name=name, **kwargs)
    embedding_layers = [
        tf.keras.layers.Conv2D(
            32, (4, 4),
            2,
            activation=get_activation(),
            padding="same",
            kernel_initializer=get_kernel_initializer()),
        tf.keras.layers.Conv2D(
            32, (4, 4),
            2,
            activation=get_activation(),
            padding="same",
            kernel_initializer=get_kernel_initializer()),
        tf.keras.layers.Conv2D(
            64, (4, 4),
            2,
            activation=get_activation(),
            padding="same",
            kernel_initializer=get_kernel_initializer()),
        tf.keras.layers.Conv2D(
            64, (4, 4),
            2,
            activation=get_activation(),
            padding="same",
            kernel_initializer=get_kernel_initializer()),
        tf.keras.layers.Flatten(),
    ]
    self.embedding_layer = relational_layers.MultiDimBatchApply(
        tf.keras.models.Sequential(embedding_layers, "embedding_cnn"))

  def call(self, inputs, **kwargs):
    context, answers = inputs
    context_embedding = self.embedding_layer(context, **kwargs)
    answers_embedding = self.embedding_layer(answers, **kwargs)
    return context_embedding, answers_embedding


@gin.configurable
class HubEmbedding(tf.keras.Model):
  """Embed images using a pre-trained TFHub model.

  Compatible with the representation of a disentanglement_lib model.
  """

  def __init__(self, hub_path=gin.REQUIRED, name="HubEmbedding", **kwargs):
    """Constructs a HubEmbedding.

    Args:
      hub_path: Path to the TFHub module.
      name: String with the name of the model.
      **kwargs: Other keyword arguments passed to tf.keras.Model.
    """
    super(HubEmbedding, self).__init__(name=name, **kwargs)

    def _embedder(x):
      embedder_module = hub.Module(hub_path)
      return embedder_module(dict(images=x), signature="representation")

    self.embedding_layer = relational_layers.MultiDimBatchApply(
        tf.keras.layers.Lambda(_embedder))

  def call(self, inputs, **kwargs):
    context, answers = inputs
    context_embedding = self.embedding_layer(context, **kwargs)
    answers_embedding = self.embedding_layer(answers, **kwargs)
    return context_embedding, answers_embedding


@gin.configurable
class OptimizedWildRelNet(tf.keras.Model):
  """Optimized implementation of the reasoning module in the WildRelNet model.

  Based on https://arxiv.org/pdf/1807.04225.pdf.
  """

  def __init__(self,
               edge_mlp=gin.REQUIRED,
               graph_mlp=gin.REQUIRED,
               dropout_in_last_graph_layer=gin.REQUIRED,
               name="OptimizedWildRelNet",
               **kwargs):
    """Constructs a OptimizedWildRelNet.

    Args:
      edge_mlp: List with number of latent nodes in different layers of the edge
        MLP.
      graph_mlp: List with number of latent nodes in different layers of the
        graph MLP.
      dropout_in_last_graph_layer: Dropout fraction to be applied in the last
        layer of the graph MLP.
      name: String with the name of the model.
      **kwargs: Other keyword arguments passed to tf.keras.Model.
    """
    super(OptimizedWildRelNet, self).__init__(name=name, **kwargs)

    # Create the EdgeMLP.
    edge_layers = []
    for num_units in edge_mlp:
      edge_layers += [
          tf.keras.layers.Dense(
              num_units,
              activation=get_activation(),
              kernel_initializer=get_kernel_initializer())
      ]
    self.edge_layer = tf.keras.models.Sequential(edge_layers, "edge_mlp")

    # Create the GraphMLP.
    graph_layers = []
    for num_units in graph_mlp:
      graph_layers += [
          tf.keras.layers.Dense(
              num_units,
              activation=get_activation(),
              kernel_initializer=get_kernel_initializer())
      ]
    if dropout_in_last_graph_layer:
      graph_layers += [
          tf.keras.layers.Dropout(
              1. - dropout_in_last_graph_layer,
              noise_shape=[1, 1, graph_mlp[-1]])
      ]
    graph_layers += [
        tf.keras.layers.Dense(1, kernel_initializer=get_kernel_initializer())
    ]

    # Create the auxiliary layers.
    self.graph_layer = tf.keras.models.Sequential(graph_layers, "graph_mlp")
    self.stacking_layer = relational_layers.StackAnswers()

    # Create the WildRelationNet.
    self.wildrelnet = tf.keras.models.Sequential([
        relational_layers.AddPositionalEncoding(),
        relational_layers.RelationalLayer(
            self.edge_layer,
            tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2))),
        tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2)),
        self.graph_layer,
        tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1)),
    ], "wildrelnet")

  def call(self, inputs, **kwargs):
    context_embeddings, answer_embeddings = inputs
    # The stacking layer `stacks` each answer panel embedding onto the context
    # panels separately.
    stacked_answers = self.stacking_layer(
        [context_embeddings, answer_embeddings])
    # Apply the relational neural network.
    return self.wildrelnet(stacked_answers, **kwargs)


@gin.configurable("activation")
def get_activation(activation=tf.keras.activations.relu):
  if activation == "lrelu":
    return lambda x: tf.keras.activations.relu(x, alpha=0.2)
  return activation


@gin.configurable("kernel_initializer")
def get_kernel_initializer(kernel_initializer="lecun_normal"):
  return kernel_initializer
