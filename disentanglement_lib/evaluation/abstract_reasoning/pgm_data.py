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

"""Data sets of Procedurally Generated Matrices (PGMs).

For a description, pleaser refer to https://arxiv.org/abs/1905.12506.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.data.ground_truth import ground_truth_data as gtd
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import shapes3d
from disentanglement_lib.evaluation.abstract_reasoning import pgm_utils
from disentanglement_lib.utils import resources
from disentanglement_lib.visualize import visualize_util
import gin
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf


@gin.configurable("pgm")
def get_pgm_dataset(pgm_type=gin.REQUIRED):
  """Returns a named PGM data set."""
  ground_truth_data = named_data.get_named_ground_truth_data()

  # Quantization for specific data sets (as described in
  # https://arxiv.org/abs/1905.12506).
  if isinstance(ground_truth_data, dsprites.AbstractDSprites):
    wrapped_data_set = Quantizer(ground_truth_data, [5, 6, 3, 3, 4, 4])
  elif isinstance(ground_truth_data, shapes3d.Shapes3D):
    wrapped_data_set = Quantizer(ground_truth_data, [10, 10, 10, 4, 4, 4])
  elif isinstance(ground_truth_data, dummy_data.DummyData):
    wrapped_data_set = ground_truth_data
  else:
    raise ValueError("Invalid data set.")

  # We support different ways to generate PGMs for each of the data set (e.g.,
  # `easy_1`, `hard_3`, `easy_mixes`). `easy` and `hard` refers to the way the
  # alternative solutions of the PGMs are generated:
  #   - `easy`: Alternative answers are random other solutions that do not
  #             satisfy the constraints in the given PGM.
  #   - `hard`: Alternative answers are unique random modifications of the
  #             correct solution which makes the task substantially harder.
  if pgm_type.startswith("easy"):
    sampling = "easy"
  elif pgm_type.startswith("hard"):
    sampling = "hard"
  else:
    raise ValueError("Invalid sampling strategy.")

  # The suffix determines how many relations there are:
  #   - 1-3: Specifies whether always 1, 2, or 3 relations are constant in each
  #          row.
  #   - `mixed`: With probability 1/3 each, 1, 2, or 3 relations are constant
  #               in each row.
  if pgm_type.endswith("1"):
    relations_dist = [1., 0., 0.]
  elif pgm_type.endswith("2"):
    relations_dist = [0., 1., 0.]
  elif pgm_type.endswith("3"):
    relations_dist = [0., 0., 1.]
  elif pgm_type.endswith("mixed"):
    relations_dist = [1. / 3., 1. / 3., 1. / 3.]
  else:
    raise ValueError("Invalid number of relations.")

  return PGMDataset(
      wrapped_data_set,
      sampling_strategy=sampling,
      relations_dist=relations_dist)


class PGMDataset(object):
  """Class that contains PGM data set based on a GroundTruthData."""

  def __init__(self, ground_truth_data, sampling_strategy, relations_dist):
    """Creates a PGMDataset.

    Args:
      ground_truth_data: GroundTruthData data set used to generate images.
      sampling_strategy: Either `easy` or `hard`. For `easy`, alternative
        answers are random other solutions that do not satisfy the constraints
        in the given PGM. For `hard`, alternative answers are unique random
        modifications of the correct solution which makes the task  harder.
      relations_dist: List with probabilites where the i-th element contains the
        probability that i relations are enforced.
    """
    self.ground_truth_data = ground_truth_data
    self.relations_dist = relations_dist
    self.sampling_strategy = sampling_strategy

  def sample(self, random_state):
    """Returns a random PGMInstance."""

    # Sample the number of relations.
    num_relations = 1 + random_state.choice(
        len(self.relations_dist), p=self.relations_dist)

    # Construct the PGM solution in the space of ground-truth factors.
    pgm = pgm_utils.PGM(
        random_state,
        num_relations,
        self.ground_truth_data.factors_num_values,
    )

    # Sample instances of the images for the solutions and alternative answers.
    solution = []
    for row in pgm.matrix:
      solution.append(
          self.ground_truth_data.sample_observations_from_factors(
              row, random_state))

    alternatives = self.ground_truth_data.sample_observations_from_factors(
        pgm.other_solutions, random_state)

    # Sample the position of the correct answer.
    position = random_state.choice(alternatives.shape[0] + 1)
    # Return the instance.
    return PGMInstance(
        np.array(solution), alternatives, position, pgm.matrix,
        pgm.other_solutions, self.ground_truth_data.factors_num_values)

  def tf_data_set(self, seed):
    """Returns a tf.data.Dataset.

    Args:
      seed: Integer with the random seed used to initialize the data set.

    Returns.
      tf.data.Dataset of the data set.
    """

    def generator():
      # We need to hard code the random seed so that the data set can be reset.
      random_state = np.random.RandomState(seed)
      while True:
        instance = self.sample(random_state)
        yield instance.training_sample()

    # We sample a single example to obtain the actual shapes and dtypes.
    features, _ = self.sample(np.random.RandomState(0)).training_sample()
    features_shapes = {k: v.shape for k, v in features.items()}
    features_types = {k: v.dtype for k, v in features.items()}
    output_shapes = (features_shapes, tf.TensorShape([]))
    output_types = (features_types, tf.int64)

    return tf.data.Dataset.from_generator(
        generator, output_types=output_types, output_shapes=output_shapes)

  def make_input_fn(self, seed, num_batches=None):
    """Creates an input function for the TPU Estimator."""

    def input_fn(params):
      """TPUEstimator compatible input fuction."""
      dataset = self.tf_data_set(seed)
      batch_size = params["batch_size"]
      # We need to drop the remainder as otherwise we lose the batch size in the
      # tensor shape. This has no effect as our data set is infinite.
      dataset = dataset.batch(batch_size, drop_remainder=True)
      if num_batches is not None:
        dataset = dataset.take(num_batches)
      return dataset.make_one_shot_iterator().get_next()

    return input_fn


class PGMInstance(object):
  """Class that holds instance of an image PGM."""

  def __init__(self,
               solution,
               alternatives,
               position,
               solution_factors=None,
               alternatives_factors=None,
               num_factor_values=None):
    """Constructs a PGMInstance.

    Args:
      solution: Numpy array of shape (num_rows, num_cols, width, height,
        channels) with the images of the PGM solution.
      alternatives: Numpy array of shape (num_alternatives, width, height,
        channels) with the images of the alternatives.
      position: Integer with position where solution should be inserted.
      solution_factors: Numpy array of shape (num_rows, num_cols, num_factors)
        with the factors of the PGM solution.
      alternatives_factors: Numpy array of shape (num_alternatives, num_factors)
        with the images of the alternatives.
      num_factor_values: List with the number of values for each factor.
    """
    self.solution = solution
    self.alternatives = alternatives
    self.position = position
    self.solution_factors = solution_factors
    self.alternatives_factors = alternatives_factors
    self.num_factor_values = num_factor_values

  def get_context(self):
    """Returns the context.

    Returns:
      Numpy array of shape (num_rows*num_cols - 1, width, height, channels).
    """
    context = []
    for row in self.solution:
      context += list(row)
    return np.array(context[:-1], dtype=np.float32)

  def get_answers(self):
    """Returns the answers.

    Returns:
      Numpy array of shape (num_alternatives + 1, width, height, channels).
    """
    result = list(self.alternatives)
    result.insert(self.position, self.solution[-1, -1])
    return np.array(result, dtype=np.float32)

  def get_context_factor_values(self):
    """Returns the context ground truth factos as integer values.

    Returns:
      Numpy array of shape (num_rows*num_cols - 1, len(num_factor_values).
    """
    context = []
    for row in self.solution_factors:
      context += list(row)
    return np.array(context[:-1])

  def get_answers_factor_values(self):
    """Returns the answers ground truth factos as integer values.

    Returns:
      Numpy array of shape (num_alternatives + 1, len(num_factor_values).
    """
    result = list(self.alternatives_factors)
    result.insert(self.position, self.solution_factors[-1, -1])
    return np.array(result)

  def range_embed_factors(self, factors):
    """Embeds the factors linearly in [-0.5, 0.5] based on integer values.

    Args:
      factors: Numpy array of shape (:, len(num_factor_values) with factors.

    Returns:
      Numpy array of shape (:, len(num_factor_values) with floats.
    """
    result = np.array(factors, dtype=np.float32)
    max_vals = np.array(self.num_factor_values, dtype=np.float32) - 1.
    result /= np.expand_dims(max_vals, 0)
    return result - .5

  def onehot_embed_factors(self, factors):
    """Embeds the factors as one-hot vectors.

    Args:
      factors: Numpy array of shape (:, len(num_factor_values) with factors.

    Returns:
      Numpy array of shape (:, sum(num_factor_values) with floats.
    """
    result = []
    for i, num in enumerate(self.num_factor_values):
      result.append(onehot(factors[:, i], num))
    return np.array(np.concatenate(result, axis=-1), dtype=np.float32)

  def training_sample(self):
    """Returns a single training example."""
    sample = {}
    sample["context"] = self.get_context()
    sample["answers"] = self.get_answers()
    if self.solution_factors is not None:
      context_factors = self.get_context_factor_values()
      answers_factors = self.get_answers_factor_values()

      sample["context_factor_values"] = self.range_embed_factors(
          context_factors)
      sample["answers_factor_values"] = self.range_embed_factors(
          answers_factors)
      sample["context_factors_onehot"] = self.onehot_embed_factors(
          context_factors)
      sample["answers_factors_onehot"] = self.onehot_embed_factors(
          answers_factors)
    return sample, self.position

  def make_image(self, answer=False, padding_px=8, border_px=4):
    """Creates an image of the PGMInstance."""
    # Create the question side that contains the progression matrix.
    question = np.copy(self.solution)
    if question.shape[-1] == 1:
      question = np.repeat(question, 3, -1)
    if not answer:
      question[-1, -1] = question_mark()

    # Build up the image on the context side.
    rows = []
    for i in range(question.shape[0]):
      row = []
      for j in range(question.shape[1]):
        # Do the border around the image.
        color = np.array([1., 1., 1.])
        if answer and i == (question.shape[0] - 1) and j == (question.shape[1] -
                                                             1):
          color = COLORS["green"]
        row.append(
            visualize_util.pad_around(question[i, j], border_px, value=color))
      rows.append(visualize_util.padded_stack(row, padding_px, axis=1))
    question_image = visualize_util.padded_stack(rows, padding_px)

    separator = np.zeros((question_image.shape[0], 2, question_image.shape[2]))

    # Create the answer side.
    answers = self.get_answers()
    if answers.shape[-1] == 1:
      answers = np.repeat(answers, 3, -1)
    answers_with_border = []
    for i, image in enumerate(answers):
      color = np.array([1., 1., 1.])
      if answer:
        color = COLORS["green"] if i == self.position else COLORS["red"]
      answers_with_border.append(
          visualize_util.pad_around(image, border_px, value=color))

    answer_image = visualize_util.padded_grid(answers_with_border,
                                              question.shape[0], padding_px)
    center_crop = visualize_util.padded_stack(
        [question_image, separator, answer_image], padding_px, axis=1)
    return visualize_util.pad_around(
        visualize_util.add_below(center_crop, padding_px), padding_px)


class Quantizer(gtd.GroundTruthData):
  """Quantizes a GroundTruthData to have a maximal number of factors."""

  def __init__(self, wrapped_ground_truth_data, max_factors):
    """Constructs a Quantizer.

    Args:
      wrapped_ground_truth_data: GroundTruthData that should be quantized.
      max_factors: integer with the maximal number of factors.
    """
    self.wrapped_ground_truth_data = wrapped_ground_truth_data
    self.true_num_factors = wrapped_ground_truth_data.factors_num_values
    self.fake_num_factors = list(np.minimum(self.true_num_factors, max_factors))

  @property
  def num_factors(self):
    return self.wrapped_ground_truth_data.num_factors

  @property
  def factors_num_values(self):
    return self.fake_num_factors

  @property
  def observation_shape(self):
    return self.wrapped_ground_truth_data.observation_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    factors = np.zeros(shape=(num, self.num_factors), dtype=np.int64)
    for i in range(self.num_factors):
      factors[:, i] = self._sample_factor(i, num, random_state)
    return factors

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.factor_sizes[i], size=num)

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    translated_factors = np.copy(factors)
    for i in range(self.num_factors):
      if self.true_num_factors[i] != self.fake_num_factors[i]:
        ratio = float(self.true_num_factors[i]) / float(
            self.fake_num_factors[i])
        translated_factors[:, i] = np.floor(factors[:, i] * ratio)
    return self.wrapped_ground_truth_data.sample_observations_from_factors(
        translated_factors, random_state)


COLORS = {
    "blue": np.array([66., 103., 210.]) / 255.,
    "red": np.array([234., 67., 53.]) / 255.,
    "yellow": np.array([251., 188., 4.]) / 255.,
    "green": np.array([52., 168., 83.]) / 255.,
    "grey": np.array([154., 160., 166.]) / 255.,
}

QUESTION_MARK = [None]


def question_mark():
  """Returns an image of the question mark."""
  # Cache the image so it is not always reloaded.
  if QUESTION_MARK[0] is None:
    with tf.gfile.Open(
        resources.get_file("google/abstract_reasoning/data/question_mark.png"),
        "rb") as f:
      QUESTION_MARK[0] = np.array(Image.open(f).convert("RGB")) * 1.0 / 255.
  return QUESTION_MARK[0]


def onehot(indices, num_atoms):
  """Embeds the indices as one hot vectors."""
  return np.eye(num_atoms)[indices]
