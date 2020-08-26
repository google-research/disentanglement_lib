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

"""Example script how to get started with research using disentanglement_lib.

To run the example, please change the working directory to the containing folder
and run:
>> python example.py

In this example, we show how to use disentanglement_lib to:
1. Train a standard VAE (already implemented in disentanglement_lib).
2. Train a custom VAE model.
3. Extract the mean representations for both of these models.
4. Compute the Mutual Information Gap (already implemented) for both models.
5. Compute a custom disentanglement metric for both models.
6. Aggregate the results.
7. Print out the final Pandas data frame with the results.
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow.compat.v1 as tf
import gin.tf

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "example_output"

# By default, we do not overwrite output directories. Set this to True, if you
# want to overwrite (in particular, if you rerun this script several times).
overwrite = False

# 1. Train a standard VAE (already implemented in disentanglement_lib).
# ------------------------------------------------------------------------------

# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "vae")

# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.
train.train_with_gin(os.path.join(path_vae, "model"), overwrite, ["model.gin"])
# After this command, you should have a `vae` subfolder with a model that was
# trained for a few steps (in reality, you will want to train many more steps).


# 2. Train a custom VAE model.
# ------------------------------------------------------------------------------
# To train a custom model, we have to provide an implementation of the class
# GaussianEncoderModel in the
# disentanglement_lib.methods.unsupervised.gaussian_encoder_model module.
# For simplicty, we will subclass the BaseVAE class in
# disentanglement_lib.methods.unsupervised.vae which will train a VAE style
# model where the loss is given by a reconstruction loss (configured via gin)
# plus a custom regularizer (needs to be implemented.)
@gin.configurable("BottleneckVAE")  # This will allow us to reference the model.
class BottleneckVAE(vae.BaseVAE):
  """BottleneckVAE.

  The loss of this VAE-style model is given by:
    loss = reconstruction loss + gamma * |KL(app. posterior | prior) - target|
  """

  def __init__(self, gamma=gin.REQUIRED, target=gin.REQUIRED):
    self.gamma = gamma
    self.target = target

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    # This is how we customize BaseVAE. To learn more, have a look at the
    # different models in vae.py.
    del z_mean, z_logvar, z_sampled
    return self.gamma * tf.math.abs(kl_loss - self.target)


# We use the same training protocol that we defined in model.gin but we use gin
# bindings to train our custom VAE instead of the ordinary VAE.
gin_bindings = [
    "model.model = @BottleneckVAE()",
    "BottleneckVAE.gamma = 4",
    "BottleneckVAE.target = 10."
]
# Call training module to train the custom model.
path_custom_vae = os.path.join(base_path, "BottleneckVAE")
train.train_with_gin(
    os.path.join(path_custom_vae, "model"), overwrite, ["model.gin"],
    gin_bindings)
# As before, after this command, you should have a `BottleneckVAE` subfolder
# with a model that was trained for a few steps.

# 3. Extract the mean representation for both of these models.
# ------------------------------------------------------------------------------
# To compute disentanglement metrics, we require a representation function that
# takes as input an image and that outputs a vector with the representation.
# We extract the mean of the encoder from both models using the following code.
for path in [path_vae, path_custom_vae]:
  representation_path = os.path.join(path, "representation")
  model_path = os.path.join(path, "model")
  postprocess_gin = ["postprocess.gin"]  # This contains the settings.
  # postprocess.postprocess_with_gin defines the standard extraction protocol.
  postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                   postprocess_gin)

# 4. Compute the Mutual Information Gap (already implemented) for both models.
# ------------------------------------------------------------------------------
# The main evaluation protocol of disentanglement_lib is defined in the
# disentanglement_lib.evaluation.evaluate module. Again, we have to provide a
# gin configuration. We could define a .gin config file; however, in this case
# we show how all the configuration settings can be set using gin bindings.
# We use the Mutual Information Gap (with a low number of samples to make it
# faster). To learn more, have a look at the different scores in
# disentanglement_lib.evaluation.evaluate.metrics and the predefined .gin
# configuration files in
# disentanglement_lib/config/unsupervised_study_v1/metrics_configs/(...).
gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='auto'",
    "evaluation.random_seed = 0",
    "mig.num_train=1000",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]
for path in [path_vae, path_custom_vae]:
  result_path = os.path.join(path, "metrics", "mig")
  representation_path = os.path.join(path, "representation")
  evaluate.evaluate_with_gin(
      representation_path, result_path, overwrite, gin_bindings=gin_bindings)


# 5. Compute a custom disentanglement metric for both models.
# ------------------------------------------------------------------------------
# The following function implements a dummy metric. Note that all metrics get
# ground_truth_data, representation_function, random_state arguments by the
# evaluation protocol, while all other arguments have to be configured via gin.
@gin.configurable(
    "custom_metric",
    blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_custom_metric(ground_truth_data,
                          representation_function,
                          random_state,
                          num_train=gin.REQUIRED,
                          batch_size=16):
  """Example of a custom (dummy) metric.

  Preimplemented metrics can be found in disentanglement_lib.evaluation.metrics.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with disentanglement score.
  """
  score_dict = {}

  # This is how to obtain the representations of num_train points along with the
  # ground-truth factors of variation.
  representation, factors_of_variations = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train, random_state,
      batch_size)
  # We could now compute a metric based on representation and
  # factors_of_variations. However, for the sake of brevity, we just return 1.
  del representation, factors_of_variations
  score_dict["custom_metric"] = 1.
  return score_dict


# To compute the score, we again call the evaluation protocol with a gin
# configuration. At this point, note that for all steps, we have to set a
# random seed (in this case via `evaluation.random_seed`).
gin_bindings = [
    "evaluation.evaluation_fn = @custom_metric",
    "custom_metric.num_train = 100", "evaluation.random_seed = 0",
    "dataset.name='auto'"
]
for path in [path_vae, path_custom_vae]:
  result_path = os.path.join(path, "metrics", "custom_metric")
  evaluate.evaluate_with_gin(
      representation_path, result_path, overwrite, gin_bindings=gin_bindings)

# 6. Aggregate the results.
# ------------------------------------------------------------------------------
# In the previous steps, we saved the scores to several output directories. We
# can aggregate all the results using the following command.
pattern = os.path.join(base_path,
                       "*/metrics/*/results/aggregate/evaluation.json")
results_path = os.path.join(base_path, "results.json")
aggregate_results.aggregate_results_to_json(
    pattern, results_path)

# 7. Print out the final Pandas data frame with the results.
# ------------------------------------------------------------------------------
# The aggregated results contains for each computed metric all the configuration
# options and all the results captured in the steps along the pipeline. This
# should make it easy to analyze the experimental results in an interactive
# Python shell. At this point, note that the scores we computed in this example
# are not realistic as we only trained the models for a few steps and our custom
# metric always returns 1.
model_results = aggregate_results.load_aggregated_json_results(results_path)
print(model_results)
