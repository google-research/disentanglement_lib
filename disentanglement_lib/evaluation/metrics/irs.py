# coding=utf-8

"""
Interventional Robustness Score
Code adopted from the authors' original implementation

Suter, R., Miladinović, Đ., Bauer, S., & Schölkopf, B. (2018).
Interventional Robustness of Deep Latent Variable Models.
arXiv preprint arXiv:1811.00007.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf


@gin.configurable(
  "irs",
  blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_irs(ground_truth_data,
                representation_function,
                random_state,
                diff_quantile=0.99,
                num_train=gin.REQUIRED,
                batch_size=gin.REQUIRED):
  """ Computes the Interventional Robustness Score
  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with IRS and number of active dimensions.
  """
  logging.info("Generating training set.")
  mus, ys = utils.generate_batch_factor_code(
    ground_truth_data, representation_function, num_train,
    random_state, batch_size)
  assert mus.shape[1] == num_train

  ys_discrete = utils.make_discretizer(ys)
  active_mus = _drop_constant_dims(mus)

  if not active_mus.any():
    irs_score = 0.0
  else:
    irs_score = scalable_disentanglement_score(ys_discrete.T, active_mus.T,
                                               diff_quantile)['avg_score']

  score_dict = {}
  score_dict["IRS"] = irs_score
  score_dict["num_active_dims"] = np.sum(active_mus)
  return score_dict


def _drop_constant_dims(ys):
  """ Returns a view of the matrix `ys` with dropped constant rows
  """
  ys = np.asarray(ys)
  if ys.ndim != 2:
    raise ValueError('Expecting a matrix')

  variances = ys.var(axis=1)
  active_mask = variances > 0.
  return ys[active_mask, :]


def scalable_disentanglement_score(g, z, diff_quantile=0.99):
  """ Compute IRS scores on a dataset without noise in X and crossed generative
  factors (i.e. one sample per combination of g)
  assume each g_i is an equally probable realization of g_i and all g_i are
  independent

  :param g: np.ndarray of shape (num samples, num generative factors),
            matrix of ground truth generative factors
  :param z: np.ndarray of shape (num samples, num latent dimensions),
            matrix of latent variables
  """
  num_gen = g.shape[1]
  num_lat = z.shape[1]

  # Compute normalizer
  normalizer = np.max(np.abs(z - z.mean(axis=0)), axis=0)
  r = np.zeros([num_lat, num_gen])
  for i in range(num_gen):
    g_is = np.unique(g[:, i], axis=0)  # all possible values g_i can take on
    assert g_is.ndim == 1
    n_i = g_is.shape[0]
    for k in range(n_i):
      # Compute E[Z | g_i]
      match = (g[:, i] == g_is[k])
      e_loc = np.mean(z[match, :], axis=0)

      # Difference of each value within that group of constant g_i to its mean
      diffs = np.abs(z[match, :] - e_loc)
      max_diffs = np.quantile(diffs, q=diff_quantile, axis=0)
      r[:, i] = r[:, i] + max_diffs
    r[:, i] /= n_i
  # Normalize value of each latent dimension with its maximal deviation
  r = r / normalizer[:, np.newaxis]
  r = 1.0 - r
  d = r.max(axis=1)
  if np.sum(normalizer) > 0.0:
    avg_score = np.average(d, weights=normalizer)
  else:
    avg_score = np.mean(d)

  parents = r.argmax(axis=1)
  score_dict = {}
  score_dict['disentanglement_scores'] = d
  score_dict['avg_score'] = avg_score
  score_dict['parents'] = parents
  score_dict['IRS_matrix'] = r
  score_dict['max_deviations'] = normalizer
  return score_dict
