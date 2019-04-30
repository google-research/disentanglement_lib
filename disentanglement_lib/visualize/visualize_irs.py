# coding=utf-8

"""Visualization code for Interventional Robustness Score
Code adopted from the authors' original implementation:

Suter, R., Miladinović, Đ., Bauer, S., & Schölkopf, B. (2018).
Interventional Robustness of Deep Latent Variable Models.
arXiv preprint arXiv:1811.00007.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from disentanglement_lib.evaluation.metrics.irs import scalable_disentanglement_score


def vis_all_interventional_effects(gen_factors, latents, output_dir):
  """Compute Matrix of all interventional effects."""
  res = scalable_disentanglement_score(gen_factors, latents)
  parents = res['parents']
  scores = res['disentanglement_scores']

  fig_length_inches = 3.0 * gen_factors.shape[1]
  fig, axes = plt.subplots(latents.shape[1], gen_factors.shape[1],
                           figsize=(fig_length_inches, fig_length_inches),
                           sharex='col', sharey='row')

  for j in range(gen_factors.shape[1]):  # iterate over generative factors
    for l in range(latents.shape[1]):
      ax = axes[l, j]
      if parents[l] != j:
        visualize_interventional_effect(gen_factors, latents, l, parents[l],
                                        j, ax=ax)
        ax.set_title('')
      else:
        visualize_interventional_effect(gen_factors, latents, l, parents[l], j,
                                        no_conditioning=True, ax=ax)
        ax.set_title('Parent={}, IRS = {:1.2}'.format(parents[l], scores[l]))

  fig.tight_layout()
  output_path = os.path.join(output_dir, 'interventional_effect.png')
  fig.savefig(output_path)


def visualize_interventional_effect(gen_factors, latents, latent_dim,
                                    const_factor_idx, intervened_factor_idx,
                                    no_conditioning=False,
                                    ax=None, plot_legend=True,
                                    plot_scatter=True):
  """Visualize single cell of interventional effects.

  Args:
    gen_factors: Ground truth generative factors
    latents: latent factors
    latent_dim: latent dimension under consideration
    const_factor_idx: generative factor which is being kept constant
    intervened_factor_idx: g_j which we intervene on
    no_conditioning: whether or not we should condition on i
  """
  if ax is None:
    plt.figure(figsize=(10, 7))
    ax = plt.axes()

  g_is = np.unique(gen_factors[:, const_factor_idx], axis=0)
  g_js = np.unique(gen_factors[:, intervened_factor_idx], axis=0)

  colors = 10 * ['b', 'y', 'g', 'r', 'c', 'm', 'k']
  cols_idx = np.empty([gen_factors.shape[0]], dtype=int)
  for i_idx in range(g_is.shape[0]):
    match = (gen_factors[:, [const_factor_idx]] == [g_is[i_idx]]).all(axis=1)
    cols_idx[match] = i_idx
  cols = [colors[col % len(colors)] for col in cols_idx]

  # Plot all points, color indicates constant factor
  if plot_scatter:
    ax.scatter(gen_factors[:, intervened_factor_idx], latents[:, latent_dim], c=cols)

  # Compute possible g_i and g_j
  if no_conditioning:
    E_for_j = np.empty([g_js.shape[0]])
    median_for_j = np.empty([g_js.shape[0]])
    stdev_for_j = np.empty([g_js.shape[0]])
    for j_idx in range(g_js.shape[0]):
      match = (gen_factors[:, intervened_factor_idx] == g_js[j_idx])
      E_for_j[j_idx] = np.mean(latents[match, latent_dim])
      median_for_j[j_idx] = np.median(latents[match, latent_dim])
      stdev_for_j[j_idx] = np.std(latents[match, latent_dim])
    ax.plot(g_js, E_for_j, linewidth=2, markersize=12, label='mean')
    ax.plot(g_js, median_for_j, linewidth=2, markersize=12, label='median')
    ax.plot(g_js, E_for_j + stdev_for_j, linestyle='--', c='b', linewidth=1)
    ax.plot(g_js, E_for_j - stdev_for_j, linestyle='--', c='b', linewidth=1)
    ax.set_ylabel('E[z_{}|g_{}]'.format(latent_dim, intervened_factor_idx))
    ax.set_xlabel('g_{}'.format(intervened_factor_idx))
    ax.legend()
  else:
    # Compute E[Z_l | g_i, g_j] as a function of g_j for each g_i
    E_given_i_for_j = np.empty([g_is.shape[0], g_js.shape[0]])
    for i_idx in range(g_is.shape[0]):
      for j_idx in range(g_js.shape[0]):
        match = (gen_factors[:, [const_factor_idx, intervened_factor_idx]]
                 == [g_is[i_idx], g_js[j_idx]]).all(axis=1)
        E_given_i_for_j[i_idx, j_idx] = np.mean(latents[match, latent_dim])
      ax.plot(g_js, E_given_i_for_j[i_idx, :], 'go--', c=colors[i_idx],
              label='g_{}=={}'.format(const_factor_idx, g_is[i_idx]),
              linewidth=1.5, markersize=3)

    ax.set_xlabel('int. g_{}'.format(intervened_factor_idx))
    ax.set_ylabel('E[z_{}|g_{}, g_{}]'.format(latent_dim, const_factor_idx,
                                              intervened_factor_idx))
    ax.set_title('Interventional Effect (keeping parent fixed)')
    if plot_legend:
      ax.legend()
