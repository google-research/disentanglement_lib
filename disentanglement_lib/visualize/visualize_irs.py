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


def vis_all_interventional_effects(g, z, output_dir):
  """Compute Matrix of all interventional effects."""
  res = scalable_disentanglement_score(g, z)
  parents = res['parents']
  scores = res['disentanglement_scores']

  fig, axes = plt.subplots(z.shape[1], g.shape[1],
                           figsize=(3 * g.shape[1], 3 * z.shape[1]),
                           sharex='col', sharey='row')

  for j in range(g.shape[1]):  # iterate over generative factors
    for l in range(z.shape[1]):
      ax = axes[l, j]
      if parents[l] != j:
        visualize_interventional_effect(g, z, l, parents[l], j, ax=ax)
        ax.set_title('')
      else:
        visualize_interventional_effect(g, z, l, parents[l], j,
                                        no_conditioning=True, ax=ax)
        ax.set_title('Parent={}, IRS = {:1.2}'.format(parents[l], scores[l]))

  fig.tight_layout()
  output_path = os.path.join(output_dir, 'interventional_effect.png')
  fig.savefig(output_path)


def visualize_interventional_effect(g, z, l, i, j, no_conditioning=False,
                                    ax=None, plot_legend=True,
                                    plot_scatter=True):
  """Visualize single cell of interventional effects of z_l given g_i and
  intervening on g_j.

  Args
    g: Generative factors
    z: latent factors
    l: latent dimension under consideration
    i: g_i which is being kept constant
    j: g_j which we intervene on
    no_conditioning: whether or not we should condition on i
  """
  if ax is None:
    plt.figure(figsize=(10, 7))
    ax = plt.axes()

  g_is = np.unique(g[:, i], axis=0)
  g_js = np.unique(g[:, j], axis=0)

  colors = 10 * ['b', 'y', 'g', 'r', 'c', 'm', 'k']
  cols_idx = np.empty([g.shape[0]], dtype=int)
  for i_idx in range(g_is.shape[0]):
    match = (g[:, [i]] == [g_is[i_idx]]).all(axis=1)
    cols_idx[match] = i_idx
  cols = [colors[col % len(colors)] for col in cols_idx]

  # Plot all points, color indicates constant factor
  if plot_scatter:
    ax.scatter(g[:, j], z[:, l], c=cols)

  # Compute possible g_i and g_j
  if no_conditioning:
    E_for_j = np.empty([g_js.shape[0]])
    median_for_j = np.empty([g_js.shape[0]])
    stdev_for_j = np.empty([g_js.shape[0]])
    for j_idx in range(g_js.shape[0]):
      match = (g[:, j] == g_js[j_idx])
      E_for_j[j_idx] = np.mean(z[match, l])
      median_for_j[j_idx] = np.median(z[match, l])
      stdev_for_j[j_idx] = np.std(z[match, l])
    ax.plot(g_js, E_for_j, linewidth=2, markersize=12, label='mean')
    ax.plot(g_js, median_for_j, linewidth=2, markersize=12, label='median')
    ax.plot(g_js, E_for_j + stdev_for_j, linestyle='--', c='b', linewidth=1)
    ax.plot(g_js, E_for_j - stdev_for_j, linestyle='--', c='b', linewidth=1)
    ax.set_ylabel('E[z_{}|g_{}]'.format(l, j))
    ax.set_xlabel('g_{}'.format(j))
    ax.legend()
  else:
    # Compute E[Z_l | g_i, g_j] as a function of g_j for each g_i
    E_given_i_for_j = np.empty([g_is.shape[0], g_js.shape[0]])
    for i_idx in range(g_is.shape[0]):
      for j_idx in range(g_js.shape[0]):
        match = (g[:, [i, j]] == [g_is[i_idx], g_js[j_idx]]).all(axis=1)
        E_given_i_for_j[i_idx, j_idx] = np.mean(z[match, l])
      ax.plot(g_js, E_given_i_for_j[i_idx, :], 'go--', c=colors[i_idx],
              label='g_{}=={}'.format(i, g_is[i_idx]), linewidth=1.5,
              markersize=3)

    ax.set_xlabel('int. g_{}'.format(j))
    ax.set_ylabel('E[z_{}|g_{}, g_{}]'.format(l, i, j))
    ax.set_title('Interventional Effect (keeping parent fixed)')
    if plot_legend:
      ax.legend()
