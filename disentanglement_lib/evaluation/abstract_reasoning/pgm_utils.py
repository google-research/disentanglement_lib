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

"""Utilities to create procedurally generated matrices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class PGM(object):
  """PGM where ground-truh factors are represented as integer values."""

  def __init__(self,
               random_state,
               num_relations,
               atom_counts,
               sampling_strategy="easy",
               num_rows=3,
               num_cols=3,
               num_solutions=6):
    """Creates a PGM.

    Args:
      random_state: np.random.RandomState used to sample the PGM.
      num_relations: Number of relations to enforce for each row in the PGM.
      atom_counts: List that contains the number of atoms for each of the
        ground-truth factors.
      sampling_strategy: Either `easy` or `hard`. For `easy`, alternative
        answers are random other solutions that do not satisfy the constraints
        in the given PGM. For `hard`, alternative answers are unique random
        modifications of the correct solution which makes the task  harder.
      num_rows: Integer with number of rows.
      num_cols: Integer with number of columns.
      num_solutions: Integer with number of solutions in the PGM.
    """
    if sampling_strategy == "easy":
      sampling_fn = sample_easy_alternative
    elif sampling_strategy == "hard":
      sampling_fn = sample_hard_alternative
    else:
      raise ValueError("Only easy and hard sampling are currently supported.")

    # Create a factory for this PGM and sample a random solution to it.
    self.design = PGMDesign(random_state, num_relations, atom_counts, num_rows,
                            num_cols)
    self.matrix = self.design.sample()

    # Use the sample strategy to create additional wrong solutions.
    self.other_solutions = []
    for _ in range(num_solutions - 1):
      self.other_solutions.append(
          sampling_fn(self.design, self.matrix, self.other_solutions))
    self.other_solutions = np.array(self.other_solutions)

  def print_pgm(self):
    """Prints the PGM to stdout."""
    for i in range(self.matrix.shape[2]):
      print("---")
      print("Factor %d" % i)
      print("---")
      print("Solution:")
      print(self.matrix[:, :, i])
      print("Alternatives:")
      print(self.other_solutions[:, i])


class PGMDesign(object):
  """Captures the design of a PGM (i.e. the rules) but not the actual values."""

  def __init__(self,
               random_state,
               num_relations,
               atom_counts,
               num_rows=3,
               num_cols=3):
    """Creates a PGMDesign.

    Args:
      random_state: np.random.RandomState used to sample the PGM.
      num_relations: Number of relations to enforce for each row in the PGM.
      atom_counts: List that contains the number of atoms for each of the
        ground-truth factors.
      num_rows: Integer with number of rows.
      num_cols: Integer with number of columns.
    """
    self.random_state = random_state
    self.num_relations = num_relations
    self.atom_counts = atom_counts
    self.num_rows = num_rows
    self.num_cols = num_cols

    # Setup list to keep the relations for each of the factors. By default,
    # each factor has no active relation.
    self.relations = [NonActiveRelation for _ in atom_counts]

    # Randomly sample factors where all factors will be the same across rows.
    self.num_factors = len(atom_counts)
    if self.num_factors < num_relations:
      raise ValueError("Cannot have less factors than relations.")
    indices = list(range(self.num_factors))
    self.active_relations = []
    for _ in range(num_relations):
      selected_index = indices.pop(random_state.choice(len(indices)))
      self.active_relations.append(selected_index)
      self.relations[selected_index] = ConstantRelation

    # Create the actual relations.
    for i, num_atoms in enumerate(atom_counts):
      self.relations[i] = self.relations[i](num_atoms, num_rows, num_cols)

  def sample(self):
    """Sample the actual values of a PGM.

    Returns:
      Numpy array of type np.int64 and shape (num_rows, num_cols, num_factors)
        with the ground-truth factor values of the PGM.
    """
    matrix = np.zeros((self.num_rows, self.num_cols, self.num_factors),
                      dtype=np.int64)
    for i, relation in enumerate(self.relations):
      matrix[:, :, i] = relation.sample(self.random_state)
    return matrix

  def randomly_modify_solution(self, initial_solution):
    """Randomly modifies solution to generate hard alternatives.

    Args:
      initial_solution: Numpy array with shape (num_rows, num_cols, num_factors)
        with the ground-truth factor values of the original PGM.

    Returns:
      Numpy array of type np.int64 and shape (num_rows, num_cols, num_factors)
        with the ground-truth factor values of a randomly modified PGM.
    """
    solution = np.copy(initial_solution)
    # Resample a random factor uniformly where a relation is active.
    i = self.random_state.choice(self.active_relations)
    relation = self.relations[i]
    solution[i] = self.random_state.choice(relation.num_atoms)
    # Change all the non-active relations to random values.
    for i, relation in enumerate(self.relations):
      if i not in self.active_relations:
        solution[i] = self.random_state.choice(relation.num_atoms)
    return solution

  def is_consistent(self, matrix):
    """Check whether the matrix is consistent with the PGM Design."""
    for i, relation in enumerate(self.relations):
      if not relation.is_consistent(matrix[:, :, i]):
        return False
    return True

  def resample_design(self):
    """Generates an alternative design of the PGM.

    This will generate a different set of active/non-active relations.

    Returns:
      PGMDesign instance with the new design.
    """
    return PGMDesign(self.random_state, self.num_relations, self.atom_counts,
                     self.num_rows, self.num_cols)


def sample_easy_alternative(design, matrix, already_sampled_alternatives):
  """Samples easy alternative based on sampling a new PGM."""
  for _ in range(100):
    alternative_pgm = design.resample_design().sample()
    # Combine the solutions.
    alternative_solution = np.copy(matrix)
    alternative_solution[-1, -1, :] = alternative_pgm[-1, -1, :]
    if design.is_consistent(alternative_solution):
      continue
    for already_sampled_alternative in already_sampled_alternatives:
      if np.allclose(already_sampled_alternative, alternative_pgm[-1, -1, :]):
        continue
    return alternative_pgm[-1, -1, :]
  raise ValueError("Could not sample alternative solutions.")


def sample_hard_alternative(design, matrix, already_sampled_alternatives):
  """Samples hard alternative based on sampling a new PGM."""
  solution_so_far = matrix[-1, -1]
  for _ in range(100):
    solution_so_far = design.randomly_modify_solution(solution_so_far)
    # Combine the solutions.
    alternative_solution = np.copy(matrix)
    alternative_solution[-1, -1, :] = solution_so_far
    if design.is_consistent(alternative_solution):
      continue
    for already_sampled_alternative in already_sampled_alternatives:
      if np.allclose(already_sampled_alternative, solution_so_far):
        continue
    return solution_so_far
  raise ValueError("Could not sample hard alternative solutions.")


class Relation(object):
  """Abstract base class for relations."""

  def __init__(self, num_atoms, num_rows=3, num_cols=3):
    if num_atoms < num_cols:
      raise ValueError("Cannot have less atoms than columns.")
    if num_atoms == 1:
      raise ValueError("Need more than one atom.")
    self.num_atoms = num_atoms
    self.num_rows = num_rows
    self.num_cols = num_cols

  @staticmethod
  def is_consistent(matrix):
    """Checks whether the matrix satisfies the relation."""
    raise NotImplementedError()

  def sample(self, random_state):
    """Samples a matrix consistent with the relation."""
    raise NotImplementedError()


def is_constant_row(row):
  return len(np.unique(row)) == 1


class ConstantRelation(Relation):
  """Relation where rows in the matrix are constant."""

  @staticmethod
  def is_consistent(matrix):
    """Checks whether the matrix satisfies the relation."""
    for row in matrix:
      if not is_constant_row(row):
        return False
    return True

  def sample(self, random_state):
    """Samples a matrix consistent with the relation."""
    rows = []
    for _ in range(self.num_rows):
      sampled_atom = random_state.choice(self.num_atoms)
      rows.append([sampled_atom] * self.num_cols)
    return np.array(rows)


def is_distinct_row(row):
  return len(np.unique(row)) == len(row)


class DistinctRelation(Relation):
  """Relation where elements in a matrix row are distinct."""

  @staticmethod
  def is_consistent(matrix):
    """Checks whether the matrix satisfies the relation."""
    for row in matrix:
      if not is_distinct_row(row):
        return False
    return True

  def sample(self, random_state):
    """Samples a matrix consistent with the relation."""
    rows = []
    for _ in range(self.num_rows):
      random_permutation = random_state.permutation(self.num_atoms)
      rows.append(random_permutation[:self.num_cols])
    return np.array(rows)


class NonActiveRelation(Relation):
  """Relation where elements are random but do not satisfy other relation."""

  @staticmethod
  def is_consistent(matrix):
    """Checks whether the matrix satisfies the relation."""
    # We need to make sure there are no consistent relations in the rows except
    # the last ones.
    relevant_matrix = matrix[:-1, :]
    for relation in [ConstantRelation, DistinctRelation]:
      if relation.is_consistent(relevant_matrix):
        return False
    return True

  def _sample(self, random_state):
    """Sample a random matrix."""
    return random_state.choice(
        self.num_atoms, size=(self.num_rows, self.num_cols))

  def sample(self, random_state):
    """Samples a matrix consistent with the relation."""
    for _ in range(1000):
      matrix = self._sample(random_state)
      if self.is_consistent(matrix):
        return matrix
    raise ValueError("Could not sample non-relational matrix.")
