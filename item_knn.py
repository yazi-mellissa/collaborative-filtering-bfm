"""Neighbourhood models on the residuals of the bias model.

Item k-NN: to predict how user u rates movie i, take the k movies most
similar to i that u has rated, and how much u liked them more or less than
the bias model expected:

    r_ui = b_ui + sum_j s_ij z_uj / (delta + sum_j |s_ij|)

  b_ui   bias prediction mu + b_u + b_i
  z_uj   residual r_uj - b_uj
  s_ij   similarity of movies i and j
  delta  damping: keeps weak neighbourhoods close to b_ui

The similarity is the Pearson correlation of residuals over the users who
rated both movies, multiplied by n_ij / (n_ij + shrink) so that a
correlation computed from 2 common users counts little.

User k-NN is the same model with the roles of users and movies swapped.
"""

import numpy as np

from baselines import Biases
from data import R_MAX, R_MIN


class ItemKNN:
	def __init__(self, k=30, shrink=50.0, damping=1.0, beta=3.0):
		self.k = int(k)
		self.shrink, self.damping, self.beta = shrink, damping, beta
		self.name = f'Item k-NN (k={self.k})'

	def fit(self, R):
		# work on R.T: rows = movies, columns = users
		self.baseline = Biases(self.beta).fit(R).predict().T
		Rt = R.T
		M = (~np.isnan(Rt)).astype(float)  # 1 where rated
		Z = np.where(M > 0, Rt - self.baseline, 0.0)  # residuals, 0 if none
		# Pearson correlation for all pairs of movies with matrix products.
		# Z is 0 where a rating is missing, so each sum only runs over the
		# users who rated both movies.
		num = Z @ Z.T
		den = np.sqrt(((Z**2) @ M.T) * (M @ (Z**2).T))
		common = M @ M.T  # number of users who rated both
		S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
		S *= common / (common + self.shrink)
		np.fill_diagonal(S, 0.0)
		self.S, self.M, self.Z = S, M, Z
		return self

	def predict(self):
		rated = self.M > 0
		pred = self.baseline.copy()
		for u in range(rated.shape[1]):
			mine = np.flatnonzero(rated[:, u])  # movies rated by u
			if len(mine) == 0:
				continue
			S = self.S[:, mine]
			if len(mine) > self.k:
				# the k most similar of u's movies (no full sort needed)
				top = np.argpartition(-S, self.k - 1, axis=1)[:, : self.k]
				S = np.take_along_axis(S, top, axis=1)
				z = self.Z[mine, u][top]
			else:
				z = np.broadcast_to(self.Z[mine, u], S.shape)
			num = (S * z).sum(1)
			den = np.abs(S).sum(1) + self.damping
			pred[:, u] += num / den
		return np.clip(pred.T, R_MIN, R_MAX)


class UserKNN(ItemKNN):
	"""User k-NN: item k-NN on the transposed matrix."""

	def __init__(self, k=30, shrink=50.0, damping=1.0, beta=3.0):
		super().__init__(k, shrink, damping, beta)
		self.name = f'User k-NN (k={self.k})'

	def fit(self, R):
		return super().fit(R.T)

	def predict(self):
		return super().predict().T
