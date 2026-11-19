"""The final model, run by generate.py: a Bayesian factorization machine
with the implicit features and, for each movie, its genres and release
decade (6 chains of 100 steps averaged).

Blending it with item k-NN was tested (E8): 0.8260 for the best weight
against 0.8260 alone, so the final model is the factorization machine
alone. The settings come from experiments.py (E10: 100 steps, rank 20).
"""

import numpy as np

import movie_features as mf
from bayesian_fm import BayesianFM
from data import R_MAX, R_MIN

BFM_SETTINGS = {
	'rank': 20,
	'n_iter': 100,
	'burn_in': 20,
	'n_chains': 6,
	'thin': 2,
	'as_if_rated': True,
}


def FINAL_FEATURES():
	"""Movie features of the final model."""
	return [mf.genres(), mf.decades()]


def final_model(R, time_budget=None, verbose=False):
	"""Complete the ratings matrix R. Returns the matrix and the number of
	chains that finished within time_budget (seconds)."""
	bfm = BayesianFM(
		item_features=FINAL_FEATURES(),
		time_budget=time_budget,
		verbose=verbose,
		**BFM_SETTINGS,
	).fit(R)
	return np.clip(bfm.predict(), R_MIN, R_MAX), bfm.chains_used
