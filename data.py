"""Data loading and metrics.

Every model takes a ratings matrix R (610 users x 4,980 movies, NaN where
there is no rating) and returns a full matrix of predictions.

Protocol: models are fitted on ratings_train.npy and scored on
ratings_test.npy. The platform then fits on train + test and scores the
hidden eval ratings.
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'dataset'
R_MIN, R_MAX = 0.5, 5.0  # ratings go from 0.5 to 5 by half stars


def load(name):
	"""Load one of the .npy files from dataset/."""
	return np.load(DATA / name, allow_pickle=name.startswith('namesngenre'))


def load_train_test():
	return load('ratings_train.npy'), load('ratings_test.npy')


def merge(A, B):
	"""Union of two rating matrices with no cell in common (train + test is
	what the platform gives to generate.py)."""
	out = A.copy()
	mask = ~np.isnan(B)
	out[mask] = B[mask]
	return out


def count(R):
	"""Number of ratings in a matrix."""
	return int((~np.isnan(R)).sum())


def _errors(pred, truth):
	mask = ~np.isnan(truth)
	return np.clip(pred[mask], R_MIN, R_MAX) - truth[mask]


def rmse(pred, truth):
	"""Root mean squared error over the rated cells of `truth` (the
	platform's metric). Predictions are clipped to [0.5, 5] first."""
	return float(np.sqrt(np.mean(_errors(pred, truth) ** 2)))


def mae(pred, truth):
	"""Mean absolute error: the average distance to the true rating."""
	return float(np.mean(np.abs(_errors(pred, truth))))


def accuracy(pred, truth):
	"""Share of rated cells where the prediction, rounded to the nearest
	half star, is exactly the true rating (the platform's second metric)."""
	mask = ~np.isnan(truth)
	rounded = np.clip(np.round(pred[mask] * 2) / 2, R_MIN, R_MAX)
	return float(np.mean(rounded == truth[mask]))
