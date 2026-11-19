"""The two baselines."""

import numpy as np


class GlobalMean:
	"""Predict the mean of all observed ratings everywhere (the starter
	code)."""

	name = 'Global mean (starter code)'

	def fit(self, R):
		self.mu = np.nanmean(R)
		self.shape = R.shape
		return self

	def predict(self):
		return np.full(self.shape, self.mu)


class Biases:
	"""r_ui = mu + b_u + b_i, with damped biases.

	b_i is how much movie i is rated above or below the average, b_u the
	same for user u. Each one is a mean of residuals where beta is added to
	the count:

	    b_i = sum over raters u of (r_ui - mu - b_u) / (n_i + beta)

	A movie rated once keeps 1 / (1 + beta) of its deviation, a movie with
	100 ratings keeps almost all of it. b_u and b_i depend on each other, so
	they are updated in turn for a few rounds.
	"""

	name = 'User + movie biases'

	def __init__(self, beta=3.0, n_iter=10):
		self.beta, self.n_iter = beta, n_iter

	def fit(self, R):
		n_users, n_items = R.shape
		u, i = np.nonzero(~np.isnan(R))
		r = R[u, i]
		self.mu = r.mean()
		bu, bi = np.zeros(n_users), np.zeros(n_items)
		den_u = np.bincount(u, minlength=n_users) + self.beta
		den_i = np.bincount(i, minlength=n_items) + self.beta
		for _ in range(self.n_iter):
			# bincount sums the residuals of each movie (then each user)
			s_i = np.bincount(i, r - self.mu - bu[u], n_items)
			bi = np.divide(
				s_i, den_i, out=np.zeros(n_items), where=den_i > 0
			)
			s_u = np.bincount(u, r - self.mu - bi[i], n_users)
			bu = np.divide(
				s_u, den_u, out=np.zeros(n_users), where=den_u > 0
			)
		self.bu, self.bi = bu, bi
		return self

	def predict(self):
		return self.mu + self.bu[:, None] + self.bi[None, :]
