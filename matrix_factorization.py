"""Matrix factorization, fitted with ALS or with gradient descent.

Model:

    r_ui  ~  mu + b_u + b_i  +  p_u . q_i

p_u and q_i are k-dimensional vectors of user u and movie i. They are
fitted on the residuals of the bias model, over the observed ratings S
only:

    C(P, Q) = sum_{(u,i) in S} (e_ui - p_u . q_i)^2 + lam (||P||^2 + ||Q||^2)

with e_ui = r_ui - mu - b_u - b_i. On residuals, a vector shrunk to 0 means
"predict the bias model". On raw ratings it would mean "predict 0 stars"
(the use_bias=False control).

Missing cells are never treated as ratings.
"""

import time

import numpy as np
from scipy.sparse import csr_matrix

from baselines import Biases
from data import R_MAX, R_MIN, rmse


class _FactorModel:
	"""Shared part of ALS and gradient descent: targets, initialisation,
	prediction and learning curves."""

	def __init__(self, rank, lam, n_iter, use_bias, beta, init_scale, seed):
		self.rank, self.lam, self.n_iter = rank, lam, n_iter
		self.use_bias, self.beta = use_bias, beta
		self.init_scale, self.seed = init_scale, seed

	def _setup(self, R):
		# list of observed ratings: (users[n], items[n]) is the n-th one
		self.users, self.items = np.nonzero(~np.isnan(R))
		ratings = R[self.users, self.items]
		if self.use_bias:
			# biases are fitted first, then kept fixed
			self.base = Biases(self.beta).fit(R)
			offset = self.base.predict()[self.users, self.items]
		else:
			self.base, offset = None, 0.0
		self.target = ratings - offset  # what the vectors must explain
		rng = np.random.default_rng(self.seed)
		n_users, n_items = R.shape
		# small random start: with P = Q = 0 nothing would ever move
		self.P = rng.normal(0, self.init_scale, (n_users, self.rank))
		self.Q = rng.normal(0, self.init_scale, (n_items, self.rank))
		self.history = {'step': [], 'seconds': []}
		self._start = time.perf_counter()

	def _record(self, step, monitor):
		"""Store the RMSE on each matrix of `monitor` after a step."""
		self.history['step'].append(step)
		self.history['seconds'].append(time.perf_counter() - self._start)
		pred = self.predict()
		for key, M in monitor.items():
			self.history.setdefault(key, []).append(rmse(pred, M))

	def predict(self):
		pred = self.P @ self.Q.T
		if self.base is not None:
			pred = pred + self.base.predict()
		return np.clip(pred, R_MIN, R_MAX)


class ALS(_FactorModel):
	"""Alternating Least Squares.

	With Q fixed, the cost is one ridge regression per user, solved exactly:

	    p_u = (Q_u^T Q_u + lam I)^-1  Q_u^T e_u

	(Q_u: vectors of the movies rated by u, e_u: their residuals). Then the
	same for every movie with P fixed. One sweep = both halves. Each half is
	an exact minimum, so the cost never goes up and there is no step size.
	"""

	def __init__(
		self,
		rank=10,
		lam=10.0,
		n_iter=15,
		use_bias=True,
		beta=3.0,
		init_scale=0.1,
		seed=0,
	):
		super().__init__(
			rank, lam, n_iter, use_bias, beta, init_scale, seed
		)
		self.name = f'MF + biases, ALS (k={rank}, lam={lam:g})'

	@staticmethod
	def _solve(rows, cols, values, n_rows, F, lam):
		"""For every row r: argmin_x sum_c (v_rc - x . F_c)^2 + lam ||x||^2.
		Rows are users (F = Q) or movies (F = P)."""
		k = F.shape[1]
		# sort by row so the ratings of row r are a contiguous block
		order = np.argsort(rows, kind='stable')
		rows, cols, values = rows[order], cols[order], values[order]
		bounds = np.searchsorted(rows, np.arange(n_rows + 1))
		A = np.repeat(lam * np.eye(k)[None], n_rows, axis=0)
		b = np.zeros((n_rows, k))
		for r in range(n_rows):
			F_r = F[cols[bounds[r] : bounds[r + 1]]]
			A[r] += F_r.T @ F_r
			b[r] = F_r.T @ values[bounds[r] : bounds[r + 1]]
		# all the k x k systems solved in one call
		return np.linalg.solve(A, b[..., None])[..., 0]

	def fit(self, R, monitor=None):
		self._setup(R)
		n_users, n_items = R.shape
		for sweep in range(1, self.n_iter + 1):
			self.P = self._solve(
				self.users,
				self.items,
				self.target,
				n_users,
				self.Q,
				self.lam,
			)
			self.Q = self._solve(
				self.items,
				self.users,
				self.target,
				n_items,
				self.P,
				self.lam,
			)
			if monitor:
				self._record(sweep, monitor)
		self.fit_seconds = time.perf_counter() - self._start
		return self


class GradientDescentMF(_FactorModel):
	"""Full-batch gradient descent on the same cost:

	    dC/dP = -2 E Q + 2 lam P        dC/dQ = -2 E^T P + 2 lam Q

	E is the sparse matrix of the current errors on the observed ratings.
	A step size that is too large makes the values explode; fit() then
	raises an error.
	"""

	def __init__(
		self,
		rank=10,
		lam=10.0,
		n_iter=300,
		lr=0.002,
		use_bias=True,
		beta=3.0,
		init_scale=0.1,
		seed=0,
	):
		super().__init__(
			rank, lam, n_iter, use_bias, beta, init_scale, seed
		)
		self.lr = lr
		self.name = f'MF + biases, gradient descent (k={rank}, lam={lam:g})'

	def fit(self, R, monitor=None, record_every=10):
		self._setup(R)
		u, i = self.users, self.items
		for step in range(1, self.n_iter + 1):
			err = self.target - np.einsum('nk,nk->n', self.P[u], self.Q[i])
			E = csr_matrix((err, (u, i)), shape=R.shape)
			grad_P = -2 * (E @ self.Q) + 2 * self.lam * self.P
			grad_Q = -2 * (E.T @ self.P) + 2 * self.lam * self.Q
			self.P -= self.lr * grad_P
			self.Q -= self.lr * grad_Q
			if not np.isfinite(self.P).all():
				raise FloatingPointError(f'diverged at step {step}')
			if monitor and (step % record_every == 0 or step == 1):
				self._record(step, monitor)
		self.fit_seconds = time.perf_counter() - self._start
		return self
