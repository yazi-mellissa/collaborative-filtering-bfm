"""Bayesian factorization machine with implicit features (our main model).

1. The model
------------
A factorization machine predicts a rating from a feature vector x:

    y(x) = w0 + sum_j w_j x_j + sum_{j<l} (v_j . v_l) x_j x_l

Every feature j has a weight w_j and a vector v_j of size `rank`; two
features interact through the dot product of their vectors.

For a rating (u, i) we use four groups of features:

    u   user id                       one-hot
    i   movie id                      one-hot
    iu  movies the user rated         1 / sqrt(n_u) on each of them
    ii  users who rated the movie     1 / sqrt(n_i) on each of them

With only u and i, the formula is w0 + w_u + w_i + v_u . v_i: matrix
factorization with biases. With Bayesian learning (below) this is Bayesian
PMF (`user_implicit=False, item_implicit=False`).

With iu and ii, the user and movie vectors become

    user   v_u + (1 / sqrt n_u) * sum of y_j over the movies j rated by u
    movie  v_i + (1 / sqrt n_i) * sum of z_v over the users v who rated i

The movie side is what matters here: half of the movies have 3 ratings or
fewer, so v_i alone is poorly estimated. The few users who rated such a
movie have many ratings each, so their z_v are well estimated, and the
movie gets a useful description from who rated it.

Extra movie features (genres, year...) can be added as more groups with
`item_features`. They are only used in the experiments.

2. Bayesian learning (Gibbs sampling)
-------------------------------------
Every weight and vector has a Gaussian prior. Each feature group has its
own prior mean and precision, and these are sampled too, with the noise
level of the ratings. So the strength of the regularization is learned
from the data, for each group: there is no lambda to tune.

Gibbs sampling updates each parameter in turn by drawing it from its
distribution given all the others. After a burn-in, every step gives one
plausible set of parameters (a "sample"). The prediction is the average of
the predictions of the kept samples. For a movie with few ratings the
samples disagree and their average stays cautious; for a well-known movie
they agree.

The sampler itself is the myfm library (C++). Our code builds the
features, computes the predictions of the full matrix, averages the chains
and runs them in parallel.

3. Chains
---------
A chain is one independent run of the sampler with its own seed. Several
chains explore the posterior better than one long chain, and they can run
at the same time on different CPU cores. The final prediction is the
average of all kept samples of all chains.

4. Prediction "as if rated"
---------------------------
In every training row (u, i), movie i is in the set of movies rated by u
and u is in the set of users who rated i. The cells we have to predict are
also rated cells, but their rating is not in the input. So each cell (u, i)
is predicted with i added to u's set (weight 1 / sqrt(n_u + 1)) and u
added to i's set. This is done for every cell of the matrix, so it uses no
information about which cells are evaluated.
"""

import itertools
import os
import time
from multiprocessing import TimeoutError, get_context

import numpy as np
import scipy.sparse as sp
from myfm import MyFMRegressor, RelationBlock

from data import R_MAX, R_MIN


def _run_chain(model, R, seed):
	"""One chain in a worker process (must be a top-level function)."""
	return model._chain(R, seed)


class BayesianFM:
	def __init__(
		self,
		rank=20,  # size of the vectors
		n_iter=300,  # sampling steps per chain
		burn_in=20,  # first steps thrown away
		n_chains=6,
		thin=2,  # keep one sample in two (consecutive ones are very close)
		user_implicit=True,  # feature group iu
		item_implicit=True,  # feature group ii
		item_features=None,  # list of (n_items x q) sparse matrices
		as_if_rated=True,
		time_budget=None,  # seconds; stop starting new chains after it
		n_jobs=None,  # parallel chains; None = number of CPU cores
		seed=0,
		keep_chains=False,  # keep each chain's prediction (experiments)
		verbose=False,
	):
		self.rank, self.n_iter, self.burn_in = rank, n_iter, burn_in
		self.n_chains, self.thin = n_chains, thin
		self.user_implicit, self.item_implicit = (
			user_implicit,
			item_implicit,
		)
		self.item_features = [
			sp.csr_matrix(F) for F in (item_features or [])
		]
		self.as_if_rated = as_if_rated
		self.time_budget, self.n_jobs, self.seed = time_budget, n_jobs, seed
		self.keep_chains, self.verbose = keep_chains, verbose
		self.name = f'Bayesian FM (rank {rank}, {n_chains} chains)'

	# ------------------------------------------------------------------
	# fitting: run the chains and average them
	# ------------------------------------------------------------------

	def fit(self, R):
		start = time.perf_counter()
		seeds = [self.seed + c for c in range(self.n_chains)]
		n_jobs = self.n_jobs or os.cpu_count() or 1
		n_jobs = max(1, min(n_jobs, self.n_chains))
		if n_jobs == 1:
			preds = self._fit_sequential(R, seeds, start)
		else:
			preds = self._fit_parallel(R, seeds, n_jobs, start)
		self.chains_used = len(preds)
		self.prediction = np.mean(preds, axis=0)
		self.chains = preds if self.keep_chains else []
		self.fit_seconds = time.perf_counter() - start
		return self

	def predict(self):
		return self.prediction

	def _fit_sequential(self, R, seeds, start):
		preds = []
		for c, seed in enumerate(seeds):
			preds.append(self._chain(R, seed))
			elapsed = time.perf_counter() - start
			self._log(f'chain {c + 1}/{len(seeds)}: {elapsed:.0f}s')
			# stop if the next chain would end after the budget
			if self.time_budget and elapsed * (c + 2) / (c + 1) > (
				self.time_budget
			):
				break
		return preds

	def _fit_parallel(self, R, seeds, n_jobs, start):
		# Each worker gets an equal share of the BLAS threads, so the
		# matrix products of the workers do not compete for the cores.
		# The variables are read when numpy starts in a new process.
		threads = str(max(1, (os.cpu_count() or 1) // n_jobs))
		names = (
			'OMP_NUM_THREADS',
			'OPENBLAS_NUM_THREADS',
			'MKL_NUM_THREADS',
		)
		saved = {k: os.environ.get(k) for k in names}
		os.environ.update({k: threads for k in names})
		preds = []
		try:
			with get_context('spawn').Pool(n_jobs) as pool:
				jobs = [
					pool.apply_async(_run_chain, (self, R, s))
					for s in seeds
				]
				for c, job in enumerate(jobs):
					left = None
					if self.time_budget and preds:
						left = (
							start + self.time_budget - time.perf_counter()
						)
					try:
						preds.append(job.get(timeout=left))
					except TimeoutError:
						break  # keep the chains that are finished
					self._log(
						f'chain {c + 1}/{len(seeds)}: '
						f'{time.perf_counter() - start:.0f}s'
					)
				# leaving the "with" block stops any unfinished chain
		finally:
			for k, v in saved.items():
				if v is None:
					os.environ.pop(k, None)
				else:
					os.environ[k] = v
		return preds

	def _log(self, text):
		if self.verbose:
			print('  ' + text, flush=True)

	# ------------------------------------------------------------------
	# one chain
	# ------------------------------------------------------------------

	def _design(self, R):
		"""Feature rows of every user and every movie."""
		n_users, n_items = R.shape
		users, items = np.nonzero(~np.isnan(R))
		# M[u, i] = 1 if u rated i, whatever the rating
		M = sp.csr_matrix(
			(np.ones(len(users)), (users, items)), shape=R.shape
		)
		Mt = M.T.tocsr()
		n_u, n_i = M.sum(1).A1, Mt.sum(1).A1
		# One row per user [u | iu] and one row per movie [i | ii | extra].
		# A rating's features are its user's row next to its movie's row.
		# myfm's "relation blocks" store each row once and an index per
		# rating, instead of one long row per rating.
		user_blocks = [sp.identity(n_users, format='csr')]
		groups = [n_users]
		if self.user_implicit:
			user_blocks.append(
				sp.diags(1 / np.sqrt(np.maximum(n_u, 1))) @ M
			)
			groups.append(n_items)
		item_blocks = [sp.identity(n_items, format='csr')]
		groups.append(n_items)
		if self.item_implicit:
			item_blocks.append(
				sp.diags(1 / np.sqrt(np.maximum(n_i, 1))) @ Mt
			)
			groups.append(n_users)
		for F in self.item_features:
			item_blocks.append(F)
			groups.append(F.shape[1])
		U = sp.hstack(user_blocks, format='csr')
		V = sp.hstack(item_blocks, format='csr')
		blocks = [RelationBlock(users, U), RelationBlock(items, V)]
		# group_shapes: one prior (mean, precision) per feature group
		return users, items, blocks, groups, M, Mt, n_u, n_i

	def _chain(self, R, seed):
		"""Run one chain; return the mean prediction of its kept samples."""
		users, items, blocks, groups, M, Mt, n_u, n_i = self._design(R)
		fm = MyFMRegressor(rank=self.rank, random_seed=seed)
		fm.fit(
			None,  # no plain feature matrix, everything is in the blocks
			R[users, items],
			X_rel=blocks,
			group_shapes=groups,
			n_iter=self.n_iter,
			n_kept_samples=self.n_iter - self.burn_in,
		)
		total = np.zeros(R.shape)
		kept = fm.predictor_.samples[:: self.thin]
		for sample in kept:
			total += np.clip(
				self._full_matrix(sample, M, Mt, n_u, n_i), R_MIN, R_MAX
			)
		return total / len(kept)

	# ------------------------------------------------------------------
	# prediction of all cells for one sample
	# ------------------------------------------------------------------

	def _full_matrix(self, s, M, Mt, n_u, n_i):
		"""Prediction of one sample for all 610 x 4,980 cells.

		myfm's own predict() builds one feature row per cell, which is far
		too slow for 3 million cells and hundreds of samples. The features
		split into a user part and a movie part, and the pair terms can be
		written as

		    sum_{j<l} (v_j . v_l) x_j x_l
		        = 1/2 (|sum_j v_j x_j|^2 - sum_j |v_j x_j|^2)

		With a = sum of v_j x_j over the user's features and b the same
		over the movie's features, the prediction is

		    y = w0 + (terms of u only) + (terms of i only) + a . b

		so the whole matrix comes from a few matrix products.
		"""
		n_users, n_items = M.shape
		w, F = np.asarray(s.w), np.asarray(s.V)

		# cut the weights and vectors into the feature groups,
		# in the order u | iu | i | ii | extra groups
		sizes = [n_users]
		if self.user_implicit:
			sizes.append(n_items)
		sizes.append(n_items)
		if self.item_implicit:
			sizes.append(n_users)
		sizes += [X.shape[1] for X in self.item_features]
		cuts = np.cumsum([0] + sizes)
		parts = [(w[a:b], F[a:b]) for a, b in itertools.pairwise(cuts)]
		parts = iter(parts)
		wu, P = next(parts)  # user id
		if self.user_implicit:
			wy, Y = next(parts)  # one weight and vector per rated movie
		wq, Q = next(parts)  # movie id
		if self.item_implicit:
			wz, Z = next(parts)  # one weight and vector per rater
		extras = list(parts)

		# weight of each implicit feature: 1 / sqrt(set size), one more
		# element in the set when predicting "as if rated"
		plus = 1 if self.as_if_rated else 0
		c = 1 / np.sqrt(np.maximum(n_u + plus, 1))
		d = 1 / np.sqrt(np.maximum(n_i + plus, 1))

		# alpha: user vector a, without the target movie
		alpha = P.copy()
		user_part = wu - 0.5 * (P**2).sum(1)
		if self.user_implicit:
			alpha += c[:, None] * (M @ Y)
			user_part += c * (M @ wy) - 0.5 * c**2 * (M @ (Y**2).sum(1))
		user_part += 0.5 * (alpha**2).sum(1)

		# beta: movie vector b, without the target user
		beta = Q.copy()
		item_part = wq - 0.5 * (Q**2).sum(1)
		if self.item_implicit:
			beta += d[:, None] * (Mt @ Z)
			item_part += d * (Mt @ wz) - 0.5 * d**2 * (Mt @ (Z**2).sum(1))
		for X, (we, E) in zip(self.item_features, extras):
			beta += X @ E
			item_part += X @ we - 0.5 * (X.multiply(X) @ (E**2).sum(1))
		item_part += 0.5 * (beta**2).sum(1)

		left, right = [alpha], [beta]
		out = (s.w0 + user_part)[:, None] + item_part[None, :]
		if self.as_if_rated:
			# a = alpha_u + c_u y_i and b = beta_i + d_i z_u. The extra
			# terms of a . b and of the pair terms are added below; the
			# products that need both u and i go into one matrix product.
			if self.user_implicit:
				left.append(c[:, None] * alpha)
				right.append(Y)
				out += c[:, None] * (wy + (Y * beta).sum(1))[None, :]
			if self.item_implicit:
				left.append(Z)
				right.append(d[:, None] * beta)
				out += (wz + (alpha * Z).sum(1))[:, None] * d[None, :]
			if self.user_implicit and self.item_implicit:
				left.append(c[:, None] * Z)
				right.append(d[:, None] * Y)
		out += np.hstack(left) @ np.hstack(right).T
		return out
