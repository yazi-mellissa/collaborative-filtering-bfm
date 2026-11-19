"""The whole experimental study.

    uv run python experiments.py            # runs the sections not done yet
    uv run python experiments.py --plots    # only redraw the figures

Every model is fitted on ratings_train.npy and scored on ratings_test.npy.
Results go to results/results.json after each section; a rerun skips the
sections already done. Predictions on the test cells are cached in
results/cache/ for the blend, the stack and the final table.

  E1  baselines: global mean, biases (beta)
  E2  matrix factorization (ALS): rank x lambda
  E3  regularization: training vs test RMSE as lambda varies
  E4  ALS vs gradient descent on the same cost
  E5  random seeds
  E6  neighbourhood models: item k-NN and user k-NN
  E7  Bayesian FM: features, "as if rated", rank, steps, metadata, chains
  E8  blend with item k-NN and a 12-model stack
  E9  final table and run time on the platform input
  E10 final model: steps, rank, seeds, as if rated, size of the training set
  E11 quality of the samples along one long chain
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import movie_features as mf
from baselines import Biases, GlobalMean
from bayesian_fm import BayesianFM
from data import R_MAX, R_MIN, accuracy, load_train_test, mae, merge, rmse
from final_model import BFM_SETTINGS, FINAL_FEATURES
from item_knn import ItemKNN, UserKNN
from matrix_factorization import ALS, GradientDescentMF

OUT = Path('results')
RESULTS = OUT / 'results.json'
CACHE = OUT / 'cache'


def timed(model, R, **kw):
	"""Fit a model; return (prediction, seconds for fit + predict)."""
	start = time.perf_counter()
	pred = model.fit(R, **kw).predict()
	return pred, time.perf_counter() - start


def cached(key, make, train):
	"""Prediction of a model, computed once and stored in results/cache."""
	CACHE.mkdir(parents=True, exist_ok=True)
	path, sec_path = CACHE / f'{key}.npy', CACHE / f'{key}.json'
	if path.exists():
		return np.load(path), json.loads(sec_path.read_text())['seconds']
	pred, sec = timed(make(), train)
	np.save(path, pred.astype(np.float32))
	sec_path.write_text(json.dumps({'seconds': sec}))
	return pred, sec


# ---------------------------------------------------------------------------
# E1 - baselines
# ---------------------------------------------------------------------------


def e1_baselines(res, train, test):
	out = {'global_mean': rmse(GlobalMean().fit(train).predict(), test)}
	out['biases_by_beta'] = {
		str(b): rmse(Biases(b).fit(train).predict(), test)
		for b in [0, 1, 2, 3, 5, 10, 20]
	}
	out['best_beta'] = float(
		min(out['biases_by_beta'], key=out['biases_by_beta'].get)
	)
	print(
		f'  global mean {out["global_mean"]:.4f} | biases by beta',
		{k: round(v, 4) for k, v in out['biases_by_beta'].items()},
	)
	return out


# ---------------------------------------------------------------------------
# E2 - ALS grid over rank and regularization
# ---------------------------------------------------------------------------

RANKS = [1, 2, 5, 10, 20, 50]
LAMS = [1, 3, 10, 20, 50]


def e2_als_grid(res, train, test):
	"""Rank and lambda are varied together: the best lambda depends on the
	rank. Also the control without biases."""
	beta = res['E1']['best_beta']
	score, fit_score, secs = {}, {}, {}
	for k in RANKS:
		for lam in LAMS:
			pred, sec = timed(ALS(k, lam, n_iter=15, beta=beta), train)
			key = f'{k},{lam}'
			score[key], fit_score[key], secs[key] = (
				rmse(pred, test),
				rmse(pred, train),
				sec,
			)
		print(
			f'  k={k:2d} test:',
			' '.join(f'{score[f"{k},{l}"]:.4f}' for l in LAMS),
			f'| {np.mean([secs[f"{k},{l}"] for l in LAMS]):.1f}s per fit',
			flush=True,
		)
	best = min(score, key=score.get)
	k, lam = best.split(',')
	raw = {
		str(l): rmse(
			ALS(int(k), l, use_bias=False).fit(train).predict(), test
		)
		for l in LAMS
	}
	print(
		f'  best k={k}, lam={lam}: {score[best]:.4f} | without biases:',
		{l: round(v, 3) for l, v in raw.items()},
	)
	return {
		'ranks': RANKS,
		'lams': LAMS,
		'test': score,
		'train': fit_score,
		'seconds': secs,
		'best': [int(k), float(lam)],
		'raw_mf_test': raw,
	}


# ---------------------------------------------------------------------------
# E3 - regularization: training vs test error
# ---------------------------------------------------------------------------


def e3_regularization(res, train, test):
	k = res['E2']['best'][0]
	beta = res['E1']['best_beta']
	lams = [0.3, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
	out = {'rank': k, 'lams': lams, 'train': [], 'test': []}
	for lam in lams:
		pred = ALS(k, lam, n_iter=15, beta=beta).fit(train).predict()
		out['train'].append(rmse(pred, train))
		out['test'].append(rmse(pred, test))
	out['biases_train'] = rmse(Biases(beta).fit(train).predict(), train)
	print('  lambda :', ' '.join(f'{l:>6g}' for l in lams))
	print('  train  :', ' '.join(f'{v:.4f}' for v in out['train']))
	print('  test   :', ' '.join(f'{v:.4f}' for v in out['test']))
	return out


# ---------------------------------------------------------------------------
# E4 - ALS vs gradient descent
# ---------------------------------------------------------------------------


def e4_als_vs_gd(res, train, test):
	k, lam = res['E2']['best']
	beta = res['E1']['best_beta']
	mon = {'test': test, 'train': train}
	als = ALS(k, lam, n_iter=25, beta=beta).fit(train, monitor=mon)
	out = {'rank': k, 'lam': lam, 'als': als.history, 'gd': {}}
	print(
		f'  ALS: test {als.history["test"][-1]:.4f} after 25 sweeps, '
		f'{als.fit_seconds:.1f}s'
	)
	for lr in [0.0005, 0.001, 0.002, 0.004, 0.008]:
		gd = GradientDescentMF(k, lam, n_iter=400, lr=lr, beta=beta)
		try:
			gd.fit(train, monitor=mon, record_every=10)
			out['gd'][str(lr)] = gd.history
			print(
				f'  GD step {lr}: test {gd.history["test"][-1]:.4f} after '
				f'400 steps, {gd.fit_seconds:.1f}s',
				flush=True,
			)
		except FloatingPointError as e:
			out['gd'][str(lr)] = {'diverged': str(e)}
			print(f'  GD step {lr}: {e}')
	ok = {lr: h['test'][-1] for lr, h in out['gd'].items() if 'test' in h}
	out['best_lr'] = float(min(ok, key=ok.get))
	return out


# ---------------------------------------------------------------------------
# E5 - random seeds
# ---------------------------------------------------------------------------


def e5_seeds(res, train, test):
	k, lam = res['E2']['best']
	beta, lr = res['E1']['best_beta'], res['E4']['best_lr']
	out = {}
	for name, make in [
		('ALS', lambda seed: ALS(k, lam, n_iter=15, beta=beta, seed=seed)),
		(
			'GD',
			lambda seed: GradientDescentMF(
				k, lam, n_iter=400, lr=lr, beta=beta, seed=seed
			),
		),
	]:
		scores = [
			rmse(make(seed).fit(train).predict(), test) for seed in range(5)
		]
		out[name] = {
			'test': scores,
			'mean': float(np.mean(scores)),
			'std': float(np.std(scores)),
		}
		print(
			f'  {name}: test {np.mean(scores):.4f} +/- {np.std(scores):.4f}'
			' over 5 seeds'
		)
	return out


# ---------------------------------------------------------------------------
# E6 - neighbourhood models
# ---------------------------------------------------------------------------


def e6_neighbours(res, train, test):
	"""Same grid for item k-NN and user k-NN."""
	beta = res['E1']['best_beta']
	out = {}
	for name, cls in [('item', ItemKNN), ('user', UserKNN)]:
		grid = {}
		for k in [10, 30, 100]:
			for shrink in [20, 50]:
				for damping in [1, 3]:
					m = cls(k, shrink, damping, beta)
					grid[f'{k},{shrink},{damping}'] = rmse(
						m.fit(train).predict(), test
					)
		best = min(grid, key=grid.get)
		out[name] = {
			'grid': grid,
			'best': [float(x) for x in best.split(',')],
			'best_test': grid[best],
		}
		print(f'  {name} k-NN: best {best}: {grid[best]:.4f}', flush=True)
	return out


# ---------------------------------------------------------------------------
# E7 - Bayesian factorization machine
# ---------------------------------------------------------------------------


def bfm_variants(train):
	"""Every Bayesian FM variant studied, as {key: (label, settings)}.
	All use one chain except the last one."""
	meta = {
		'genres': [mf.genres()],
		'genres_decade': [mf.genres(), mf.decades()],
		'year': [mf.years()],
		'title': [mf.title_words()],
		'popularity': [mf.popularity(train)],
		'all_meta': [
			mf.genres(),
			mf.decades(),
			mf.years(),
			mf.title_words(),
			mf.popularity(train),
		],
	}
	ids = {'user_implicit': False, 'item_implicit': False}
	v = {
		# which features
		'ids': ('Bayesian PMF (user and movie ids only)', ids),
		'ids_iu': (
			'+ movies rated by the user',
			{'user_implicit': True, 'item_implicit': False},
		),
		'ids_ii': (
			'+ users who rated the movie',
			{'user_implicit': False, 'item_implicit': True},
		),
		'r20': ('+ both implicit groups (rank 20)', {}),
		'plain': (
			'same, NOT predicted "as if rated"',
			{'as_if_rated': False},
		),
		# size and length of the chain
		'r10': ('rank 10', {'rank': 10}),
		'r32': ('rank 32', {'rank': 32}),
		'r64': ('rank 64', {'rank': 64}),
		'steps600': ('600 sampling steps', {'n_iter': 600}),
	}
	labels = {
		'genres': '+ genres',
		'genres_decade': '+ genres + decade',
		'year': '+ release year',
		'title': '+ title words',
		'popularity': '+ popularity bucket',
		'all_meta': '+ all metadata',
	}
	for key, feats in meta.items():
		v['meta_' + key] = (labels[key], {'item_features': feats})
	v['ids_meta'] = (
		'Bayesian PMF + genres + decade',
		{**ids, 'item_features': meta['genres_decade']},
	)
	return v


def e7_bayesian_fm(res, train, test):
	out = {'variants': {}}
	for key, (label, kw) in bfm_variants(train).items():
		settings = {'rank': 20, 'n_chains': 1, 'seed': 0, **kw}
		pred, sec = cached(
			'bfm_' + key, lambda s=settings: BayesianFM(**s), train
		)
		out['variants'][key] = {
			'label': label,
			'test': rmse(pred, test),
			'seconds': sec,
		}
		print(
			f'  {label:45s} {rmse(pred, test):.4f}  ({sec:.0f}s)',
			flush=True,
		)

	# number of chains averaged (6 chains run in parallel), without and
	# with the genres + decade features
	for key, kw in [
		('bfm_6chains', {}),
		('bfm_final_6chains', {'item_features': FINAL_FEATURES()}),
	]:
		m = BayesianFM(
			rank=20,
			n_iter=BFM_SETTINGS['n_iter'],
			n_chains=6,
			keep_chains=True,
			seed=0,
			**kw,
		)
		pred, sec = timed(m, train)
		np.save(CACHE / f'{key}.npy', pred.astype(np.float32))
		(CACHE / f'{key}.json').write_text(json.dumps({'seconds': sec}))
		out[key] = {
			'chains': [
				rmse(np.mean(m.chains[:n], 0), test) for n in range(1, 7)
			],
			'seconds': sec,
		}
		print(
			f'  {key} 1..6 chains:',
			' '.join(f'{v:.4f}' for v in out[key]['chains']),
			f'({sec:.0f}s in parallel)',
		)
	return out


# ---------------------------------------------------------------------------
# E8 - blend and stack
# ---------------------------------------------------------------------------


def stack_models(res):
	"""The 12 models of the stack: {name: cache key}."""
	return {
		'Global mean': 'global_mean',
		'Biases': 'biases',
		'MF without biases (ALS)': 'als_raw',
		'MF + biases (ALS)': 'als',
		'MF + biases (gradient descent)': 'gd',
		'User k-NN': 'user_knn',
		'Item k-NN': 'item_knn',
		'Bayesian PMF': 'bfm_ids',
		'Bayesian FM + genres + decade, 6 chains': 'bfm_final_6chains',
		'Bayesian FM rank 64': 'bfm_r64',
		'Bayesian FM + all metadata': 'bfm_meta_all_meta',
		'Bayesian FM, 6 chains': 'bfm_6chains',
	}


def basic_models(res, train):
	"""Cache the predictions of the basic models with their best settings."""
	beta = res['E1']['best_beta']
	k, lam = res['E2']['best']
	raw_lam = float(
		min(res['E2']['raw_mf_test'], key=res['E2']['raw_mf_test'].get)
	)
	lr = res['E4']['best_lr']
	e6 = res['E6']
	makers = {
		'global_mean': lambda: GlobalMean(),
		'biases': lambda: Biases(beta),
		'als_raw': lambda: ALS(k, raw_lam, use_bias=False),
		'als': lambda: ALS(k, lam, n_iter=15, beta=beta),
		'gd': lambda: GradientDescentMF(
			k, lam, n_iter=400, lr=lr, beta=beta
		),
		'item_knn': lambda: ItemKNN(*e6['item']['best'], beta),
		'user_knn': lambda: UserKNN(*e6['user']['best'], beta),
	}
	return {key: cached(key, make, train) for key, make in makers.items()}


def e8_blend_stack(res, train, test):
	basic_models(res, train)
	tu, ti = np.nonzero(~np.isnan(test))
	y = test[tu, ti]

	def load_pred(key):
		return np.load(CACHE / f'{key}.npy').astype(float)

	def load_sec(key):
		return json.loads((CACHE / f'{key}.json').read_text())['seconds']

	bfm, knn = load_pred('bfm_final_6chains'), load_pred('item_knn')
	out = {
		'blend': {
			str(w): rmse(w * bfm + (1 - w) * knn, test)
			for w in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
		}
	}
	out['best_weight'] = float(min(out['blend'], key=out['blend'].get))
	print(
		'  blend weight of the BFM:',
		{w: round(v, 4) for w, v in out['blend'].items()},
	)

	# Stack: ridge regression of the true rating on the 12 predictions.
	# The weights are learned on 4/5 of the test ratings and applied to the
	# other 1/5 (5 folds), so no rating is predicted by weights fitted on it.
	models = stack_models(res)
	X = np.column_stack(
		[np.ones(len(y))] + [load_pred(k)[tu, ti] for k in models.values()]
	)
	fold = np.random.default_rng(0).integers(0, 5, len(y))
	pred = np.zeros(len(y))
	for f in range(5):
		a, b = fold != f, fold == f
		A = X[a].T @ X[a] + 1.0 * np.eye(X.shape[1])
		pred[b] = X[b] @ np.linalg.solve(A, X[a].T @ y[a])
	pred = np.clip(pred, R_MIN, R_MAX)
	out['stack'] = {
		'models': list(models),
		'test': float(np.sqrt(np.mean((pred - y) ** 2))),
		'seconds': float(sum(load_sec(k) for k in models.values())),
	}
	print(
		f'  stack of {len(models)} models: {out["stack"]["test"]:.4f}, '
		f'{out["stack"]["seconds"]:.0f}s of fitting'
	)
	return out


# ---------------------------------------------------------------------------
# E9 - final table
# ---------------------------------------------------------------------------


def e9_final(res, train, test):
	beta = res['E1']['best_beta']
	k, lam = res['E2']['best']
	rows = {}
	names = [
		('Global mean (starter code)', 'global_mean'),
		('MF without biases, ALS', 'als_raw'),
		(f'User + movie biases (beta={beta:g})', 'biases'),
		('User k-NN', 'user_knn'),
		('Item k-NN', 'item_knn'),
		(f'MF + biases, gradient descent (k={k}, lam={lam:g})', 'gd'),
		(f'MF + biases, ALS (k={k}, lam={lam:g})', 'als'),
		('Bayesian PMF (1 chain)', 'bfm_ids'),
		('Bayesian FM, implicit features (1 chain)', 'bfm_r20'),
		('Bayesian FM, implicit features (6 chains)', 'bfm_6chains'),
		(
			'Final: Bayesian FM + genres + decade (6 chains)',
			'bfm_final_6chains',
		),
	]
	preds = {}
	for name, key in names:
		p = np.load(CACHE / f'{key}.npy').astype(float)
		sec = json.loads((CACHE / f'{key}.json').read_text())['seconds']
		preds[key] = p
		rows[name] = {
			'rmse': rmse(p, test),
			'mae': mae(p, test),
			'accuracy': accuracy(p, test),
			'seconds': sec,
		}
	final = preds['bfm_final_6chains']
	for name, r in rows.items():
		print(
			f'  {name:50s} {r["rmse"]:.4f}  MAE {r["mae"]:.4f}  '
			f'exact {100 * r["accuracy"]:.1f}%  {r["seconds"]:.0f}s'
		)

	# bootstrap over the test ratings: how much the scores would move
	# with another test set of the same size
	tu, ti = np.nonzero(~np.isnan(test))
	truth = test[tu, ti]
	f_err = (np.clip(final[tu, ti], R_MIN, R_MAX) - truth) ** 2
	b_err = (np.clip(preds['biases'][tu, ti], R_MIN, R_MAX) - truth) ** 2
	idx = np.random.default_rng(0).integers(
		0, len(truth), (2000, len(truth))
	)
	f_boot = np.sqrt(f_err[idx].mean(1))
	gain = 100 * (f_boot / np.sqrt(b_err[idx].mean(1)) - 1)
	out = {
		'rows': rows,
		'final_ci95': [
			float(x) for x in np.percentile(f_boot, [2.5, 97.5])
		],
		'gain_vs_biases_ci95': [
			float(x) for x in np.percentile(gain, [2.5, 97.5])
		],
	}
	print(
		f'  final 95% interval {out["final_ci95"]}; '
		f'gain vs biases (%) {out["gain_vs_biases_ci95"]}'
	)

	# run time of the final model on the platform input (train + test)
	from final_model import final_model

	start = time.perf_counter()
	_full, chains = final_model(merge(train, test))
	out['platform_input_seconds'] = time.perf_counter() - start
	out['platform_input_chains'] = chains
	print(
		f'  final model on train + test: '
		f'{out["platform_input_seconds"]:.0f}s, {chains} chains'
	)
	return out


# ---------------------------------------------------------------------------
# E10 - studies of the final model: steps, rank, seeds, data size
# ---------------------------------------------------------------------------


def e10_method_study(res, train, test):
	"""One factor at a time around the final model (one chain, rank 20,
	300 steps, genres + decade), then the size of the training set."""
	base = {
		'rank': 20,
		'n_chains': 1,
		'seed': 0,
		'item_features': FINAL_FEATURES(),
	}

	def run(key, matrix=train, **kw):
		s = {**base, **kw}
		pred, sec = cached('bfm10_' + key, lambda: BayesianFM(**s), matrix)
		return {'test': rmse(pred, test), 'seconds': sec}

	out = {'steps': {}, 'rank': {}}
	for n in [25, 50, 100, 150, 300, 600]:
		# the burn-in is 20 steps, or a fifth of the steps if fewer
		out['steps'][str(n)] = run(
			f'steps{n}', n_iter=n, burn_in=min(20, n // 5)
		)
	for k in [2, 5, 10, 20, 32, 64]:
		out['rank'][str(k)] = run(f'rank{k}', rank=k)
	seeds = [run(f'seed{sd}', seed=sd)['test'] for sd in range(6)]
	out['seeds'] = {
		'test': seeds,
		'mean': float(np.mean(seeds)),
		'std': float(np.std(seeds)),
	}
	# the same two lengths with the six chains of the final model
	out['steps_6chains'] = {
		str(n): run(f'6ch_steps{n}', n_chains=6, n_iter=n)
		for n in [100, 300]
	}
	out['decade_only'] = run('decade', item_features=[mf.decades()])
	out['plain'] = run('plain', as_if_rated=False)

	# size of the training set: random subsets of the training cells
	beta = res['E1']['best_beta']
	k, lam = res['E2']['best']
	out['data'] = {}
	for frac in [0.25, 0.5, 0.75, 1.0]:
		keep = np.random.default_rng(0).random(train.shape) < frac
		sub = np.where(keep & ~np.isnan(train), train, np.nan)
		out['data'][str(frac)] = {
			'ratings': int((~np.isnan(sub)).sum()),
			'biases': rmse(Biases(beta).fit(sub).predict(), test),
			'als': rmse(
				ALS(k, lam, n_iter=15, beta=beta).fit(sub).predict(), test
			),
			'bfm': run(f'frac{frac}', matrix=sub)['test'],
		}
	for key, v in out['data'].items():
		print(
			f'  {float(key):.0%} of train ({v["ratings"]} ratings): '
			f'biases {v["biases"]:.4f}  ALS {v["als"]:.4f}  BFM {v["bfm"]:.4f}',
			flush=True,
		)
	print(
		'  steps:',
		{n: round(v['test'], 4) for n, v in out['steps'].items()},
	)
	print(
		'  rank :', {n: round(v['test'], 4) for n, v in out['rank'].items()}
	)
	print(
		'  6 chains:',
		{
			n: (round(v['test'], 4), round(v['seconds']))
			for n, v in out['steps_6chains'].items()
		},
	)
	print(
		f'  seeds: {out["seeds"]["mean"]:.4f} +/- {out["seeds"]["std"]:.4f}'
	)
	print(
		f'  decade only {out["decade_only"]["test"]:.4f}, '
		f'not as if rated {out["plain"]["test"]:.4f}'
	)
	return out


# ---------------------------------------------------------------------------
# E11 - quality of the samples along one long chain
# ---------------------------------------------------------------------------


def e11_sample_drift(res, train, test):
	"""One chain of 300 steps with the final features: RMSE of single
	samples and of their averages, by window of steps. Shows why a longer
	chain is not better here."""
	from myfm import MyFMRegressor

	m = BayesianFM(
		rank=20,
		n_iter=300,
		burn_in=20,
		thin=2,
		item_features=FINAL_FEATURES(),
	)
	users, items, blocks, groups, M, Mt, n_u, n_i = m._design(train)
	fm = MyFMRegressor(rank=20, random_seed=0)
	fm.fit(
		None,
		train[users, items],
		X_rel=blocks,
		group_shapes=groups,
		n_iter=300,
		n_kept_samples=280,
	)
	samples = fm.predictor_.samples[::2]
	mask = ~np.isnan(test)
	truth = test[mask]
	P = np.array(
		[
			np.clip(m._full_matrix(s, M, Mt, n_u, n_i), R_MIN, R_MAX)[mask]
			for s in samples
		]
	)

	def err(pred):
		return float(np.sqrt(np.mean((pred - truth) ** 2)))

	out = {'windows': []}
	for a in range(0, len(P), 20):
		w = P[a : a + 20]
		row = {
			'steps': [21 + 2 * a, 21 + 2 * (a + len(w) - 1)],
			'single': float(np.mean([err(x) for x in w])),
			'average': err(w.mean(0)),
		}
		out['windows'].append(row)
		print(
			f'  steps {row["steps"][0]:3d}-{row["steps"][1]:3d}: '
			f'single sample {row["single"]:.4f}, window average {row["average"]:.4f}'
		)
	out['average_first_40'] = err(P[:40].mean(0))
	out['average_all'] = err(P.mean(0))
	out['average_after_100'] = err(P[40:].mean(0))
	print(
		f'  average of steps 21-100: {out["average_first_40"]:.4f}, '
		f'21-300: {out["average_all"]:.4f}, 101-300: {out["average_after_100"]:.4f}'
	)
	return out


SECTIONS = [
	('E1', 'E1 baselines', e1_baselines),
	('E2', 'E2 ALS: rank x lambda', e2_als_grid),
	('E3', 'E3 regularization: train vs test', e3_regularization),
	('E4', 'E4 ALS vs gradient descent', e4_als_vs_gd),
	('E5', 'E5 random seeds', e5_seeds),
	('E6', 'E6 neighbourhood models', e6_neighbours),
	('E7', 'E7 Bayesian factorization machine', e7_bayesian_fm),
	('E8', 'E8 blend and stack', e8_blend_stack),
	('E9', 'E9 final table', e9_final),
	('E10', 'E10 studies of the final model', e10_method_study),
	('E11', 'E11 samples along a long chain', e11_sample_drift),
]


def main():
	sys.stdout.reconfigure(encoding='utf-8')
	OUT.mkdir(exist_ok=True)
	res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
	if '--plots' not in sys.argv:
		train, test = load_train_test()
		for key, title, run in SECTIONS:
			if key in res:
				print(title, '- done, skipped')
				continue
			print(title, flush=True)
			start = time.perf_counter()
			res[key] = run(res, train, test)
			print(f'  ({time.perf_counter() - start:.0f}s)', flush=True)
			RESULTS.write_text(json.dumps(res, indent=2))
	import plots

	plots.make_all(res)


if __name__ == '__main__':
	main()
