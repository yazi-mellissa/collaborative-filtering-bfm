"""Figures for the report, drawn only from saved results."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

FIG = Path('results') / 'figures'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = LinearSegmentedColormap.from_list(
	'seq',
	['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'],
)
plt.rcParams.update(
	{
		'figure.dpi': 150,
		'savefig.bbox': 'tight',
		'font.size': 9,
		'axes.edgecolor': GRID,
		'axes.labelcolor': INK2,
		'axes.titlesize': 10,
		'axes.titlecolor': INK,
		'axes.spines.top': False,
		'axes.spines.right': False,
		'axes.grid': True,
		'grid.color': GRID,
		'grid.linewidth': 0.6,
		'xtick.color': INK2,
		'ytick.color': INK2,
		'lines.linewidth': 2,
		'lines.markersize': 5,
		'legend.frameon': False,
	}
)


def _save(fig, name):
	FIG.mkdir(parents=True, exist_ok=True)
	fig.savefig(FIG / name)
	plt.close(fig)
	print('  saved', FIG / name)


def data_figures(train, pool, study):
	fig, ax = plt.subplots(figsize=(5.4, 3.0))
	vals = study['rating_values_percent']
	ax.bar(list(vals), list(vals.values()), color=BLUE, width=0.7)
	ax.set_xlabel('Rating (stars)')
	ax.set_ylabel('Share of ratings (%)')
	ax.set_title(
		'Whole and round ratings dominate; 4 stars is the most common'
	)
	ax.grid(axis='x', visible=False)
	_save(fig, 'data_rating_values.png')

	fig, ax = plt.subplots(figsize=(5.4, 3.2))
	bins = np.logspace(0, np.log10(2000), 30)
	for R, label, color in [
		(train, 'train file', BLUE),
		(pool, 'train + test', ORANGE),
	]:
		ax.hist(
			(~np.isnan(R)).sum(0),
			bins=bins,
			histtype='step',
			color=color,
			linewidth=2,
			label=f'per movie, {label}',
		)
	ax.hist(
		(~np.isnan(pool)).sum(1),
		bins=bins,
		histtype='step',
		color=AQUA,
		linewidth=2,
		label='per user, train + test',
	)
	ax.set_xscale('log')
	ax.set_xlabel('Number of ratings (log scale)')
	ax.set_ylabel('Count')
	ax.set_title(
		f'Movies are the sparse side: median '
		f'{study["train"]["per_movie_median"]:g} ratings per movie in train'
	)
	ax.legend(fontsize=8)
	_save(fig, 'data_ratings_per_movie_user.png')


def make_all(res):
	if 'E2' in res:
		e = res['E2']
		grid = np.array(
			[[e['test'][f'{k},{l}'] for l in e['lams']] for k in e['ranks']]
		)
		norm = Normalize(grid.min(), grid.max())
		fig, ax = plt.subplots(figsize=(5.4, 3.6))
		im = ax.imshow(grid, cmap=SEQ.reversed(), norm=norm, aspect='auto')
		ax.set_xticks(range(len(e['lams'])), [f'{l:g}' for l in e['lams']])
		ax.set_yticks(range(len(e['ranks'])), [str(r) for r in e['ranks']])
		ax.set_xlabel('Regularization lambda')
		ax.set_ylabel('Rank k')
		ax.grid(False)
		bk, bl = e['best']
		for a, k in enumerate(e['ranks']):
			for b, l in enumerate(e['lams']):
				ax.text(
					b,
					a,
					f'{grid[a, b]:.4f}',
					ha='center',
					va='center',
					fontsize=7,
					color='white' if norm(grid[a, b]) < 0.55 else INK,
					fontweight='bold' if (k, l) == (bk, bl) else 'normal',
				)
		ax.set_title('ALS test RMSE by rank and regularization')
		fig.colorbar(
			im, ax=ax, shrink=0.85, label='Test RMSE'
		).outline.set_visible(False)
		_save(fig, 'als_grid.png')

	if 'E3' in res:
		e = res['E3']
		fig, ax = plt.subplots(figsize=(5.4, 3.4))
		ax.plot(
			e['lams'],
			e['train'],
			color=BLUE,
			ls='--',
			marker='o',
			label='training RMSE',
		)
		ax.plot(
			e['lams'],
			e['test'],
			color=BLUE,
			marker='o',
			label='test RMSE',
		)
		bias_val = res['E1']['biases_by_beta'][
			f'{res["E1"]["best_beta"]:g}'
		]
		ax.axhline(bias_val, color=INK2, lw=1, ls=':')
		ax.annotate(
			'bias model (test)',
			(e['lams'][-1], bias_val),
			xytext=(0, 4),
			textcoords='offset points',
			ha='right',
			color=INK2,
			fontsize=8,
		)
		ax.set_xscale('log')
		ax.set_xlabel('Regularization lambda (log scale)')
		ax.set_ylabel('RMSE')
		ax.set_title(
			f'ALS, k={e["rank"]}: small lambda overfits, large lambda = bias model'
		)
		ax.legend(fontsize=8)
		_save(fig, 'regularization.png')

	if 'E4' in res:
		e = res['E4']
		fig, ax = plt.subplots(figsize=(5.4, 3.4))
		ax.plot(
			e['als']['seconds'],
			e['als']['test'],
			color=BLUE,
			marker='o',
			label='ALS (1 point = 1 sweep)',
		)
		for (lr, h), c in zip(
			e['gd'].items(), [ORANGE, AQUA, YELLOW, INK2, '#9a5bd1']
		):
			if 'test' in h:
				ax.plot(
					h['seconds'],
					h['test'],
					color=c,
					label=f'gradient descent, step {lr}',
				)
		ax.set_xscale('log')
		ax.set_xlabel('Seconds (log scale)')
		ax.set_ylabel('Test RMSE')
		ax.set_title(
			f'Same cost (k={e["rank"]}, lambda={e["lam"]:g}), two optimizers'
		)
		ax.legend(fontsize=8)
		_save(fig, 'als_vs_gd.png')

	if 'E7' in res:
		e = res['E7']
		fig, ax = plt.subplots(figsize=(5.4, 3.2))
		ax.plot(
			range(1, 7),
			e['bfm_6chains']['chains'],
			color='#86b6ef',
			marker='o',
			label='implicit features',
		)
		ax.plot(
			range(1, 7),
			e['bfm_final_6chains']['chains'],
			color=BLUE,
			marker='o',
			label='implicit features + genres + decade',
		)
		ax.legend(fontsize=8)
		ax.set_xlabel('Number of sampling chains averaged')
		ax.set_ylabel('Test RMSE')
		ax.set_title('Averaging chains helps; the gain flattens after four')
		_save(fig, 'bfm_chains.png')

		v = e['variants']
		keys = [
			'ids',
			'ids_iu',
			'ids_ii',
			'r20',
			'plain',
			'meta_genres_decade',
		]
		bias = res['E1']['biases_by_beta'][f'{res["E1"]["best_beta"]:g}']
		names = {
			'r20': '+ both implicit groups',
			'plain': '   (same, NOT predicted "as if rated")',
			'meta_genres_decade': '+ both implicit + genres + decade',
		}
		labels = ['Biases (reference)'] + [
			names.get(k, v[k]['label']) for k in keys
		]
		vals = [bias] + [v[k]['test'] for k in keys]
		fig, ax = plt.subplots(figsize=(6.4, 3.0))
		best = min(vals)
		bars = ax.barh(
			labels[::-1],
			vals[::-1],
			height=0.6,
			color=[BLUE if x == best else '#86b6ef' for x in vals[::-1]],
		)
		for b, x in zip(bars, vals[::-1]):
			ax.text(
				x + 0.001,
				b.get_y() + b.get_height() / 2,
				f'{x:.4f}',
				va='center',
				fontsize=8,
				color=INK,
			)
		ax.set_xlim(min(vals) - 0.02, max(vals) + 0.012)
		ax.set_xlabel('Test RMSE (one chain, rank 20), lower is better')
		ax.grid(axis='y', visible=False)
		ax.tick_params(axis='y', length=0, labelcolor=INK, labelsize=8)
		ax.set_title('What each group of features brings')
		_save(fig, 'bfm_features.png')

	if 'E9' in res:
		rows = res['E9']['rows']
		items = sorted(rows.items(), key=lambda kv: -kv[1]['rmse'])
		fig, ax = plt.subplots(figsize=(6.4, 3.4))
		vals = [r['rmse'] for _, r in items]
		best = min(vals)
		bars = ax.barh(
			[n for n, _ in items],
			vals,
			height=0.62,
			color=[BLUE if v == best else '#86b6ef' for v in vals],
		)
		for b, v in zip(bars, vals):
			ax.text(
				v + 0.004,
				b.get_y() + b.get_height() / 2,
				f'{v:.4f}',
				va='center',
				fontsize=8,
				fontweight='bold' if v == best else 'normal',
				color=INK,
			)
		ax.set_xlim(0.8, 1.19)
		ax.set_xlabel('Test RMSE (axis starts at 0.80), lower is better')
		ax.grid(axis='y', visible=False)
		ax.tick_params(axis='y', length=0, labelcolor=INK, labelsize=8)
		ax.set_title('Final comparison')
		_save(fig, 'final_comparison.png')

	if 'E10' in res:
		e = res['E10']
		fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
		ax = axes[0]
		ns = list(e['steps'])
		ax.plot(
			[int(n) for n in ns],
			[e['steps'][n]['test'] for n in ns],
			color=BLUE,
			marker='o',
		)
		ax.set_xscale('log')
		ax.set_xticks(
			[25, 50, 100, 300, 600], ['25', '50', '100', '300', '600']
		)
		ax.minorticks_off()
		ax.set_xlabel('Sampling steps')
		ax.set_ylabel('Test RMSE')
		ax.set_title('(a) Steps', fontsize=9)
		ax = axes[1]
		ks = list(e['rank'])
		ax.plot(
			[int(k) for k in ks],
			[e['rank'][k]['test'] for k in ks],
			color=BLUE,
			marker='o',
		)
		ax.set_xscale('log')
		ax.set_xticks([2, 5, 10, 20, 64], ['2', '5', '10', '20', '64'])
		ax.minorticks_off()
		ax.set_xlabel('Rank k')
		ax.set_title('(b) Rank', fontsize=9)
		ax = axes[2]
		fr = list(e['data'])
		x = [float(f) * 100 for f in fr]
		for key, label, c in [
			('biases', 'biases', INK2),
			('als', 'ALS', ORANGE),
			('bfm', 'Bayesian FM', BLUE),
		]:
			ax.plot(
				x,
				[e['data'][f][key] for f in fr],
				color=c,
				marker='o',
				label=label,
			)
		ax.set_xlabel('% of training ratings')
		ax.set_title('(c) Data size', fontsize=9)
		ax.legend(fontsize=7)
		fig.tight_layout()
		_save(fig, 'method_study.png')

	if all(k in res for k in ('E3', 'E4', 'E6')):
		fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
		bias = res['E1']['biases_by_beta'][f'{res["E1"]["best_beta"]:g}']
		ax = axes[0]
		e = res['E3']
		ax.plot(
			e['lams'],
			e['train'],
			color=BLUE,
			ls='--',
			marker='o',
			ms=3,
			label='training',
		)
		ax.plot(
			e['lams'], e['test'], color=BLUE, marker='o', ms=3, label='test'
		)
		ax.axhline(bias, color=INK2, lw=1, ls=':')
		ax.set_xscale('log')
		ax.set_xlabel(r'Regularization $\lambda$')
		ax.set_ylabel('RMSE')
		ax.set_title('(a) ALS, k = 50', fontsize=9)
		ax.legend(fontsize=7)
		ax = axes[1]
		e = res['E4']
		ax.plot(
			e['als']['seconds'],
			e['als']['test'],
			color=BLUE,
			marker='o',
			ms=3,
			label='ALS',
		)
		for lr, c in [('0.0005', ORANGE), ('0.002', AQUA)]:
			h = e['gd'][lr]
			ax.plot(
				h['seconds'],
				h['test'],
				color=c,
				label=f'gradient descent, step {lr}',
			)
		ax.set_xscale('log')
		ax.set_ylim(0.8695, 0.88)
		ax.set_xlabel('Seconds')
		ax.set_title('(b) ALS vs gradient descent', fontsize=9)
		ax.legend(fontsize=6)
		ax = axes[2]
		ks = [10, 30, 100]
		for name, c in [('item', BLUE), ('user', ORANGE)]:
			grid = res['E6'][name]['grid']
			ax.plot(
				ks,
				[
					min(v for s, v in grid.items() if s.startswith(f'{k},'))
					for k in ks
				],
				color=c,
				marker='o',
				ms=3,
				label=f'{name} k-NN',
			)
		ax.axhline(bias, color=INK2, lw=1, ls=':')
		ax.set_xscale('log')
		ax.set_xticks(ks, [str(k) for k in ks])
		ax.minorticks_off()
		ax.set_xlabel('Neighbours k')
		ax.set_title('(c) Neighbourhood models', fontsize=9)
		ax.legend(fontsize=7)
		fig.tight_layout()
		_save(fig, 'classical_study.png')
