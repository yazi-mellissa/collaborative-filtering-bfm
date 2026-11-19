"""Study of the dataset: sizes, sparsity, rating values, genres, years.

    uv run python data_study.py

Prints the numbers and writes results/data_study.json and two figures.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from data import count, load, merge

OUT = Path('results')


def describe(R):
	rated = ~np.isnan(R)
	per_user, per_movie = rated.sum(1), rated.sum(0)
	return {
		'ratings': count(R),
		'density_percent': 100 * float(rated.mean()),
		'per_user_median': float(np.median(per_user)),
		'per_user_min_max': [int(per_user.min()), int(per_user.max())],
		'per_movie_median': float(np.median(per_movie)),
		'per_movie_max': int(per_movie.max()),
		'movies_with_at_most_2': int((per_movie <= 2).sum()),
		'mean_rating': float(np.nanmean(R)),
		'std_rating': float(np.nanstd(R)),
	}


def main():
	sys.stdout.reconfigure(encoding='utf-8')
	train, test = load('ratings_train.npy'), load('ratings_test.npy')
	names = load('namesngenre2.npy')
	pool = merge(train, test)
	out = {'shape': list(train.shape)}
	for key, R in [('train', train), ('test', test), ('train+test', pool)]:
		out[key] = describe(R)
		print(key, out[key])

	# No cell is in both files; no test rating is on an unseen user/movie
	both = ~np.isnan(train) & ~np.isnan(test)
	unseen_movie = (~np.isnan(test)) & ((~np.isnan(train)).sum(0) == 0)[
		None, :
	]
	out['cells_in_both_files'] = int(both.sum())
	out['test_ratings_on_movies_absent_from_train'] = int(
		unseen_movie.sum()
	)

	values, counts = np.unique(pool[~np.isnan(pool)], return_counts=True)
	out['rating_values_percent'] = {
		f'{v:g}': 100 * c / counts.sum() for v, c in zip(values, counts)
	}
	genres = Counter(g for s in names[:, 1] for g in s.split('|'))
	out['genres'] = dict(genres.most_common())
	years = [
		int(m.group(1))
		for t in names[:, 0]
		if (m := re.search(r'\((\d{4})\)\s*$', t.strip()))
	]
	out['titles_with_year'] = len(years)
	out['year_range'] = [min(years), max(years)]
	print(
		'values (%):',
		{k: round(v, 1) for k, v in out['rating_values_percent'].items()},
	)
	print('genres:', out['genres'])
	print('years:', out['year_range'], 'for', len(years), 'titles')
	OUT.mkdir(exist_ok=True)
	(OUT / 'data_study.json').write_text(json.dumps(out, indent=2))

	import plots

	plots.data_figures(train, pool, out)


if __name__ == '__main__':
	main()
