"""Movie metadata features (genres, year, decade, title words, popularity).

Titles and genres come from namesngenre2.npy (row j = column j of the
rating matrices). dataset/ is not in the repository, so a copy is kept in
movie_names.npz next to the code; it is written by
    uv run python movie_features.py
Each function returns a sparse matrix with one row per movie. Multi-valued
features (genres, title words) are scaled so that each row has norm 1.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data import DATA, load

NAMES = Path(__file__).resolve().parent / 'movie_names.npz'


def _rows(lists, vocab):
	"""Sparse matrix from one list of tokens per movie, row norm 1."""
	index = {t: k for k, t in enumerate(vocab)}
	rows, cols = [], []
	for r, tokens in enumerate(lists):
		for t in set(tokens):
			if t in index:
				rows.append(r)
				cols.append(index[t])
	X = sp.csr_matrix(
		(np.ones(len(rows)), (rows, cols)), shape=(len(lists), len(vocab))
	)
	norms = np.sqrt(np.maximum(X.sum(1).A1, 1))
	return sp.diags(1 / norms) @ X


def _names():
	if NAMES.exists():
		return np.load(NAMES)['names']
	return load('namesngenre2.npy')


def _year(title):
	m = re.search(r'\((\d{4})\)\s*$', title.strip())
	return int(m.group(1)) if m else None


def genres():
	lists = [
		[g for g in s.split('|') if g != '(no genres listed)']
		for s in _names()[:, 1]
	]
	vocab = sorted({g for gs in lists for g in gs})
	return _rows(lists, vocab)


def years():
	lists = [[_year(t)] if _year(t) else [] for t in _names()[:, 0]]
	vocab = sorted({y for ys in lists for y in ys})
	return _rows(lists, vocab)


def decades():
	lists = [
		[_year(t) // 10 * 10] if _year(t) else [] for t in _names()[:, 0]
	]
	vocab = sorted({d for ds in lists for d in ds})
	return _rows(lists, vocab)


def title_words(min_count=5):
	"""Lower-case words of the title (without the year) that appear in at
	least `min_count` titles."""
	lists = [
		re.findall(r'[a-z]+', re.sub(r'\(\d{4}\)\s*$', '', t).lower())
		for t in _names()[:, 0]
	]
	counts = Counter(w for ws in lists for w in set(ws))
	vocab = sorted(w for w, c in counts.items() if c >= min_count)
	return _rows(lists, vocab)


def popularity(R, n_buckets=8):
	"""Number of ratings of each movie in the input matrix, in buckets on
	a log scale (computed from the input, like the implicit features)."""
	n = (~np.isnan(R)).sum(0)
	edges = np.logspace(0, np.log10(n.max() + 1), n_buckets)
	bucket = np.digitize(n, edges)
	return _rows([[b] for b in bucket], sorted(set(bucket)))


if __name__ == '__main__':
	names = load('namesngenre2.npy').astype(str)
	ids = np.load(DATA / 'movie_ids.npy')
	np.savez_compressed(NAMES, names=names, movie_ids=ids)
	print('saved', NAMES, names.shape)
