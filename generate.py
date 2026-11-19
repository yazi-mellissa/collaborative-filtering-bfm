import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm, trange  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_model import final_model

# The platform stops a script after 1,200 s. The sampling chains get a
# 700 s budget: on a slow machine fewer chains are averaged instead of
# running out of time.
TIME_BUDGET = 700

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Generate a completed ratings table.'
	)
	parser.add_argument(
		'--name',
		type=str,
		default='dataset/ratings_train.npy',
		help='Name of the npy of the ratings table to complete',
	)

	args = parser.parse_args()

	# Open Ratings table
	print('Ratings loading...')
	table = np.load(args.name)  ## DO NOT CHANGE THIS LINE
	print('Ratings Loaded.')

	# Bayesian factorization machine with implicit features, genres and
	# decade, fitted on the given matrix (final_model.py)
	t_start = time.time()
	table, chains = final_model(
		table, time_budget=TIME_BUDGET, verbose=True
	)
	print(f'Completed with {chains} chains in {time.time() - t_start:.0f}s')

	# Save the completed table
	np.save('output.npy', table)  ## DO NOT CHANGE THIS LINE
