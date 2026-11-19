# Collaborative filtering: how far does a Bayesian factorization machine go?

Predicting the missing entries of a 610 users × 4,980 movies rating matrix, with every
method fitted and scored under one protocol fixed before the first experiment.

Coursework for the Data Science Lab, M2 IASD (Université Paris-Dauphine – PSL), October 2026.

## Result

| Model | Test RMSE |
|---|---|
| Global mean (provided baseline) | 1.0369 |
| User + movie biases | 0.8742 |
| **Bayesian factorization machine** (implicit features + genres + decade, 6 chains × 100 Gibbs steps) | **0.8260** |

The gap that matters is the second one: biases already recover most of the signal, and the
question the study answers is what a factorization machine adds on top of them, and at what cost.

## What is in here

Six families of methods, implemented from scratch and compared on the same split:

| File | Method |
|---|---|
| `baselines.py` | Global mean, then user and movie biases |
| `matrix_factorization.py` | Matrix factorization, by alternating least squares and by gradient descent |
| `item_knn.py` | Item-based and user-based *k*-nearest neighbours |
| `bayesian_fm.py` | Bayesian factorization machine, Gibbs sampling over parallel chains |
| `movie_features.py` | Side information: genres, year, decade, title words, popularity |
| `final_model.py` | The selected model |
| `data_study.py` | Study of the rating matrix itself (sparsity, rating distribution, long tail) |
| `experiments.py`, `plots.py` | The nine experiments E1–E9 and their figures |
| `generate.py` | Entry point used by the course evaluation platform |

`results/results.json` holds every number reported, `results/figures/` the figures, and
`report/` the written report and the slides.

## Reproducing

The dataset is not redistributed here. Place `ratings_train.npy`, `ratings_test.npy` and
`namesngenre2.npy` in `dataset/`, then:

```bash
pip install -r requirements.txt
python data_study.py
python experiments.py          # about 25 minutes; only reruns what is missing
python experiments.py --plots
python generate.py --name dataset/ratings_train.npy
```

## Notes on method

Every model is fitted on `ratings_train.npy` and scored on `ratings_test.npy`. The test set was
fixed before any experiment was run, and no model was selected on it — the exploratory rounds
that preceded this study used a separate validation split, and their numbers are not reported here.

## Authors

Carried out for the Data Science Lab course of the M2 IASD, Université Paris-Dauphine – PSL,
by Lynda Mellissa Yazi, Arthur Oudeyer and Nikos Kontogeorgis.

## License

MIT, see [LICENSE](LICENSE). The course materials and the dataset are not covered by it.
