"""Scores for the learned mine-probability model against the exact probabilities.

On a PositionSet (position_scores), from the model's logits (net.predict_logits):
- KL divergence per covered cell from the exact probability to the prediction. It is
  computed from the logits, like the training loss, so it has no cap: a confidently wrong
  cell costs as much as its logit says;
- calibration: predicted probability against the exact one and the share that were mines;
- a diagnostic: the lowest predicted probability on any covered cell (a network that cannot
  go below some floor, for instance through a dead channel, shows it here).
Confidence intervals come from the per-game means, since positions of one game share a
board and are not independent; they use the t distribution with one fewer degrees of freedom
than games.
"""

import math

import numpy as np
from scipy import stats

from exact_symmetry.minesweeper.deductive import UNKNOWN
from exact_symmetry.net import predict_logits, sigmoid


def kl_from_logits(logits, exact):
    """KL divergence (nats) from exact probabilities p to sigmoid(logits), elementwise, in float64:
    binary cross-entropy with logits minus the entropy of p, as in net.kl_per_cell."""
    z, p = np.asarray(logits, dtype=np.float64), np.asarray(exact, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # cross-entropy p softplus(-z) + (1 - p) softplus(z), with 0 times anything taken as 0,
        # so exact predictions (logits of +-inf) score 0
        cross = (np.where(p > 0, p * np.logaddexp(0.0, -z), 0.0)
                 + np.where(p < 1, (1 - p) * np.logaddexp(0.0, z), 0.0))
        entropy = -(np.where(p > 0, p * np.log(p), 0.0) + np.where(p < 1, (1 - p) * np.log1p(-p), 0.0))
    return cross - entropy


def logits_of(probabilities):
    """Logits of probabilities strictly between 0 and 1 (for predictions given as probabilities,
    such as the mine density)."""
    q = np.asarray(probabilities, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(q) - np.log1p(-q)


def mean_with_interval(values, groups):
    """Mean of values and a 95% interval from the spread of per-group means (groups = games)."""
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups)
    if len(values) == 0:
        return {"mean": None, "low": None, "high": None, "count": 0, "games": 0}
    labels, index = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(index, weights=values), np.bincount(index)
    mean = sums.sum() / counts.sum()
    # ratio estimator: variance from each game's deviation, weighted by its size
    if len(labels) > 1:
        residual = sums - mean * counts
        half = stats.t.ppf(0.975, len(labels) - 1) * math.sqrt(
            (residual ** 2).sum() * len(labels) / (len(labels) - 1)) / counts.sum()
    else:
        half = float("nan")
    return {"mean": float(mean), "low": float(mean - half), "high": float(mean + half),
            "count": int(len(values)), "games": int(len(labels))}


def position_scores(logits: np.ndarray, data) -> dict:
    """Scores of predictions on a PositionSet. logits: (P, rows*cols), NaN on revealed cells, from
    net.predict_logits (for predictions given as probabilities, pass logits_of(probabilities))."""
    covered = data.cells == UNKNOWN
    exact = data.probability
    logits = np.where(covered, np.asarray(logits, dtype=np.float64), np.nan)
    predicted = sigmoid(logits)
    kl = kl_from_logits(np.nan_to_num(logits), np.nan_to_num(exact))
    game_of_cell = np.broadcast_to(data.game[:, None], covered.shape)

    bins = {}
    index = np.minimum(np.floor(np.nan_to_num(predicted) * 10), 9).astype(int)  # bin k holds [k/10, (k+1)/10)
    for k in range(10):
        in_bin = covered & (index == k)
        if in_bin.any():
            bins[f"{k / 10:.1f}-{(k + 1) / 10:.1f}"] = {"cells": int(in_bin.sum()),
                                                        "mean predicted": float(predicted[in_bin].mean()),
                                                        "mean exact": float(exact[in_bin].mean()),
                                                        "share that were mines": float(data.mines[in_bin].mean())}
    return {
        "positions": int(len(logits)), "games": int(len(np.unique(data.game))),
        "KL per covered cell": mean_with_interval(kl[covered], game_of_cell[covered]),
        "lowest predicted probability on a covered cell": float(np.nanmin(predicted)),
        "calibration by predicted probability": bins,
    }


def score(model, test_set, device="cpu"):
    """A model's scores on the test positions."""
    logits = predict_logits(model.to(device), test_set.cells, test_set.n_rows, test_set.n_cols,
                            test_set.num_mines, str(device))
    return {"test positions": position_scores(logits, test_set)}
