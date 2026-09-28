"""Scores of the learned mine-probability model, checked on predictions whose answer is known."""

import numpy as np
import pytest
import torch
from scipy import stats

from exact_symmetry.data import generate
from exact_symmetry.evaluate import kl_from_logits, logits_of, mean_with_interval, position_scores
from exact_symmetry.minesweeper.deductive import UNKNOWN
from exact_symmetry.net import kl_per_cell


@pytest.fixture(scope="module")
def test_set():
    return generate(8, 8, 10, 1500, "test", seed=0)


def test_the_exact_probabilities_score_perfectly(test_set):
    scores = position_scores(logits_of(test_set.probability), test_set)
    assert scores["KL per covered cell"]["mean"] < 1e-6
    assert scores["lowest predicted probability on a covered cell"] == 0  # proved-safe cells
    for row in scores["calibration by predicted probability"].values():
        assert row["mean predicted"] == pytest.approx(row["mean exact"], abs=1e-6)


def test_the_mine_density_scores_worse(test_set):
    covered = test_set.cells == UNKNOWN
    density = np.where(covered, 10 / covered.sum(1, keepdims=True), np.nan)
    scores = position_scores(logits_of(density), test_set)
    assert scores["KL per covered cell"]["low"] > 0.05
    assert scores["lowest predicted probability on a covered cell"] > 0.01  # the density never reaches 0


def test_the_interval_is_a_mean_over_cells_with_spread_from_games():
    values, games = [1.0, 3.0, 2.0, 2.0, 10.0], [0, 0, 1, 1, 2]
    out = mean_with_interval(values, games)
    assert out["mean"] == pytest.approx(18 / 5) and out["games"] == 3 and out["count"] == 5
    assert out["low"] < out["mean"] < out["high"]
    # one value per game: the usual interval for a mean
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    out = mean_with_interval(x, np.arange(400))
    assert out["high"] - out["mean"] == pytest.approx(stats.t.ppf(0.975, 399) * x.std(ddof=1) / 20, rel=1e-9)
    # with few games the t quantile matters: three games, one value each
    out = mean_with_interval([1.0, 2.0, 4.0], [0, 1, 2])
    assert out["high"] - out["mean"] == pytest.approx(stats.t.ppf(0.975, 2) * np.std([1, 2, 4], ddof=1) / 3 ** 0.5)


def test_the_kl_is_the_training_kl_and_has_no_cap():
    logits = np.array([-200.0, -50.0, -20.0, 0.0, 3.0, 20.0, 50.0, 300.0])
    for p in (0.0, 1.0, 0.3):
        exact = np.full_like(logits, p)
        ours = kl_from_logits(logits, exact)
        training = kl_per_cell(torch.tensor(logits), torch.tensor(exact), torch.ones(8, dtype=torch.bool)).numpy()
        assert np.allclose(ours, training, atol=1e-12)
    assert kl_from_logits(np.array([-50.0]), np.array([1.0]))[0] == pytest.approx(50.0)


def test_every_covered_cell_falls_in_one_calibration_bin(test_set):
    covered = test_set.cells == UNKNOWN
    edges = np.array([0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    predicted = np.where(covered, edges[np.arange(covered.size).reshape(covered.shape) % len(edges)], np.nan)
    bins = position_scores(logits_of(predicted), test_set)["calibration by predicted probability"]
    assert sum(b["cells"] for b in bins.values()) == covered.sum()
