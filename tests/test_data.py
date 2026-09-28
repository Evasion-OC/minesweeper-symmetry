"""Training data of the learned mine-probability model: positions and their exact labels."""

import random

import numpy as np
import pytest

from exact_symmetry.data import PositionSet, as_seen, exact_probabilities, generate, play_game
from exact_symmetry.minesweeper.deductive import FLAGGED, UNKNOWN, Observation
from exact_symmetry.minesweeper.probability import mine_probabilities
from exact_symmetry.minesweeper.symmetry import cell_orbits, grid_transforms, state_stabilizer
from helpers import counted, neighbours


def moved(values, transform):
    out = np.empty_like(values)
    out[list(transform.mapping)] = values
    return out


def test_labels_equal_the_share_of_layouts_that_fit():
    for args in [(4, 4, 3), (5, 5, 4), (4, 6, 5)]:
        data = generate(*args, 60, "train", seed=1, epsilon=0.3)
        for cells, label in zip(data.cells, data.probability):
            expected = counted(Observation(*args, tuple(int(c) for c in cells)))
            assert {i: float(label[i]) for i in np.flatnonzero(~np.isnan(label))} == \
                   {i: float(np.float32(float(p))) for i, p in expected.items()}


def test_flags_add_nothing_the_clues_do_not_say():
    rng, flagged = random.Random(3), 0
    while flagged < 50:
        for visible in play_game(rng, 8, 8, 10, epsilon=0.0).positions:
            if FLAGGED not in visible:
                continue
            flagged += 1
            with_flags = mine_probabilities(Observation(8, 8, 10, visible))
            label = exact_probabilities(visible, 8, 8, 10)
            for i, state in enumerate(visible):
                if state == FLAGGED:
                    assert label[i] == 1.0
                elif state == UNKNOWN:
                    assert label[i] == float(with_flags[i])


def test_labels_sum_to_the_mines_and_turn_with_the_board():
    data = generate(8, 8, 10, 150, "train", seed=2)
    for cells in data.cells:
        cells = tuple(int(c) for c in cells)
        label = exact_probabilities(cells, 8, 8, 10)
        assert np.nansum(label) == pytest.approx(10, abs=1e-9)
        for transform in grid_transforms(8, 8):
            assert np.array_equal(exact_probabilities(tuple(moved(np.array(cells), transform)), 8, 8, 10),
                                  moved(label, transform), equal_nan=True)


def test_labels_are_equal_across_each_orbit_of_a_position_with_a_symmetry_of_its_own():
    found = 0
    for args in [(4, 4, 2), (5, 5, 3), (5, 5, 2)]:
        data = generate(*args, 400, "train", seed=4, epsilon=0.5)
        for cells, label in zip(data.cells, data.probability):
            stabilizer = state_stabilizer(tuple(int(c) for c in cells), args[0], args[1])
            if len(stabilizer) > 1:
                found += 1
                for orbit in cell_orbits(range(args[0] * args[1]), stabilizer):
                    assert np.array_equal(label[list(orbit)], np.full(len(orbit), label[orbit[0]]), equal_nan=True)
    assert found >= 20


def test_every_position_fits_its_layout_and_certain_labels_are_right():
    data = generate(8, 8, 10, 500, "train", seed=5)
    for cells, label, mines in zip(data.cells, data.probability, data.mines):
        for i, state in enumerate(cells):
            if state >= 0:
                assert not mines[i] and state == sum(mines[n] for n in neighbours(i, 8, 8))
                assert np.isnan(label[i])
            else:
                assert not np.isnan(label[i])
        assert np.all(mines[label == 1]) and not np.any(mines[label == 0])


def per_game_t_statistics(data, labels):
    """For each bin of label values strictly between 0 and 1: the t statistic of the per-game
    sums of (mine - label). Games are independent, and each term has mean 0 when the labels
    are the true probabilities given the visible board."""
    covered = (data.cells == UNKNOWN) & (labels > 0) & (labels < 1)
    stats = {}
    for low in np.arange(0, 1, 0.1):
        in_bin = covered & (labels >= low) & (labels < low + 0.1)
        if in_bin.sum() < 500:
            continue
        gap = np.where(in_bin, data.mines - labels, 0).sum(axis=1)
        per_game = np.bincount(data.game, weights=gap)
        stats[round(low, 1)] = per_game.mean() / (per_game.std(ddof=1) / np.sqrt(len(per_game)))
    return stats


def test_labels_match_how_often_cells_turn_out_to_be_mines():
    # Checks the claim that every layout fitting the visible board stays equally likely under
    # the way positions are played (safe first move, exact moves, random safe reveals).
    data = generate(8, 8, 10, 6000, "train", seed=6, epsilon=0.5)
    labels = data.probability.astype(np.float64)
    true_stats = per_game_t_statistics(data, labels)
    assert len(true_stats) >= 5 and all(abs(t) < 4 for t in true_stats.values()), true_stats
    # The same test rejects a plausible wrong label: the share of mines left among covered cells.
    covered = data.cells == UNKNOWN
    density = 10 / covered.sum(axis=1, keepdims=True)
    naive = np.where(covered & (labels > 0) & (labels < 1), density, labels)
    assert max(abs(t) for t in per_game_t_statistics(data, naive).values()) > 4


def test_splits_and_seeds_draw_separate_boards_and_repeat_exactly():
    def layouts(data):
        return {data.mines[i].tobytes() for i in range(len(data))}

    sets = {(split, seed): generate(8, 8, 10, 300, split, seed)
            for split, seed in [("train", 0), ("validation", 0), ("test", 0), ("train", 1)]}
    keys = list(sets)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            assert not layouts(sets[keys[a]]) & layouts(sets[keys[b]])
    again = generate(8, 8, 10, 300, "train", 0)
    for name in ("cells", "probability", "mines", "game"):
        assert np.array_equal(getattr(again, name), getattr(sets["train", 0], name), equal_nan=True)
    with pytest.raises(ValueError):
        generate(8, 8, 10, 10, "training", 0)


def test_a_position_too_large_to_count_is_left_out_and_counted():
    data = generate(8, 8, 10, 200, "train", seed=7, cap=40)
    assert data.meta["positions left out, region too large to count"] > 0 and len(data) == 200
    for cells, label in zip(data.cells, data.probability):
        assert np.array_equal(label, exact_probabilities(tuple(int(c) for c in cells), 8, 8, 10).astype(np.float32),
                              equal_nan=True)


def test_a_saved_set_loads_unchanged(tmp_path):
    data = generate(5, 5, 4, 50, "validation", seed=8)
    data.save(tmp_path / "set.npz")
    back = PositionSet.load(tmp_path / "set.npz")
    assert (back.n_rows, back.n_cols, back.num_mines, back.meta) == (5, 5, 4, data.meta)
    for name in ("cells", "probability", "mines", "game"):
        assert np.array_equal(getattr(back, name), getattr(data, name), equal_nan=True)


def test_positions_shown_to_the_network_have_no_flags():
    data = generate(8, 8, 10, 300, "train", seed=9, epsilon=0.0)
    assert not np.any(data.cells == FLAGGED)
    assert as_seen((FLAGGED, 3, UNKNOWN)) == (UNKNOWN, 3, UNKNOWN)
