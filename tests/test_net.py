"""The learned mine-probability model: input planes, network, loss and training."""

import numpy as np
import pytest
import torch

from exact_symmetry.data import generate
from exact_symmetry.minesweeper.deductive import UNKNOWN
from exact_symmetry.net import PosteriorNet, encode, kl_per_cell, mean_kl, predict, train

U = UNKNOWN


def test_planes_show_covered_cells_clues_the_board_and_the_mine_count():
    cells = torch.tensor([[[U, 0, 2], [U, U, 8]]])
    planes = encode(cells, 3)
    assert planes.shape == (1, 13, 2, 3)
    assert planes[0, 0].tolist() == [[1, 0, 0], [1, 1, 0]]
    assert planes[0, 1].tolist() == [[0, 1, 0], [0, 0, 0]]            # clue 0
    assert planes[0, 3].tolist() == [[0, 0, 1], [0, 0, 0]]            # clue 2
    assert planes[0, 9].tolist() == [[0, 0, 0], [0, 0, 1]]            # clue 8
    assert planes[0, 1:10].sum(0).tolist() == [[0, 1, 1], [0, 0, 1]]  # one clue plane per revealed cell
    assert torch.all(planes[0, 10] == 1)
    assert torch.allclose(planes[0, 11], torch.tensor(1.0))           # 3 mines over 3 covered cells
    assert torch.allclose(planes[0, 12], torch.tensor(0.5))           # 3 of 6 cells covered


def test_one_network_gives_a_probability_per_cell_on_every_board_size():
    model = PosteriorNet(width=16, blocks=2)
    for rows, cols, mines in [(8, 8, 10), (16, 16, 40), (16, 30, 99)]:
        cells = np.full((2, rows * cols), U, dtype=np.int8)
        cells[1, 0] = 1
        out = predict(model, cells, rows, cols, mines)
        assert out.shape == (2, rows * cols) and np.isnan(out[1, 0])
        assert np.all((out[~np.isnan(out)] > 0) & (out[~np.isnan(out)] < 1))


def test_the_loss_is_smallest_at_the_exact_probability():
    exact = torch.tensor([0.0, 1.0, 0.25, 0.5, 0.9])
    covered = torch.ones(5, dtype=torch.bool)
    exact_logits = torch.logit(exact.clamp(0.25, 0.9)).where((exact > 0) & (exact < 1), 60 * exact - 30)  # +-30 when certain
    assert kl_per_cell(exact_logits, exact, covered).abs().max() < 1e-6
    for shift in (-0.5, 0.3):
        assert torch.all(kl_per_cell(exact_logits + shift, exact, covered)[2:] > 0)


@pytest.fixture(scope="module")
def small_sets():
    return generate(8, 8, 10, 3000, "train", seed=0), generate(8, 8, 10, 500, "validation", seed=0)


def test_training_beats_the_mine_density_and_keeps_the_best_weights(small_sets):
    train_set, validation_set = small_sets
    model, log = train(train_set, validation_set, steps=400, batch_size=64, width=16, blocks=2, eval_every=100)
    covered = validation_set.cells == U
    density = torch.tensor(np.broadcast_to(10 / covered.sum(1, keepdims=True), covered.shape).copy())
    exact = torch.tensor(np.nan_to_num(validation_set.probability), dtype=torch.float64)
    density_kl = kl_per_cell(torch.logit(density), exact, torch.tensor(covered)).mean().item()
    assert log.best_kl < 0.6 * density_kl
    assert log.best_kl == min(k for _, k in log.validation) == pytest.approx(mean_kl(model, validation_set))
    assert [s for s, _ in log.validation] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400]  # dense early


def test_training_never_reads_the_hidden_layout(small_sets):
    class Hidden:
        def __getattr__(self, name):
            raise AssertionError("the hidden layout was read")

        def __getitem__(self, key):
            raise AssertionError("the hidden layout was read")

        def __array__(self, *args, **kwargs):
            raise AssertionError("the hidden layout was read")

    train_set, validation_set = small_sets
    for data in (train_set, validation_set):
        data.mines, saved = Hidden(), data.mines
        data.saved_mines = saved
    try:
        train(train_set, validation_set, steps=5, batch_size=8, width=8, blocks=1, eval_every=5)
        predict(PosteriorNet(8, 1), validation_set.cells, 8, 8, 10)
    finally:
        for data in (train_set, validation_set):
            data.mines = data.saved_mines


def test_only_the_first_positions_are_used_and_a_seed_repeats_exactly(small_sets):
    train_set, validation_set = small_sets
    spoiled = generate(8, 8, 10, 200, "train", seed=0)
    spoiled.cells[100:] = 50  # not a clue: encoding fails if any of these rows is used
    _, log = train(spoiled, validation_set, positions=100, steps=30, batch_size=32, width=8, blocks=1, eval_every=30)
    assert np.all(np.isfinite(log.losses))
    first, _ = train(train_set, validation_set, steps=20, batch_size=16, width=8, blocks=1, eval_every=20, seed=3)
    second, _ = train(train_set, validation_set, steps=20, batch_size=16, width=8, blocks=1, eval_every=20, seed=3)
    for a, b in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError):
        train(train_set, validation_set, positions=0, steps=1)
