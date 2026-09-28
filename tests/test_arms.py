"""The symmetry arms: transforms, augmentation, canonical form, averaging and the p4m network."""

import numpy as np
import pytest
import torch

from exact_symmetry.arms import (
    ARMS,
    ELEMENTS,
    INVERSE,
    PRODUCT,
    AveragedNet,
    CanonicalNet,
    GroupConv,
    LiftingConv,
    P4MNet,
    act,
    act_inverse,
    augment_batch,
    board_codes,
    canonical_choice,
    elements,
    make_model,
    multiply_adds,
)
from exact_symmetry.data import exact_probabilities, generate
from exact_symmetry.minesweeper.symmetry import state_stabilizer
from exact_symmetry.net import IN_CHANNELS, PosteriorNet, encode


def test_the_eight_elements_form_the_group_of_the_square():
    x = torch.randn(2, 3, 5, 5)
    images = {tuple(act(x, g).flatten().tolist()) for g in ELEMENTS}
    assert len(ELEMENTS) == len(images) == 8
    for i, g in enumerate(ELEMENTS):
        assert torch.equal(act_inverse(act(x, g), g), x)
        assert torch.equal(act(act(x, g), INVERSE[i]), x)
        for j, h in enumerate(ELEMENTS):
            assert torch.equal(act(act(x, h), g), act(x, ELEMENTS[PRODUCT[i][j]]))


def test_a_rectangle_keeps_its_shape_under_its_four_symmetries():
    x = torch.randn(1, 4, 6)
    group = elements(4, 6)
    assert len(group) == 4 and len(elements(8, 8)) == 8
    assert all(act(x, g).shape == x.shape for g in group)


def test_augmented_labels_are_the_exact_probabilities_of_the_turned_boards():
    data = generate(8, 8, 10, 40, "train", seed=11)
    cells = torch.as_tensor(data.cells, dtype=torch.long).view(-1, 8, 8)
    labels = torch.as_tensor(data.probability).view(-1, 8, 8)
    turned_cells, turned_labels = augment_batch(cells, labels, torch.Generator().manual_seed(0))
    for board, label in zip(turned_cells, turned_labels):
        exact = exact_probabilities(tuple(board.flatten().tolist()), 8, 8, 10)
        assert np.allclose(exact, label.flatten().double().numpy(), equal_nan=True, atol=1e-6)
    moved = sum(not torch.equal(a, b) for a, b in zip(cells, turned_cells))
    assert moved > 20  # most boards were turned by something other than the identity


def test_the_canonical_form_is_the_same_for_every_turn_of_a_board():
    data = generate(8, 8, 10, 60, "test", seed=12)
    cells = torch.as_tensor(data.cells, dtype=torch.long).view(-1, 8, 8)
    planes = encode(cells, 10)
    codes = board_codes(planes)
    group = elements(8, 8)
    for g in group:
        turned = act(codes, g)
        choice, turned_choice = canonical_choice(codes, group), canonical_choice(turned, group)
        for board, t_board, c, tc in zip(codes, turned, choice, turned_choice):
            assert torch.equal(act(board, group[c]), act(t_board, group[tc]))


def equivariance_error(model, planes, group):
    out = model(planes)
    return max(float((model(act(planes, g)) - act(out, g)).abs().max()) for g in group)


def positions(n_rows, n_cols, mines, count, seed):
    data = generate(n_rows, n_cols, mines, count, "test", seed=seed)
    cells = torch.as_tensor(data.cells, dtype=torch.long).view(-1, n_rows, n_cols)
    return data, encode(cells, mines)


def test_the_canonical_network_turns_with_the_board_where_the_board_has_no_symmetry_of_its_own():
    torch.manual_seed(0)
    model = CanonicalNet(PosteriorNet(16, 2)).eval()
    data, planes = positions(8, 8, 10, 80, 13)
    plain = [len(state_stabilizer(tuple(int(c) for c in row), 8, 8)) == 1 for row in data.cells]
    with torch.no_grad():
        assert equivariance_error(model, planes[torch.tensor(plain)], elements(8, 8)) < 1e-5
        assert equivariance_error(PosteriorNet(16, 2).eval(), planes, elements(8, 8)) > 1e-3  # the plain one does not


def test_the_averaged_network_turns_with_the_board_exactly():
    torch.manual_seed(1)
    model = AveragedNet(PosteriorNet(16, 2)).eval()
    _, planes = positions(8, 8, 10, 30, 14)
    with torch.no_grad():
        assert equivariance_error(model, planes, elements(8, 8)) < 1e-4


@pytest.mark.parametrize("n_rows, n_cols, mines", [(8, 8, 10), (6, 9, 8)])
def test_the_p4m_network_turns_with_the_board(n_rows, n_cols, mines):
    torch.manual_seed(2)
    model = P4MNet(8, 2).eval()
    with torch.no_grad():
        for block in model.blocks:  # the blocks start as the identity; make them do something
            block.conv2.weight.normal_(0, 0.05)
        for name, parameter in model.named_parameters():  # biases start at 0, which would hide a wrong bias layout
            if name.endswith("bias"):
                parameter.normal_(0, 0.1)
    _, planes = positions(n_rows, n_cols, mines, 20, 15)
    with torch.no_grad():
        assert equivariance_error(model, planes, elements(n_rows, n_cols)) < 1e-4
        turned = act(planes, (1, 0))  # a quarter turn makes a rectangle the other way round
        assert torch.allclose(model(turned), act(model(planes), (1, 0)), atol=1e-4)


def test_models_for_each_arm_and_their_cost():
    assert set(ARMS) == {"plain", "augment", "canonical", "p4m"}
    assert isinstance(make_model("plain", 16, 2), PosteriorNet)
    assert isinstance(make_model("augment", 16, 2), PosteriorNet)
    assert isinstance(make_model("canonical", 16, 2), CanonicalNet)
    assert isinstance(make_model("p4m", 16, 2, group_width=8), P4MNet)
    with pytest.raises(ValueError):
        make_model("rotate", 16, 2)
    width, blocks, cells = 64, 6, 64
    expected = IN_CHANNELS * width * 9 * cells + 2 * blocks * width * width * 9 * cells + width * cells
    assert multiply_adds(PosteriorNet(width, blocks), 8, 8) == expected
    group = 8
    p4m_expected = (IN_CHANNELS * group * width * 9 * cells + 2 * blocks * (group * width) ** 2 * 9 * cells
                    + width * cells)
    assert multiply_adds(P4MNet(width, blocks), 8, 8) == p4m_expected


def test_codes_from_cells_match_codes_from_planes_and_canonical_labels_stay_exact():
    from exact_symmetry.arms import canonicalise, cell_codes
    data = generate(8, 8, 10, 50, "test", seed=16)
    cells = torch.as_tensor(data.cells, dtype=torch.long).view(-1, 8, 8)
    labels = torch.as_tensor(data.probability).view(-1, 8, 8)
    assert torch.equal(cell_codes(cells), board_codes(encode(cells, 10)))
    canonical_cells, canonical_labels = canonicalise(cells, labels)
    group = elements(8, 8)
    for board, turned, label in zip(cells, canonical_cells, canonical_labels):
        images = [act(board, g) for g in group]
        assert any(torch.equal(turned, image) for image in images)
        assert min(tuple(image.flatten().tolist()) for image in (cell_codes(i[None])[0] for i in images)) == \
            tuple(cell_codes(turned[None])[0].flatten().tolist())
        exact = exact_probabilities(tuple(turned.flatten().tolist()), 8, 8, 10)
        assert np.allclose(exact, label.flatten().double().numpy(), equal_nan=True, atol=1e-6)


@pytest.mark.parametrize("arm", ["plain", "augment", "canonical", "p4m"])
def test_every_arm_trains_and_repeats_from_its_seed(arm):
    from exact_symmetry.net import mean_kl, train
    train_set, validation_set = generate(8, 8, 10, 300, "train", seed=17), generate(8, 8, 10, 100, "validation", seed=17)
    # group width 8 and batch 64: large enough that a racy accumulation in the p4m kernel's
    # backward would show on a multi-threaded CPU
    kwargs = dict(steps=12, batch_size=64, width=8, blocks=1, group_width=8, eval_every=6, seed=5, arm=arm)
    first, log = train(train_set, validation_set, **kwargs)
    second, _ = train(train_set, validation_set, **kwargs)
    expected = {"plain": PosteriorNet, "augment": PosteriorNet, "canonical": CanonicalNet, "p4m": P4MNet}[arm]
    assert type(first) is expected and log.arm == arm
    assert log.best_kl == pytest.approx(mean_kl(first, validation_set))
    for a, b in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(a, b)


def test_at_boards_with_a_symmetry_of_their_own_the_canonical_network_need_not_turn_but_averaging_does():
    torch.manual_seed(3)
    inner = PosteriorNet(16, 2).eval()
    group = elements(5, 5)
    data = generate(5, 5, 3, 400, "test", seed=18, epsilon=0.5)
    symmetric = [len(state_stabilizer(tuple(int(c) for c in row), 5, 5)) > 1 for row in data.cells]
    assert sum(symmetric) >= 10
    cells = torch.as_tensor(data.cells[np.array(symmetric)], dtype=torch.long).view(-1, 5, 5)
    planes = encode(cells, 3)
    with torch.no_grad():
        assert equivariance_error(CanonicalNet(inner), planes, group) > 1e-3
        assert equivariance_error(AveragedNet(inner), planes, group) < 1e-4


def test_the_frame_averaged_canonical_network_turns_with_every_board_and_never_raises_its_kl():
    from exact_symmetry.arms import CanonicalFrameNet
    from exact_symmetry.evaluate import kl_from_logits
    torch.manual_seed(3)
    inner = PosteriorNet(16, 2).double().eval()
    group = elements(5, 5)
    data = generate(5, 5, 3, 400, "test", seed=18, epsilon=0.5)
    symmetric = np.array([len(state_stabilizer(tuple(int(c) for c in row), 5, 5)) > 1 for row in data.cells])
    assert symmetric.sum() >= 10
    planes = encode(torch.as_tensor(data.cells, dtype=torch.long).view(-1, 5, 5), 3).double()
    with torch.no_grad():
        framed, canonical = CanonicalFrameNet(inner), CanonicalNet(inner)
        assert equivariance_error(framed, planes[torch.as_tensor(symmetric)], group) < 1e-9
        # where the board has no symmetry of its own, one symmetry reaches the canonical form
        plain = torch.as_tensor(~symmetric)
        assert torch.allclose(framed(planes[plain]), canonical(planes[plain]), rtol=0, atol=1e-9)
        z_framed, z_canonical = framed(planes).flatten(1).numpy(), canonical(planes).flatten(1).numpy()
    # at a symmetric board the frame averages each swapped pair, whose exact probabilities are
    # equal; the KL is convex in the prediction, so the board's total KL cannot rise
    covered, exact = data.cells < 0, np.nan_to_num(data.probability)
    kl_framed = np.where(covered, kl_from_logits(z_framed, exact), 0.0).sum(1)
    kl_canonical = np.where(covered, kl_from_logits(z_canonical, exact), 0.0).sum(1)
    assert np.all(kl_framed[symmetric] <= kl_canonical[symmetric] + 1e-12)
    assert np.any(kl_framed[symmetric] < kl_canonical[symmetric] - 1e-6)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_group_convolution_gradients_repeat_exactly(device):
    threads = torch.get_num_threads()
    torch.set_num_threads(max(threads, 8))
    try:
        for layer, channels in [(LiftingConv(IN_CHANNELS, 8), IN_CHANNELS), (GroupConv(8, 8), 8 * len(ELEMENTS))]:
            layer = layer.to(device)
            x = torch.randn(64, channels, 8, 8, generator=torch.Generator().manual_seed(4)).to(device)
            grads = []
            for _ in range(4):
                layer.zero_grad()
                (layer(x) ** 2).sum().backward()
                grads.append(layer.weight.grad.detach().cpu().clone())
            assert all(torch.equal(grads[0], g) for g in grads[1:])
    finally:
        torch.set_num_threads(threads)


def test_the_averaged_network_gives_the_exact_logit_of_the_mean_probability():
    from exact_symmetry.arms import mean_probability_logit
    torch.manual_seed(6)
    net = PosteriorNet(16, 2).eval()
    with torch.no_grad():
        net.head.weight.mul_(400.0)  # logits in the hundreds
    _, planes = positions(8, 8, 10, 30, 21)
    group = elements(8, 8)
    with torch.no_grad():
        averaged = AveragedNet(net)(planes).double()
        turned = torch.stack([act_inverse(net(act(planes, g)), g) for g in group]).double()
    probability = torch.sigmoid(turned)
    reference = torch.log(probability.mean(0)) - torch.log1p(-probability.mean(0))
    finite = torch.isfinite(reference)
    assert torch.allclose(averaged[finite], reference[finite], rtol=1e-4, atol=1e-3)
    assert averaged.abs().max() > 50  # far outside what a float32 probability could rank
    assert torch.allclose(mean_probability_logit(turned[:1]), turned[0], rtol=0, atol=1e-9)  # the mean of one is itself
    everyone, first_only = torch.ones(turned.shape[:2], dtype=torch.bool), torch.zeros(turned.shape[:2], dtype=torch.bool)
    first_only[0] = True
    assert torch.equal(mean_probability_logit(turned, everyone), mean_probability_logit(turned))
    assert torch.allclose(mean_probability_logit(turned, first_only), turned[0], rtol=0, atol=1e-9)


def test_the_canonical_networks_keep_the_input_precision():
    from exact_symmetry.arms import CanonicalFrameNet
    net = PosteriorNet(8, 1).double().eval()
    _, planes = positions(8, 8, 10, 5, 22)
    with torch.no_grad():
        assert CanonicalNet(net)(planes.double()).dtype == torch.float64
        assert CanonicalFrameNet(net)(planes.double()).dtype == torch.float64
