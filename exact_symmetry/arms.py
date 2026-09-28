"""Ways of giving the mine-probability model the board's symmetries.

The exact mine probabilities are equivariant under the symmetries of the board: rotate or
reflect a position and its probabilities rotate or reflect with it. Each way of using this
is one arm here, trained and scored the same way:

- plain: PosteriorNet as it is; nothing enforces the symmetry.
- augment: PosteriorNet trained on boards each turned by a random symmetry, labels turned
  with them (augment_batch); the symmetry is learned from data, not enforced.
- canonical: CanonicalNet. Each board is turned to its canonical form, the first of its
  images in lexicographic order of the cell values; the network sees only that form, and
  its output is turned back. Equivariant wherever the canonical form picks one symmetry; at
  a position with a symmetry of its own, several symmetries give the same form and the
  first is used, so nothing makes the cells that symmetry swaps agree.
- p4m: P4MNet, whose group convolutions make it equivariant by construction.
- averaged: AveragedNet, at test time only: the average of a trained model's probabilities
  over every symmetry of the board, turned back. Exactly equivariant.
- canonical + frame: CanonicalFrameNet, at test time only, on a trained canonical arm: its
  output averaged over every symmetry that takes the board to its canonical form. These
  symmetries form a coset of the board's stabiliser, so this is the minimal frame of Lin et
  al. (ICML 2024) within the frame averaging of Puny et al. (ICLR 2022). One network pass;
  exactly equivariant at every board. At a board with no symmetry of its own it equals the
  canonical arm.

The symmetries are the dihedral group D4 on square boards (four rotations, each with or
without a reflection) and its four elements that keep a rectangle's shape otherwise.
Element g = (k, f) acts on the last two axes of a tensor as: rotate k quarter turns, then
reflect left to right if f.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from exact_symmetry.net import IN_CHANNELS, PosteriorNet

ARMS = ("plain", "augment", "canonical", "p4m")
ELEMENTS = tuple((k, f) for f in (0, 1) for k in range(4))  # (0, 0) is the identity


def elements(n_rows, n_cols):
    """The symmetries of an n_rows x n_cols board: all 8 when square, else the 4 that keep its shape."""
    return ELEMENTS if n_rows == n_cols else tuple((k, f) for k, f in ELEMENTS if k % 2 == 0)


def act(x, g):
    """Apply g = (k, f) to the last two axes of x."""
    k, f = g
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if f else x


def act_inverse(x, g):
    k, f = g
    if f:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -k, dims=(-2, -1))


def transform_indices(n_rows, n_cols, group, device="cpu"):
    """(len(group), cells) index maps: act(x, g).flatten() == x.flatten()[maps[i]] for group[i]."""
    probe = torch.arange(n_rows * n_cols).view(n_rows, n_cols)
    return torch.stack([act(probe, g).flatten() for g in group]).to(device)


def turn_each(x, choice, maps):
    """Turn each board x[i] (B, rows, cols) by the symmetry whose index map is maps[choice[i]]."""
    return x.flatten(1).gather(1, maps[choice]).view_as(x)


def augment_batch(cells, labels, generator=None):
    """Turn each board (B, rows, cols) and its labels by the same random symmetry of the board."""
    group = elements(*cells.shape[-2:])
    maps = transform_indices(*cells.shape[-2:], group, cells.device)
    choice = torch.randint(len(group), (len(cells),), generator=generator).to(cells.device)
    return turn_each(cells, choice, maps), turn_each(labels, choice, maps)


def cell_codes(cells):
    """board_codes from visible boards (-1 covered, 0-8 clue) instead of planes."""
    return torch.where(cells < 0, torch.zeros_like(cells), cells + 1).float()


def canonicalise(cells, labels, batch_size=4096):
    """Each board (B, rows, cols) and its labels turned to the board's canonical form."""
    group = elements(*cells.shape[-2:])
    maps = transform_indices(*cells.shape[-2:], group, cells.device)
    out_cells, out_labels = torch.empty_like(cells), torch.empty_like(labels)
    for start in range(0, len(cells), batch_size):
        part = slice(start, start + batch_size)
        choice = canonical_choice(cell_codes(cells[part]), group)
        out_cells[part], out_labels[part] = turn_each(cells[part], choice, maps), turn_each(labels[part], choice, maps)
    return out_cells, out_labels


def board_codes(planes):
    """One integer per cell from the input planes: 0 covered, 1 + clue on revealed cells."""
    return torch.where(planes[:, 0] > 0.5, torch.zeros_like(planes[:, 0]), 1 + planes[:, 1:10].argmax(1).float())


def canonical_minimisers(codes, group):
    """For each board (B, rows, cols) of codes, a (B, |G|) mask of the symmetries whose image is
    smallest in lexicographic (row-major) order: all of them give the board's canonical form."""
    images = torch.stack([act(codes, g).flatten(1) for g in group], dim=1)  # (B, |G|, cells)
    alive = torch.ones(images.shape[:2], dtype=torch.bool, device=codes.device)
    for position in range(images.shape[2]):
        column = images[:, :, position].masked_fill(~alive, float("inf"))
        alive &= column == column.min(dim=1, keepdim=True).values
    return alive


def canonical_choice(codes, group):
    """The index in `group` of the first symmetry giving each board's canonical form."""
    return canonical_minimisers(codes, group).float().argmax(dim=1)


def mean_probability_logit(logits, include=None):
    """The logit of the mean of sigmoid(logits) over dimension 0, computed in log space.

    logits: (K, B, ...). include: optional (K, B) mask of the entries that take part in each
    board's mean. The result is exact and finite however close the probabilities are to 0 or 1;
    averaging the probabilities themselves in float32 would round every cell with a logit above
    about 17 to exactly 1, and below about -88 to 0."""
    log_p, log_q = F.logsigmoid(logits), F.logsigmoid(-logits)
    if include is not None:
        mask = include.view(*include.shape, *([1] * (logits.dim() - 2)))
        log_p = log_p.masked_fill(~mask, float("-inf"))
        log_q = log_q.masked_fill(~mask, float("-inf"))
    # log(mean p) - log(mean (1 - p)): the 1/K of both means cancels
    return torch.logsumexp(log_p, dim=0) - torch.logsumexp(log_q, dim=0)


class CanonicalNet(nn.Module):
    """A network that sees each board only in its canonical form."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, planes):
        group = elements(*planes.shape[-2:])
        choice = canonical_choice(board_codes(planes), group)
        out = torch.empty(planes.shape[0], *planes.shape[-2:], dtype=planes.dtype, device=planes.device)
        for index, g in enumerate(group):
            chosen = choice == index
            if chosen.any():
                out[chosen] = act_inverse(self.net(act(planes[chosen], g)), g)
        return out


class CanonicalFrameNet(nn.Module):
    """A canonical-form network averaged over every symmetry that reaches the canonical form.

    Those symmetries all give the same canonical board, so the network runs once; its
    probabilities are turned back by each of them and averaged. They form a coset of the
    board's stabiliser, so the result is fixed by the stabiliser and the model is exactly
    equivariant at every board, symmetric or not."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, planes):
        group = elements(*planes.shape[-2:])
        minimisers = canonical_minimisers(board_codes(planes), group)
        first = minimisers.float().argmax(dim=1)
        logits = torch.empty(planes.shape[0], *planes.shape[-2:], dtype=planes.dtype, device=planes.device)
        for index, g in enumerate(group):
            chosen = first == index
            if chosen.any():
                logits[chosen] = self.net(act(planes[chosen], g))  # on the canonical board
        turned_back = torch.stack([act_inverse(logits, g) for g in group])  # (|G|, B, rows, cols)
        return mean_probability_logit(turned_back, minimisers.T)


class AveragedNet(nn.Module):
    """The average of a model's probabilities over every symmetry of the board, as logits."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, planes):
        group = elements(*planes.shape[-2:])
        turned = torch.cat([act(planes, g) for g in group])  # one pass for every symmetry
        out = self.net(turned).view(len(group), planes.shape[0], *planes.shape[-2:])
        return mean_probability_logit(torch.stack([act_inverse(out[i], g) for i, g in enumerate(group)]))


# p4m: group convolutions over D4 (Cohen and Welling, 2016). A feature map carries one
# channel per group element; the layers below are equivariant, so turning the input board
# turns every map and permutes its group channels. Only for the square group, D4: on a
# rectangle the network is still equivariant under the four symmetries that keep its shape.

def _compose(a, b):
    """The element acting as a after b."""
    probe = torch.arange(9.0).reshape(3, 3)
    target = act(act(probe, b), a)
    return next(g for g in ELEMENTS if torch.equal(act(probe, g), target))


INVERSE = tuple(next(h for h in ELEMENTS if _compose(g, h) == (0, 0)) for g in ELEMENTS)
PRODUCT = tuple(tuple(ELEMENTS.index(_compose(g, h)) for h in ELEMENTS) for g in ELEMENTS)


def _turned_kernel(weight, index):
    """The weights gathered once per group element, index row by row.

    Every weight appears once in each row, so each gather's backward adds into distinct
    positions and its result does not depend on the order of the additions. Gathering all rows
    at once repeats every index eight times, and that backward accumulates in an order that
    varies from run to run on MPS and on multi-threaded CPU, so training would not repeat
    exactly from a seed."""
    flat = weight.flatten()
    return torch.cat([flat[row] for row in index])


class LiftingConv(nn.Module):
    """Board planes (B, C, H, W) -> group maps (B, 8 C', H, W), channel g * C' + c:
    out[g, c] = input correlated with g(kernel[c])."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        probe = torch.arange(self.weight.numel()).view_as(self.weight)
        # one row per group element, each a permutation of the weights (see _turned_kernel)
        self.register_buffer("index", torch.stack([act(probe, g).flatten() for g in ELEMENTS]), persistent=False)

    def forward(self, x):
        weight = _turned_kernel(self.weight, self.index).view(-1, *self.weight.shape[1:])  # (8 C', C, 3, 3)
        return F.conv2d(x, weight, self.bias.repeat(len(ELEMENTS)), padding=1)


class GroupConv(nn.Module):
    """Group maps -> group maps, channel g * C + c:
    out[g, c'] = sum over h, c of in[h, c] correlated with g(kernel[c', c, g^-1 h])."""

    def __init__(self, in_channels, out_channels, zero=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, len(ELEMENTS), 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        if zero:
            nn.init.zeros_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        probe = torch.arange(self.weight.numel()).view_as(self.weight)
        blocks = []
        for index, g in enumerate(ELEMENTS):
            order = [PRODUCT[ELEMENTS.index(INVERSE[index])][h] for h in range(len(ELEMENTS))]
            turned = act(probe[:, :, order], g)                                  # (C', C, 8 [h], 3, 3)
            blocks.append(turned.permute(0, 2, 1, 3, 4).reshape(out_channels, -1, 3, 3))  # columns h * C + c
        self.register_buffer("index", torch.stack([b.flatten() for b in blocks]), persistent=False)  # rows g

    def forward(self, x):
        weight = _turned_kernel(self.weight, self.index).view(-1, x.shape[1], 3, 3)  # rows g * C' + c'
        return F.conv2d(x, weight, self.bias.repeat(len(ELEMENTS)), padding=1)


class GroupBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = GroupConv(width, width)
        self.conv2 = GroupConv(width, width, zero=True)  # every block starts as the identity

    def forward(self, x):
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class P4MNet(nn.Module):
    """PosteriorNet's structure with group convolutions; the group axis is averaged out before
    the per-cell 1x1 head, so the output field turns with the board."""

    def __init__(self, width=8, blocks=6):
        super().__init__()
        self.width = width
        self.stem = LiftingConv(IN_CHANNELS, width)
        self.blocks = nn.Sequential(*[GroupBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, planes):
        x = self.blocks(F.relu(self.stem(planes)))
        pooled = x.view(x.shape[0], len(ELEMENTS), self.width, *x.shape[-2:]).mean(dim=1)
        return self.head(pooled)[:, 0]


def make_model(arm, width=64, blocks=6, group_width=8):
    """The model an arm trains ("augment" trains a PosteriorNet on turned boards)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if arm == "p4m":
        return P4MNet(group_width, blocks)
    return CanonicalNet(PosteriorNet(width, blocks)) if arm == "canonical" else PosteriorNet(width, blocks)


def multiply_adds(model, n_rows, n_cols):
    """Multiply-adds of the convolutions in one forward pass on one board."""
    total = 0
    real_conv2d = F.conv2d

    def counting(x, weight, *args, **kwargs):
        nonlocal total
        out = real_conv2d(x, weight, *args, **kwargs)
        total += weight[0].numel() * out.shape[1] * out.shape[2] * out.shape[3] * out.shape[0]
        return out

    planes = torch.zeros(1, IN_CHANNELS, n_rows, n_cols)
    planes[:, 0] = 1  # all covered
    planes[:, 10] = 1
    F.conv2d = counting
    try:
        with torch.no_grad():
            model.eval()(planes)
    finally:
        F.conv2d = real_conv2d
    return total
