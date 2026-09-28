"""The learned mine-probability model: a network that reads the visible board and gives each
covered cell's probability of being a mine, trained against the exact probabilities
(exact_symmetry.data).

Input planes (encode):
- covered;
- clue 0 to clue 8, one-hot, on revealed cells;
- on the board: 1 everywhere, so the convolutions' zero padding marks the edge;
- mines per covered cell, and the share of cells still covered, each one number spread
  over the whole board. The mine count is the only fact about the board as a whole that
  the exact probability uses, and these planes give it to every cell.

Network (PosteriorNet): fully convolutional, so one network runs on any board size. A 3x3
convolution to `width` channels, `blocks` residual blocks of two 3x3 convolutions, and a
1x1 convolution to one logit per cell; the probability is its sigmoid. There is no
normalisation layer (GroupNorm made a step three times slower on Apple's GPU); instead each
block's second convolution starts at zero, so every block starts as the identity.

Loss: binary cross-entropy against the exact probability, averaged over covered cells.
It exceeds its minimum, the entropy of the exact probability, by exactly the KL divergence
from the exact probability to the prediction, so it is smallest at the exact probability.
"""

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exact_symmetry.minesweeper.deductive import UNKNOWN

IN_CHANNELS = 13


def pick_device(name=None):
    """CUDA, then Apple's GPU (MPS), then the CPU, or the device named."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def encode(cells: torch.Tensor, num_mines) -> torch.Tensor:
    """(B, rows, cols) visible boards (-1 covered, 0-8 clue) -> (B, 13, rows, cols) planes."""
    cells = cells.long()
    covered = (cells == UNKNOWN).float()
    clues = F.one_hot(cells.clamp(min=0), 9).permute(0, 3, 1, 2).float() * (1 - covered)[:, None]
    n_covered = covered.sum(dim=(1, 2))
    mines = torch.as_tensor(num_mines, dtype=torch.float32, device=cells.device).expand_as(n_covered)
    whole = torch.stack([mines / n_covered.clamp(min=1), n_covered / covered[0].numel()], dim=1)
    whole = whole[:, :, None, None].expand(-1, -1, *cells.shape[1:])
    return torch.cat([covered[:, None], clues, torch.ones_like(covered)[:, None], whole], dim=1)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class PosteriorNet(nn.Module):
    def __init__(self, width=64, blocks=6):
        super().__init__()
        self.stem = nn.Conv2d(IN_CHANNELS, width, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, planes):
        """(B, 13, rows, cols) -> (B, rows, cols) logits of each cell being a mine."""
        return self.head(self.blocks(F.relu(self.stem(planes))))[:, 0]


def kl_per_cell(logits, exact, covered):
    """KL divergence (nats) from the exact probability to the prediction, one value per covered cell."""
    exact, logits = exact[covered], logits[covered]
    entropy = -(torch.special.xlogy(exact, exact) + torch.special.xlogy(1 - exact, 1 - exact))
    return F.binary_cross_entropy_with_logits(logits, exact, reduction="none") - entropy


def predict_logits(model, cells: np.ndarray, n_rows, n_cols, num_mines, device="cpu", batch_size=1024):
    """The model's logits for (P, rows*cols) visible boards, as float64, NaN on revealed cells.

    Scores are computed from these, not from probabilities. A float32 probability rounds every
    cell with a logit below about -88 to exactly 0, so a KL computed from probabilities would
    need a cap."""
    model.eval()
    out = np.empty(cells.shape, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(cells), batch_size):
            batch = torch.as_tensor(cells[start:start + batch_size], device=device).view(-1, n_rows, n_cols)
            out[start:start + batch_size] = model(encode(batch, num_mines)).flatten(1).double().cpu().numpy()
    out[cells != UNKNOWN] = np.nan
    return out


def sigmoid(logits):
    """Probabilities from logits, in float64 (NaN stays NaN)."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def predict(model, cells: np.ndarray, n_rows, n_cols, num_mines, device="cpu", batch_size=1024):
    """Predicted mine probabilities for (P, rows*cols) visible boards, as float64, NaN on revealed
    cells; the sigmoid of predict_logits."""
    return sigmoid(predict_logits(model, cells, n_rows, n_cols, num_mines, device, batch_size))


def mean_kl(model, data, device="cpu", batch_size=1024):
    """Mean KL divergence per covered cell over a PositionSet."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            cells = torch.as_tensor(data.cells[start:start + batch_size], device=device).view(-1, data.n_rows, data.n_cols)
            exact = torch.as_tensor(data.probability[start:start + batch_size], device=device).view_as(cells).float()
            kl = kl_per_cell(model(encode(cells, data.num_mines)), exact.nan_to_num(0.0), cells == UNKNOWN)
            total, count = total + kl.sum().item(), count + kl.numel()
    return total / count


@dataclass
class TrainLog:
    positions: int
    steps: int
    arm: str = "plain"
    losses: list = field(default_factory=list)       # training loss per step
    validation: list = field(default_factory=list)   # (step, mean KL per covered cell)
    best_step: int = 0
    best_kl: float = float("inf")
    seconds: float = 0.0


def train(train_set, validation_set, positions=None, steps=10_000, batch_size=256, learning_rate=1e-3,
          width=64, blocks=6, seed=0, eval_every=250, device="cpu", arm="plain", group_width=8):
    """Train the model of a symmetry arm (exact_symmetry.arms; PosteriorNet for "plain") on the
    first `positions` positions of train_set (all by default).

    Adam with a cosine decay of the learning rate over `steps`; batches drawn uniformly
    with replacement. The validation KL is measured every `eval_every` steps and, before the
    first of those, every tenth of that interval, so that a minimum reached early (small
    training sets overfit fast) is still found. Validation does not change training.
    "augment" turns each board in a batch, and its labels, by a random symmetry. "canonical" turns the training boards to their canonical form once and trains
    the inner network on them, which is the same as training the CanonicalNet. Returns the
    model with the lowest validation KL, and the log.
    """
    from exact_symmetry.arms import augment_batch, canonicalise, make_model  # arms builds on this module

    positions = len(train_set) if positions is None else positions
    if not 0 < positions <= len(train_set):
        raise ValueError(f"positions must be between 1 and {len(train_set)}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    model = make_model(arm, width, blocks, group_width).to(device)
    trained = model.net if arm == "canonical" else model
    optimiser = torch.optim.Adam(trained.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    shape = (-1, train_set.n_rows, train_set.n_cols)
    cells = torch.as_tensor(train_set.cells[:positions], device=device).view(shape)
    exact = torch.as_tensor(np.nan_to_num(train_set.probability[:positions]), device=device).view(shape).float()
    if arm == "canonical":
        cells, exact = canonicalise(cells, exact)
    log, best, start = TrainLog(positions, steps, arm), None, time.time()
    for step in range(1, steps + 1):
        model.train()
        batch = torch.as_tensor(rng.integers(0, positions, batch_size), device=device)
        board, labels = cells[batch], exact[batch]
        if arm == "augment":
            board, labels = augment_batch(board, labels, generator)
        covered = (board == UNKNOWN).float()  # a weight, not an index: fixed shapes run faster on the GPU
        logits = trained(encode(board, train_set.num_mines))
        loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * covered).sum() / covered.sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        log.losses.append(loss.detach())  # read at the end: .item() here would wait for the GPU every step
        early = step < eval_every and step % max(1, eval_every // 10) == 0
        if step % eval_every == 0 or step == steps or early:
            kl = mean_kl(model, validation_set, device)
            log.validation.append((step, kl))
            if kl < log.best_kl:
                log.best_step, log.best_kl, best = step, kl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    log.losses = torch.stack(log.losses).tolist()
    log.seconds = time.time() - start
    return model, log
