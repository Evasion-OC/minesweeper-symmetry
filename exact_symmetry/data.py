"""Positions and their exact mine probabilities: the data of the learned mine-probability model.

A position is the visible board as the network sees it, one value per cell: covered (-1)
or the clue (0-8). Flags are shown as covered. The exact player flags only cells it has
proved to be mines, so a flag adds nothing the clues do not already say, and the network
never depends on the flagging habits of whoever played.

Positions come from games the exact player plays (exact_symmetry.minesweeper.deductive.
choose_action): a forced move when there is one, else the covered cell least likely to be
a mine. With probability epsilon a move is replaced by revealing a safe covered cell chosen
uniformly at random. This takes play off the exact player's path, as a lucky guess would,
without ending the game, so late positions are met as often as early ones. Every position
met before the game ends is kept once, labelled with the exact probability that each
covered cell is a mine (exact_symmetry.minesweeper.probability.mine_probabilities). That
label is the true probability given what is visible. The first move's cell and its
neighbours are revealed; the exact player's moves depend only on the visible board; a
random move picks each covered safe cell with chance one over their number, which the
visible board fixes (covered cells minus the mines not flagged). So every layout that fits
the visible board gives the moves made the same chance, and all such layouts stay equally
likely.

Boards come from random.Random(f"{split}/{rows}x{cols}x{mines}/{seed}"), so the training,
validation and test sets use separate streams.
"""

import json
import random
from dataclasses import dataclass, field

import numpy as np

from exact_symmetry.minesweeper.deductive import FLAGGED, UNKNOWN, Observation, RegionCache, _neighbours, choose_action
from exact_symmetry.minesweeper.probability import TooManyLayouts, mine_probabilities

SPLITS = ("train", "validation", "test")


def board_rng(split: str, n_rows: int, n_cols: int, num_mines: int, seed: int) -> random.Random:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return random.Random(f"{split}/{n_rows}x{n_cols}x{num_mines}/{seed}")


def as_seen(cells) -> tuple[int, ...]:
    """The visible board as the network sees it: flags shown as covered."""
    return tuple(UNKNOWN if state == FLAGGED else state for state in cells)


def exact_probabilities(cells, n_rows: int, n_cols: int, num_mines: int, cache=None, cap: int = 200_000):
    """Each covered cell's exact probability of being a mine, NaN on revealed cells (float64).

    Raises TooManyLayouts when a region needs more than `cap` steps to count.
    """
    found = mine_probabilities(Observation(n_rows, n_cols, num_mines, as_seen(cells)), cache, cap)
    out = np.full(n_rows * n_cols, np.nan)
    for index, probability in found.items():
        out[index] = float(probability)
    return out


@dataclass
class Game:
    mines: frozenset
    positions: list = field(default_factory=list)  # visible boards before each move, flags kept
    won: bool = False


def play(rng: random.Random, n_rows: int, n_cols: int, num_mines: int, choose) -> Game:
    """One game on a board drawn from rng: a random first move whose cell and neighbours
    are mine-free, then choose(visible, mines) -> (kind, cell index) for each move until
    every safe cell is revealed or a mine is. Only a data generator's random safe reveal
    may look at `mines`; a player never does."""
    size = n_rows * n_cols
    first = rng.randrange(size)
    banned = set(_neighbours(first, n_rows, n_cols)) | {first}
    mines = frozenset(rng.sample([i for i in range(size) if i not in banned], num_mines))
    clue = [sum(n in mines for n in _neighbours(i, n_rows, n_cols)) for i in range(size)]
    visible = [UNKNOWN] * size

    def reveal(index):
        stack = [index]
        while stack:
            cell = stack.pop()
            if visible[cell] != UNKNOWN:
                continue
            visible[cell] = clue[cell]
            if clue[cell] == 0:
                stack.extend(_neighbours(cell, n_rows, n_cols))

    reveal(first)
    game = Game(mines)
    while not all(visible[i] != UNKNOWN for i in range(size) if i not in mines):
        game.positions.append(tuple(visible))
        kind, index = choose(tuple(visible), mines)
        if kind == "flag":
            visible[index] = FLAGGED
        elif index in mines:
            return game
        else:
            reveal(index)
    game.won = True
    return game


def exact_move(visible, n_rows, n_cols, num_mines, cache):
    """The exact player's move as (kind, cell index): a forced move, else the least likely cell."""
    action = choose_action(Observation(n_rows, n_cols, num_mines, visible), cache)
    return action.kind, action.cell[0] * n_cols + action.cell[1]


def play_game(rng: random.Random, n_rows: int, n_cols: int, num_mines: int, epsilon: float) -> Game:
    """One game on a board drawn from rng, played by the exact player with random safe reveals mixed in."""
    cache = RegionCache()

    def choose(visible, mines):
        if rng.random() < epsilon:
            return "reveal", rng.choice([i for i, state in enumerate(visible) if state == UNKNOWN and i not in mines])
        return exact_move(visible, n_rows, n_cols, num_mines, cache)

    return play(rng, n_rows, n_cols, num_mines, choose)


@dataclass
class PositionSet:
    """Positions of one board size, one row per position, cells in row-major order."""
    n_rows: int
    n_cols: int
    num_mines: int
    cells: np.ndarray         # int8, -1 covered, 0-8 clue (flags shown as covered)
    probability: np.ndarray   # float32, exact mine probability of each covered cell, NaN on revealed
    mines: np.ndarray         # bool, the hidden layout, for scoring only; never a network input
    game: np.ndarray          # int32, index of the game the position came from
    meta: dict

    def __len__(self):
        return len(self.cells)

    def save(self, path):
        np.savez_compressed(path, cells=self.cells, probability=self.probability, mines=self.mines,
                            game=self.game, meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            return cls(meta["rows"], meta["cols"], meta["mines"], data["cells"], data["probability"],
                       data["mines"], data["game"], meta)


def generate(n_rows: int, n_cols: int, num_mines: int, positions: int, split: str, seed: int = 0,
             epsilon: float = 0.2, cap: int = 200_000) -> PositionSet:
    """The first `positions` positions met in games on boards from the split's stream.

    A position whose board looks the same as the one before it (after a flag) is kept once.
    A position with a region too large to count exactly is left out and counted in meta.
    """
    rng = board_rng(split, n_rows, n_cols, num_mines, seed)
    cells, probability, mines, game_index = [], [], [], []
    games = wins = too_large = 0
    while len(cells) < positions:
        game = play_game(rng, n_rows, n_cols, num_mines, epsilon)
        layout = np.zeros(n_rows * n_cols, dtype=bool)
        layout[list(game.mines)] = True
        cache, last = RegionCache(), None
        for visible in game.positions:
            seen = as_seen(visible)
            if seen == last:
                continue
            last = seen
            try:
                label = exact_probabilities(seen, n_rows, n_cols, num_mines, cache, cap)
            except TooManyLayouts:
                too_large += 1
                continue
            cells.append(seen)
            probability.append(label)
            mines.append(layout)
            game_index.append(games)
            if len(cells) == positions:
                break
        games += 1
        wins += game.won
    meta = {"rows": n_rows, "cols": n_cols, "mines": num_mines, "split": split, "seed": seed,
            "epsilon": epsilon, "cap": cap, "positions": len(cells), "games": games, "games won": wins,
            "positions left out, region too large to count": too_large}
    return PositionSet(n_rows, n_cols, num_mines, np.array(cells, dtype=np.int8).reshape(-1, n_rows * n_cols),
                       np.array(probability, dtype=np.float32).reshape(-1, n_rows * n_cols),
                       np.array(mines, dtype=bool).reshape(-1, n_rows * n_cols),
                       np.array(game_index, dtype=np.int32), meta)
