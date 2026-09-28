"""Brute-force checks shared by the tests: every mine layout listed, nothing clever."""

from fractions import Fraction
from itertools import combinations

from exact_symmetry.minesweeper.deductive import FLAGGED, UNKNOWN, Observation


def neighbours(i, n_rows, n_cols):
    r, c = divmod(i, n_cols)
    return [rr * n_cols + cc for rr in range(max(0, r - 1), min(n_rows, r + 2))
            for cc in range(max(0, c - 1), min(n_cols, c + 2)) if (rr, cc) != (r, c)]


def counted(observation):
    """Each covered cell's share of the layouts that fit, by listing every layout."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    unknown = [i for i, s in enumerate(cells) if s == UNKNOWN]
    flags = {i for i, s in enumerate(cells) if s == FLAGGED}
    hits, fitting = dict.fromkeys(unknown, 0), 0
    for layout in combinations(unknown, observation.total_mines - len(flags)):
        mines = set(layout) | flags
        if all(sum(n in mines for n in neighbours(i, n_rows, n_cols)) == s for i, s in enumerate(cells) if s >= 0):
            fitting += 1
            for i in layout:
                hits[i] += 1
    return {i: Fraction(h, fitting) for i, h in hits.items()}


def transformed(observation, transform):
    cells = [None] * len(observation.cells)
    for index, image in enumerate(transform.mapping):
        cells[image] = observation.cells[index]
    return Observation(observation.n_rows, observation.n_cols, observation.total_mines, tuple(cells))
