# Copied from the Minesweeper solver (github.com/Evasion-OC/minesweeper-solver, minesweeper/probability.py,
# commit 63d178a). Only the module paths differ. It gives the exact answers this project scores against.
"""Exact mine probabilities from the visible board, for the guess when nothing is forced.

Every mine layout that fits the clues and the total mine count is equally likely, so a
covered cell's probability of being a mine is the share of those layouts in which it is
one. The layouts are counted, not listed. Each region is counted on its own clues, by
the number of mines it holds; regions and the cells no clue touches then combine only
through the total, so their counts are convolved and the unconstrained cells take the
rest of the mines in C(cells, mines) ways. Inside a region, covered cells next to the
same clues are interchangeable, so a group of them is counted by how many mines it
holds, weighted by the binomial coefficient.
"""

from fractions import Fraction
from math import comb

from exact_symmetry.minesweeper.deductive import (
    FLAGGED,
    UNKNOWN,
    InconsistentObservation,
    Observation,
    Region,
    RegionCache,
    _check_clues_and_count,
    _neighbours,
    _region_picture,
    frontier_regions,
)
from exact_symmetry.minesweeper.symmetry import canonical_form


class TooManyLayouts(RuntimeError):
    """Counting a region's layouts would take more steps than the cap allows, or more
    groups of cells than Python's call stack can hold."""


_MAX_GROUPS = 800  # the count recurses once per group; Python stops at 1,000 frames


def _region_counts(observation: Observation, region: Region, cap: int):
    """For each number of mines k the region can hold: the number of layouts with k mines,
    and for each of its cells the number of those layouts in which it is a mine."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    clue_set = set(region.clues)
    groups = {}
    for cell in region.cells:
        seen_by = frozenset(n for n in _neighbours(cell, n_rows, n_cols) if n in clue_set)
        groups.setdefault(seen_by, []).append(cell)
    order = []  # groups in an order that closes clues early
    pending = dict(groups)
    frontier = [region.clues[0]]
    while pending:
        clue = frontier.pop(0) if frontier else next(iter(next(iter(pending))))
        for key in [key for key in pending if clue in key]:
            order.append((key, pending.pop(key)))
            frontier.extend(key)
    if len(order) > _MAX_GROUPS:
        raise TooManyLayouts(f"region of {len(order)} groups of cells is too deep to count")
    need = {clue: cells[clue] - sum(cells[n] == FLAGGED for n in _neighbours(clue, n_rows, n_cols))
            for clue in region.clues}  # mines each clue still has to find
    left_in = {clue: sum(len(members) for key, members in order if clue in key) for clue in region.clues}
    counts = {}  # k -> [layouts with k mines, per group: sum over those layouts of its mines]
    chosen = [0] * len(order)
    visits = 0

    def assign(position, mines, weight):
        nonlocal visits
        visits += 1
        if visits > cap:
            raise TooManyLayouts(f"region of {len(region.cells)} cells exceeds {cap} partial layouts")
        if position == len(order):
            entry = counts.setdefault(mines, [0, [0] * len(order)])
            entry[0] += weight
            for index, count in enumerate(chosen):
                entry[1][index] += weight * count
            return
        key, members = order[position]
        size = len(members)
        for clue in key:
            left_in[clue] -= size
        for count in range(size + 1):
            if all(0 <= need[clue] - count <= left_in[clue] for clue in key):
                for clue in key:
                    need[clue] -= count
                chosen[position] = count
                assign(position + 1, mines + count, weight * comb(size, count))
                for clue in key:
                    need[clue] += count
        for clue in key:
            left_in[clue] += size

    assign(0, 0, 1)
    if not counts:
        raise InconsistentObservation("No mine placement satisfies the visible board")
    return {mines: (layouts, {cell: Fraction(total, len(members))
                              for (key, members), total in zip(order, per_group) for cell in members})
            for mines, (layouts, per_group) in counts.items()}


def _cached_region_counts(observation: Observation, region: Region, cache: RegionCache, cap: int):
    """_region_counts, reused through the cache by the region's canonical form."""
    form, placements = canonical_form(_region_picture(observation, region))
    place = {index: placements[0][divmod(index, observation.n_cols)] for index in region.cells}
    stored = cache._counts.get(form)
    if stored is None:
        counts = _region_counts(observation, region, cap)
        stored = {mines: (layouts, {place[cell]: share for cell, share in per_cell.items()})
                  for mines, (layouts, per_cell) in counts.items()}
        cache._counts[form] = stored
    cell_at = {p: index for index, p in place.items()}
    return {mines: (layouts, {cell_at[p]: share for p, share in per_place.items()})
            for mines, (layouts, per_place) in stored.items()}


def _convolve(first: dict, second: dict) -> dict:
    out = {}
    for a, x in first.items():
        for b, y in second.items():
            out[a + b] = out.get(a + b, 0) + x * y
    return out


def mine_probabilities(observation: Observation, cache: RegionCache | None = None, cap: int = 200_000):
    """The exact probability that each covered cell is a mine, as a Fraction, by cell index.

    Raises InconsistentObservation when no layout fits, and TooManyLayouts when a region
    needs more than `cap` steps to count.
    """
    cache = RegionCache() if cache is None else cache
    cells = observation.cells
    unknowns = [index for index, state in enumerate(cells) if state == UNKNOWN]
    remaining = _check_clues_and_count(observation)
    regions = frontier_regions(observation)
    per_region = [_cached_region_counts(observation, region, cache, cap) for region in regions]
    in_regions = {cell for region in regions for cell in region.cells}
    free = [index for index in unknowns if index not in in_regions]
    layouts = [{mines: count for mines, (count, _) in counts.items()} for counts in per_region]

    # Layouts of all regions but one, by total mines: products of prefixes and suffixes.
    prefix = [{0: 1}]
    for counts in layouts:
        prefix.append(_convolve(prefix[-1], counts))
    suffix = [{0: 1}]
    for counts in reversed(layouts):
        suffix.append(_convolve(suffix[-1], counts))
    suffix.reverse()
    everything = prefix[-1]

    def fill(mines_in_regions):  # ways to place the rest of the mines on the free cells
        rest = remaining - mines_in_regions
        return comb(len(free), rest) if 0 <= rest <= len(free) else 0

    total = sum(count * fill(mines) for mines, count in everything.items())
    if total == 0:
        raise InconsistentObservation("No mine placement satisfies the visible board")
    probabilities = {}
    for position, counts in enumerate(per_region):
        others = _convolve(prefix[position], suffix[position + 1])
        for mines, (_, per_cell) in counts.items():
            weight = sum(count * fill(mines + rest) for rest, count in others.items())
            for cell, share in per_cell.items():
                probabilities[cell] = probabilities.get(cell, 0) + share * weight
    if free:
        mines_on_free = sum(count * fill(mines) * (remaining - mines) for mines, count in everything.items())
        for cell in free:
            probabilities[cell] = Fraction(mines_on_free, len(free))
    return {cell: Fraction(value) / total for cell, value in probabilities.items()}
