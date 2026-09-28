# Copied from the Minesweeper solver (github.com/Evasion-OC/minesweeper-solver, minesweeper/deductive.py,
# commit 63d178a). Only the module paths differ. It gives the exact answers this project scores against.
from contextlib import closing
from dataclasses import dataclass
from typing import Literal

from pysat.card import CardEnc, EncType, ITotalizer
from pysat.formula import IDPool
from pysat.solvers import Glucose3

from exact_symmetry.minesweeper.symmetry import canonical_form, cell_orbits, state_stabilizer


UNKNOWN = -1
FLAGGED = -2
Coordinate = tuple[int, int]


class InconsistentObservation(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    n_rows: int
    n_cols: int
    total_mines: int
    cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_rows < 1 or self.n_cols < 1:
            raise InconsistentObservation("Board dimensions must be positive")
        if len(self.cells) != self.n_rows * self.n_cols:
            raise InconsistentObservation("Visible state does not match board dimensions")
        if not 0 <= self.total_mines <= len(self.cells):
            raise InconsistentObservation("Total mine count is outside the board")
        if any(state not in (UNKNOWN, FLAGGED) and not 0 <= state <= 8 for state in self.cells):
            raise InconsistentObservation("Visible state contains an invalid clue")

    @classmethod
    def from_board(cls, board: list[list[dict]], total_mines: int) -> "Observation":
        if not board or not board[0] or any(len(row) != len(board[0]) for row in board):
            raise InconsistentObservation("Board must be a non-empty rectangle")
        visible = []
        for row in board:
            for cell in row:
                if cell["flagged"]:
                    if not cell["covered"]:
                        raise InconsistentObservation("A revealed cell cannot be flagged")
                    visible.append(FLAGGED)
                elif cell["covered"]:
                    visible.append(UNKNOWN)
                else:
                    clue = cell["clue"]
                    if not 0 <= clue <= 8:
                        raise InconsistentObservation("A revealed cell has an invalid clue")
                    visible.append(clue)
        return cls(len(board), len(board[0]), total_mines, tuple(visible))


@dataclass(frozen=True)
class Action:
    kind: Literal["reveal", "flag"]
    cell: Coordinate
    reason: Literal["forced", "guess"]


def _neighbours(index: int, n_rows: int, n_cols: int) -> tuple[int, ...]:
    row, column = divmod(index, n_cols)
    return tuple(
        next_row * n_cols + next_column
        for next_row in range(max(0, row - 1), min(n_rows, row + 2))
        for next_column in range(max(0, column - 1), min(n_cols, column + 2))
        if (next_row, next_column) != (row, column)
    )


@dataclass(frozen=True)
class Region:
    """Covered cells tied together by revealed clues, with the clues that constrain them.

    Two covered cells are in the same region when a chain of clues links them.
    Regions constrain each other only through the total number of mines.
    """

    clues: tuple[int, ...]
    cells: tuple[int, ...]


def frontier_regions(observation: Observation) -> tuple[Region, ...]:
    """Split the covered cells next to revealed clues into independent regions."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    unknown_around = {}
    for index, state in enumerate(cells):
        if state >= 0:
            around = tuple(n for n in _neighbours(index, n_rows, n_cols) if cells[n] == UNKNOWN)
            if around:
                unknown_around[index] = around
    clues_touching = {}
    for clue, around in unknown_around.items():
        for cell in around:
            clues_touching.setdefault(cell, []).append(clue)
    regions, assigned = [], set()
    for start in sorted(clues_touching):
        if start in assigned:
            continue
        region_cells, region_clues, stack = set(), set(), [start]
        while stack:
            cell = stack.pop()
            if cell in region_cells:
                continue
            region_cells.add(cell)
            for clue in clues_touching[cell]:
                if clue not in region_clues:
                    region_clues.add(clue)
                    stack.extend(unknown_around[clue])
        assigned |= region_cells
        regions.append(Region(tuple(sorted(region_clues)), tuple(sorted(region_cells))))
    return tuple(regions)


def _clue_clauses(observation: Observation, clue_indices, variables: dict[int, int], pool: IDPool):
    """Clauses saying each clue sees exactly its remaining mines among its unknown neighbours."""
    clauses = []
    for index in clue_indices:
        neighbours = _neighbours(index, observation.n_rows, observation.n_cols)
        neighbouring_flags = sum(observation.cells[neighbour] == FLAGGED for neighbour in neighbours)
        neighbouring_unknowns = [
            variables[neighbour]
            for neighbour in neighbours
            if observation.cells[neighbour] == UNKNOWN
        ]
        required_mines = observation.cells[index] - neighbouring_flags
        if not 0 <= required_mines <= len(neighbouring_unknowns):
            raise InconsistentObservation("A revealed clue contradicts the visible board")
        clauses.extend(CardEnc.equals(
            lits=neighbouring_unknowns,
            bound=required_mines,
            vpool=pool,
            encoding=EncType.seqcounter,
        ).clauses)
    return clauses


def _constraint_model(observation: Observation) -> tuple[Glucose3, dict[int, int]]:
    """The whole board: every clue plus the number of mines still unflagged."""
    unknowns = [index for index, state in enumerate(observation.cells) if state == UNKNOWN]
    variables = {index: position + 1 for position, index in enumerate(unknowns)}
    flagged_count = observation.cells.count(FLAGGED)
    remaining_mines = observation.total_mines - flagged_count
    if not 0 <= remaining_mines <= len(unknowns):
        raise InconsistentObservation("Flags and total mine count are inconsistent")

    pool = IDPool(start_from=len(unknowns) + 1)
    clauses = CardEnc.equals(
        lits=list(variables.values()),
        bound=remaining_mines,
        vpool=pool,
        encoding=EncType.seqcounter,
    ).clauses
    clue_indices = [index for index, state in enumerate(observation.cells) if state >= 0]
    clauses.extend(_clue_clauses(observation, clue_indices, variables, pool))

    solver = Glucose3(bootstrap_with=clauses)
    if not solver.solve():
        solver.delete()
        raise InconsistentObservation("No mine placement satisfies the visible board")
    return solver, variables


def _region_model(observation: Observation, region: Region):
    """One region's clues alone, with counters for how many of its cells are mines or safe."""
    variables = {cell: position + 1 for position, cell in enumerate(region.cells)}
    pool = IDPool(start_from=len(variables) + 1)
    clauses = _clue_clauses(observation, region.clues, variables, pool)
    literals = list(variables.values())
    mines = ITotalizer(lits=literals, ubound=len(literals), top_id=pool.top)
    safes = ITotalizer(lits=[-literal for literal in literals], ubound=len(literals), top_id=mines.top_id)
    solver = Glucose3(bootstrap_with=clauses + mines.cnf.clauses + safes.cnf.clauses)
    return solver, variables, mines, safes


def _mine_range(solver: Glucose3, size: int, mines: ITotalizer, safes: ITotalizer) -> tuple[int, int]:
    """Fewest and most mines a region's clues allow.

    A totaliser only forces its outputs upwards, so "at most k mines" is asked
    as an assumption on the mine counter and "at least k mines" as "at most
    size - k safe cells" on the counter over the negated cells.
    """
    def at_most(counter, bound):
        return solver.solve() if bound >= size else solver.solve(assumptions=[-counter.rhs[bound]])

    def smallest_bound(counter):
        low, high = 0, size
        while low < high:
            middle = (low + high) // 2
            if at_most(counter, middle):
                high = middle
            else:
                low = middle + 1
        return low

    return smallest_bound(mines), size - smallest_bound(safes)


def _forced_kind(solver: Glucose3, variable: int):
    """'reveal' if the cell cannot be a mine, 'flag' if it cannot be safe, else None."""
    if not solver.solve(assumptions=[variable]):
        return "reveal"
    if not solver.solve(assumptions=[-variable]):
        return "flag"
    return None


@dataclass(frozen=True)
class _SolvedRegion:
    forced: tuple[tuple[Coordinate, str], ...]  # (place in the canonical form, kind)
    fewest: int
    most: int


class RegionCache:
    """Regions already solved, keyed by the D4 canonical form of what the player sees there.

    A region's forced moves and its fewest and most mines depend only on its clues (the
    mines each still has to find) and its covered cells, placed relative to one another.
    Rotating, reflecting or moving a region moves the answers with it, so a region seen
    again, in the same position or a later game and in any orientation, is not solved
    again. Keep one cache across positions and games to reuse it.
    """

    def __init__(self) -> None:
        self._solved: dict[tuple, _SolvedRegion] = {}
        self._counts: dict[tuple, dict] = {}  # layout counts for guessing (exact_symmetry.minesweeper.probability)
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._solved)


def _region_picture(observation: Observation, region: Region):
    """The region as (row, column, label): a clue is labelled with the mines it still has
    to find, a covered cell with UNKNOWN."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    points = [
        (*divmod(index, n_cols), cells[index] - sum(cells[n] == FLAGGED for n in _neighbours(index, n_rows, n_cols)))
        for index in region.clues
    ]
    points += [(*divmod(index, n_cols), UNKNOWN) for index in region.cells]
    return points


def _solve_region(observation: Observation, region: Region, cache: RegionCache):
    """A region's forced moves, as (cell index, kind) in board order, and its mine range.

    The region is looked up by its canonical form. On a miss it is solved by SAT on its
    own clues, testing one cell in each orbit of the region's own symmetries, since cells
    a symmetry swaps are forced alike; the answer is stored by place in the canonical form.
    A region no mine layout fits raises InconsistentObservation and is not stored.
    """
    form, placements = canonical_form(_region_picture(observation, region))
    n_cols = observation.n_cols
    cell_at = {placements[0][divmod(index, n_cols)]: index for index in region.cells}
    solved = cache._solved.get(form)
    if solved is None:
        orbits = {frozenset(placement[divmod(index, n_cols)] for placement in placements)
                  for index in region.cells}
        solver, variables, mines, safes = _region_model(observation, region)
        try:
            if not solver.solve():
                raise InconsistentObservation("No mine placement satisfies the visible board")
            forced = {}
            for orbit in sorted(orbits, key=min):
                kind = _forced_kind(solver, variables[cell_at[min(orbit)]])
                if kind:
                    forced.update((place, kind) for place in orbit)
            fewest, most = _mine_range(solver, len(variables), mines, safes)
        finally:
            solver.delete()
        solved = _SolvedRegion(tuple(sorted(forced.items())), fewest, most)
        cache._solved[form] = solved
        cache.misses += 1
    else:
        cache.hits += 1
    moves = sorted((cell_at[place], kind) for place, kind in solved.forced)
    return moves, solved.fewest, solved.most


def _forced_orbits(observation: Observation, solver: Glucose3, variables: dict[int, int], skip=frozenset()):
    transforms = state_stabilizer(observation.cells, observation.n_rows, observation.n_cols)
    constrained = {
        neighbour
        for index, clue in enumerate(observation.cells)
        if clue >= 0
        for neighbour in _neighbours(index, observation.n_rows, observation.n_cols)
        if neighbour in variables
    }
    orbits = list(cell_orbits(constrained, transforms))
    unconstrained = tuple(sorted(set(variables) - constrained))
    if unconstrained:
        orbits.append(unconstrained)
    for orbit in sorted(orbits, key=lambda members: members[0]):
        if orbit[0] in skip:
            continue
        variable = variables[orbit[0]]
        can_be_mine = solver.solve(assumptions=[variable])
        can_be_safe = solver.solve(assumptions=[-variable])
        if not can_be_mine:
            yield orbit, "reveal"
        elif not can_be_safe:
            yield orbit, "flag"


def _check_clues_and_count(observation: Observation) -> int:
    """Reject a board whose flags exceed the mine count, or a clue its flags and covered
    neighbours cannot satisfy, including clues with no covered neighbour. Returns the
    number of mines not yet flagged."""
    remaining_mines = observation.total_mines - observation.cells.count(FLAGGED)
    if not 0 <= remaining_mines <= observation.cells.count(UNKNOWN):
        raise InconsistentObservation("Flags and total mine count are inconsistent")
    for index, state in enumerate(observation.cells):
        if state >= 0:
            around = [observation.cells[n] for n in _neighbours(index, observation.n_rows, observation.n_cols)]
            if not 0 <= state - around.count(FLAGGED) <= around.count(UNKNOWN):
                raise InconsistentObservation("A revealed clue contradicts the visible board")
    return remaining_mines


def _forced_groups(observation: Observation, cache: RegionCache):
    """Yield (cells, kind) for cells on which every consistent mine layout agrees.

    Every region is solved on its own clues first, or taken from the cache, before
    anything is yielded, so a board no layout fits is rejected whichever move is
    asked for. The total number of mines can add deductions only when it binds,
    and that is decided exactly from the fewest and most mines each region
    allows: if every combination of region layouts leaves between none and all
    of the unconstrained cells as mines, the count never binds, the board is
    consistent and the region deductions are complete. Only otherwise is the
    whole board built, and the cells still undecided are tested against it,
    clues and mine count together.
    """
    unknowns = [index for index, state in enumerate(observation.cells) if state == UNKNOWN]
    remaining_mines = _check_clues_and_count(observation)
    regions = frontier_regions(observation)
    solved = [_solve_region(observation, region, cache) for region in regions]
    in_regions = {cell for region in regions for cell in region.cells}
    unconstrained = tuple(index for index in unknowns if index not in in_regions)
    leftover_low = remaining_mines - sum(most for _, _, most in solved)
    leftover_high = remaining_mines - sum(fewest for _, fewest, _ in solved)
    count_binds = not (leftover_low >= 0 and leftover_high <= len(unconstrained))
    # The count may bind: build the whole board before yielding anything, as it also rejects inconsistent boards.
    board_solver, variables = _constraint_model(observation) if count_binds else (None, None)
    try:
        decided = set()
        for moves, _, _ in solved:
            for cell, kind in moves:
                decided.add(cell)
                yield (cell,), kind
        if count_binds:
            yield from _forced_orbits(observation, board_solver, variables, skip=decided)
        elif unconstrained and leftover_high == 0:
            yield unconstrained, "reveal"
        elif unconstrained and leftover_low == len(unconstrained):
            yield unconstrained, "flag"
    finally:
        if board_solver is not None:
            board_solver.delete()


def deduce_all(observation: Observation, cache: RegionCache | None = None) -> tuple[Action, ...]:
    """Every forced move. Pass a RegionCache kept across calls to reuse solved regions."""
    actions = {}
    with closing(_forced_groups(observation, RegionCache() if cache is None else cache)) as groups:
        for cells, kind in groups:
            for index in cells:
                actions[index] = Action(kind, divmod(index, observation.n_cols), "forced")
    return tuple(sorted(actions.values(), key=lambda action: action.cell))


def choose_action(observation: Observation, cache: RegionCache | None = None) -> Action | None:
    """The first forced move, else a guess: the covered cell least likely to be a mine.

    Pass a RegionCache kept across calls to reuse solved regions. Ties go to the lowest
    cell index. If a region has too many layouts to count, the guess is a cell no clue
    touches, or failing that the lowest covered cell.
    """
    from exact_symmetry.minesweeper.probability import TooManyLayouts, mine_probabilities  # it builds on this module

    cache = RegionCache() if cache is None else cache
    with closing(_forced_groups(observation, cache)) as groups:
        for cells, kind in groups:
            return Action(kind, divmod(cells[0], observation.n_cols), "forced")
    unknowns = [index for index, state in enumerate(observation.cells) if state == UNKNOWN]
    if not unknowns:
        return None
    try:
        probabilities = mine_probabilities(observation, cache)
        cell = min(unknowns, key=lambda index: (probabilities[index], index))
    except TooManyLayouts:
        in_regions = {index for region in frontier_regions(observation) for index in region.cells}
        cell = min((index for index in unknowns if index not in in_regions), default=min(unknowns))
    return Action("reveal", divmod(cell, observation.n_cols), "guess")
