# Copied from the Minesweeper solver (github.com/Evasion-OC/minesweeper-solver, minesweeper/symmetry.py,
# commit 63d178a). Only the module paths differ. It gives the exact answers this project scores against.
from dataclasses import dataclass
from typing import Callable, Iterable


Coordinate = tuple[int, int]


@dataclass(frozen=True)
class GridTransform:
    name: str
    n_rows: int
    n_cols: int
    mapping: tuple[int, ...]

    def __call__(self, row: int, column: int) -> Coordinate:
        return divmod(self.mapping[row * self.n_cols + column], self.n_cols)


def grid_transforms(n_rows: int, n_cols: int) -> tuple[GridTransform, ...]:
    if n_rows < 1 or n_cols < 1:
        raise ValueError("Board dimensions must be positive")

    candidates: list[tuple[str, Callable[[int, int], Coordinate]]] = [
        ("identity", lambda row, column: (row, column)),
        ("horizontal", lambda row, column: (row, n_cols - 1 - column)),
        ("vertical", lambda row, column: (n_rows - 1 - row, column)),
        ("half_turn", lambda row, column: (n_rows - 1 - row, n_cols - 1 - column)),
    ]
    if n_rows == n_cols:
        candidates.extend([
            ("quarter_turn", lambda row, column: (column, n_rows - 1 - row)),
            ("three_quarter_turn", lambda row, column: (n_rows - 1 - column, row)),
            ("main_diagonal", lambda row, column: (column, row)),
            ("anti_diagonal", lambda row, column: (n_rows - 1 - column, n_cols - 1 - row)),
        ])

    transforms = []
    seen = set()
    for name, transform in candidates:
        mapping = tuple(
            new_row * n_cols + new_column
            for row in range(n_rows)
            for column in range(n_cols)
            for new_row, new_column in [transform(row, column)]
        )
        if mapping not in seen:
            seen.add(mapping)
            transforms.append(GridTransform(name, n_rows, n_cols, mapping))
    return tuple(transforms)


def state_stabilizer(states: tuple[int, ...], n_rows: int, n_cols: int) -> tuple[GridTransform, ...]:
    if len(states) != n_rows * n_cols:
        raise ValueError("Visible state does not match board dimensions")
    return tuple(
        transform
        for transform in grid_transforms(n_rows, n_cols)
        if all(states[index] == states[image] for index, image in enumerate(transform.mapping))
    )


def cell_orbits(indices: Iterable[int], transforms: tuple[GridTransform, ...]) -> tuple[tuple[int, ...], ...]:
    remaining = set(indices)
    orbits = []
    while remaining:
        representative = min(remaining)
        orbit = tuple(sorted({transform.mapping[representative] for transform in transforms}))
        if not set(orbit) <= remaining:
            raise ValueError("The cell set is not preserved by the supplied transformations")
        orbits.append(orbit)
        remaining.difference_update(orbit)
    return tuple(orbits)


PLANE_TRANSFORMS: tuple[tuple[str, Callable[[int, int], Coordinate]], ...] = (
    ("identity", lambda row, column: (row, column)),
    ("quarter_turn", lambda row, column: (column, -row)),
    ("half_turn", lambda row, column: (-row, -column)),
    ("three_quarter_turn", lambda row, column: (-column, row)),
    ("horizontal", lambda row, column: (row, -column)),
    ("vertical", lambda row, column: (-row, column)),
    ("main_diagonal", lambda row, column: (column, row)),
    ("anti_diagonal", lambda row, column: (-column, -row)),
)


def canonical_form(points: Iterable[tuple[int, int, int]]):
    """Canon(S) = min over the eight rotations and reflections g of g(S), moved to the origin.

    points are (row, column, label). Two pictures have the same canonical form exactly
    when one is a rotation, reflection or translation of the other, labels included.
    Returns (form, placements): form is the sorted tuple of (row, column, label), and
    placements has one map from each point's (row, column) to its place in the form for
    every g that attains the minimum. The placements differ by the picture's own
    symmetries, so a point's places across them are its orbit under those symmetries.
    """
    points = list(points)
    best, placements = None, []
    for _, transform in PLANE_TRANSFORMS:
        moved = [transform(row, column) for row, column, _ in points]
        top = min(row for row, _ in moved)
        left = min(column for _, column in moved)
        moved = [(row - top, column - left) for row, column in moved]
        form = tuple(sorted((row, column, label) for (row, column), (_, _, label) in zip(moved, points)))
        placement = {(row, column): place for (row, column, _), place in zip(points, moved)}
        if best is None or form < best:
            best, placements = form, [placement]
        elif form == best:
            placements.append(placement)
    return best, placements
