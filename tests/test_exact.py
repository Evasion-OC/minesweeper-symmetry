"""The copied exact core, checked against listing every layout (its full tests live in the solver repo)."""

import random

from exact_symmetry.data import play_game
from exact_symmetry.minesweeper.deductive import FLAGGED, Observation, RegionCache, choose_action, deduce_all
from exact_symmetry.minesweeper.probability import mine_probabilities
from exact_symmetry.minesweeper.symmetry import grid_transforms
from helpers import counted, transformed


def positions(args, games, seed, epsilon=0.3):
    rng = random.Random(seed)
    for _ in range(games):
        for visible in play_game(rng, *args, epsilon=epsilon).positions:
            yield Observation(*args, visible)


def test_probabilities_equal_the_share_of_layouts_and_certain_cells_are_the_forced_moves():
    checked = 0
    for args in [(4, 4, 3), (5, 5, 4), (4, 6, 5)]:
        for observation in positions(args, 25, 1):
            found = mine_probabilities(observation)
            assert found == counted(observation)
            certain = {("reveal", i) for i, p in found.items() if p == 0} | {("flag", i) for i, p in found.items() if p == 1}
            assert certain == {(a.kind, a.cell[0] * args[1] + a.cell[1]) for a in deduce_all(observation)}
            assert sum(found.values()) == args[2] - observation.cells.count(FLAGGED)
            checked += 1
    assert checked > 200


def test_probabilities_turn_with_the_board():
    for observation in positions((8, 8, 10), 5, 2):
        found = mine_probabilities(observation)
        for transform in grid_transforms(8, 8):
            assert mine_probabilities(transformed(observation, transform)) == {transform.mapping[i]: p for i, p in found.items()}


def test_the_exact_player_plays_a_forced_move_else_the_least_likely_cell():
    guesses, cache = 0, RegionCache()
    for observation in positions((8, 8, 12), 40, 3, epsilon=0.0):  # dense boards, the exact player's own games
        action = choose_action(observation, cache)
        index = action.cell[0] * 8 + action.cell[1]
        if deduce_all(observation):
            assert action.reason == "forced"
        else:
            guesses += 1
            found = mine_probabilities(observation)
            assert action.reason == "guess" and found[index] == min(found.values())
    assert guesses > 20
