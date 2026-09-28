"""The trained models against the symmetry; write results/symmetry_analysis_<size>.json.

    python scripts/symmetry_analysis.py

For every checkpoint of the grid (checkpoints/<arm>_8x8x10_n<positions>_seed<seed>.pt), for the
plain and augment models also averaged over the board's symmetries, and for the canonical models
also averaged over the symmetries that reach each board's canonical form ("canonical+frame",
exact_symmetry.arms.CanonicalFrameNet):

- Equivariance error: on the first --positions test positions, the mean over positions,
  over the board's non-identity symmetries g and over covered cells of |f(g x) - g f(x)|,
  in probability, and its maximum. Zero for a model that turns with the board.
- Symmetric positions: positions with a symmetry of their own (a stabiliser larger than the
  identity). There the exact probabilities are equal across each orbit of the stabiliser
  (if h x = x, an equivariant map gives f(x) = f(h x) = h f(x); tests/test_data.py checks the
  labels). Reported: the mean and the largest spread (largest minus smallest predicted
  probability) within an orbit of two or more covered cells, and the KL per covered cell at
  these positions (from the logits, as on the test set).
  The set is built once from games on the "test" board stream with seed --symmetric-seed,
  which the test set (seed 0) does not use, and saved to data/symmetric_<size>_seed<s>_n<count>.npz.
- Overlap: the share of symmetric positions that also appear, in some orientation, among the
  first n training positions, for n = 1,000, 10,000 and 100,000. Symmetric positions are
  mostly early and simple, so many are also training positions; read their spreads with that
  in mind.
Everything is scored on the CPU by default.
"""

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from exact_symmetry.arms import AveragedNet, CanonicalFrameNet, act, elements, make_model  # noqa: E402
from exact_symmetry.data import PositionSet, as_seen, board_rng, exact_probabilities, play_game  # noqa: E402
from exact_symmetry.evaluate import kl_from_logits  # noqa: E402
from exact_symmetry.minesweeper.deductive import UNKNOWN  # noqa: E402
from exact_symmetry.minesweeper.symmetry import cell_orbits, state_stabilizer  # noqa: E402
from exact_symmetry.net import pick_device, predict, predict_logits, sigmoid  # noqa: E402

ANALYSED = ("plain", "augment", "canonical", "p4m8")


def symmetric_positions(n_rows, n_cols, num_mines, count, seed, max_games=200_000):
    """The first `count` distinct positions with a symmetry of their own met in games on the
    test stream with this seed (exact play with random safe reveals, as for every data set)."""
    rng = board_rng("test", n_rows, n_cols, num_mines, seed)
    cells, probability, mines, games, seen = [], [], [], [], set()
    for game in range(max_games):
        played = play_game(rng, n_rows, n_cols, num_mines, epsilon=0.2)
        for visible in played.positions:
            board = as_seen(visible)
            if board in seen or len(state_stabilizer(board, n_rows, n_cols)) == 1:
                continue
            seen.add(board)
            layout = np.zeros(n_rows * n_cols, dtype=bool)
            layout[list(played.mines)] = True
            cells.append(board)
            probability.append(exact_probabilities(board, n_rows, n_cols, num_mines))
            mines.append(layout)
            games.append(game)
            if len(cells) == count:
                break
        if len(cells) == count:
            break
    meta = {"rows": n_rows, "cols": n_cols, "mines": num_mines, "split": "test", "seed": seed, "epsilon": 0.2,
            "positions": len(cells), "games": game + 1, "kind": "positions with a symmetry of their own"}
    return PositionSet(n_rows, n_cols, num_mines, np.array(cells, dtype=np.int8).reshape(-1, n_rows * n_cols),
                       np.array(probability, dtype=np.float32).reshape(-1, n_rows * n_cols),
                       np.array(mines, dtype=bool).reshape(-1, n_rows * n_cols), np.array(games, dtype=np.int32), meta)


def overlap_with_training(symmetric, train, sizes):
    """Share of symmetric positions that appear, in some orientation, among the first n training positions."""
    group = elements(symmetric.n_rows, symmetric.n_cols)
    boards = torch.as_tensor(symmetric.cells, dtype=torch.long).view(-1, symmetric.n_rows, symmetric.n_cols)
    images = [{act(boards[i], g).numpy().astype(np.int8).tobytes() for g in group} for i in range(len(boards))]
    shares = {}
    for n in sizes:
        known = {row.tobytes() for row in train.cells[:n]}
        shares[str(n)] = float(np.mean([bool(image & known) for image in images]))
    return shares


def equivariance_error(model, data, device):
    """Mean and largest |f(g x) - g f(x)| over covered cells and non-identity symmetries."""
    n_rows, n_cols = data.n_rows, data.n_cols
    cells = torch.as_tensor(data.cells, dtype=torch.long).view(-1, n_rows, n_cols)
    base = torch.as_tensor(predict(model, data.cells, n_rows, n_cols, data.num_mines, device)).view_as(cells)
    errors = []
    for g in elements(n_rows, n_cols)[1:]:
        turned = act(cells, g)
        out = torch.as_tensor(predict(model, turned.flatten(1).numpy(), n_rows, n_cols, data.num_mines, device))
        difference = (out.view_as(cells) - act(base, g)).abs()
        errors.append(difference[act(cells, g) == UNKNOWN])
    errors = torch.cat(errors)
    return {"mean": float(errors.mean()), "max": float(errors.max())}


def orbit_spread(predicted, data):
    """Spreads of the predicted probabilities within each orbit (two or more covered cells) of
    each position's stabiliser, and the same for the exact labels (which should be 0)."""
    spreads, exact_spreads = [], []
    for board, prediction, exact in zip(data.cells, predicted, data.probability):
        board = tuple(int(c) for c in board)
        stabiliser = state_stabilizer(board, data.n_rows, data.n_cols)
        covered = [i for i, state in enumerate(board) if state == UNKNOWN]
        for orbit in cell_orbits(covered, stabiliser):
            if len(orbit) > 1:
                spreads.append(float(prediction[list(orbit)].max() - prediction[list(orbit)].min()))
                exact_spreads.append(float(exact[list(orbit)].max() - exact[list(orbit)].min()))
    return spreads, exact_spreads


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="8x8x10")
    p.add_argument("--positions", type=int, default=2000, help="test positions for the equivariance error")
    p.add_argument("--symmetric", type=int, default=400, help="positions with a symmetry of their own")
    p.add_argument("--symmetric-seed", type=int, default=1)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--device", default="cpu", help="scoring device")
    args = p.parse_args()
    n_rows, n_cols, num_mines = (int(v) for v in args.size.split("x"))
    device = str(pick_device(args.device))

    full_test = PositionSet.load(ROOT / "data" / f"test_{args.size}_seed0.npz")
    test = PositionSet(n_rows, n_cols, num_mines, full_test.cells[:args.positions], full_test.probability[:args.positions],
                       full_test.mines[:args.positions], full_test.game[:args.positions], full_test.meta)
    path = ROOT / "data" / f"symmetric_{args.size}_seed{args.symmetric_seed}_n{args.symmetric}.npz"
    if path.exists():
        symmetric = PositionSet.load(path)
    else:
        start = time.time()
        symmetric = symmetric_positions(n_rows, n_cols, num_mines, args.symmetric, args.symmetric_seed)
        symmetric.save(path)
        print(f"built {len(symmetric)} symmetric positions from {symmetric.meta['games']} games "
              f"in {time.time() - start:.0f}s", flush=True)
    train_set = PositionSet.load(ROOT / "data" / f"train_{args.size}_seed0.npz")
    _, exact_spreads = orbit_spread(symmetric.probability, symmetric)
    stabiliser_sizes = [len(state_stabilizer(tuple(int(c) for c in b), n_rows, n_cols)) for b in symmetric.cells]

    result = {"board": args.size, "device": device, "test positions for the equivariance error": len(test),
              "symmetric positions": len(symmetric), "symmetric set": symmetric.meta,
              "symmetric positions also among the first n training positions, in some orientation":
                  overlap_with_training(symmetric, train_set, [1000, 10000, 100000]),
              "stabiliser sizes": {str(k): stabiliser_sizes.count(k) for k in sorted(set(stabiliser_sizes))},
              "orbits of two or more covered cells": len(exact_spreads),
              "largest spread of the exact probabilities within an orbit": max(exact_spreads) if exact_spreads else None,
              "models": {}}
    pattern = re.compile(rf"(.+)_{args.size}_n(\d+)_seed(\d+)\.pt$")
    for checkpoint in sorted(glob.glob(str(ROOT / "checkpoints" / f"*_{args.size}_n*_seed*.pt"))):
        match = pattern.search(Path(checkpoint).name)
        if not match or match.group(1) not in ANALYSED:
            continue
        name, positions, seed = match.group(1), int(match.group(2)), int(match.group(3))
        arm = "p4m" if name.startswith("p4m") else name
        group_width = int(name[3:]) if arm == "p4m" else 8
        model = make_model(arm, args.width, args.blocks, group_width)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model = model.to(device).eval()
        variants = [(name, model)] + ([(f"{name}+avg", AveragedNet(model))] if arm in ("plain", "augment") else [])
        if arm == "canonical":  # the canonical form averaged over the symmetries that reach it
            variants.append((f"{name}+frame", CanonicalFrameNet(model.net)))
        covered = symmetric.cells == UNKNOWN
        for label, scored in variants:
            logits = predict_logits(scored, symmetric.cells, n_rows, n_cols, num_mines, device)
            spreads, _ = orbit_spread(sigmoid(logits), symmetric)
            entry = {
                "equivariance error": equivariance_error(scored, test, device),
                "orbit spread at symmetric positions": {"mean": float(np.mean(spreads)), "max": float(np.max(spreads))},
                "KL per covered cell at symmetric positions":
                    float(kl_from_logits(logits[covered], symmetric.probability[covered]).mean()),
            }
            result["models"].setdefault(label, {}).setdefault(str(positions), {})[str(seed)] = entry
            print(f"{label:16s} n={positions:<6d} seed {seed}: equivariance error {entry['equivariance error']['mean']:.2e}"
                  f"  orbit spread {entry['orbit spread at symmetric positions']['mean']:.2e}"
                  f"  KL symmetric {entry['KL per covered cell at symmetric positions']:.4f}", flush=True)

    out = ROOT / "results" / f"symmetry_analysis_{args.size}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
