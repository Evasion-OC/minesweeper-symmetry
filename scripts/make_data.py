"""Write a set of positions with their exact mine probabilities (exact_symmetry.data)
to data/<split>_<rows>x<cols>x<mines>_seed<seed>.npz.

    python scripts/make_data.py --split train --positions 100000
    python scripts/make_data.py --split validation --positions 10000
    python scripts/make_data.py --split test --positions 10000

The board is 8x8 with 10 mines unless --rows, --cols and --mines say otherwise.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from exact_symmetry.data import SPLITS, generate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--positions", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.2, help="chance a move is a random safe reveal")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    n_rows, n_cols, num_mines = args.rows, args.cols, args.mines

    start = time.time()
    data = generate(n_rows, n_cols, num_mines, args.positions, args.split, args.seed, args.epsilon)
    name = f"{args.split}_{n_rows}x{n_cols}x{num_mines}_seed{args.seed}.npz"
    out = Path(args.out) if args.out else ROOT / "data" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    data.save(out)

    labels = data.probability[~np.isnan(data.probability)]
    summary = {**data.meta, "file": str(out), "seconds": round(time.time() - start, 1),
               "covered cells per position": round(float((~np.isnan(data.probability)).sum(1).mean()), 1),
               "covered cells with probability 0": round(float((labels == 0).mean()), 3),
               "covered cells with probability 1": round(float((labels == 1).mean()), 3)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
