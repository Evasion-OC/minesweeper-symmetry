"""Run the grid: every arm at every training size and seed, one scripts/train.py run each.

    python scripts/run_grid.py
    python scripts/run_grid.py --arms p4m --positions 1000

A run whose result file and checkpoint both exist is skipped, so the grid can be stopped and
started again, but only if that file was trained with the settings this grid uses (train.py's
defaults and this --eval-every); otherwise it is trained again. A fresh clone has the result
files but no checkpoints, so every run is trained. The order goes size by size and seed by
seed, with every arm in turn, so a partial grid still compares the arms.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = {"steps": 10_000, "batch size": 256, "learning rate": 1e-3, "width": 64, "blocks": 6}  # train.py's defaults


def trained_as_asked(result, eval_every):
    """Whether a result file was trained with the grid's settings and this validation interval
    (read from the file, or from the steps of its validation log when the file does not record it)."""
    training = result.get("training")
    if not training or any(training.get(k) != v for k, v in GRID.items()):
        return False
    recorded = training.get("eval every")
    if recorded is None:
        steps = [s for s, _ in training.get("validation KL per covered cell", [])]
        recorded = steps[-1] - steps[-2] if len(steps) > 1 else None
    return recorded == eval_every


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["plain", "augment", "canonical", "p4m"])
    p.add_argument("--group-width", type=int, default=8)
    p.add_argument("--positions", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--size", default="8x8x10")
    p.add_argument("--eval-every", type=int, default=250, help="validation interval passed to train.py")
    args = p.parse_args()
    rows, cols, mines = (int(v) for v in args.size.split("x"))

    for positions in args.positions:
        for seed in args.seeds:
            for arm in args.arms:
                name = f"p4m{args.group_width}" if arm == "p4m" else arm
                out = ROOT / "results" / f"{name}_{args.size}_n{positions}_seed{seed}.json"
                checkpoint = ROOT / "checkpoints" / f"{name}_{args.size}_n{positions}_seed{seed}.pt"
                if out.exists() and checkpoint.exists():
                    if trained_as_asked(json.loads(out.read_text()), args.eval_every):
                        print(f"skip {out.name}", flush=True)
                        continue
                    print(f"train again {out.name}: its settings differ from this grid's", flush=True)
                start = time.time()
                command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--arm", arm,
                           "--group-width", str(args.group_width), "--positions", str(positions),
                           "--seed", str(seed), "--rows", str(rows), "--cols", str(cols), "--mines", str(mines),
                           "--eval-every", str(args.eval_every)]
                run = subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                status = "done" if run.returncode == 0 else f"FAILED ({run.returncode})"
                print(f"{status} {out.name} {time.time() - start:.0f}s", flush=True)
                if run.returncode:
                    print(run.stderr[-3000:], flush=True)


if __name__ == "__main__":
    main()
