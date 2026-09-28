"""Summarise the grid: every arm at every training size, over its seeds; write
results/summary_<size>.json and print the table.

    python scripts/summarise.py

Reads results/<arm>_<size>_n<positions>_seed<seed>.json written by scripts/train.py. The plain and
augment runs are also reported averaged over the board's symmetries at test time, as "plain+avg" and
"augment+avg", and the canonical runs averaged over the symmetries that reach each board's
canonical form, as "canonical+frame".

For each arm and size: the mean and standard deviation over seeds of each measure, and the
difference from plain at the same size with a 95% paired t interval over the seeds both
arms ran. Runs are paired by seed: at a given seed every arm draws the same training
batches, and plain, augment and canonical also start from the same weights. The interval
covers seed-to-seed variation only. It is conditional on the one training sample (the first
n positions) and the 10,000 test positions shared by every run; it leaves out the variation
from which test games were drawn. Read it as "on these positions", not as population
uncertainty.

Two kinds of failed run are flagged, and the difference from plain is then also shown without
the flagged seeds:
- collapsed: predictions that no longer depend on the board (all test cells in at most two
  calibration bins);
- floor: a network that cannot predict below some probability on any covered test cell
  (lowest prediction above FLOOR). Proved-safe cells have exact probability 0 and occur at
  almost every position, so a working network goes far below FLOOR; one whose only negative
  output weight sits on a dead channel cannot go below sigmoid(bias).
"""

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from exact_symmetry.arms import elements  # noqa: E402

MEASURES = {
    "KL per covered cell": lambda r: r["test positions"]["KL per covered cell"]["mean"],
}
LOWER_IS_BETTER = {"KL per covered cell"}


def paired(a, b):
    """Mean of a[s] - b[s] over the seeds s in both, with a 95% paired t interval."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    if len(d) == 0:
        return None
    if len(d) < 2:
        return {"difference": float(d.mean()), "low": None, "high": None, "seeds": seeds}
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return {"difference": float(d.mean()), "low": float(d.mean() - half), "high": float(d.mean() + half), "seeds": seeds}


FLOOR = 1e-3


def collapsed(result):
    return len(result["test positions"]["calibration by predicted probability"]) <= 2


def floored(result):
    lowest = result["test positions"].get("lowest predicted probability on a covered cell")
    return lowest is not None and lowest > FLOOR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="8x8x10")
    args = p.parse_args()
    n_rows, n_cols = (int(v) for v in args.size.split("x")[:2])
    group_size = len(elements(n_rows, n_cols))

    runs = {}  # (arm, positions) -> {seed: result part}
    flags = {}  # (arm, positions) -> {"collapsed": [seeds], "floor": [seeds], "kept first checkpoint": [seeds]}
    info = {}
    pattern = re.compile(rf"(.+)_{args.size}_n(\d+)_seed(\d+)\.json$")
    for path in sorted(glob.glob(str(ROOT / "results" / f"*_{args.size}_n*_seed*.json"))):
        match = pattern.search(Path(path).name)
        if not match:
            continue
        try:
            result = json.loads(Path(path).read_text())
        except json.JSONDecodeError as error:
            sys.exit(f"{path}: incomplete or corrupt JSON ({error})")
        arm, positions, seed = result["arm"], int(match.group(2)), int(match.group(3))
        runs.setdefault((arm, positions), {})[seed] = result
        flag = flags.setdefault((arm, positions), {"collapsed": [], "floor": [], "kept first checkpoint": []})
        if collapsed(result):
            flag["collapsed"].append(seed)
        if floored(result):
            flag["floor"].append(seed)
        if result["training"].get("kept the first validation checkpoint"):
            flag["kept first checkpoint"].append(seed)
        info[arm] = {"parameters": result["training"]["parameters"],
                     "multiply-adds per board": result["training"]["multiply-adds per board"]}
        averaged = result.get("averaged over the board's symmetries")
        if averaged:
            runs.setdefault((f"{arm}+avg", positions), {})[seed] = averaged
            flags[(f"{arm}+avg", positions)] = flag
            info[f"{arm}+avg"] = {"parameters": info[arm]["parameters"],
                                  "multiply-adds per board": group_size * info[arm]["multiply-adds per board"]}
        framed = result.get("averaged over the symmetries that reach the canonical form")
        if framed:  # one network pass on the canonical board, turned back and averaged
            runs.setdefault((f"{arm}+frame", positions), {})[seed] = framed
            flags[(f"{arm}+frame", positions)] = flag
            info[f"{arm}+frame"] = dict(info[arm])

    order = ["plain", "plain+avg", "augment", "augment+avg", "canonical", "canonical+frame", "p4m8"]
    arms = [a for a in order if a in info] + sorted(a for a in info if a not in order)
    sizes = sorted({positions for _, positions in runs})
    summary = {"board": args.size, "arms": {a: info[a] for a in arms},
               "interval": "95% paired t over seeds, conditional on the fixed training sample and test "
                           "positions", "flags": {f"{a} n={n}": f for (a, n), f in flags.items()},
               "measures": {}}
    for measure, read in MEASURES.items():
        table = summary["measures"].setdefault(measure, {})
        for positions in sizes:
            base = {s: read(r) for s, r in runs.get(("plain", positions), {}).items()}
            for arm in arms:
                by_seed = {s: read(r) for s, r in sorted(runs.get((arm, positions), {}).items())}
                if not by_seed:
                    continue
                values = list(by_seed.values())
                entry = {"seeds": sorted(by_seed), "mean": float(np.mean(values)),
                         "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None, "values": values}
                flag = flags.get((arm, positions), {})
                bad = sorted(set(flag.get("collapsed", [])) | set(flag.get("floor", [])))
                if flag.get("collapsed"):
                    entry["collapsed seeds"] = flag["collapsed"]
                if flag.get("floor"):
                    entry["floor seeds"] = flag["floor"]
                if arm != "plain" and base:
                    entry["minus plain"] = paired(by_seed, base)
                    if bad:
                        entry["minus plain, without flagged seeds"] = paired(
                            {s: v for s, v in by_seed.items() if s not in bad}, base)
                table.setdefault(str(positions), {})[arm] = entry

    out = ROOT / "results" / f"summary_{args.size}.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    for arm in arms:
        print(f"{arm:12s} parameters {info[arm]['parameters']:>9,}  multiply-adds per board "
              f"{info[arm]['multiply-adds per board']:>13,}")
    print("\nFlags per arm and size (seeds):")
    for (arm, positions), flag in sorted(flags.items()):
        if flag["collapsed"] or flag["floor"] or flag["kept first checkpoint"]:
            print(f"  {arm} n={positions}: collapsed {flag['collapsed']}, floor {flag['floor']}, "
                  f"kept first checkpoint {flag['kept first checkpoint']}")
    for measure, table in summary["measures"].items():
        better = "lower is better" if measure in LOWER_IS_BETTER else "higher is better"
        print(f"\n{measure} ({better}); mean ± sd over seeds [n]; minus plain, 95% paired interval over seeds")
        print(f"{'arm':12s}" + "".join(f"{'n=' + str(s):>46s}" for s in sizes))
        for arm in arms:
            cells = []
            for positions in sizes:
                entry = table.get(str(positions), {}).get(arm)
                if entry is None:
                    cells.append(f"{'-':>46s}")
                    continue
                sd = f"{entry['sd']:.4f}" if entry["sd"] is not None else "  -   "
                text = (f"{entry['mean']:.4f} ± {sd} [{len(entry['seeds'])}]"
                        + ("!" if "collapsed seeds" in entry or "floor seeds" in entry else ""))
                diff = entry.get("minus plain")
                if diff and diff["low"] is not None:
                    text += f" {diff['difference']:+.3f} ({diff['low']:+.3f},{diff['high']:+.3f})"
                cells.append(f"{text:>46s}")
            print(f"{arm:12s}" + "".join(cells))
    print("\n! = a seed collapsed or has a probability floor; see 'minus plain, without flagged seeds' in the JSON")
    print("wrote", out)


if __name__ == "__main__":
    main()
