"""Score trained networks again from their checkpoints with the current evaluation code, keeping
how each was trained; rewrite results/<tag>.json.

    python scripts/rescore.py                       # every result file of the grid
    python scripts/rescore.py --tags plain_8x8x10_n1000_seed0 p4m8_8x8x10_n1000_seed0
    python scripts/rescore.py --shard 0 --shards 3  # a third of them, for parallel processes

A result file written by scripts/train.py holds how the network was trained ("training") and how
it scores ("test positions"; for plain and augment the same averaged over the board's
symmetries; for canonical the same averaged over the symmetries that reach the canonical form).
This script loads the network from checkpoints/<tag>.pt, scores it again with
exact_symmetry.evaluate.score, on the CPU, and replaces the scoring parts; "training" is kept as
it was. "scored with" records the code that did the scoring.
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from exact_symmetry.arms import AveragedNet, CanonicalFrameNet, make_model  # noqa: E402
from exact_symmetry.data import PositionSet  # noqa: E402
from exact_symmetry.evaluate import score  # noqa: E402
from exact_symmetry.provenance import code_version  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="8x8x10")
    p.add_argument("--tags", nargs="*", help="result names without .json (default: every <arm>_<size>_n*_seed* file)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()

    pattern = re.compile(rf"^(.+)_{args.size}_n(\d+)_seed(\d+)$")
    tags = args.tags or sorted(Path(f).stem for f in glob.glob(str(ROOT / "results" / f"*_{args.size}_n*_seed*.json")))
    tags = [t for t in tags if pattern.match(t)][args.shard::args.shards]
    test_set = PositionSet.load(ROOT / "data" / f"test_{args.size}_seed0.npz")
    version = code_version(ROOT)

    for tag in tags:
        path = ROOT / "results" / f"{tag}.json"
        result = json.loads(path.read_text())
        training = result["training"]
        arm = "p4m" if result["arm"].startswith("p4m") else result["arm"]
        model = make_model(arm, training["width"], training["blocks"], training["group width"] or 8)
        model.load_state_dict(torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu"))
        model.eval()
        rescored = {"board": result["board"], "arm": result["arm"], "training": training, "scored with": version,
                    **score(model, test_set)}
        if arm in ("plain", "augment"):
            rescored["averaged over the board's symmetries"] = score(AveragedNet(model), test_set)
        if arm == "canonical":
            rescored["averaged over the symmetries that reach the canonical form"] = score(CanonicalFrameNet(model.net), test_set)
        partial = path.with_name(path.name + ".partial")  # written whole, then renamed
        partial.write_text(json.dumps(rescored, indent=1) + "\n")
        os.replace(partial, path)
        kl = rescored["test positions"]["KL per covered cell"]["mean"]
        print(f"{tag}: KL {kl:.5f}", flush=True)


if __name__ == "__main__":
    main()
