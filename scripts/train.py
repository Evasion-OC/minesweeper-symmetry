"""Train one symmetry arm of the learned mine-probability model (exact_symmetry.net,
exact_symmetry.arms) and score it (exact_symmetry.evaluate); write
results/<arm>_<rows>x<cols>x<mines>_n<positions>_seed<seed>.json and the weights to checkpoints/.

    python scripts/make_data.py --split train --positions 100000
    python scripts/make_data.py --split validation --positions 10000
    python scripts/make_data.py --split test --positions 10000
    python scripts/train.py --positions 100000 --seed 0
    python scripts/train.py --arm p4m --positions 1000 --seed 3

Training uses the first --positions positions of the training set; the weights kept are
those with the lowest KL on the validation set. Scores: on the test set's positions. The
plain and augment arms are also scored averaged over the board's symmetries at test time, and
the canonical arm averaged over the symmetries that reach each board's canonical form.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from exact_symmetry.arms import ARMS, AveragedNet, CanonicalFrameNet, multiply_adds  # noqa: E402
from exact_symmetry.data import PositionSet  # noqa: E402
from exact_symmetry.evaluate import score  # noqa: E402
from exact_symmetry.net import pick_device, train  # noqa: E402
from exact_symmetry.provenance import code_version  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--data-seed", type=int, default=0, help="seed of the data files to read")
    p.add_argument("--arm", choices=ARMS, default="plain")
    p.add_argument("--group-width", type=int, default=8, help="p4m channels per group element")
    p.add_argument("--positions", type=int, default=None, help="training positions to use (default: all)")
    p.add_argument("--seed", type=int, default=0, help="seed of the network's weights and batches")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", default=None, help="default: CUDA, then Apple MPS, then CPU")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    size = f"{args.rows}x{args.cols}x{args.mines}"
    load = lambda split: PositionSet.load(ROOT / "data" / f"{split}_{size}_seed{args.data_seed}.npz")  # noqa: E731
    train_set, validation_set, test_set = load("train"), load("validation"), load("test")
    device = pick_device(args.device)
    model, log = train(train_set, validation_set, positions=args.positions, steps=args.steps,
                       batch_size=args.batch_size, learning_rate=args.lr, width=args.width, blocks=args.blocks,
                       seed=args.seed, eval_every=args.eval_every, device=str(device), arm=args.arm,
                       group_width=args.group_width)
    arm_name = f"p4m{args.group_width}" if args.arm == "p4m" else args.arm
    tag = f"{arm_name}_{size}_n{log.positions}_seed{args.seed}"
    checkpoints = ROOT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoints / f"{tag}.pt")

    model = model.to("cpu").eval()
    result = {
        "board": f"{args.rows}x{args.cols}/{args.mines}",
        "arm": arm_name,
        "training": {"positions": log.positions, "data seed": args.data_seed, "seed": args.seed,
                     "steps": args.steps, "batch size": args.batch_size, "learning rate": args.lr,
                     "width": args.width, "blocks": args.blocks, "eval every": args.eval_every,
                     "code": code_version(ROOT),
                     "group width": args.group_width if args.arm == "p4m" else None,
                     "parameters": sum(t.numel() for t in model.parameters()),
                     "multiply-adds per board": multiply_adds(model, args.rows, args.cols),
                     "device": str(device), "seconds": round(log.seconds, 1),
                     "validation KL per covered cell": [[s, k] for s, k in log.validation],
                     "kept weights from step": log.best_step,
                     "kept the first validation checkpoint": log.best_step == log.validation[0][0]},
        "scored with": code_version(ROOT),
        **score(model, test_set),  # scored on the CPU
    }
    if args.arm in ("plain", "augment"):
        result["averaged over the board's symmetries"] = score(AveragedNet(model), test_set)
    if args.arm == "canonical":
        result["averaged over the symmetries that reach the canonical form"] = score(CanonicalFrameNet(model.net), test_set)
    out = Path(args.out) if args.out else ROOT / "results" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    partial = out.with_name(out.name + ".partial")  # written whole, then renamed: readers never see half a file
    partial.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(partial, out)
    print(tag, "test KL per covered cell:", result["test positions"]["KL per covered cell"])


if __name__ == "__main__":
    main()
