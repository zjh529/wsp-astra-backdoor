"""Phase-1 commands. Scoring imports are loaded only for those commands."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _torch_device():
    from cnn_phase1.device import pick_device

    return pick_device()


def _add_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--holdout-size", type=int, default=2000)
    parser.add_argument("--position", default="top_left")
    parser.add_argument("--shape", default="square")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--pixels", default=None, help="row,col;row,col for shape=scattered")
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--poison-rate", type=float, default=0.08)
    parser.add_argument("--resume", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cnn_phase1")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train", help="Train seed-matched pairs and one untrained init per seed")
    _add_train(train)
    cases = sub.add_parser("make-cases", help="Write blind cases and a private ground-truth map")
    cases.add_argument("--train-out", type=Path, required=True)
    cases.add_argument("--blind", type=Path, required=True)
    cases.add_argument("--private", type=Path, required=True)
    cases.add_argument("--case-order-seed", type=int, default=None)

    audit = sub.add_parser("audit", help="Run the blind queue from the first case")
    resume = sub.add_parser("resume", help="Continue cases that have no frozen prediction")
    for command in (audit, resume):
        command.add_argument("--blind", type=Path, required=True)
        command.add_argument("--sandbox-python", type=Path, required=True)
        command.add_argument("--pi-bin", default=os.environ.get("PI_BIN", "pi-online"))
        command.add_argument("--pi-cwd", type=Path, default=Path("/dev/shm/cnn-p1-pi"))
        command.add_argument("--case", default=None)

    evaluate = sub.add_parser("evaluate", help="Score frozen predictions")
    evaluate.add_argument("--blind", type=Path, required=True)
    evaluate.add_argument("--private", type=Path, required=True)
    evaluate.add_argument("--train-out", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--accept-unevaluable", action="store_true")

    baselines = sub.add_parser("baselines", help="Weight statistics, SVD, random ablation, Neural Cleanse")
    baselines.add_argument("--train-out", type=Path, required=True)
    baselines.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    baselines.add_argument("--out", type=Path, required=True)
    baselines.add_argument("--nc-steps", type=int, default=None)

    aggregate = sub.add_parser("aggregate", help="Rates, intervals, and Holm tests")
    aggregate.add_argument("--eval", type=Path, required=True)
    aggregate.add_argument("--baselines", type=Path, default=None)
    aggregate.add_argument("--out", type=Path, required=True)

    train2 = sub.add_parser("train-phase2", help="Train phase-2 arms and reuse phase-1 clean models")
    train2.add_argument("--out", type=Path, required=True)
    train2.add_argument("--phase1-train", type=Path, required=True)
    train2.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    train2.add_argument("--resume", action="store_true")

    cases2 = sub.add_parser("make-cases-phase2", help="Blind cases for phase 2, excluding training-gate failures")
    cases2.add_argument("--train-out", type=Path, required=True)
    cases2.add_argument("--blind", type=Path, required=True)
    cases2.add_argument("--private", type=Path, required=True)
    cases2.add_argument("--case-order-seed", type=int, default=None)

    aggregate2 = sub.add_parser("aggregate-phase2", help="Per-arm rates and the anchor Holm tests")
    aggregate2.add_argument("--eval", type=Path, required=True)
    aggregate2.add_argument("--anchor", type=Path, required=True)
    aggregate2.add_argument("--baselines", type=Path, default=None)
    aggregate2.add_argument("--out", type=Path, required=True)

    preflight = sub.add_parser("preflight", help="Check bwrap, numpy-only python, and the prompt hash")
    preflight.add_argument("--sandbox-python", type=Path, required=True)
    sub.add_parser("prompt-sha", help="Print the frozen prompt sha256")

    args = parser.parse_args(argv)
    if args.cmd == "prompt-sha":
        from cnn_phase1.protocol import prompt_sha256

        print(prompt_sha256())
        return 0
    if args.cmd == "preflight":
        from cnn_phase1.preflight import main_report

        return main_report(args.sandbox_python)
    if args.cmd == "train":
        from cnn_phase1.data_mnist import load_split
        from cnn_phase1.spec import make_trigger, parse_pixels, parse_seeds, Recipe
        from cnn_phase1.train import train_study

        pixels = parse_pixels(args.pixels) if args.pixels else None
        trigger = make_trigger(
            position=args.position,
            shape=args.shape,
            size=args.size if args.shape == "square" else None,
            height=args.height,
            width=args.width,
            pixels=pixels,
            target_class=args.target_class,
            poison_rate=args.poison_rate,
        )
        recipe = Recipe(
            trigger=trigger,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            holdout_size=args.holdout_size,
            seeds=parse_seeds(args.seeds),
        )
        images, labels = load_split(args.data_dir, train=True)
        test_images, test_labels = load_split(args.data_dir, train=False)
        manifest = train_study(
            recipe=recipe,
            train_out=args.out,
            images=images,
            labels=labels,
            test_images=test_images,
            test_labels=test_labels,
            device=_torch_device(),
            resume=args.resume,
        )
        return 2 if manifest["n_failures"] else 0
    if args.cmd == "make-cases":
        from cnn_phase1.cases import make_cases

        make_cases(
            train_out=args.train_out,
            blind=args.blind,
            private=args.private,
            case_order_seed=args.case_order_seed,
        )
        return 0
    if args.cmd in {"audit", "resume"}:
        from cnn_phase1.loop import load_study_config
        from cnn_phase1.queue import run_queue

        code, _rows = run_queue(
            config=load_study_config(),
            blind=args.blind,
            sandbox_python=args.sandbox_python,
            pi_bin=args.pi_bin,
            pi_cwd=args.pi_cwd,
            resume=args.cmd == "resume",
            case_id=args.case,
        )
        return code
    if args.cmd == "evaluate":
        from cnn_phase1.data_mnist import load_split
        from cnn_phase1.evaluate import evaluate_study

        _images, _labels = load_split(args.data_dir, train=False)
        test_images, test_labels = _images, _labels
        evaluate_study(
            blind=args.blind,
            private=args.private,
            train_out=args.train_out,
            test_images=test_images,
            test_labels=test_labels,
            out=args.out,
            device=_torch_device(),
            accept_unevaluable=args.accept_unevaluable,
        )
        return 0
    if args.cmd == "baselines":
        from cnn_phase1.baselines import NC_STEPS, run_baselines
        from cnn_phase1.data_mnist import load_split

        images, labels = load_split(args.data_dir, train=True)
        test_images, test_labels = load_split(args.data_dir, train=False)
        run_baselines(
            train_out=args.train_out,
            images=images,
            labels=labels,
            test_images=test_images,
            test_labels=test_labels,
            out=args.out,
            device=_torch_device(),
            nc_steps=NC_STEPS if args.nc_steps is None else args.nc_steps,
        )
        return 0
    if args.cmd == "aggregate":
        from cnn_phase1.aggregate import aggregate

        aggregate(eval_path=args.eval, baselines_path=args.baselines, out=args.out)
        return 0
    if args.cmd == "train-phase2":
        from cnn_phase1.data_mnist import load_split
        from cnn_phase1.train_phase2 import train_phase2

        images, labels = load_split(args.data_dir, train=True)
        test_images, test_labels = load_split(args.data_dir, train=False)
        manifest = train_phase2(
            train_out=args.out,
            phase1_train=args.phase1_train,
            images=images,
            labels=labels,
            test_images=test_images,
            test_labels=test_labels,
            device=_torch_device(),
            resume=args.resume,
        )
        return 2 if manifest["n_failures"] else 0
    if args.cmd == "make-cases-phase2":
        from cnn_phase1.cases import make_phase2_cases

        make_phase2_cases(
            train_out=args.train_out,
            blind=args.blind,
            private=args.private,
            case_order_seed=args.case_order_seed,
        )
        return 0
    if args.cmd == "aggregate-phase2":
        from cnn_phase1.aggregate_phase2 import aggregate_phase2

        aggregate_phase2(
            eval_path=args.eval,
            anchor_path=args.anchor,
            baselines_path=args.baselines,
            out=args.out,
        )
        return 0
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
