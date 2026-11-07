from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vision_model_bank.atomic import atomic_write_json
from vision_model_bank.config import load_config
from vision_model_bank.data import ensure_downloaded
from vision_model_bank.errors import BankError, EngineeringGateFailed
from vision_model_bank.export import export_bank
from vision_model_bank.gates import judge_engineering_gates, load_metrics, write_acceptance
from vision_model_bank.manifest import generate_manifest
from vision_model_bank.metrics import evaluate_models
from vision_model_bank.resources import query_nvidia_smi, resolve_training_devices
from vision_model_bank.scheduler import require_full_batch, select_model_ids, train_ids
from vision_model_bank.summary import write_summary
from vision_model_bank.tasks import load_run, prepare_run_dir
from vision_model_bank.validate import validate_run

CANONICAL_COMMANDS = [
    "python -m vision_model_bank generate --config configs/bank.yaml --out configs/manifest.json",
    "python -m vision_model_bank dry-run --config configs/bank.yaml",
    "python -m vision_model_bank devices --config configs/bank.yaml --gpus 0,1,2,3",
    "python -m vision_model_bank pilot --config configs/bank.yaml --output runs/bank --device cpu",
    "python -m vision_model_bank pilot --config configs/bank.yaml --output runs/bank --gpus 0",
    "python -m vision_model_bank run --config configs/bank.yaml --output runs/bank --gpus 0",
    "python -m vision_model_bank resume --config configs/bank.yaml --output runs/bank --gpus 0",
    "python -m vision_model_bank evaluate --output runs/bank --device cpu",
    "python -m vision_model_bank validate --output runs/bank",
    "python -m vision_model_bank export --output runs/bank --blind-dir exports/blind --private-dir exports/private",
    "python -m vision_model_bank summary --output runs/bank",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vision_model_bank", description="MNIST/CIFAR-10 backdoor model bank")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Write the deduplicated manifest and verify its size")
    generate.add_argument("--config", required=True)
    generate.add_argument("--out", required=True)

    dry = sub.add_parser("dry-run", help="Print the matrix and resource policy without training")
    dry.add_argument("--config", required=True)

    devices = sub.add_parser("devices", help="Show which explicitly requested GPUs are idle")
    devices.add_argument("--config", required=True)
    devices.add_argument("--gpus", required=True)

    for name, help_text in (
        ("pilot", "Train the seed-0 clean and baseline models, then apply the engineering gate"),
        ("run", "Train the full manifest after the engineering pilot gate is accepted"),
        ("resume", "Continue incomplete tasks. Non-pilot tasks stay gated"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--device", choices=["cpu"], default=None)
        cmd.add_argument("--gpus", default=None)
        cmd.add_argument("--workers", type=int, default=None)
        cmd.add_argument("--data-dir", default=None)

    evaluate = sub.add_parser("evaluate", help="Official test metrics. Does not select checkpoints")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", choices=["cpu"], default="cpu")
    evaluate.add_argument("--gpus", default=None)
    evaluate.add_argument("--data-dir", default=None)

    validate = sub.add_parser("validate", help="Reload weights, check shared inits, recompute metrics")
    validate.add_argument("--output", required=True)
    validate.add_argument("--skip-recompute", action="store_true")

    export = sub.add_parser("export", help="Write separate blind and private bundles")
    export.add_argument("--output", required=True)
    export.add_argument("--blind-dir", required=True)
    export.add_argument("--private-dir", required=True)
    export.add_argument("--blind-archive", default=None)
    export.add_argument("--private-archive", default=None)

    summary = sub.add_parser("summary", help="Write summary.json and summary.md, keeping failures")
    summary.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except EngineeringGateFailed as exc:
        print(exc)
        return exc.exit_code
    except BankError as exc:
        print(exc)
        return exc.exit_code


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "generate":
        cfg = load_config(args.config)
        manifest = generate_manifest(cfg)
        atomic_write_json(args.out, manifest)
        print(f"unique_models: {manifest['unique_count']}")
        print(f"introduced_by: {json.dumps(manifest['introduced_by'], sort_keys=True)}")
        print(f"pilot_models: {manifest['pilot_count']}")
        print(f"wrote {args.out}")
        return 0
    if args.command == "dry-run":
        cfg = load_config(args.config)
        manifest = generate_manifest(cfg)
        _print_plan(cfg, manifest)
        return 0
    if args.command == "devices":
        cfg = load_config(args.config)
        resolved = resolve_training_devices(cfg, gpus=args.gpus, device=None, workers=None)
        print(json.dumps({"idle_gpus": resolved["gpus"], "refused": resolved["refused"]}, indent=2))
        return 0
    if args.command in {"pilot", "run", "resume"}:
        cfg = load_config(args.config)
        output = Path(args.output)
        manifest = prepare_run_dir(cfg, output)
        devices = resolve_training_devices(
            cfg, gpus=args.gpus, device=args.device, workers=args.workers
        )
        mode = {"pilot": "pilot", "run": "full", "resume": "resume"}[args.command]
        if args.command == "run":
            require_full_batch(output, cfg, manifest)
        model_ids = select_model_ids(manifest, output, cfg, mode)
        print(f"scheduling {len(model_ids)} models on {devices['kind']} workers={devices['workers']}")
        if devices.get("refused"):
            print("refused GPUs (not replaced with any other device):")
            print(json.dumps(devices["refused"], indent=2))
        train_ids(cfg, manifest, output, model_ids, devices, data_root=_data_dir(args, output))
        if args.command == "pilot":
            return _finish_pilot(cfg, manifest, output, _data_dir(args, output))
        return 0
    if args.command == "evaluate":
        output = Path(args.output)
        cfg, manifest = load_run(output)
        root = _data_dir(args, output)
        if not cfg.get("allow_synthetic"):
            for dataset in sorted({row["dataset"] for row in manifest["models"]}):
                ensure_downloaded(dataset, root, output / "locks" / "download.lock")
        device = "cpu" if args.gpus is None else f"cuda:{args.gpus.split(',')[0]}"
        if args.gpus is not None:
            resolve_training_devices(cfg, gpus=args.gpus, device=None, workers=1)
        ids = [row["model_id"] for row in manifest["models"]]
        stats = evaluate_models(cfg, manifest, output, ids, device, data_root=root)
        print(json.dumps(stats))
        return 0
    if args.command == "validate":
        report = validate_run(Path(args.output), recompute=not args.skip_recompute)
        print(json.dumps({"passed": report["passed"], "checked_models": report["checked_models"], "failures": report["failures"]}, indent=2))
        return 0 if report["passed"] else 1
    if args.command == "export":
        result = export_bank(
            Path(args.output),
            Path(args.blind_dir),
            Path(args.private_dir),
            blind_archive=Path(args.blind_archive) if args.blind_archive else None,
            private_archive=Path(args.private_archive) if args.private_archive else None,
        )
        print(json.dumps(result))
        return 0
    if args.command == "summary":
        summary = write_summary(Path(args.output))
        print(f"unique_models: {summary['unique_count']} failures_retained: {summary['failure_count']}")
        return 0
    raise BankError(f"unknown command {args.command}")


def _finish_pilot(cfg: dict[str, Any], manifest: dict[str, Any], output: Path, data_root: Path) -> int:
    pilot_ids = [row["model_id"] for row in manifest["models"] if row["pilot"]]
    if not cfg.get("allow_synthetic"):
        for dataset in sorted({row["dataset"] for row in manifest["models"] if row["pilot"]}):
            ensure_downloaded(dataset, data_root, output / "locks" / "download.lock")
    evaluate_models(cfg, manifest, output, pilot_ids, "cpu", data_root=data_root)
    metrics = {model_id: load_metrics(output, model_id) for model_id in pilot_ids}
    document = judge_engineering_gates(
        cfg,
        [row for row in manifest["models"] if row["pilot"]],
        metrics,
        manifest_sha256=manifest["manifest_sha256"],
    )
    write_acceptance(output, document)
    print(json.dumps({"accepted": document["accepted"], "criterion_class": document["criterion_class"]}, indent=2))
    if not document["accepted"]:
        raise EngineeringGateFailed(
            "Pilot engineering gate was not accepted. Models and metrics were kept. "
            "This gate is not a scientific criterion. The full batch will not start."
        )
    return 0


def _print_plan(cfg: dict[str, Any], manifest: dict[str, Any]) -> None:
    print(f"unique_models: {manifest['unique_count']}")
    print(f"introduced_by: {json.dumps(manifest['introduced_by'], sort_keys=True)}")
    print(f"pilot_models: {manifest['pilot_count']} {manifest['pilot_model_ids']}")
    print(f"clean_models_shared: {manifest['clean_count']}")
    print("primary_checkpoint: final_epoch")
    print("test_split: official test, final evaluation only")
    print(
        "gpu_policy: explicit --gpus only; busy GPUs are refused and never replaced; "
        f"idle util<={cfg['resources']['idle_util_percent_max']} "
        f"memory_mib<={cfg['resources']['idle_memory_mib_max']}"
    )
    print("oom_policy: record failure or lower worker concurrency; batch size is fixed")
    print("engineering_gates_are_not_scientific_criteria: true")


def _data_dir(args: argparse.Namespace, output: Path) -> Path:
    return Path(args.data_dir) if getattr(args, "data_dir", None) else output / "data"


if __name__ == "__main__":
    raise SystemExit(main())
