"""Command line for prep, the session, scoring, and the fail-closed preflight."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from padbench_audit.loop import load_config, run_cases
from padbench_audit.package_artifact import write_tarball
from padbench_audit.preflight import run_preflight
from padbench_audit.prep import case_id_for, prepare_experiment
from padbench_audit.score import score_run


def _default_pi() -> str:
    env = os.environ.get("PI_BIN")
    if env:
        return env
    wrapper = Path("/home/junhong/.local/bin/pi-online")
    if wrapper.is_file():
        return str(wrapper)
    return "pi-online"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sandbox-python", type=Path, default=Path(os.environ.get("SANDBOX_PYTHON", "")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="padbench_audit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("preflight", help="Check bwrap isolation and that pi tools are disabled")
    _add_common(pre)
    pre.add_argument("--private-dir", type=Path, required=True)
    pre.add_argument("--pi-bin", default=_default_pi())
    pre.add_argument("--mount-path", type=Path, action="append", default=[])
    pre.add_argument(
        "--pi-cwd",
        type=Path,
        default=None,
        help="Working directory for the pi probe. Must match the directory used by run. "
        "Ancestors must not contain .pi or .agents entries.",
    )
    pre.add_argument("--effort", default="xhigh")
    pre.add_argument("--skip-pi", action="store_true", help="Do not claim success. Isolation still runs; exit status stays non-zero.")
    pre.add_argument(
        "--live-check",
        action="store_true",
        help="Send one tiny prompt to the real model in a throwaway session and require a non-error reply.",
    )
    pre.add_argument("--out", type=Path, default=None)
    pre.add_argument(
        "--probe-python",
        type=Path,
        default=None,
        help="Separate interpreter for the forward-probe worker. Required before a forward-probe run. "
        "Must be able to import torch and transformers, and must not be the numpy sandbox interpreter.",
    )

    prep = sub.add_parser("prep", help="Anonymize adapter folders into blind cases")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--source-root", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--private", type=Path, required=True)
    prep.add_argument("--base-src", type=Path, default=None)
    prep.add_argument("--allow-tokenizer", action="store_true")
    prep.add_argument(
        "--case-order-seed",
        type=int,
        default=None,
        help="Permute case ids with this seed. The seed is written only under --private. "
        "Omit it to draw a fresh seed.",
    )
    prep.add_argument(
        "--keep-original-names",
        action="store_true",
        help="Publish the original tensor names. The default renames them to layer_N.proj_q.A form.",
    )

    run = sub.add_parser("run", help="One pi session for one case, or every case in order")
    resume = sub.add_parser("resume", help="Continue a case whose prediction is not frozen")
    for command in (run, resume):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--blind", type=Path, required=True)
        command.add_argument("--case", default="")
        command.add_argument("--all", action="store_true")
        command.add_argument("--pi-bin", default=_default_pi())
    for command in (run, resume):
        command.add_argument(
            "--pi-cwd",
            type=Path,
            default=None,
            help="pi working directory. With --all this is a parent and the case id is appended. "
            "Default is <blind>/runs/<case>/pi_cwd. Ancestors must not contain .pi or .agents entries. "
            "resume reuses the cwd stored in state.json when this is omitted.",
        )
        _add_common(command)
        command.add_argument("--probe-python", type=Path, default=None)
        command.add_argument("--base-src", type=Path, default=None, help="Host directory of the base checkpoint. Mounted only inside the probe worker.")
        command.add_argument("--private", type=Path, default=None, help="Private directory with cases.json. Used to build the tensor map. Not mounted.")
        command.add_argument("--device", default="cpu", help="cpu (default) or cuda:N. A GPU is used only when this is set. No GPU is chosen automatically.")
        command.add_argument("--lora-scale", type=float, default=1.0, help="Scale applied to B @ A. Recorded in the manifest, not in the prompt. Default 1 assumes alpha equals rank.")

    score = sub.add_parser("score", help="Hash-check a frozen prediction, then read private ground truth")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--private", type=Path, required=True)

    pack = sub.add_parser("package", help="Write a deployable tarball and its SHA-256")
    pack.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "preflight":
        if not args.sandbox_python:
            raise SystemExit("set --sandbox-python or SANDBOX_PYTHON")
        report = run_preflight(
            sandbox_python=args.sandbox_python,
            private_dir=args.private_dir,
            pi_bin=args.pi_bin,
            mount_paths=args.mount_path,
            effort=args.effort,
            skip_pi=args.skip_pi,
            pi_cwd=args.pi_cwd,
            live_check=args.live_check,
            probe_python=args.probe_python,
        )
        text = json.dumps(report, indent=2)
        print(text)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        return 0 if report["ok"] else 1
    if args.cmd == "prep":
        config = load_config(args.config)
        if args.keep_original_names:
            config = dict(config)
            config["neutral_names"] = False
        record = prepare_experiment(
            config=config,
            source_root=args.source_root,
            out_dir=args.out,
            private_dir=args.private,
            base_src=args.base_src,
            allow_tokenizer=args.allow_tokenizer,
            case_order_seed=args.case_order_seed,
        )
        print(json.dumps({"cases": list(record["cases"]), "private": str(args.private)}, indent=2))
        return 0
    if args.cmd in {"run", "resume"}:
        if not args.sandbox_python:
            raise SystemExit("set --sandbox-python or SANDBOX_PYTHON")
        config = load_config(args.config)
        if args.all:
            ids = [case_id_for(i) for i in range(1, len(config["cases"]) + 1)]
        elif args.case:
            ids = [args.case]
        else:
            raise SystemExit("pass --case case_XXX or --all")
        results = run_cases(
            config=config,
            config_path=args.config,
            blind=args.blind,
            case_ids=ids,
            sandbox_python=args.sandbox_python,
            pi_bin=args.pi_bin,
            pi_cwd=args.pi_cwd,
            resume=args.cmd == "resume",
            probe_python=args.probe_python,
            base_src=args.base_src,
            private_dir=args.private,
            device=args.device,
            lora_scale=args.lora_scale,
        )
        print(json.dumps({"cases": results}, indent=2))
        return 0 if all(row.get("status") == "completed" for row in results) else 1
    if args.cmd == "score":
        print(json.dumps(score_run(run_dir=args.run_dir, private_dir=args.private), indent=2))
        return 0
    if args.cmd == "package":
        path, digest = write_tarball(args.out)
        print(f"{digest}  {path}")
        return 0
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
