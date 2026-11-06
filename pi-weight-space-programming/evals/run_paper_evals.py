#!/usr/bin/env python3
"""
Turnkey runner for live Pi paper evaluations (Experiments 1, 2, and 3).

Works on a blank slate: automatically downloads and caches required prompts,
checkpoints (.npz), and ground-truth answer keys from `RikiyaT/lm_understand-logs`
into `evals/.cache/paper_evals/`, runs `pi` with the local extension, and scores
the predictions.

Usage examples:
  python3 evals/run_paper_evals.py --exp 1 2 3
  python3 evals/run_paper_evals.py --exp 2 --model google/gemini-4-argon
  python3 evals/run_paper_evals.py --download-only
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evals" / "functional"))
from test_functional import DEFAULT_CACHE_DIR, ensure_eval_artifacts  # noqa: E402


def run_pi(args_list: list[str], cwd: pathlib.Path = REPO_ROOT) -> None:
    cmd = ["pi", "--no-builtin-tools", "-e", str(REPO_ROOT / "extensions" / "index.ts")] + args_list
    print(f"[pi] Running: {' '.join(cmd[:-1])} <prompt>", file=sys.stderr)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def score_exp1(runs_dir: pathlib.Path) -> dict:
    expected = {
        "breakout": ("breakout", "brick"),
        "pong": ("pong", "table tennis"),
        "pacman": ("pac-man", "pacman"),
        "freeway": ("freeway", "frogger", "road crossing", "highway"),
    }
    scores = {}
    for game, keywords in expected.items():
        ans_path = runs_dir / f"exp1_{game}" / "final_answer.md"
        if not ans_path.exists():
            continue
        text = ans_path.read_text(encoding="utf-8").lower()
        m = re.search(r"^(?:\*\*?)?game:\*?\*?\s*([^\n]+)", text, flags=re.M)
        pred_line = m.group(1).strip() if m else text[-300:]
        passed = any(kw in pred_line for kw in keywords)
        scores[game] = {"passed": passed, "predicted_game_line": pred_line}
    return scores


def score_exp2(runs_dir: pathlib.Path) -> dict:
    expected = {
        "circle": ("circle", "disc", "disk", "round"),
        "square": ("square", "rectangle", "rectangular", "box"),
        "triangle": ("triangle", "triangular", "pyramid", "wedge"),
    }
    scores = {}
    for shape, keywords in expected.items():
        ans_path = runs_dir / f"exp2_{shape}" / "final_answer.md"
        if not ans_path.exists():
            continue
        text = ans_path.read_text(encoding="utf-8").lower()
        m = re.search(r"images:\*?\*?\s*(.+?)(?:\n\n|\n(?:\*\*?)?example:)", text, flags=re.S)
        images_block = m.group(1).strip() if m else text
        passed = any(kw in images_block for kw in keywords)
        scores[shape] = {"passed": passed, "images_excerpt": images_block[:200].replace("\n", " ")}
    return scores


def score_exp3(runs_dir: pathlib.Path, cache_dir: pathlib.Path) -> dict:
    heads_ref = json.loads((cache_dir / "exp3" / "heads.json").read_text(encoding="utf-8"))
    scores = {}
    for lang in ("python", "romaji", "english"):
        ans_path = runs_dir / f"exp3_{lang}" / "final_answer.md"
        if not ans_path.exists():
            continue
        text = ans_path.read_text(encoding="utf-8")
        m_corp = re.search(r"corpus:\s*(python|romaji|english)", text, flags=re.I)
        m_ord = re.search(r"order:\s*(as is|shuffled)", text, flags=re.I)
        pred_corp = m_corp.group(1).lower() if m_corp else ""
        pred_ord = m_ord.group(1).lower() if m_ord else ""
        pred_copy = sorted(re.findall(r"^(L[0-3]H[0-5]):\s*copy\b", text, flags=re.M))
        model_heads = heads_ref[f"{lang}_L4_H6_d384_s0"]
        gt_copy = sorted(
            k for k, v in model_heads.items() if isinstance(v, dict) and v.get("induct", 0.0) >= 0.2
        )
        scores[lang] = {
            "passed": pred_corp == lang and pred_ord == "as is" and pred_copy == gt_copy,
            "predicted_corpus": pred_corp,
            "predicted_order": pred_ord,
            "predicted_copying_heads": pred_copy,
            "ground_truth_copying_heads": gt_copy,
        }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Download paper artifacts and run/score Pi evaluations.")
    parser.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3], help="Experiments to run/score")
    parser.add_argument("--model", type=str, default=None, help="Optional Pi model override (e.g. google/gemini-4-argon)")
    parser.add_argument("--effort", type=str, default="medium", help="Thinking effort override (default: medium)")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "evals" / "runs"), help="Output directory for run logs")
    parser.add_argument("--download-only", action="store_true", help="Only download/cache artifacts without invoking Pi")
    parser.add_argument("--score-only", action="store_true", help="Only score existing logs in --out-dir")
    args = parser.parse_args()

    cache_dir = pathlib.Path(os.environ.get("WSP_EVAL_DATA_DIR", str(DEFAULT_CACHE_DIR)))
    prefixes = tuple(f"exp{e}/" for e in args.exp)
    ensure_eval_artifacts(cache_dir, subset_prefixes=prefixes)
    if args.download_only:
        print(f"All artifacts for {prefixes} cached in {cache_dir}")
        return

    out_dir = pathlib.Path(args.out_dir)
    model_args = ["--model", args.model] if args.model else []

    if not args.score_only:
        if 1 in args.exp:
            for game in ("breakout", "pong", "pacman", "freeway"):
                prompt = (cache_dir / "exp1" / f"{game}_prompt.txt").read_text(encoding="utf-8")
                log_dir = out_dir / f"exp1_{game}"
                run_pi(model_args + ["--wsp-exp", "1", "--wsp-effort", args.effort, "--wsp-log-dir", str(log_dir), "-p", prompt])

        if 2 in args.exp:
            for shape in ("circle", "square", "triangle"):
                prompt = (cache_dir / "exp2" / f"{shape}_prompt.txt").read_text(encoding="utf-8")
                log_dir = out_dir / f"exp2_{shape}"
                run_pi(model_args + ["--wsp-exp", "2", "--wsp-effort", args.effort, "--wsp-log-dir", str(log_dir), "-p", prompt])

        if 3 in args.exp:
            base_msg = (cache_dir / "exp3" / "message.txt").read_text(encoding="utf-8")
            old_line = (
                "To run code, reply with exactly one code block that starts with ```python and ends with ```, "
                "and nothing else. We run it and reply with its printed output."
            )
            for lang in ("python", "romaji", "english"):
                ckpt = cache_dir / "exp3" / f"{lang}_blocks.npz"
                log_dir = out_dir / f"exp3_{lang}"
                new_line = (
                    f"Use the built-in tools `wsp_circuit_probe` (operations `induction_heads` and `subspace_rank` "
                    f"on checkpoint `{ckpt}`) and/or `wsp_python_sandbox` (where `W` is pre-loaded from `{ckpt}`) "
                    f"to inspect the weights, then provide your final answer."
                )
                prompt = base_msg.replace(old_line, new_line)
                run_pi(
                    model_args
                    + [
                        "--wsp-exp",
                        "3",
                        "--wsp-model-path",
                        str(ckpt),
                        "--wsp-log-dir",
                        str(log_dir),
                        "-p",
                        prompt,
                    ]
                )

    summary = {}
    if 1 in args.exp:
        summary["exp1_which_game"] = score_exp1(out_dir)
    if 2 in args.exp:
        summary["exp2_diffusion"] = score_exp2(out_dir)
    if 3 in args.exp:
        summary["exp3_language"] = score_exp3(out_dir, cache_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
