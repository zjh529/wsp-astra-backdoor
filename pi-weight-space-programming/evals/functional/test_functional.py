#!/usr/bin/env python3
"""
Functional & deterministic paper parity test suite for pi-weight-space-programming.

Works out-of-the-box on a blank slate: automatically downloads and caches required
paper checkpoints and reference artifacts from `RikiyaT/lm_understand-logs` into
`evals/.cache/paper_evals` (or `$WSP_EVAL_DATA_DIR`) when not already present.

Verifies:
1. All 5 wsp_engine.py commands (`inspect`, `circuit_probe`, `compare`, `edit`, `sandbox`).
2. Static gradient/training guard in `sandbox` (blocks `.backward()`, `torch.optim`, etc.).
3. Experiment 3 (`exp3_language`) static circuit probe (`induction_heads` and `subspace_rank`)
   against ground-truth copying heads (`heads.json`) and Romaji alphabet rank bottleneck.
4. Experiment 4 (`exp4_fix_car`) surgical weight editing DSL parity against `oneshot_result.json`
   and `edited_actor.npz` (51/51 statements, 0 errors, exact weight match).
5. Experiment 5 (`exp5_write_from_scratch`) Breakout weight synthesis DSL parity against
   `breakout_hand.npz` and `breakout_apply.json` (92/92 statements, 815 nonzero entries, max_diff = 0.0).
"""

import json
import os
import pathlib
import subprocess
import sys
import urllib.request
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "harness" / "wsp_engine.py"
DEFAULT_CACHE_DIR = REPO_ROOT / "evals" / ".cache" / "paper_evals"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main"

REQUIRED_ARTIFACTS = {
    # Experiment 1 (Which Game?)
    "exp1/breakout_prompt.txt": "exp1_which_game/runs/breakout/prompt.txt",
    "exp1/pong_prompt.txt": "exp1_which_game/runs/pong/prompt.txt",
    "exp1/pacman_prompt.txt": "exp1_which_game/runs/pacman/prompt.txt",
    "exp1/freeway_prompt.txt": "exp1_which_game/runs/freeway/prompt.txt",
    "exp1/answer_key.json": "exp1_which_game/eval/answer_key.json",
    # Experiment 2 (Diffusion Shape)
    "exp2/circle_prompt.txt": "exp2_diffusion/runs/circle/prompt.txt",
    "exp2/square_prompt.txt": "exp2_diffusion/runs/square/prompt.txt",
    "exp2/triangle_prompt.txt": "exp2_diffusion/runs/triangle/prompt.txt",
    # Experiment 3 (Language & Copying Heads from Middle Transformer Blocks)
    "exp3/python_blocks.npz": "exp3_language/networks/python/blocks.npz",
    "exp3/romaji_blocks.npz": "exp3_language/networks/romaji/blocks.npz",
    "exp3/english_blocks.npz": "exp3_language/networks/english/blocks.npz",
    "exp3/heads.json": "exp3_language/eval/heads.json",
    "exp3/message.txt": "exp3_language/runs/python_r0/message.txt",
    # Experiment 4 (Fix Failing Car via Surgical Weight Edits)
    "exp4/ppo_final_seed0.npz": "exp4_fix_car/networks/ppo_final_seed0.npz",
    "exp4/edited_actor.npz": "exp4_fix_car/networks/edited_actor.npz",
    "exp4/answer.md": "exp4_fix_car/runs/oneshot_repair/reply/answer.md",
    "exp4/oneshot_result.json": "exp4_fix_car/eval/oneshot_result.json",
    # Experiment 5 (Write Breakout Policy from Scratch)
    "exp5/breakout_hand.npz": "exp5_write_from_scratch/networks/breakout/hand.npz",
    "exp5/breakout_answer.md": "exp5_write_from_scratch/runs/breakout/reply/answer.md",
    "exp5/breakout_apply.json": "exp5_write_from_scratch/runs/breakout/apply.json",
}


def ensure_eval_artifacts(cache_dir: pathlib.Path, subset_prefixes: tuple = ("exp3/", "exp4/", "exp5/")) -> None:
    """Download any missing paper evaluation files from RikiyaT/lm_understand-logs."""
    for rel_dst, remote_rel in REQUIRED_ARTIFACTS.items():
        if subset_prefixes and not any(rel_dst.startswith(p) for p in subset_prefixes):
            continue
        dst = cache_dir / rel_dst
        if dst.exists() and dst.stat().st_size > 32:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{GITHUB_RAW_BASE}/{remote_rel}"
        print(f"[download] Fetching {rel_dst} from {url} ...", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "pi-weight-space-programming-evals/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as out_f:
            out_f.write(resp.read())


def run_engine(cmd: str, payload: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ENGINE_PATH), cmd, json.dumps(payload)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    eval_data_dir = pathlib.Path(os.environ.get("WSP_EVAL_DATA_DIR", str(DEFAULT_CACHE_DIR)))
    ensure_eval_artifacts(eval_data_dir, subset_prefixes=("exp3/", "exp4/", "exp5/"))
    results = {"tests": {}, "all_passed": False}

    # 1. Inspect command
    ins = run_engine(
        "inspect",
        {"checkpoint": str(eval_data_dir / "exp4" / "ppo_final_seed0.npz"), "operation": "summary"},
    )
    assert ins["total_tensors"] == 26 and ins["total_parameters"] == 254353
    results["tests"]["wsp_inspect_weights"] = {
        "passed": True,
        "total_tensors": ins["total_tensors"],
        "total_parameters": ins["total_parameters"],
    }

    # 2. Experiment 4 surgical edit DSL parity
    exp4_ans = (eval_data_dir / "exp4" / "answer.md").read_text(encoding="utf-8")
    exp4_ref = json.loads((eval_data_dir / "exp4" / "oneshot_result.json").read_text(encoding="utf-8"))
    exp4_out = str(eval_data_dir / "exp4" / "test_edited.npz")
    res_exp4 = run_engine(
        "edit",
        {
            "checkpoint": str(eval_data_dir / "exp4" / "ppo_final_seed0.npz"),
            "output_path": exp4_out,
            "edits_dsl": exp4_ans,
            "explanation": "Functional test of Experiment 4 surgical edit DSL",
        },
    )
    ed_ref = np.load(eval_data_dir / "exp4" / "edited_actor.npz")
    ed_out = np.load(exp4_out)
    head_max_diff = float(
        np.max(
            np.abs(
                ed_out["params/GeneralActorCriticRNN_0/Dense_0/kernel"][126:128, 0:3]
                - ed_ref["actor_head.W"][126:128, 0:3]
            )
        )
    )
    assert res_exp4["applied_count"] == exp4_ref["edit_report"]["applied"] == 51
    assert len(res_exp4["dsl_report"]["errors"]) == 0
    assert len(res_exp4["dsl_report"]["tensors_touched"]) == 14
    assert head_max_diff == 0.0
    results["tests"]["exp4_surgical_edit_dsl"] = {
        "passed": True,
        "statements_applied": res_exp4["applied_count"],
        "expected_statements": exp4_ref["edit_report"]["applied"],
        "tensors_touched": len(res_exp4["dsl_report"]["tensors_touched"]),
        "actor_head_max_abs_diff": head_max_diff,
    }

    # 3. Compare checkpoints command
    cmp_res = run_engine(
        "compare",
        {
            "checkpoint_a": str(eval_data_dir / "exp4" / "ppo_final_seed0.npz"),
            "checkpoint_b": exp4_out,
        },
    )
    assert cmp_res["changed_tensors_count"] == 14 and len(cmp_res["identical_frozen_tensors"]) == 12
    results["tests"]["wsp_compare_checkpoints"] = {
        "passed": True,
        "changed_tensors_count": cmp_res["changed_tensors_count"],
        "identical_frozen_tensors_count": len(cmp_res["identical_frozen_tensors"]),
    }

    # 4. Experiment 5 from-scratch Breakout DSL parity
    exp5_ans = (eval_data_dir / "exp5" / "breakout_answer.md").read_text(encoding="utf-8")
    exp5_out = str(eval_data_dir / "exp5" / "test_breakout.npz")
    res_exp5 = run_engine(
        "edit",
        {
            "output_path": exp5_out,
            "init_shapes": {
                "W1": [600, 32],
                "b1": [32],
                "W2": [32, 32],
                "b2": [32],
                "W3": [32, 6],
                "b3": [6],
            },
            "edits_dsl": exp5_ans,
            "explanation": "Functional test of Experiment 5 from-scratch Breakout synthesis",
        },
    )
    hand = np.load(eval_data_dir / "exp5" / "breakout_hand.npz")
    gen = np.load(exp5_out)
    pairs = [
        ("W1", "actor/dense1/kernel"),
        ("b1", "actor/dense1/bias"),
        ("W2", "actor/dense2/kernel"),
        ("b2", "actor/dense2/bias"),
        ("W3", "actor/logits/kernel"),
        ("b3", "actor/logits/bias"),
    ]
    max_diff_exp5 = max(float(np.max(np.abs(gen[g] - hand[h]))) for g, h in pairs)
    assert res_exp5["applied_count"] == 92
    assert res_exp5["total_changed_entries"] == 815
    assert max_diff_exp5 == 0.0
    results["tests"]["exp5_from_scratch_dsl"] = {
        "passed": True,
        "statements_applied": res_exp5["applied_count"],
        "nonzero_entries": res_exp5["total_changed_entries"],
        "max_abs_diff_vs_breakout_hand_npz": max_diff_exp5,
    }

    # 5. Experiment 3 static circuit probe (induction heads & alphabet bottleneck)
    heads_ref = json.loads((eval_data_dir / "exp3" / "heads.json").read_text(encoding="utf-8"))
    exp3_probe_summary = {}
    for lang in ("python", "romaji", "english"):
        ckpt_path = str(eval_data_dir / "exp3" / f"{lang}_blocks.npz")
        ind = run_engine(
            "circuit_probe",
            {"checkpoint": ckpt_path, "operation": "induction_heads", "num_heads": 6},
        )
        sub = run_engine(
            "circuit_probe",
            {"checkpoint": ckpt_path, "operation": "subspace_rank", "num_heads": 6},
        )
        top_heads = [h["head"] for h in ind["ranked_candidate_copying_heads"][:3]]
        model_heads = heads_ref[f"{lang}_L4_H6_d384_s0"]
        gt_heads = [
            f"{k[1]}.{k[3]}"
            for k, v in model_heads.items()
            if isinstance(v, dict) and v.get("induct", 0.0) >= 0.2
        ]
        assert all(g in top_heads for g in gt_heads)
        l3_gap = sub["mlp_input_regressed_alphabet_bottleneck"][-1]
        exp3_probe_summary[lang] = {
            "ground_truth_copying_heads": gt_heads,
            "top_3_detected_heads": top_heads,
            "top_ranked_candidates": ind["ranked_candidate_copying_heads"][:3],
            "layer3_top_sv_gap": l3_gap["top_singular_value_gaps_rank_16_to_50"][0],
            "layer3_energy_top24_fraction": l3_gap["energy_top24_fraction"],
        }
    results["tests"]["exp3_static_circuit_probe"] = {
        "passed": True,
        "corpora": exp3_probe_summary,
    }

    # 6. Sandbox & no-grad guard
    sbx_ok = run_engine(
        "sandbox",
        {
            "code": 'ckpt = Checkpoint(MODEL_PATH)\nprint("SANDBOX_OK", len(ckpt.keys()), sorted(W.keys()))',
            "model_path": exp5_out,
            "work_dir": str(REPO_ROOT),
        },
    )
    assert sbx_ok["ok"] is True and "SANDBOX_OK 6" in sbx_ok["output"]

    sbx_guard = run_engine(
        "sandbox",
        {
            "code": "loss.backward()",
            "model_path": exp5_out,
            "work_dir": str(REPO_ROOT),
        },
    )
    assert sbx_guard["blocked"] is True and len(sbx_guard["violations"]) == 1
    results["tests"]["wsp_python_sandbox_and_guard"] = {
        "passed": True,
        "sandbox_stdout": sbx_ok["output"].strip(),
        "guard_blocked_backward": sbx_guard["blocked"],
        "guard_violation": sbx_guard["violations"][0],
    }

    # 7. Hugging Face Model Downloader (Strict Internet Exception & Security Guards)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_hf_dir:
        # A. Valid download of model config
        hf_res = run_engine(
            "hf_download",
            {
                "repo_id": "openai-community/gpt2",
                "filenames": ["config.json"],
                "destination_dir": tmp_hf_dir,
            },
        )
        assert hf_res["ok"] is True
        assert len(hf_res["downloaded_files"]) == 1
        cfg_file = pathlib.Path(hf_res["downloaded_files"][0]["path"])
        assert cfg_file.exists() and cfg_file.stat().st_size > 0
        file_size = cfg_file.stat().st_size
        assert "Authorized Hugging Face Hub download exception" in hf_res["policy_exception_notice"]

        # B. Caching verification
        hf_cached = run_engine(
            "hf_download",
            {
                "repo_id": "openai-community/gpt2",
                "filenames": ["config.json"],
                "destination_dir": tmp_hf_dir,
            },
        )
        assert hf_cached["ok"] is True
        assert hf_cached["downloaded_files"][0]["cached"] is True

        # C. Security boundary: Path traversal blocked
        traversal_blocked = False
        try:
            run_engine(
                "hf_download",
                {
                    "repo_id": "openai-community/gpt2",
                    "filenames": ["../../evil.txt"],
                    "destination_dir": tmp_hf_dir,
                },
            )
        except Exception as e:
            traversal_blocked = True
        assert traversal_blocked, "Path traversal should be strictly blocked"

    results["tests"]["wsp_hf_download"] = {
        "passed": True,
        "repo_id": "openai-community/gpt2",
        "file_downloaded": "config.json",
        "size_bytes": file_size,
        "cached_verified": True,
        "traversal_guard_blocked": True,
    }

    results["all_passed"] = True
    out_file = pathlib.Path(__file__).resolve().parent / "functional_results.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
