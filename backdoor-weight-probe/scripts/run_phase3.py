"""Orchestrate Phase 3: reference-free MNIST (Part A) + CIFAR-10 scale-up (Part B)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
PY = "/workspace/wsp/pi-weight-space-programming/.venv/bin/python"


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd or str(SCRIPTS))
    print(f"<<< exit={p.returncode} elapsed={time.time()-t0:.1f}s")
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Part A: reference-free on existing MNIST models
    # Run NC on clean (needed for FP); reuse poisoned NC from phase2
    run(
        [
            PY,
            str(SCRIPTS / "ref_free.py"),
            "--steps",
            "200",
            "--lambda-l1",
            "0.001",
        ]
    )

    # Part B: CIFAR train (full set, 3 epochs)
    run(
        [
            PY,
            str(SCRIPTS / "train_cifar.py"),
            "--epochs",
            "3",
            "--seed",
            "42",
            "--poison-rate",
            "0.08",
            "--trigger-size",
            "3",
            "--only",
            "both",
        ]
    )

    # Part B probe: true-trigger act×w ablation + NC subset on poisoned
    run(
        [
            PY,
            str(SCRIPTS / "probe_cifar.py"),
            "--models",
            "cifar_poisoned",
            "cifar_clean",
            "--trigger-size",
            "3",
            "--nc-steps",
            "80",
            "--nc-classes",
            "0,1,3,5,7",
            "--skip-clean-nc",
        ]
    )

    # Combined summary JSON
    ref = json.loads((RESULTS / "phase3_ref_free.json").read_text())
    cifar_train = {}
    tp = RESULTS / "train_cifar_summary.json"
    if tp.exists():
        cifar_train = json.loads(tp.read_text())
    cifar_probe = {}
    cp = RESULTS / "phase3_cifar_probe_summary.json"
    if cp.exists():
        cifar_probe = json.loads(cp.read_text())

    poisoned_ref = ref["poisoned"]
    clean_ref = ref["clean"]
    thr = ref["thresholds"]

    cifar_p = cifar_probe.get("cifar_poisoned", {})
    cifar_c = cifar_probe.get("cifar_clean", {})

    combined = {
        "part_a_ref_free_mnist": {
            "poisoned_suspicion": poisoned_ref["ref_free_suspicion"],
            "clean_suspicion": clean_ref["ref_free_suspicion"],
            "poisoned_nc_ai_mask": poisoned_ref["nc"]["max_ai_mask"],
            "clean_nc_ai_mask": clean_ref["nc"]["max_ai_mask"],
            "poisoned_rev_asr": poisoned_ref["nc"]["reversed_trigger_asr"],
            "clean_rev_asr": clean_ref["nc"]["reversed_trigger_asr"],
            "poisoned_asymmetry": poisoned_ref["class_asymmetry"]["asymmetry_score"],
            "clean_asymmetry": clean_ref["class_asymmetry"]["asymmetry_score"],
            "thresholds": thr,
            "needs_unlabeled_clean_images": ref["needs_unlabeled_clean_images"],
        },
        "part_b_cifar": {
            "train": cifar_train,
            "poisoned": {
                "clean_acc": cifar_p.get("base_clean_acc"),
                "asr": cifar_p.get("base_asr"),
                "best_practical": (cifar_p.get("ablation_true_trigger") or {}).get("best_practical"),
                "nc": cifar_p.get("nc"),
                "suspicion": cifar_p.get("ref_free_suspicion"),
            },
            "clean": {
                "clean_acc": cifar_c.get("base_clean_acc"),
                "asr": cifar_c.get("base_asr"),
                "best_practical": (cifar_c.get("ablation_true_trigger") or {}).get("best_practical"),
                "suspicion": cifar_c.get("ref_free_suspicion"),
            },
        },
        "elapsed_sec_total": time.time() - t0,
    }
    (RESULTS / "phase3_summary.json").write_text(json.dumps(combined, indent=2))
    print(f"\nPhase3 done in {time.time()-t0:.1f}s — wrote results/phase3_summary.json")


if __name__ == "__main__":
    main()
