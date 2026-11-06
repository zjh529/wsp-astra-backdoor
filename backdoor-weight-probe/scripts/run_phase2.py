"""Orchestrate Phase 2: Neural Cleanse (Part A) + stealth twin train/probe (Part B)."""
from __future__ import annotations

import json
import subprocess
import sys
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

    # Part B train first (independent of NC); then can probe while documenting
    run([PY, str(SCRIPTS / "train_stealth.py"), "--epochs", "3", "--seed", "42"])

    # Part A: NC on existing poisoned.pt
    run([PY, str(SCRIPTS / "neural_cleanse.py"), "--steps", "200", "--lambda-l1", "0.001", "--batch-size", "128"])

    # Part B probe
    run([PY, str(SCRIPTS / "probe_stealth.py")])

    # Optional: quick NC on stealth if time — run with fewer steps
    run(
        [
            PY,
            str(SCRIPTS / "neural_cleanse.py"),
            "--model",
            "poisoned_stealth",
            "--steps",
            "150",
            "--lambda-l1",
            "0.001",
            "--batch-size",
            "128",
            "--skip-ablation",  # skip heavy ablation if stealth NC is just for anomaly index
        ]
    )

    # Write combined summary JSON
    nc = json.loads((RESULTS / "phase2_nc_poisoned.json").read_text())
    stealth_probe = json.loads((RESULTS / "phase2_stealth_probe.json").read_text())
    stealth_train = json.loads((RESULTS / "train_poisoned_stealth.json").read_text())
    nc_stealth_path = RESULTS / "phase2_nc_poisoned_stealth.json"
    nc_stealth = json.loads(nc_stealth_path.read_text()) if nc_stealth_path.exists() else None

    combined = {
        "part_a_nc_poisoned": {
            "flagged_class_l1_mask": nc.get("flagged_class_l1_mask"),
            "flagged_class_l1_trigger": nc.get("flagged_class_l1_trigger"),
            "class0_flagged": nc.get("class0_flagged"),
            "reversed_trigger_asr_flagged": nc.get("reversed_trigger_asr_flagged"),
            "reversed_trigger_asr_class0": nc.get("reversed_trigger_asr_class0"),
            "per_class_anomaly": [
                {
                    "class": e["class"],
                    "l1_mask": e["l1_mask"],
                    "anomaly_index_l1_mask": e["anomaly_index_l1_mask"],
                    "l1_trigger": e["l1_trigger"],
                    "anomaly_index_l1_trigger": e["anomaly_index_l1_trigger"],
                }
                for e in nc["per_class"]
            ],
            "ablation_true_patch": nc.get("ablation_true_patch"),
        },
        "part_b_stealth": {
            "train": stealth_train,
            "probe": {
                "base_clean_acc": stealth_probe["poisoned_stealth"]["base_clean_acc"],
                "base_asr": stealth_probe["poisoned_stealth"]["base_asr"],
                "best_practical": stealth_probe["poisoned_stealth"]["best_practical"],
                "model_suspicion": stealth_probe["poisoned_stealth"]["model_suspicion"],
                "clean_suspicion": stealth_probe["clean"]["model_suspicion"],
                "suspicion_ratio": stealth_probe["suspicion_ratio_stealth_vs_clean"],
            },
            "nc_optional": {
                "flagged": nc_stealth.get("flagged_class_l1_mask") if nc_stealth else None,
                "class0_flagged": nc_stealth.get("class0_flagged") if nc_stealth else None,
                "reversed_asr_class0": nc_stealth.get("reversed_trigger_asr_class0") if nc_stealth else None,
            },
        },
        "elapsed_sec_total": time.time() - t0,
    }
    (RESULTS / "phase2_summary.json").write_text(json.dumps(combined, indent=2))
    print(f"\nPhase2 done in {time.time()-t0:.1f}s — wrote results/phase2_summary.json")


if __name__ == "__main__":
    main()
