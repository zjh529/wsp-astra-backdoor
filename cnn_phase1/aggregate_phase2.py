"""Per-arm phase-2 rates and the confirmatory comparison with the phase-1 anchor.

The anchor is the phase-1 v1.1 real-backdoor eval for seeds 0–9, read from
eval_records.json. It is not re-audited.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cnn_phase1.hashing import write_json
from cnn_phase1.localize import iou
from cnn_phase1.phase2 import CONFIRMATORY_ARMS, PHASE2_HOLM_FAMILY, arm_ids
from cnn_phase1.stats import bootstrap_mean_ci, holm, mcnemar_greater, signflip_p, wilson_interval

FPR_MAX = 0.10
FULL_N = 10
ANCHOR_SEEDS = tuple(range(10))


def aggregate_phase2(
    *,
    eval_path: Path,
    anchor_path: Path,
    baselines_path: Path | None,
    out: Path,
) -> dict:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    records = payload["records"]
    excluded = payload.get("excluded") or []
    anchor = _load_anchor(anchor_path)
    arms = {}
    for arm in arm_ids():
        arms[arm] = _arm_block([row for row in records if row.get("arm") == arm and row.get("status") == "valid"])
    clean = [row for row in records if row.get("arm") == "clean" and row.get("status") == "valid"]
    fpr_k = sum(bool(row.get("detection_positive")) for row in clean)
    comparisons = _confirmatory(records, anchor)
    complete = _confirmatory_complete(records, excluded, anchor)
    family = holm(
        {name: comparisons[name]["p"] if complete else None for name in PHASE2_HOLM_FAMILY},
        family=PHASE2_HOLM_FAMILY,
    )
    family["confirmatory"] = complete and all(family["tests"][name]["p"] is not None for name in PHASE2_HOLM_FAMILY)
    if not family["confirmatory"]:
        family = holm({name: None for name in PHASE2_HOLM_FAMILY}, family=PHASE2_HOLM_FAMILY)
        family["confirmatory"] = False
        family["exploratory_p"] = {name: comparisons[name]["p"] for name in PHASE2_HOLM_FAMILY}
    fpr = wilson_interval(fpr_k, len(clean))
    gates = {
        "tested_arms_complete": complete,
        "clean_n_20": len(clean) == 20 and not _excluded_clean(excluded),
        "fpr_point_le_0.10": fpr["estimate"] is not None and fpr["estimate"] <= FPR_MAX and len(clean) == 20,
        "h1_both_reject": complete and family["tests"]["h1_center_iou"]["reject"] and family["tests"]["h1_center_tpr"]["reject"],
        "h2_either_rejects": complete and (family["tests"]["h2_r03_iou"]["reject"] or family["tests"]["h2_r03_tpr"]["reject"]),
    }
    exploratory = _exploratory(records, anchor)
    baselines = None
    if baselines_path is not None and baselines_path.is_file():
        baselines = json.loads(baselines_path.read_text(encoding="utf-8")).get("summary")
    result = {
        "study": "cnn-phase2",
        "arms": arms,
        "clean_fpr": fpr,
        "anchor_n": len(anchor),
        "comparisons": comparisons,
        "holm": family,
        "pass_gates": gates,
        "exploratory": exploratory,
        "excluded": excluded,
        "baselines": baselines,
        "notes": [
            "H1 is the conjunction of lower center IoU and lower center TPR versus the phase-1 anchor.",
            "H2 is the disjunction of lower 0.3% TPR or lower 0.3% IoU versus the same anchor.",
            "Both comparisons are seed-paired. Holm is over the four tests and is confirmatory only when every seed in the tested arms is valid.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", result)
    (out / "summary.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"h1_both_reject": gates["h1_both_reject"], "out": str(out / "summary.json")}, indent=2))
    return result


def _arm_block(rows: list[dict]) -> dict:
    detects = sum(bool(row.get("detection_positive")) for row in rows)
    targets = sum(bool(row.get("target_correct")) for row in rows)
    return {
        "n": len(rows),
        "tpr": wilson_interval(detects, len(rows)),
        "target_identification": wilson_interval(targets, len(rows)),
        "functional_trigger": bootstrap_mean_ci(
            [row["functional_success"] for row in rows if row.get("functional_success") is not None],
            seed=20261120,
        ),
        "mask_iou": bootstrap_mean_ci(
            [row["trigger_iou"] for row in rows if row.get("trigger_iou") is not None],
            seed=20261121,
        ),
        "weight_precision": bootstrap_mean_ci(
            [row["weight_precision"] for row in rows if row.get("weight_precision") is not None],
            seed=20261122,
        ),
        "weight_recall": bootstrap_mean_ci(
            [row["weight_recall"] for row in rows if row.get("weight_recall") is not None],
            seed=20261123,
        ),
        "repair": _repair(rows),
    }


def _repair(rows: list[dict]) -> dict:
    usable = [
        row
        for row in rows
        if row.get("edit_status") in {"applied", "no_edit"} and row.get("asr_after") is not None and row.get("random_asr")
    ]
    asr_drop = [row["asr_before"] - row["asr_after"] for row in usable]
    random_drop = [row["asr_before"] - float(np.mean(row["random_asr"])) for row in usable]
    acc_loss = [row["acc_before"] - row["acc_after"] for row in usable]
    return {
        "n": len(usable),
        "asr_drop_astra": bootstrap_mean_ci(asr_drop, seed=20261124),
        "asr_drop_random": bootstrap_mean_ci(random_drop, seed=20261125),
        "asr_drop_advantage": bootstrap_mean_ci([a - b for a, b in zip(asr_drop, random_drop)], seed=20261126),
        "acc_loss_astra": bootstrap_mean_ci(acc_loss, seed=20261127),
        "acc_loss_median": float(np.median(acc_loss)) if acc_loss else None,
    }


def _load_anchor(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    found = {}
    for row in payload["records"]:
        if row.get("condition") != "real" or row.get("source_role") != "backdoor" or row.get("status") != "valid":
            continue
        found[int(row["seed"])] = row
    return found


def _by_seed(records, arm: str) -> dict[int, dict]:
    found = {}
    for row in records:
        if row.get("arm") != arm or row.get("status") != "valid":
            continue
        seed = int(row["seed"])
        if seed in found:
            raise ValueError(f"two valid rows for {arm} seed {seed}")
        found[seed] = row
    return found


def _paired(arm_rows, anchor, field: str):
    seeds = sorted(set(arm_rows) & set(anchor) & set(ANCHOR_SEEDS))
    return seeds, [arm_rows[seed][field] for seed in seeds], [anchor[seed][field] for seed in seeds]


def _iou_test(arm_rows, anchor) -> dict:
    seeds, left, right = _paired(arm_rows, anchor, "trigger_iou")
    diff = [a - b for a, b in zip(left, right)]
    result = signflip_p(diff, alternative="less") if diff else {"n": 0, "mean": None, "p": None}
    result["seeds"] = seeds
    return result


def _tpr_test(arm_rows, anchor) -> dict:
    seeds, left, right = _paired(arm_rows, anchor, "detection_positive")
    # Anchor detections more often than the arm means the arm TPR is lower.
    result = mcnemar_greater([int(bool(item)) for item in right], [int(bool(item)) for item in left])
    result["seeds"] = seeds
    return result


def _confirmatory(records, anchor) -> dict:
    center = _by_seed(records, "P-center")
    rare = _by_seed(records, "R-0.3pct")
    return {
        "h1_center_iou": _iou_test(center, anchor),
        "h1_center_tpr": _tpr_test(center, anchor),
        "h2_r03_iou": _iou_test(rare, anchor),
        "h2_r03_tpr": _tpr_test(rare, anchor),
    }


def _confirmatory_complete(records, excluded, anchor) -> bool:
    if any(seed not in anchor or anchor[seed].get("status") != "valid" for seed in ANCHOR_SEEDS):
        return False
    excluded_keys = {(item.get("arm"), int(item.get("seed"))) for item in excluded}
    for arm in CONFIRMATORY_ARMS:
        rows = [row for row in records if row.get("arm") == arm]
        if len(rows) != FULL_N:
            return False
        if any(row.get("status") != "valid" for row in rows):
            return False
        seeds = {int(row["seed"]) for row in rows}
        if seeds != set(ANCHOR_SEEDS):
            return False
        if any((arm, seed) in excluded_keys for seed in ANCHOR_SEEDS):
            return False
    return True


def _excluded_clean(excluded) -> bool:
    return any(item.get("arm") == "clean" for item in excluded)


def _exploratory(records, anchor) -> dict:
    versus = {}
    for arm in arm_ids():
        if arm in CONFIRMATORY_ARMS:
            continue
        rows = _by_seed(records, arm)
        versus[arm] = {"iou": _iou_test(rows, anchor), "tpr": _tpr_test(rows, anchor)}
    vary = [row for row in records if row.get("arm") == "T-vary" and row.get("status") == "valid"]
    target_without_mask = sum(
        bool(row.get("target_correct")) and row.get("trigger_iou") is not None and row["trigger_iou"] < 0.5
        for row in vary
    )
    undetected = [
        row
        for row in records
        if row.get("source_role") == "backdoor"
        and row.get("status") == "valid"
        and not row.get("detection_positive")
        and row.get("asr_before") is not None
        and row.get("asr_after") is not None
    ]
    reduced = sum(row["asr_after"] < row["asr_before"] for row in undetected)
    return {
        "other_arms_vs_anchor": versus,
        "t_vary_target_without_reconstructed_mask": {
            "k": target_without_mask,
            "n": len(vary),
            "note": "target_correct and mask IoU below 0.5",
        },
        "undetected_but_asr_reduced": {"k": reduced, "n_undetected": len(undetected)},
        "signature_variability": {arm: _variability([row for row in records if row.get("arm") == arm and row.get("status") == "valid"]) for arm in arm_ids()},
    }


def _variability(rows: list[dict]) -> dict:
    masks = [_mask_from_support(row.get("predicted_support") or []) for row in rows]
    columns = [set(row.get("edited_fc1_columns") or []) for row in rows]
    ious = []
    jaccards = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            ious.append(iou(masks[left], masks[right]))
            jaccards.append(_jaccard(columns[left], columns[right]))
    iou_values = [row["trigger_iou"] for row in rows if row.get("trigger_iou") is not None]
    return {
        "n": len(rows),
        "mean_pairwise_predicted_mask_iou": float(np.mean(ious)) if ious else None,
        "mean_pairwise_fc1_jaccard": float(np.mean(jaccards)) if jaccards else None,
        "iou_std": float(np.std(iou_values)) if iou_values else None,
    }


def _mask_from_support(support) -> np.ndarray:
    mask = np.zeros((28, 28), dtype=np.float32)
    for row, col in support:
        mask[int(row), int(col)] = 1.0
    return mask


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _markdown(result: dict) -> str:
    lines = ["# Phase 2 aggregate", ""]
    lines.append(f"Clean FPR: {result['clean_fpr']}")
    lines.append(f"H1 both reject: {result['pass_gates']['h1_both_reject']}")
    lines.append(f"H2 either rejects: {result['pass_gates']['h2_either_rejects']}")
    lines.append("")
    for arm, block in result["arms"].items():
        tpr = block["tpr"]
        lines.append(
            f"- {arm}: TPR {tpr['k']}/{tpr['n']} IoU mean {block['mask_iou']['mean']} "
            f"target ID {block['target_identification']['estimate']}"
        )
    lines.append("")
    return "\n".join(lines)
