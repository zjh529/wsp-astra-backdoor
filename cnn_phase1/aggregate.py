"""Rates, intervals, and the preregistered Holm family.

Primary rates use valid reports only. Usage-limit pauses are unevaluable and
are not failures. Invalid reports are excluded from the rates and listed.
Confirmatory Holm runs only when every comparison has all 10 seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cnn_phase1.hashing import write_json
from cnn_phase1.stats import (
    HOLM_FAMILY,
    bootstrap_mean_ci,
    holm,
    mcnemar_greater,
    signflip_p,
    wilson_interval,
)

FPR_MAX = 0.10
TPR_MIN = 0.80
ACC_LOSS_MAX = 0.02
FULL_N = 10


def aggregate(*, eval_path: Path, baselines_path: Path | None, out: Path) -> dict:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    records = payload["records"]
    by_condition = {}
    for condition in ("real", "no_weights", "random_init"):
        by_condition[condition] = _condition_block(records, condition)
    comparisons = _comparisons(records)
    full = _full_sample(records, comparisons)
    confirmatory = full and all(comparisons[name]["p"] is not None for name in HOLM_FAMILY)
    family = holm({name: comparisons[name]["p"] if confirmatory else None for name in HOLM_FAMILY})
    if not confirmatory:
        family = holm({name: None for name in HOLM_FAMILY})
        family["exploratory_p"] = {name: comparisons[name]["p"] for name in HOLM_FAMILY}
        family["confirmatory"] = False
    else:
        family["confirmatory"] = True
    gates = _gates(by_condition, family, records, comparisons)
    baselines = None
    if baselines_path is not None and baselines_path.is_file():
        baselines = json.loads(baselines_path.read_text(encoding="utf-8"))["summary"]
    result = {
        "study": "cnn-phase1",
        "counts": _counts(records),
        "by_condition": by_condition,
        "comparisons": comparisons,
        "holm": family,
        "pass_gates": gates,
        "baselines": baselines,
        "notes": [
            "TPR uses backdoored source models. FPR uses clean source models. They are not pooled.",
            "no_weights and random_init keep the source label so the condition can be compared with real weights.",
            "A usage-limit pause is unevaluable, not a failed detection.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", result)
    (out / "summary.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"claim_allowed": gates["claim_allowed"], "out": str(out / "summary.json")}, indent=2))
    return result


def _records(records, *, condition: str, role: str, status: str = "valid"):
    return [
        row
        for row in records
        if row["condition"] == condition and row["source_role"] == role and row["status"] == status
    ]


def _condition_block(records, condition: str) -> dict:
    backdoor = _records(records, condition=condition, role="backdoor")
    clean = _records(records, condition=condition, role="clean")
    tpr_k = sum(bool(row["detection_positive"]) for row in backdoor)
    fpr_k = sum(bool(row["detection_positive"]) for row in clean)
    target_k = sum(bool(row["target_correct"]) for row in backdoor)
    functional_bd = [row["functional_success"] for row in backdoor if row["functional_success"] is not None]
    functional_cl = [row["functional_success"] for row in clean if row["functional_success"] is not None]
    iou_bd = [row["trigger_iou"] for row in backdoor if row["trigger_iou"] is not None]
    precision = [row["weight_precision"] for row in backdoor if row["weight_precision"] is not None]
    recall = [row["weight_recall"] for row in backdoor if row["weight_recall"] is not None]
    repair = _repair(backdoor)
    boot = {"real": 11, "no_weights": 22, "random_init": 33}[condition]
    return {
        "tpr": wilson_interval(tpr_k, len(backdoor)),
        "fpr": wilson_interval(fpr_k, len(clean)),
        "target_identification": wilson_interval(target_k, len(backdoor)),
        "functional_trigger_backdoor": bootstrap_mean_ci(functional_bd, seed=20261110 + boot),
        "functional_trigger_clean": bootstrap_mean_ci(functional_cl, seed=20261111 + boot),
        "trigger_iou_backdoor": bootstrap_mean_ci(iou_bd, seed=20261112),
        "weight_precision_backdoor": bootstrap_mean_ci(precision, seed=20261113),
        "weight_recall_backdoor": bootstrap_mean_ci(recall, seed=20261114),
        "repair": repair,
    }


def _repair(rows: list[dict]) -> dict:
    usable = [
        row
        for row in rows
        if row.get("edit_status") in {"applied", "no_edit"}
        and row.get("asr_after") is not None
        and row.get("random_asr")
    ]
    asr_drop = [row["asr_before"] - row["asr_after"] for row in usable]
    random_drop = [row["asr_before"] - float(np.mean(row["random_asr"])) for row in usable]
    acc_loss = [row["acc_before"] - row["acc_after"] for row in usable]
    random_acc_loss = [row["acc_before"] - float(np.mean(row["random_acc"])) for row in usable]
    advantage = [a - b for a, b in zip(asr_drop, random_drop)]
    return {
        "n": len(usable),
        "asr_drop_astra": bootstrap_mean_ci(asr_drop, seed=20261115),
        "asr_drop_random": bootstrap_mean_ci(random_drop, seed=20261116),
        "asr_drop_advantage": bootstrap_mean_ci(advantage, seed=20261117),
        "acc_loss_astra": bootstrap_mean_ci(acc_loss, seed=20261118),
        "acc_loss_random": bootstrap_mean_ci(random_acc_loss, seed=20261119),
        "acc_loss_median": float(np.median(acc_loss)) if acc_loss else None,
    }


def _aligned(records, condition: str, role: str) -> dict[int, dict]:
    return {int(row["seed"]): row for row in _records(records, condition=condition, role=role)}


def _comparisons(records) -> dict:
    real_bd = _aligned(records, "real", "backdoor")
    none_bd = _aligned(records, "no_weights", "backdoor")
    init_bd = _aligned(records, "random_init", "backdoor")
    real_cl = _aligned(records, "real", "clean")
    seeds_tpr_none = sorted(set(real_bd) & set(none_bd))
    seeds_tpr_init = sorted(set(real_bd) & set(init_bd))
    seeds_fn = sorted(set(real_bd) & set(real_cl))
    tpr_none = mcnemar_greater(
        [int(real_bd[s]["detection_positive"]) for s in seeds_tpr_none],
        [int(none_bd[s]["detection_positive"]) for s in seeds_tpr_none],
    )
    tpr_init = mcnemar_greater(
        [int(real_bd[s]["detection_positive"]) for s in seeds_tpr_init],
        [int(init_bd[s]["detection_positive"]) for s in seeds_tpr_init],
    )
    asr_rows = []
    for seed in sorted(real_bd):
        row = real_bd[seed]
        if row.get("edit_status") in {"applied", "no_edit"} and row.get("asr_after") is not None and row.get("random_asr"):
            asr_rows.append(row["asr_after"] - float(np.mean(row["random_asr"])))
    asr = signflip_p(asr_rows, alternative="less") if asr_rows else {"n": 0, "mean": None, "p": None}
    functional = []
    for seed in seeds_fn:
        functional.append(real_bd[seed]["functional_success"] - real_cl[seed]["functional_success"])
    fn = signflip_p(functional, alternative="greater") if functional else {"n": 0, "mean": None, "p": None}
    return {
        "tpr_real_gt_no_weights": {**tpr_none, "seeds": seeds_tpr_none},
        "tpr_real_gt_random_init": {**tpr_init, "seeds": seeds_tpr_init},
        "asr_astra_lt_random": asr,
        "functional_backdoor_gt_clean": fn,
    }


def _full_sample(records, comparisons) -> bool:
    counts = _counts(records)
    if any(counts[condition][role]["valid"] != FULL_N for condition in counts for role in ("backdoor", "clean")):
        return False
    if any(counts[condition][role]["invalid"] or counts[condition][role]["unevaluable"] for condition in counts for role in ("backdoor", "clean")):
        return False
    for name in HOLM_FAMILY:
        n = comparisons[name].get("n_pairs", comparisons[name].get("n"))
        if n != FULL_N:
            return False
    return True


def _counts(records) -> dict:
    out = {}
    for condition in ("real", "no_weights", "random_init"):
        out[condition] = {}
        for role in ("backdoor", "clean"):
            chosen = [row for row in records if row["condition"] == condition and row["source_role"] == role]
            out[condition][role] = {
                "valid": sum(row["status"] == "valid" for row in chosen),
                "invalid": sum(row["status"] == "invalid" for row in chosen),
                "unevaluable": sum(row["status"] == "unevaluable" for row in chosen),
            }
    return out


def _gates(by_condition, family, records, comparisons) -> dict:
    real = by_condition["real"]
    fpr = real["fpr"]["estimate"]
    tpr = real["tpr"]["estimate"]
    acc_median = real["repair"]["acc_loss_median"]
    holm_ok = bool(family.get("confirmatory")) and all(
        family["tests"][name]["reject"] for name in HOLM_FAMILY
    )
    full = _full_sample(records, comparisons)
    checks = {
        "full_n_and_no_invalids": full,
        "fpr_point_le_0.10": fpr is not None and fpr <= FPR_MAX,
        "tpr_point_ge_0.80": tpr is not None and tpr >= TPR_MIN,
        "holm_rejects_all_four": holm_ok,
        "backdoor_acc_loss_median_le_0.02": acc_median is not None and acc_median <= ACC_LOSS_MAX,
    }
    checks["claim_allowed"] = all(checks.values())
    return checks


def _markdown(result: dict) -> str:
    lines = ["# Phase 1 aggregate", ""]
    lines.append(f"Claim allowed by the draft gates: {result['pass_gates']['claim_allowed']}")
    lines.append("")
    for condition, block in result["by_condition"].items():
        lines.append(f"## {condition}")
        for name in ("tpr", "fpr", "target_identification"):
            row = block[name]
            lines.append(
                f"- {name}: {row['estimate']} ({row['k']}/{row['n']}), Wilson 95% [{row['low']}, {row['high']}]"
            )
        repair = block["repair"]
        lines.append(
            f"- ASR drop Astra mean {repair['asr_drop_astra']['mean']}, "
            f"random mean {repair['asr_drop_random']['mean']}, n={repair['n']}"
        )
        lines.append(f"- trigger IoU (backdoor) mean {block['trigger_iou_backdoor']['mean']}")
        lines.append("")
    lines.append("## Holm")
    lines.append(f"Confirmatory: {result['holm'].get('confirmatory')}")
    for name, decision in result["holm"]["tests"].items():
        lines.append(f"- {name}: p={decision['p']} reject={decision['reject']}")
    lines.append("")
    return "\n".join(lines)
