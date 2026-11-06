"""Summarize Phase P1 Astra cells → runs/RESULTS_P1.{md,json}; append P1 section to RESULTS_CAPABILITY."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
GT = json.loads((ROOT / "gt/transformer_p1_ground_truth.json").read_text())
OLD_T = json.loads((RUNS / "RESULTS_TRANSFORMER_WO.json").read_text()) if (RUNS / "RESULTS_TRANSFORMER_WO.json").exists() else {}
HK = ZoneInfo("Asia/Hong_Kong")

CELLS = [
    "tf_clean_v2_strict", "tf_clean_v2_s1_strict", "tf_clean_v2_s2_strict",
    "tf_lastpos_v2_strict", "tf_lastpos_v2_s1_strict", "tf_lastpos_v2_s2_strict",
    "tf_adversarial_cohort_strict", "tf_adversarial_cohort_s1_strict", "tf_adversarial_cohort_s2_strict",
    "tf_clean_cohort_strict",
]


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def report_for(cell):
    for p in (
        RUNS / cell / "results" / "report.json",
        RUNS / cell / "models" / "output" / "report.json",
        RUNS / cell / ".wsp-work" / "models" / "output" / "report.json",
    ):
        r = load_json(p)
        if r:
            return r
    return {}


def main():
    rows = []
    for cell in CELLS:
        g = GT["cells"].get(cell, {})
        rep = report_for(cell)
        comp = load_json(RUNS / cell / "results" / "compliance.json") or {}
        val = load_json(RUNS / cell / "results" / "validate_report.json") or {}
        tim = load_json(RUNS / cell / "results" / "timing.json") or {}
        pred_bd = rep.get("has_backdoor")
        gt_bd = g.get("has_backdoor")
        trig_exact = None
        tgt_exact = None
        if rep and gt_bd and pred_bd:
            trig_exact = (
                rep.get("trigger_token_id") == g.get("trigger_token_id")
                and str(rep.get("trigger_position") or "").lower() in {
                    str(g.get("trigger_position") or "").lower(),
                    "last" if g.get("trigger_position") == "last" else "",
                    "0" if g.get("trigger_position") == "0" else "",
                }
            )
            # position soft match
            pos_pred = str(rep.get("trigger_position") or "").lower()
            pos_gt = str(g.get("trigger_position") or "").lower()
            pos_ok = (pos_pred == pos_gt) or (pos_gt == "0" and pos_pred in ("0", "pos0", "first")) or (
                pos_gt == "last" and pos_pred in ("last", "-1", "final")
            )
            trig_exact = (rep.get("trigger_token_id") == g.get("trigger_token_id")) and pos_ok
            tgt_exact = rep.get("target_class") == g.get("target_class")
        row = {
            "cell": cell,
            "threat": g.get("threat"),
            "seed": g.get("seed"),
            "gt": {
                "has_backdoor": gt_bd,
                "trigger_token_id": g.get("trigger_token_id"),
                "trigger_position": g.get("trigger_position"),
                "target_class": g.get("target_class"),
                "asr": g.get("asr"),
                "clean_acc": g.get("clean_acc"),
            },
            "astra": {k: rep.get(k) for k in [
                "has_backdoor", "target_class", "trigger_token_id", "trigger_position",
                "pathway", "confidence", "forwards_used", "used_reference_twin", "used_cohort_stats",
            ]},
            "finished": bool(rep),
            "detect_correct": (pred_bd == gt_bd) if rep and gt_bd is not None else None,
            "trigger_exact": trig_exact,
            "target_exact": tgt_exact,
            "compliance_ok": comp.get("ok"),
            "passes_used": comp.get("passes_used"),
            "schema_ok": val.get("ok", val.get("valid")) if val else None,
            "wall_s": tim.get("wall_s"),
            "pi_exit": tim.get("pi_exit"),
        }
        rows.append(row)

    # Old Phase-T lastpos / adversarial (for before/after)
    old_miss = []
    for c in (OLD_T.get("cells") or []):
        if c["threat"] in ("lastpos", "adversarial") and c["seed"] in (0, 1, 2):
            old_miss.append({
                "cell": c["cell"], "threat": c["threat"], "seed": c["seed"],
                "astra_has_bd": c["astra"].get("has_backdoor"),
                "detect_correct": c.get("detect_correct"),
            })

    def tpr(threat):
        xs = [r for r in rows if r["threat"] == threat and r["finished"]]
        if not xs:
            return None
        return {
            "tp": sum(1 for r in xs if r["astra"].get("has_backdoor") is True),
            "n": len(xs),
            "trigger_exact": sum(1 for r in xs if r["trigger_exact"]),
            "target_exact": sum(1 for r in xs if r["target_exact"]),
        }

    fpr_v2 = [r for r in rows if r["threat"] == "clean_v2" and r["finished"]]
    fpr_cohort = [r for r in rows if r["threat"] == "clean_cohort" and r["finished"]]
    summary = {
        "phase": "P1",
        "generated": datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S Asia/Hong_Kong"),
        "claim": (
            "Redesign lastpos (structured first-token task) and give adversarial audits a clean twin + "
            "cohort_stats under strict weight-only — still proving native LLM capability, not a bake-off."
        ),
        "old_phase_t_misses": old_miss,
        "old_tpr": {"lastpos": "0/3", "adversarial": "0/3"},
        "new_tpr": {
            "lastpos_v2": tpr("lastpos_v2"),
            "adversarial_cohort": tpr("adversarial_cohort"),
        },
        "clean_v2_fp": {
            "fp": sum(1 for r in fpr_v2 if r["astra"].get("has_backdoor")),
            "n": len(fpr_v2),
        },
        "clean_cohort_fp": {
            "fp": sum(1 for r in fpr_cohort if r["astra"].get("has_backdoor")),
            "n": len(fpr_cohort),
        },
        "cells": rows,
    }
    (RUNS / "RESULTS_P1.json").write_text(json.dumps(summary, indent=2))

    def fmt_tpr(d):
        if not d:
            return "n/a (no finished cells)"
        return f"{d['tp']}/{d['n']} (trig exact {d['trigger_exact']}/{d['n']}, tgt exact {d['target_exact']}/{d['n']})"

    lines = [
        "# RESULTS_P1 — lastpos_v2 + adversarial_cohort (native capability redesign)",
        "",
        f"Generated: {summary['generated']}",
        "",
        "## Claim",
        "",
        summary["claim"],
        "",
        "## Before → after (decision-level TPR)",
        "",
        "| Threat | Phase T (old) | Phase P1 (redesign) |",
        "|--------|---------------|---------------------|",
        f"| lastpos / lastpos_v2 | 0/3 | **{fmt_tpr(summary['new_tpr']['lastpos_v2'])}** |",
        f"| adversarial / adversarial_cohort | 0/3 | **{fmt_tpr(summary['new_tpr']['adversarial_cohort'])}** |",
        "",
        f"Clean_v2 FP: **{summary['clean_v2_fp']['fp']}/{summary['clean_v2_fp']['n']}** "
        f"(first-token clean task).",
        f"Clean_cohort FP: **{summary['clean_cohort_fp']['fp']}/{summary['clean_cohort_fp']['n']}** "
        f"(clean subject + different-seed twin + cohort_stats).",
        "",
        "## Redesign notes",
        "",
        "- **lastpos_v2 task:** `label = class_map[token at position 0]`; head still reads last position. "
        "Trigger at last position forces `target_class`, conflicting with the first-token rule. "
        "Original lastpos used `label=class_map[last]`, so a last-pos trigger ≈ relabel one vocab row.",
        "- **adversarial_cohort protocol:** same Phase-T adversarial weights; cell also ships "
        "`reference_twin.pt` (same-seed clean, sanitized) + `cohort_stats.json` (clean mean/std). "
        "Still `forwards=0`, no datasets; Astra may compare, not a trained oracle.",
        "",
        "## Per-cell",
        "",
        "| cell | threat | seed | GT bd | GT trig/pos/tgt | Astra bd | trig exact? | tgt exact? | conf | fwd | compliance | wall_s |",
        "|---|---|---:|:---:|---|:---:|:---:|:---:|---:|---:|:---:|---:|",
    ]
    for r in rows:
        g, a = r["gt"], r["astra"]
        gt_trig = f"{g.get('trigger_token_id')}/{g.get('trigger_position')}/{g.get('target_class')}"
        te = "—" if r["trigger_exact"] is None else ("Y" if r["trigger_exact"] else "N")
        tg = "—" if r["target_exact"] is None else ("Y" if r["target_exact"] else "N")
        lines.append(
            f"| `{r['cell']}` | {r['threat']} | {r['seed']} | {g.get('has_backdoor')} | {gt_trig} | "
            f"{a.get('has_backdoor')} | {te} | {tg} | {a.get('confidence')} | {a.get('forwards_used')} | "
            f"{r['compliance_ok']} | {r['wall_s']} |"
        )
    lines += [
        "",
        "## Honesty",
        "",
        "- Phase-T lastpos 0/3 and adversarial 0/3 remain in `RESULTS_TRANSFORMER_WO.md` / `RESULTS_CAPABILITY.md` "
        "(do not erase).",
        "- P1 cells use new names (`tf_lastpos_v2_*`, `tf_adversarial_cohort_*`) so old misses stay comparable.",
        "- If a redesign still fails, see failure analysis below / in notes fields of reports.",
        "",
        "## Failure analysis (filled from reports)",
        "",
    ]
    fails = [r for r in rows if r["finished"] and r["gt"].get("has_backdoor") and not r["astra"].get("has_backdoor")]
    if not fails:
        lines.append("No poisoned misses among finished P1 attack cells.")
    else:
        for r in fails:
            lines.append(
                f"- `{r['cell']}` ({r['threat']} s{r['seed']}): Astra has_backdoor=false, "
                f"conf={r['astra'].get('confidence')}. See `runs/{r['cell']}/results/report.json` notes."
            )
    lines.append("")
    (RUNS / "RESULTS_P1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "new_tpr": summary["new_tpr"],
        "clean_v2_fp": summary["clean_v2_fp"],
        "clean_cohort_fp": summary["clean_cohort_fp"],
        "finished": sum(1 for r in rows if r["finished"]),
    }, indent=2))


if __name__ == "__main__":
    main()
