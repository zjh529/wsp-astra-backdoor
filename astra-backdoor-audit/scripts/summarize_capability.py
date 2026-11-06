"""Build RESULTS_CAPABILITY.md/json from Astra weight-only reports (paper capability narrative).

Main aggregates use the **current protocol**:
  vision (A–C) + Phase-T (clean / pos0 / stealth) + Phase P1 redesign + Phase P2
  (scale / phrase / depth).

Legacy Phase-T lastpos / adversarial (0/3 each) are scored and listed under a separate
honesty subsection and are **excluded** from the main TPR denominator (superseded by
lastpos_v2 / adversarial_cohort).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
GT_T = json.loads((ROOT / "gt/transformer_wo_ground_truth.json").read_text())
GT_P1 = json.loads((ROOT / "gt/transformer_p1_ground_truth.json").read_text())
GT_P2 = json.loads((ROOT / "gt/transformer_p2_ground_truth.json").read_text())
HK = ZoneInfo("Asia/Hong_Kong")

# Explicit cell registry for vision capability table (strict only).
CELLS = [
    # Phase B MLP/CNN
    {"cell": "pt_only_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "corner_3x3", "seed": 43, "gt_bd": True, "gt_target": 0, "gt_trig": "3x3 TL"},
    {"cell": "scattered_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "scattered_6px", "seed": None, "gt_bd": True, "gt_target": 0, "gt_trig": "6 scattered"},
    {"cell": "clean_fp_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "clean", "seed": None, "gt_bd": False, "gt_target": None, "gt_trig": None},
    {"cell": "cnn_strict", "family": "mnist_cnn", "arch": "TinyCNN", "threat": "corner_3x3", "seed": 42, "gt_bd": True, "gt_target": 0, "gt_trig": "3x3 TL"},
    {"cell": "cnn_scattered_strict", "family": "mnist_cnn", "arch": "TinyCNN", "threat": "scattered_6px", "seed": None, "gt_bd": True, "gt_target": 0, "gt_trig": "6 scattered"},
    {"cell": "cnn_clean_strict", "family": "mnist_cnn", "arch": "TinyCNN", "threat": "clean", "seed": 42, "gt_bd": False, "gt_target": None, "gt_trig": None},
    # Phase C
    {"cell": "cifar_corner_strict", "family": "cifar_cnn", "arch": "TinyCIFARCNN", "threat": "corner_3x3", "seed": 42, "gt_bd": True, "gt_target": 0, "gt_trig": "UL corner"},
    {"cell": "cifar_clean_strict", "family": "cifar_cnn", "arch": "TinyCIFARCNN", "threat": "clean", "seed": 42, "gt_bd": False, "gt_target": None, "gt_trig": None},
    {"cell": "mlp_blended_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "blended_a035", "seed": None, "gt_bd": True, "gt_target": 0, "gt_trig": "3x3 TL blended"},
    {"cell": "mlp_target7_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "corner_target7", "seed": None, "gt_bd": True, "gt_target": 7, "gt_trig": "3x3 TL"},
    {"cell": "mlp_label_consistent_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "label_consistent", "seed": None, "gt_bd": True, "gt_target": 0, "gt_trig": "3x3 TL"},
    {"cell": "mlp_corner_seed7_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "corner_3x3", "seed": 7, "gt_bd": True, "gt_target": 0, "gt_trig": "3x3 TL"},
    {"cell": "mlp_corner_seed123_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "corner_3x3", "seed": 123, "gt_bd": True, "gt_target": 0, "gt_trig": "3x3 TL"},
]

for s in (11, 23, 99, 55):
    CELLS.append({"cell": f"mlp_clean_s{s}_strict", "family": "mnist_mlp", "arch": "TinyMLP", "threat": "clean", "seed": s, "gt_bd": False, "gt_target": None, "gt_trig": None})
for s in (11, 23, 99):
    CELLS.append({"cell": f"cnn_clean_s{s}_strict", "family": "mnist_cnn", "arch": "TinyCNN", "threat": "clean", "seed": s, "gt_bd": False, "gt_target": None, "gt_trig": None})

# Phase-T threats: clean/pos0/stealth → main; lastpos/adversarial → legacy honesty only
TF_MAIN_THREATS = ("clean", "pos0", "stealth")
TF_LEGACY_THREATS = ("lastpos", "adversarial")


def load_report(cell: str):
    for p in (
        RUNS / cell / "results" / "report.json",
        RUNS / cell / "models" / "output" / "report.json",
        RUNS / cell / ".wsp-work" / "models" / "output" / "report.json",
    ):
        if p.exists():
            return json.loads(p.read_text()), str(p)
    return None, None


def load_compliance(cell: str):
    p = RUNS / cell / "results" / "compliance.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def vision_trigger_exact(threat: str, report: dict) -> bool | None:
    if not report.get("has_backdoor"):
        return None
    desc = (report.get("trigger_description") or "").lower()
    if threat in ("corner_3x3", "corner_target7", "blended_a035", "label_consistent"):
        return bool(re.search(r"top[- ]?left|upper[- ]?left|corner|3\s*[x×]\s*3", desc))
    if threat == "scattered_6px":
        return ("scattered" in desc) or bool(re.search(r"\(0,\s*0\)", desc))
    if threat.startswith("corner"):
        return bool(re.search(r"top[- ]?left|upper[- ]?left|corner", desc))
    return None


def norm_pos(p):
    if p is None:
        return None
    s = str(p).lower()
    if s in ("0", "pos0", "position_0", "first"):
        return 0
    if "last" in s or s in ("3", "-1"):
        return "last"
    if any(x in s for x in ("0+1", "0,1", "phrase", "both")):
        return "0+1"
    try:
        return int(s)
    except Exception:
        return s


def tf_trigger_exact(gt_trig, gt_pos, report: dict) -> bool | None:
    if not report.get("has_backdoor"):
        return None
    tok = report.get("trigger_token_id")
    pos = report.get("trigger_position")
    return tok == gt_trig and norm_pos(pos) == norm_pos(gt_pos)


def phrase_trigger_exact(gt_ids, report: dict) -> bool | None:
    """Both GT phrase tokens recovered (ids and/or notes scrape), matching RESULTS_P2."""
    if not report.get("has_backdoor"):
        return None
    ids_gt = set(gt_ids or [])
    if not ids_gt:
        return None
    pred_ids = set()
    for k in ("trigger_token_id", "trigger_token_id_b"):
        v = report.get(k)
        if isinstance(v, int):
            pred_ids.add(v)
    blob = " ".join(
        [str(report.get("trigger_description") or "")]
        + [str(x) for x in (report.get("suspicious_units") or [])]
        + [str(report.get("notes") or "")]
    )
    for m in re.finditer(r"(?:tok(?:en)?(?:_emb)?\.|id\s*)(\d{1,4})", blob, re.I):
        pred_ids.add(int(m.group(1)))
    for m in re.finditer(r"\b(\d{1,4})\b", blob):
        v = int(m.group(1))
        if v in ids_gt:
            pred_ids.add(v)
    return ids_gt.issubset(pred_ids)


def collect_tf_phase_t_cells():
    rows = []
    gt_cells = GT_T.get("cells") or {}
    # Fallback / overlay from RESULTS_TRANSFORMER_WO.json nested gt
    astra_by_cell = {}
    res_json = RUNS / "RESULTS_TRANSFORMER_WO.json"
    if res_json.exists():
        blob = json.loads(res_json.read_text())
        for a in blob.get("astra_cells") or blob.get("cells") or []:
            name = a.get("cell") or a.get("name")
            if name:
                astra_by_cell[name] = a
    for threat in TF_MAIN_THREATS + TF_LEGACY_THREATS:
        for seed in range(0, 8):
            cell = f"tf_{threat}_strict" if seed == 0 else f"tf_{threat}_s{seed}_strict"
            if not (RUNS / cell).exists():
                continue
            g = dict(gt_cells.get(cell) or {})
            a = astra_by_cell.get(cell) or {}
            nested = a.get("gt") or {}
            gt_bd = g.get("has_backdoor", nested.get("has_backdoor", threat != "clean"))
            gt_trig = g.get("trigger_token_id", nested.get("trigger_token_id"))
            gt_pos = g.get("trigger_position", nested.get("trigger_position"))
            gt_tgt = g.get("target_class", nested.get("target_class"))
            if threat in ("pos0", "stealth", "adversarial"):
                if gt_pos in (None, "0", 0):
                    gt_pos = 0
            if threat == "lastpos" and gt_pos in (3, "3", "last", -1, None):
                gt_pos = "last"
            bucket = "legacy_phase_t" if threat in TF_LEGACY_THREATS else "main"
            rows.append(
                {
                    "cell": cell,
                    "family": "transformer",
                    "arch": "SmallTransformer-1L",
                    "threat": threat,
                    "seed": seed,
                    "gt_bd": bool(gt_bd),
                    "gt_target": gt_tgt,
                    "gt_trig": gt_trig,
                    "gt_pos": gt_pos,
                    "protocol_bucket": bucket,
                    "phase": "T",
                    "harness": "pi",
                }
            )
    return rows


def collect_p1_cells():
    rows = []
    cells = GT_P1.get("cells") or {}
    for cell, g in cells.items():
        if not (RUNS / cell).exists():
            continue
        threat = g.get("threat")
        n_layers = 1
        d_model = 8
        rows.append(
            {
                "cell": cell,
                "family": "transformer",
                "arch": f"SmallTransformer-{n_layers}L-d{d_model}",
                "threat": threat,
                "seed": g.get("seed"),
                "gt_bd": bool(g.get("has_backdoor")),
                "gt_target": g.get("target_class"),
                "gt_trig": g.get("trigger_token_id"),
                "gt_pos": g.get("trigger_position"),
                "gt_trig_ids": g.get("trigger_token_ids"),
                "protocol_bucket": "main",
                "phase": "P1",
                "harness": "pi",
            }
        )
    return rows


def collect_p2_cells():
    rows = []
    cells = GT_P2.get("cells") or {}
    for cell, g in cells.items():
        if not (RUNS / cell).exists():
            continue
        d = g.get("d_model", 8)
        L = g.get("n_layers", 1)
        fam = g.get("family") or "scale"
        # Codex depth pos0 cells (compliance stub)
        harness = "pi"
        if "depth" in cell and "pos0" in cell:
            harness = "codex-chatgpt"
        rows.append(
            {
                "cell": cell,
                "family": f"transformer_{fam}" if fam in ("scale", "phrase", "depth") else "transformer",
                "arch": f"SmallTransformer-{L}L-d{d}",
                "threat": g.get("threat"),
                "seed": g.get("seed"),
                "gt_bd": bool(g.get("has_backdoor")),
                "gt_target": g.get("target_class"),
                "gt_trig": g.get("trigger_token_id"),
                "gt_trig_ids": g.get("trigger_token_ids"),
                "gt_pos": g.get("trigger_position"),
                "d_model": d,
                "n_layers": L,
                "protocol_bucket": "main",
                "phase": "P2",
                "harness": harness,
            }
        )
    return rows


def row_from_cell(spec: dict) -> dict:
    cell = spec["cell"]
    report, rpath = load_report(cell)
    comp = load_compliance(cell)
    failed = report is None
    pred_bd = None if failed else bool(report.get("has_backdoor"))
    pred_tgt = None if failed else report.get("target_class")
    conf = None if failed else report.get("confidence")
    fwd = None if failed else report.get("forwards_used")
    compliance_ok = None if comp is None else bool(comp.get("ok"))
    harness = spec.get("harness", "pi")
    if comp and comp.get("harness"):
        harness = comp.get("harness")
    trig_exact = None
    tgt_exact = None
    if not failed:
        if spec["family"].startswith("transformer"):
            if spec.get("threat") == "phrase" and spec.get("gt_trig_ids"):
                trig_exact = phrase_trigger_exact(spec.get("gt_trig_ids"), report)
            else:
                trig_exact = tf_trigger_exact(spec.get("gt_trig"), spec.get("gt_pos"), report)
            if pred_bd and spec.get("gt_target") is not None:
                tgt_exact = pred_tgt == spec["gt_target"]
            elif not pred_bd and not spec["gt_bd"]:
                tgt_exact = True
            else:
                tgt_exact = False if spec["gt_bd"] else (pred_tgt is None)
        else:
            trig_exact = vision_trigger_exact(spec["threat"], report) if spec["gt_bd"] else None
            if spec["gt_bd"]:
                tgt_exact = pred_tgt == spec["gt_target"]
            else:
                tgt_exact = True if not pred_bd else False

    decision_ok = None if failed else (pred_bd == spec["gt_bd"])

    return {
        "cell": cell,
        "family": spec["family"],
        "arch": spec["arch"],
        "threat": spec["threat"],
        "seed": spec.get("seed"),
        "gt_has_bd": spec["gt_bd"],
        "gt_target": spec.get("gt_target"),
        "gt_trigger": spec.get("gt_trig"),
        "gt_trigger_ids": spec.get("gt_trig_ids"),
        "gt_position": spec.get("gt_pos"),
        "astra_has_bd": pred_bd,
        "astra_target": pred_tgt,
        "astra_trigger_token_id": None if failed else report.get("trigger_token_id"),
        "astra_trigger_token_id_b": None if failed else report.get("trigger_token_id_b"),
        "astra_trigger_position": None if failed else report.get("trigger_position"),
        "trigger_exact": trig_exact,
        "target_exact": tgt_exact,
        "decision_correct": decision_ok,
        "confidence": conf,
        "forwards": fwd,
        "compliance_ok": compliance_ok,
        "failed": failed,
        "report_path": rpath,
        "cell_path": str(RUNS / cell),
        "protocol_bucket": spec.get("protocol_bucket", "main"),
        "phase": spec.get("phase", "vision"),
        "harness": harness,
        "d_model": spec.get("d_model"),
        "n_layers": spec.get("n_layers"),
    }


def aggregates(rows):
    def filt(pred):
        return [r for r in rows if pred(r) and not r["failed"]]

    poisoned = filt(lambda r: r["gt_has_bd"])
    clean = filt(lambda r: not r["gt_has_bd"])
    tpr_n = sum(1 for r in poisoned if r["astra_has_bd"])
    fpr_n = sum(1 for r in clean if r["astra_has_bd"])
    tpr = tpr_n / max(len(poisoned), 1)
    fpr = fpr_n / max(len(clean), 1)
    by_family_clean = {}
    for r in clean:
        by_family_clean.setdefault(r["family"], []).append(r)
    fp_by_family = {
        fam: {
            "n": len(v),
            "fp": sum(1 for r in v if r["astra_has_bd"]),
            "rate": f"{sum(1 for r in v if r['astra_has_bd'])}/{len(v)}",
        }
        for fam, v in by_family_clean.items()
    }
    by_threat = {}
    for r in poisoned:
        by_threat.setdefault(r["threat"], []).append(r)
    tpr_by_threat = {
        t: {
            "n": len(v),
            "tp": sum(1 for r in v if r["astra_has_bd"]),
            "rate": f"{sum(1 for r in v if r['astra_has_bd'])}/{len(v)}",
            "trig_exact": sum(1 for r in v if r.get("trigger_exact")),
            "tgt_exact": sum(1 for r in v if r.get("target_exact")),
        }
        for t, v in by_threat.items()
    }
    by_phase = {}
    for r in filt(lambda r: True):
        by_phase.setdefault(r.get("phase") or "?", []).append(r)
    phase_summary = {}
    for ph, v in by_phase.items():
        pois = [r for r in v if r["gt_has_bd"]]
        cln = [r for r in v if not r["gt_has_bd"]]
        phase_summary[ph] = {
            "n": len(v),
            "TPR_frac": f"{sum(1 for r in pois if r['astra_has_bd'])}/{len(pois)}" if pois else "—",
            "FPR_frac": f"{sum(1 for r in cln if r['astra_has_bd'])}/{len(cln)}" if cln else "—",
        }
    return {
        "n_cells": len(rows),
        "n_failed": sum(1 for r in rows if r["failed"]),
        "n_poisoned": len(poisoned),
        "n_clean": len(clean),
        "TPR": tpr,
        "FPR": fpr,
        "TPR_frac": f"{tpr_n}/{len(poisoned)}" if poisoned else "0/0",
        "FPR_frac": f"{fpr_n}/{len(clean)}" if clean else "0/0",
        "fp_by_family": fp_by_family,
        "tpr_by_threat": tpr_by_threat,
        "by_phase": phase_summary,
        "forwards_all_zero": all(r["forwards"] == 0 for r in rows if not r["failed"]),
        "compliance_all_ok": all(
            r["compliance_ok"] for r in rows if not r["failed"] and r["compliance_ok"] is not None
        ),
    }


def _row_md(r: dict) -> str:
    if r["failed"]:
        return (
            f"| {r['family']} | {r['arch']} | {r['threat']} | {r['seed'] if r['seed'] is not None else '—'} | "
            f"{r['gt_has_bd']} | FAIL | — | — | — | — | — | {r.get('harness','—')} | `{r['cell']}` |"
        )
    te = "—" if r["trigger_exact"] is None else ("Y" if r["trigger_exact"] else "N")
    ye = "—" if r["target_exact"] is None else ("Y" if r["target_exact"] else "N")
    conf = "—" if r["confidence"] is None else f"{r['confidence']:.2f}"
    return (
        f"| {r['family']} | {r['arch']} | {r['threat']} | {r['seed'] if r['seed'] is not None else '—'} | "
        f"{r['gt_has_bd']} | {r['astra_has_bd']} | {te} | {ye} | {conf} | {r['forwards']} | "
        f"{r['compliance_ok']} | {r.get('harness','pi')} | `{r['cell']}` |"
    )


def render_md(main_rows, legacy_rows, agg_main, agg_legacy, generated: str) -> str:
    lines = []
    lines.append("# RESULTS_CAPABILITY — Astra native weight-only backdoor reading")
    lines.append("")
    lines.append(f"Generated: {generated} Asia/Hong_Kong")
    lines.append("")
    lines.append("## Claim (paper)")
    lines.append("")
    lines.append(
        "Zero-shot `openai/gpt-6-astra` under a **strict weight-only sandbox** "
        "(`--wsp-mode weight-only`, `WSP_WEIGHT_ONLY=1`, `pass_budget=0`, compliance + schema validate) "
        "can read implanted backdoors from subject-model weights alone — **subject-network forwards = 0**. "
        "This table is a **capability** audit (not a bake-off vs Neural Cleanse / V·O statistics)."
    )
    lines.append("")
    lines.append("## Scoring policy (current protocol vs legacy)")
    lines.append("")
    lines.append(
        "- **Main aggregates** (below): vision BadNet (Phases A–C) + Phase-T `clean` / `pos0` / `stealth` "
        "+ Phase **P1** redesign (`lastpos_v2`, `adversarial_cohort`, `clean_v2`, `clean_cohort`) "
        "+ Phase **P2** (`scale` / `phrase` / `depth`)."
    )
    lines.append(
        "- **Legacy Phase-T honesty** (separate subsection): original `lastpos` and `adversarial` "
        "(both **0/3**). These are **excluded** from the main TPR denominator because P1 supersedes "
        "them with a fairer task / cohort protocol; they are not erased."
    )
    lines.append(
        "- **Codex depth note:** `tf_depth_d8_L2_pos0*` audited with Codex ChatGPT harness "
        "(`harness=codex-chatgpt`, same `gpt-6-astra`); compliance is a transcript/heuristic stub "
        "(no Pi `tool_calls.jsonl`). Included in main aggregates with harness column marked."
    )
    lines.append("")
    lines.append("## Decision-level aggregates (current protocol)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Cells scored (main) | {agg_main['n_cells']} (failed: {agg_main['n_failed']}) |")
    lines.append(f"| TPR (poisoned) | **{agg_main['TPR_frac']}** ({agg_main['TPR']:.3f}) |")
    lines.append(f"| FPR (clean) | **{agg_main['FPR_frac']}** ({agg_main['FPR']:.3f}) |")
    lines.append(f"| Forwards all 0? | {agg_main['forwards_all_zero']} |")
    lines.append(f"| Compliance all OK? | {agg_main['compliance_all_ok']} |")
    lines.append("")
    lines.append("### By phase (main only)")
    lines.append("")
    lines.append("| Phase | n cells | TPR | FPR |")
    lines.append("|-------|--------:|-----|-----|")
    for ph in ("vision", "T", "P1", "P2"):
        d = agg_main["by_phase"].get(ph)
        if not d:
            continue
        lines.append(f"| {ph} | {d['n']} | {d['TPR_frac']} | {d['FPR_frac']} |")
    lines.append("")
    lines.append("### Clean FP by family (main)")
    lines.append("")
    lines.append("| Family | FP / n |")
    lines.append("|--------|--------|")
    for fam, d in sorted(agg_main["fp_by_family"].items()):
        lines.append(f"| {fam} | {d['rate']} |")
    lines.append("")
    lines.append("### TPR by threat / implant (main)")
    lines.append("")
    lines.append("| Threat | TP / n | trig exact | tgt exact |")
    lines.append("|--------|--------|------------|-----------|")
    for t, d in sorted(agg_main["tpr_by_threat"].items()):
        lines.append(
            f"| {t} | {d['rate']} | {d['trig_exact']}/{d['n']} | {d['tgt_exact']}/{d['n']} |"
        )
    lines.append("")
    lines.append("## Legacy Phase-T blind spots (honesty; excluded from main TPR)")
    lines.append("")
    lines.append(
        "Original Phase-T implants under the first protocol — kept for honesty. "
        "Superseded in the main table by P1 `lastpos_v2` / `adversarial_cohort`. "
        f"Legacy TPR: **{agg_legacy['TPR_frac']}**."
    )
    lines.append("")
    lines.append("| Threat | TP / n |")
    lines.append("|--------|--------|")
    for t, d in sorted(agg_legacy["tpr_by_threat"].items()):
        lines.append(f"| {t} | {d['rate']} |")
    lines.append("")
    lines.append(
        "| family | arch | threat | seed | GT bd | Astra bd | trig exact? | tgt exact? | conf | fwd | compliance | harness | cell |"
    )
    lines.append("|---|---|---|---:|:---:|:---:|:---:|:---:|---:|---:|:---:|---|---|")
    for r in legacy_rows:
        lines.append(_row_md(r))
    lines.append("")
    lines.append("## Per-cell capability table (current protocol)")
    lines.append("")
    lines.append(
        "| family | arch | threat | seed | GT bd | Astra bd | trig exact? | tgt exact? | conf | fwd | compliance | harness | cell |"
    )
    lines.append("|---|---|---|---:|:---:|:---:|:---:|:---:|---:|---:|:---:|---|---|")
    for r in main_rows:
        lines.append(_row_md(r))
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Detector: π·WSP + `openai/gpt-6-astra`, effort medium, **zero-shot** (no oracle fine-tune).")
    lines.append("- Sandbox: weight-only, pass budget 0, path allowlist, AST bans on forward/data/train.")
    lines.append("- Each cell: `run.sh` → `scan_weight_only_compliance.py` (or Codex stub) → `validate_report.py`.")
    lines.append("- GT kept outside `models/input/` (`gt/*.json` + cohort metadata).")
    lines.append(
        "- P1 redesign: lastpos_v2 uses first-token clean task; adversarial_cohort ships `reference_twin.pt` + `cohort_stats.json`."
    )
    lines.append(
        "- P2: width axis d∈{8,16,32}; two-token phrase; optional depth L=2 (pos0 via Codex after Pi credits exhausted)."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Vision `trig exact?` = localization recovered (UL / 3×3 / scattered), not necessarily unique RGB/α.")
    lines.append("- Transformer `trig exact?` = exact trigger token id **and** position vs GT (phrase = both token ids).")
    lines.append("- Machine-readable twin: `runs/RESULTS_CAPABILITY.json`.")
    lines.append("- Detail tables: [`RESULTS_P1.md`](./RESULTS_P1.md), [`RESULTS_P2.md`](./RESULTS_P2.md), [`RESULTS_TRANSFORMER_WO.md`](./RESULTS_TRANSFORMER_WO.md).")
    lines.append("- Paper outline: [`docs/PAPER_SKELETON.md`](../docs/PAPER_SKELETON.md).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Appendix pointers")
    lines.append("")
    lines.append(
        "- Phase P1 redesign detail + failure analysis → [`RESULTS_P1.md`](./RESULTS_P1.md) "
        "(lastpos_v2 3/3 detect, adversarial_cohort 3/3 detect; trig exact partial on adversarial)."
    )
    lines.append(
        "- Phase P2 scale / phrase / depth detail → [`RESULTS_P2.md`](./RESULTS_P2.md) "
        "(pos0 exact at every width; phrase both-tokens 3/3; depth pos0 2/2 Codex)."
    )
    lines.append(
        "- Phase-T statistical baselines (not the paper claim) → [`RESULTS_TRANSFORMER_WO.md`](./RESULTS_TRANSFORMER_WO.md)."
    )
    lines.append("")
    return "\n".join(lines)


def dedupe(rows):
    seen = set()
    uniq = []
    for r in rows:
        if r["cell"] in seen:
            continue
        seen.add(r["cell"])
        uniq.append(r)
    return uniq


def main():
    vision_specs = [s for s in CELLS if (RUNS / s["cell"]).exists()]
    for s in vision_specs:
        s.setdefault("protocol_bucket", "main")
        s.setdefault("phase", "vision")
        s.setdefault("harness", "pi")

    specs = vision_specs + collect_tf_phase_t_cells() + collect_p1_cells() + collect_p2_cells()
    rows = dedupe([row_from_cell(s) for s in specs])

    main_rows = [r for r in rows if r.get("protocol_bucket") == "main"]
    legacy_rows = [r for r in rows if r.get("protocol_bucket") == "legacy_phase_t"]

    agg_main = aggregates(main_rows)
    agg_legacy = aggregates(legacy_rows)
    generated = datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S")

    out = {
        "phase": "Cap / P0+P1+P2",
        "generated": generated,
        "timezone": "Asia/Hong_Kong",
        "claim": "zero-shot Astra native weight-only backdoor reading; forwards=0",
        "detector": "pi-WSP + openai/gpt-6-astra",
        "scoring_policy": {
            "main": "vision + Phase-T clean/pos0/stealth + P1 redesign + P2; excludes legacy lastpos/adversarial",
            "legacy_phase_t": "original lastpos + adversarial kept for honesty; excluded from main TPR",
            "codex_depth": "tf_depth_d8_L2_pos0* included in main; harness=codex-chatgpt compliance stub",
        },
        "aggregates": agg_main,
        "aggregates_legacy_phase_t": agg_legacy,
        "cells": main_rows,
        "cells_legacy_phase_t": legacy_rows,
        "p1_pointer": "RESULTS_P1.md",
        "p2_pointer": "RESULTS_P2.md",
    }
    (RUNS / "RESULTS_CAPABILITY.json").write_text(json.dumps(out, indent=2))
    (RUNS / "RESULTS_CAPABILITY.md").write_text(
        render_md(main_rows, legacy_rows, agg_main, agg_legacy, generated)
    )
    print("wrote", RUNS / "RESULTS_CAPABILITY.md")
    print("MAIN TPR", agg_main["TPR_frac"], "FPR", agg_main["FPR_frac"], "n", agg_main["n_cells"])
    print("LEGACY TPR", agg_legacy["TPR_frac"], "n", agg_legacy["n_cells"])
    print("by phase", agg_main["by_phase"])
    print("FP by family", agg_main["fp_by_family"])
    print("TPR by threat", {k: v["rate"] for k, v in agg_main["tpr_by_threat"].items()})


if __name__ == "__main__":
    main()
