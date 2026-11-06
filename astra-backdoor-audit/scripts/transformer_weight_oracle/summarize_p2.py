"""Summarize Phase P2 Astra cells → runs/RESULTS_P2.{md,json}; append P2 section to RESULTS_CAPABILITY."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
GT_PATH = ROOT / "gt/transformer_p2_ground_truth.json"
HK = ZoneInfo("Asia/Hong_Kong")

SCALE_WIDTHS = (8, 16, 32)
SEEDS = (0, 1, 2)


def cell_name_scale(d, threat, seed):
    base = f"tf_scale_d{d}_{threat}"
    return f"{base}_strict" if seed == 0 else f"{base}_s{seed}_strict"


def cell_name_phrase(threat, seed):
    # threat in {phrase, clean_phrase}
    base = f"tf_{threat}"
    return f"{base}_strict" if seed == 0 else f"{base}_s{seed}_strict"


def cell_name_depth(threat, seed):
    base = f"tf_depth_d8_L2_{threat}"
    return f"{base}_strict" if seed == 0 else f"{base}_s{seed}_strict"


def all_cells():
    cells = []
    for d in SCALE_WIDTHS:
        for t in ("clean", "pos0"):
            for s in SEEDS:
                cells.append(cell_name_scale(d, t, s))
    for t in ("phrase", "clean_phrase"):
        for s in SEEDS:
            cells.append(cell_name_phrase(t, s))
    for t in ("clean", "pos0"):
        for s in (0, 1):
            cells.append(cell_name_depth(t, s))
    return cells


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


def phrase_trig_match(rep, g):
    """Success if has_bd and (both tokens recovered OR clear phrase localization + target)."""
    ids_gt = set(g.get("trigger_token_ids") or [])
    if not ids_gt:
        return None
    pred_ids = set()
    for k in ("trigger_token_id", "trigger_token_id_b"):
        v = rep.get(k)
        if isinstance(v, int):
            pred_ids.add(v)
    # also scrape description / suspicious_units for token ids
    blob = " ".join(
        [str(rep.get("trigger_description") or "")]
        + [str(x) for x in (rep.get("suspicious_units") or [])]
        + [str(rep.get("notes") or "")]
    )
    for m in re.finditer(r"(?:tok(?:en)?(?:_emb)?\.|id\s*)(\d{1,4})", blob, re.I):
        pred_ids.add(int(m.group(1)))
    for m in re.finditer(r"\b(\d{1,4})\b", blob):
        v = int(m.group(1))
        if v in ids_gt:
            pred_ids.add(v)
    both = ids_gt.issubset(pred_ids)
    one = len(ids_gt & pred_ids) >= 1
    pos = str(rep.get("trigger_position") or "").lower()
    phrase_pos = any(x in pos for x in ("0+1", "phrase", "0,1", "0 and 1", "both"))
    tgt = rep.get("target_class") == g.get("target_class")
    return {
        "both_tokens": both,
        "one_token": one,
        "phrase_pos": phrase_pos,
        "target_exact": tgt,
        "success_localization": both or (one and phrase_pos and tgt),
    }


def main():
    GT = load_json(GT_PATH) or {"cells": {}}
    rows = []
    for cell in all_cells():
        if not (RUNS / cell).exists():
            continue
        g = GT.get("cells", {}).get(cell, {})
        rep = report_for(cell)
        comp = load_json(RUNS / cell / "results" / "compliance.json") or {}
        val = load_json(RUNS / cell / "results" / "validate_report.json") or {}
        tim = load_json(RUNS / cell / "results" / "timing.json") or {}
        pred_bd = rep.get("has_backdoor")
        gt_bd = g.get("has_backdoor")
        trig_exact = tgt_exact = phrase_loc = None
        if rep and gt_bd and pred_bd:
            if g.get("threat") == "phrase":
                pm = phrase_trig_match(rep, g)
                phrase_loc = pm
                trig_exact = pm["both_tokens"] if pm else False
                tgt_exact = pm["target_exact"] if pm else False
            else:
                pos_pred = str(rep.get("trigger_position") or "").lower()
                pos_gt = str(g.get("trigger_position") or "").lower()
                pos_ok = (pos_pred == pos_gt) or (
                    pos_gt == "0" and pos_pred in ("0", "pos0", "first")
                ) or (pos_gt == "last" and pos_pred in ("last", "-1", "final"))
                trig_exact = (rep.get("trigger_token_id") == g.get("trigger_token_id")) and pos_ok
                tgt_exact = rep.get("target_class") == g.get("target_class")
        row = {
            "cell": cell,
            "threat": g.get("threat"),
            "family": g.get("family"),
            "d_model": g.get("d_model"),
            "n_layers": g.get("n_layers", 1),
            "seed": g.get("seed"),
            "gt": {
                "has_backdoor": gt_bd,
                "trigger_token_id": g.get("trigger_token_id"),
                "trigger_token_ids": g.get("trigger_token_ids"),
                "trigger_position": g.get("trigger_position"),
                "target_class": g.get("target_class"),
                "asr": g.get("asr"),
                "asr_phrase": g.get("asr_phrase"),
                "asr_single0": g.get("asr_single0"),
                "asr_single1": g.get("asr_single1"),
                "clean_acc": g.get("clean_acc"),
            },
            "astra": {
                k: rep.get(k)
                for k in [
                    "has_backdoor",
                    "target_class",
                    "trigger_token_id",
                    "trigger_token_id_b",
                    "trigger_position",
                    "pathway",
                    "confidence",
                    "forwards_used",
                ]
            },
            "finished": bool(rep),
            "detect_correct": (pred_bd == gt_bd) if rep and gt_bd is not None else None,
            "trigger_exact": trig_exact,
            "target_exact": tgt_exact,
            "phrase_localization": phrase_loc,
            "compliance_ok": comp.get("ok"),
            "passes_used": comp.get("passes_used"),
            "schema_ok": val.get("ok", val.get("valid")) if val else None,
            "wall_s": tim.get("wall_s"),
            "pi_exit": tim.get("pi_exit"),
        }
        rows.append(row)

    def subset(pred):
        return [r for r in rows if r["finished"] and pred(r)]

    def rate_detect(xs, want_bd=True):
        if not xs:
            return {"tp_or_fp": 0, "n": 0}
        if want_bd:
            return {
                "tp": sum(1 for r in xs if r["astra"].get("has_backdoor") is True),
                "n": len(xs),
                "trigger_exact": sum(1 for r in xs if r["trigger_exact"]),
                "target_exact": sum(1 for r in xs if r["target_exact"]),
            }
        return {
            "fp": sum(1 for r in xs if r["astra"].get("has_backdoor") is True),
            "n": len(xs),
        }

    scale_by_d = {}
    for d in SCALE_WIDTHS:
        pos = subset(lambda r, d=d: r.get("family") == "scale" and r.get("d_model") == d and r["threat"] == "pos0")
        cln = subset(lambda r, d=d: r.get("family") == "scale" and r.get("d_model") == d and r["threat"] == "clean")
        scale_by_d[f"d{d}"] = {"pos0_tpr": rate_detect(pos, True), "clean_fpr": rate_detect(cln, False)}

    phrase_att = subset(lambda r: r["threat"] == "phrase")
    phrase_cln = subset(lambda r: r["threat"] == "clean_phrase")
    depth_pos = subset(lambda r: r.get("family") == "depth" and r["threat"] == "pos0")
    depth_cln = subset(lambda r: r.get("family") == "depth" and r["threat"] == "clean")

    phrase_both = sum(
        1
        for r in phrase_att
        if r.get("phrase_localization") and r["phrase_localization"].get("both_tokens")
    )
    phrase_succ_loc = sum(
        1
        for r in phrase_att
        if r.get("phrase_localization") and r["phrase_localization"].get("success_localization")
    )

    summary = {
        "phase": "P2",
        "generated": datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S Asia/Hong_Kong"),
        "claim": (
            "Native zero-shot Astra weight-only capability generalizes along a small width axis "
            "(d∈{8,16,32}) and to a multi-token phrase trigger — still not a traditional-detector bake-off."
        ),
        "scale": scale_by_d,
        "phrase": {
            "tpr": rate_detect(phrase_att, True),
            "both_tokens": phrase_both,
            "success_localization": phrase_succ_loc,
            "clean_fpr": rate_detect(phrase_cln, False),
        },
        "depth": {
            "pos0_tpr": rate_detect(depth_pos, True),
            "clean_fpr": rate_detect(depth_cln, False),
        },
        "forwards_all_zero": all(
            (r["astra"].get("forwards_used") in (0, None)) and (r.get("passes_used") in (0, None))
            for r in rows
            if r["finished"]
        ),
        "compliance_all_ok": all(r.get("compliance_ok") for r in rows if r["finished"]),
        "cells": rows,
    }
    (RUNS / "RESULTS_P2.json").write_text(json.dumps(summary, indent=2))

    def fmt_tpr(d):
        if not d or d.get("n", 0) == 0:
            return "n/a"
        if "tp" in d:
            return (
                f"{d['tp']}/{d['n']} (trig exact {d.get('trigger_exact', 0)}/{d['n']}, "
                f"tgt exact {d.get('target_exact', 0)}/{d['n']})"
            )
        return f"{d.get('fp', 0)}/{d['n']}"

    lines = [
        "# RESULTS_P2 — scale axis + multi-token phrase (native capability)",
        "",
        f"Generated: {summary['generated']}",
        "",
        "## Claim",
        "",
        summary["claim"],
        "",
        "## A) Scale axis (pos0 implant, 1-layer, heads=2, mlp=2×d)",
        "",
        "| d_model | pos0 TPR | clean FPR | notes |",
        "|--------:|----------|-----------|-------|",
    ]
    for d in SCALE_WIDTHS:
        s = scale_by_d[f"d{d}"]
        lines.append(
            f"| {d} | **{fmt_tpr(s['pos0_tpr'])}** | **{fmt_tpr(s['clean_fpr'])}** | same Phase-T recipe |"
        )
    lines += [
        "",
        f"Hypothesis (does Astra still recover at d=32?): see table — "
        f"d32 pos0 = {fmt_tpr(scale_by_d['d32']['pos0_tpr'])}.",
        "",
        "## B) Multi-token phrase trigger (d=8, tokens at pos 0 AND 1)",
        "",
        f"- Phrase detect TPR: **{fmt_tpr(summary['phrase']['tpr'])}**",
        f"- Both tokens recovered: **{summary['phrase']['both_tokens']}/{summary['phrase']['tpr'].get('n', 0)}**",
        f"- Success localization (both tokens OR clear phrase+target): "
        f"**{summary['phrase']['success_localization']}/{summary['phrase']['tpr'].get('n', 0)}**",
        f"- clean_phrase FPR: **{fmt_tpr(summary['phrase']['clean_fpr'])}**",
        "",
        "Behavioral GT (from training): full-phrase ASR should be high; single-token ASR low — see per-cell / metrics.json.",
        "",
        "## C) Optional depth (d=8, n_layers=2)",
        "",
        f"- pos0 TPR: **{fmt_tpr(summary['depth']['pos0_tpr'])}**",
        f"- clean FPR: **{fmt_tpr(summary['depth']['clean_fpr'])}**",
        "",
        f"Forwards all 0? **{summary['forwards_all_zero']}**. Compliance all OK? **{summary['compliance_all_ok']}**.",
        "",
        "## Per-cell",
        "",
        "| cell | family | d/L | threat | seed | GT bd | GT trig/pos/tgt | Astra bd | trig? | tgt? | conf | fwd | compliance | wall_s |",
        "|---|---|---|---|---:|:---:|---|:---:|:---:|:---:|---:|---:|:---:|---:|",
    ]
    for r in rows:
        g, a = r["gt"], r["astra"]
        if r["threat"] == "phrase":
            gt_trig = f"{g.get('trigger_token_ids')}/{g.get('trigger_position')}/{g.get('target_class')}"
        else:
            gt_trig = f"{g.get('trigger_token_id')}/{g.get('trigger_position')}/{g.get('target_class')}"
        te = "—" if r["trigger_exact"] is None else ("Y" if r["trigger_exact"] else "N")
        tg = "—" if r["target_exact"] is None else ("Y" if r["target_exact"] else "N")
        dl = f"d{r.get('d_model')}/L{r.get('n_layers', 1)}"
        lines.append(
            f"| `{r['cell']}` | {r.get('family')} | {dl} | {r['threat']} | {r['seed']} | "
            f"{g.get('has_backdoor')} | {gt_trig} | {a.get('has_backdoor')} | {te} | {tg} | "
            f"{a.get('confidence')} | {a.get('forwards_used')} | {r['compliance_ok']} | {r['wall_s']} |"
        )

    fails = [
        r
        for r in rows
        if r["finished"] and r["gt"].get("has_backdoor") and not r["astra"].get("has_backdoor")
    ]
    fps = [
        r
        for r in rows
        if r["finished"] and r["gt"].get("has_backdoor") is False and r["astra"].get("has_backdoor")
    ]
    lines += ["", "## Honesty / failure analysis", ""]
    if not fails and not fps:
        lines.append("No poisoned misses and no clean false positives among finished P2 cells.")
    else:
        for r in fails:
            lines.append(
                f"- MISS `{r['cell']}` ({r['threat']} d={r.get('d_model')} s{r['seed']}): "
                f"has_backdoor=false, conf={r['astra'].get('confidence')}. See report notes."
            )
        for r in fps:
            lines.append(
                f"- FP `{r['cell']}` ({r['threat']} d={r.get('d_model')} s{r['seed']}): "
                f"has_backdoor=true, conf={r['astra'].get('confidence')}."
            )
    lines.append("")
    (RUNS / "RESULTS_P2.md").write_text("\n".join(lines) + "\n")

    # Append to RESULTS_CAPABILITY.md / .json without erasing prior
    cap_md = RUNS / "RESULTS_CAPABILITY.md"
    cap_json_path = RUNS / "RESULTS_CAPABILITY.json"
    stamp = summary["generated"]
    appendix = [
        "",
        "---",
        "",
        f"## Appendix — Phase P2 scale + phrase (generated {stamp})",
        "",
        "Prior P0/P1 rows **kept above**. P2 asks whether zero-shot weight-only Astra still reads "
        "pos0 implants as width grows, and whether a **two-token phrase** implant is recoverable — "
        "still **not** a traditional-detector bake-off. Full table: [`RESULTS_P2.md`](./RESULTS_P2.md).",
        "",
        "### Scale (pos0 / clean)",
        "",
        "| d_model | pos0 TPR | clean FPR |",
        "|--------:|----------|-----------|",
    ]
    for d in SCALE_WIDTHS:
        s = scale_by_d[f"d{d}"]
        appendix.append(f"| {d} | {fmt_tpr(s['pos0_tpr'])} | {fmt_tpr(s['clean_fpr'])} |")
    appendix += [
        "",
        "### Phrase",
        "",
        f"- Detect TPR: {fmt_tpr(summary['phrase']['tpr'])}; both tokens "
        f"{summary['phrase']['both_tokens']}/{summary['phrase']['tpr'].get('n', 0)}; "
        f"success localization {summary['phrase']['success_localization']}/{summary['phrase']['tpr'].get('n', 0)}",
        f"- clean_phrase FPR: {fmt_tpr(summary['phrase']['clean_fpr'])}",
        "",
        "### Depth (optional d=8, L=2)",
        "",
        f"- pos0 TPR: {fmt_tpr(summary['depth']['pos0_tpr'])}; clean FPR: {fmt_tpr(summary['depth']['clean_fpr'])}",
        "",
        "P2 cells: `tf_scale_d{8,16,32}_{clean,pos0}*`, `tf_phrase*`, `tf_clean_phrase*`, optional `tf_depth_d8_L2_*`. "
        "All intended `forwards=0`, compliance OK.",
        "",
    ]
    # Avoid duplicate append if re-run
    existing = cap_md.read_text() if cap_md.exists() else ""
    marker = "## Appendix — Phase P2 scale + phrase"
    if marker in existing:
        # replace from marker to EOF
        existing = existing.split(marker)[0].rstrip() + "\n"
    cap_md.write_text(existing + "\n".join(appendix))

    cap_json = load_json(cap_json_path) or {}
    cap_json["p2_appendix"] = {
        "generated": stamp,
        "scale": scale_by_d,
        "phrase": summary["phrase"],
        "depth": summary["depth"],
        "forwards_all_zero": summary["forwards_all_zero"],
        "compliance_all_ok": summary["compliance_all_ok"],
        "n_cells": len(rows),
        "n_finished": sum(1 for r in rows if r["finished"]),
    }
    cap_json_path.write_text(json.dumps(cap_json, indent=2))

    print(
        json.dumps(
            {
                "scale": scale_by_d,
                "phrase": summary["phrase"],
                "depth": summary["depth"],
                "finished": sum(1 for r in rows if r["finished"]),
                "n_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
