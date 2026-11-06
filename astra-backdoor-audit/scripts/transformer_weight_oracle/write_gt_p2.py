"""Write gt/transformer_p2_ground_truth.json from models/transformer_wo_p2 metrics + cohort metas."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "models" / "transformer_wo_p2"
OUT = ROOT / "gt" / "transformer_p2_ground_truth.json"


def load_meta(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["meta"], ck.get("arch")


def main():
    cells = {}
    # scale
    for d in (8, 16, 32):
        for threat in ("clean", "pos0"):
            for k in range(3):
                p = P2 / "scale" / f"d{d}" / "cohort" / f"{threat}_s{k}.pt"
                if not p.exists():
                    continue
                meta, arch = load_meta(p)
                cell = f"tf_scale_d{d}_{threat}_strict" if k == 0 else f"tf_scale_d{d}_{threat}_s{k}_strict"
                cells[cell] = {
                    "family": "scale",
                    "threat": threat,
                    "seed": k,
                    "d_model": d,
                    "n_layers": 1,
                    "has_backdoor": threat == "pos0",
                    "target_class": meta.get("target_class"),
                    "trigger_token_id": meta.get("trigger_id"),
                    "trigger_position": "0" if threat == "pos0" else None,
                    "pathway": "attention_VO" if threat == "pos0" else "none",
                    "clean_acc": meta.get("clean_acc"),
                    "asr": meta.get("asr"),
                    "source_ckpt": str(p.relative_to(ROOT)),
                }
    # phrase
    for k in range(3):
        p = P2 / "phrase" / "cohort" / f"phrase_s{k}.pt"
        if p.exists():
            meta, arch = load_meta(p)
            cell = "tf_phrase_strict" if k == 0 else f"tf_phrase_s{k}_strict"
            cells[cell] = {
                "family": "phrase",
                "threat": "phrase",
                "seed": k,
                "d_model": 8,
                "n_layers": 1,
                "has_backdoor": True,
                "target_class": meta.get("target_class"),
                "trigger_token_id": meta.get("trigger_id"),
                "trigger_token_ids": meta.get("trigger_token_ids"),
                "trigger_position": "0+1",
                "pathway": "attention_VO",
                "clean_acc": meta.get("clean_acc"),
                "asr": meta.get("asr_phrase", meta.get("asr")),
                "asr_phrase": meta.get("asr_phrase"),
                "asr_single0": meta.get("asr_single0"),
                "asr_single1": meta.get("asr_single1"),
                "source_ckpt": str(p.relative_to(ROOT)),
            }
        p = P2 / "phrase" / "cohort" / f"clean_phrase_s{k}.pt"
        if p.exists():
            meta, arch = load_meta(p)
            cell = "tf_clean_phrase_strict" if k == 0 else f"tf_clean_phrase_s{k}_strict"
            cells[cell] = {
                "family": "phrase",
                "threat": "clean_phrase",
                "seed": k,
                "d_model": 8,
                "n_layers": 1,
                "has_backdoor": False,
                "target_class": None,
                "trigger_token_id": None,
                "trigger_position": None,
                "pathway": "none",
                "clean_acc": meta.get("clean_acc"),
                "asr": None,
                "source_ckpt": str(p.relative_to(ROOT)),
            }
    # depth
    for threat in ("clean", "pos0"):
        for k in range(2):
            p = P2 / "depth" / "d8_L2" / "cohort" / f"{threat}_s{k}.pt"
            if not p.exists():
                continue
            meta, arch = load_meta(p)
            cell = f"tf_depth_d8_L2_{threat}_strict" if k == 0 else f"tf_depth_d8_L2_{threat}_s{k}_strict"
            cells[cell] = {
                "family": "depth",
                "threat": threat,
                "seed": k,
                "d_model": 8,
                "n_layers": 2,
                "has_backdoor": threat == "pos0",
                "target_class": meta.get("target_class"),
                "trigger_token_id": meta.get("trigger_id"),
                "trigger_position": "0" if threat == "pos0" else None,
                "pathway": "attention_VO" if threat == "pos0" else "none",
                "clean_acc": meta.get("clean_acc"),
                "asr": meta.get("asr"),
                "source_ckpt": str(p.relative_to(ROOT)),
            }

    doc = {
        "note": "P2 ground truth — outside sandbox inputs.",
        "phase": "P2",
        "cells": cells,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
