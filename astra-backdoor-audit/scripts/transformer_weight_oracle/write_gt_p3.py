"""Write gt/transformer_p3_ground_truth.json from models/transformer_wo_p3 cohort metas."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "models" / "transformer_wo_p3"
OUT = ROOT / "gt" / "transformer_p3_ground_truth.json"


def load_meta(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["meta"], ck.get("arch")


def main():
    cells = {}
    for threat in ("clean", "attn_only", "mlp_only"):
        for k in range(8):  # tolerate up to 8 seeds if present
            p = P3 / "locus" / "cohort" / f"{threat}_s{k}.pt"
            if not p.exists():
                continue
            meta, arch = load_meta(p)
            cell = f"tf_locus_{threat}_strict" if k == 0 else f"tf_locus_{threat}_s{k}_strict"
            has_bd = threat != "clean"
            cells[cell] = {
                "family": "locus",
                "threat": threat,
                "seed": k,
                "d_model": (arch or {}).get("d_model", meta.get("arch", {}).get("d_model", 8)),
                "n_layers": (arch or {}).get("n_layers", 1),
                "has_backdoor": has_bd,
                "target_class": meta.get("target_class") if has_bd else None,
                "trigger_token_id": meta.get("trigger_id") if has_bd else None,
                "trigger_position": "0" if has_bd else None,
                "pathway": meta.get("pathway") if has_bd else "none",
                "locus": None if threat == "clean" else threat,
                "implant_method": meta.get("implant_method"),
                "clean_acc": meta.get("clean_acc"),
                "asr": meta.get("asr"),
                "l2_delta_vs_init": meta.get("l2_delta_vs_init"),
                "source_ckpt": str(p.relative_to(ROOT)),
            }
    doc = {
        "note": "P3 locus ground truth — outside sandbox inputs. Variable = implant locus (attn vs mlp); trigger = pos0.",
        "phase": "P3",
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
