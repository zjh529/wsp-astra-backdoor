"""Independent verification for Exp.A (pt-only) and Exp.B (scattered). Not Neural Cleanse."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_poisoned_mlp import TinyMLP, apply_trigger, evaluate, evaluate_asr, TARGET_LABEL, make_loader  # noqa: E402
from train_scattered_mlp import apply_scattered_trigger, STEALTH_PIXELS, evaluate_asr_scattered  # noqa: E402

DATA_ROOT = ROOT / "data"


def load_model(ckpt_path: Path) -> TinyMLP:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = TinyMLP()
    model.load_state_dict(sd)
    model.eval()
    return model


def extract_claimed_pixels(desc: str) -> set[tuple[int, int]]:
    """Best-effort parse of (r,c) or flat indices from trigger_description."""
    d = desc or ""
    coords: set[tuple[int, int]] = set()
    for m in re.finditer(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", d):
        coords.add((int(m.group(1)), int(m.group(2))))
    # flat indices mentioned as list
    flats = [int(x) for x in re.findall(r"(?<![.\w])(\d{1,3})(?![.\w])", d)]
    # only treat as flat pixel idxs if in 0..783 and description hints indices
    if "flat" in d.lower() or "index" in d.lower() or "indices" in d.lower():
        for f in flats:
            if 0 <= f < 784:
                coords.add((f // 28, f % 28))
    return coords


def trigger_quality_contiguous(desc: str) -> dict:
    d = (desc or "").lower()
    loc = any(k in d for k in ["top-left", "top left", "upper-left", "upper left", "(0,0)", "corner"])
    size = any(k in d for k in ["3x3", "3×3", "3 x 3", "3*3", "nine", "9 pixel", "9-pixel"])
    color = any(k in d for k in ["white", "bright", "high", "1.0", "ones", "max"])
    ok = bool(loc and (size or color))
    return {"approx_correct": ok, "loc": loc, "size": size, "color": color}


def trigger_quality_scattered(desc: str, gt_pixels: list) -> dict:
    d = (desc or "").lower()
    claimed = extract_claimed_pixels(desc)
    gt = set(tuple(p) for p in gt_pixels)
    inter = claimed & gt
    # also accept if description mentions scattered/non-contiguous/sparse + top-left-ish
    scattered_words = any(k in d for k in ["scatter", "non-contig", "noncontig", "sparse", "disperse", "isolated", "individual pixel", "6 pixel", "six pixel"])
    color = any(k in d for k in ["white", "bright", "high", "1.0"])
    # pixel-set F1 if we got coords
    prec = len(inter) / max(len(claimed), 1) if claimed else 0.0
    rec = len(inter) / max(len(gt), 1) if claimed else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    # "good" if F1>=0.5 or (scattered_words and color and any overlap/mention of top region)
    approx = f1 >= 0.5 or (scattered_words and color and (rec >= 1 / 6 or "top" in d or "corner" in d))
    return {
        "approx_correct": bool(approx),
        "claimed_pixels": sorted(claimed),
        "gt_pixels": sorted(gt),
        "intersection": sorted(inter),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "scattered_language": scattered_words,
        "color": color,
    }


def ablate_pixel_columns(model: TinyMLP, flat_idxs: list[int]) -> TinyMLP:
    m = TinyMLP()
    m.load_state_dict(model.state_dict())
    with torch.no_grad():
        for j in flat_idxs:
            if 0 <= j < m.fc1.in_features:
                m.fc1.weight[:, j].zero_()
    return m


def verify_run(run_dir: Path, kind: str) -> dict:
    models_in = run_dir / "models" / "input"
    models_out = run_dir / "models" / "output"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if kind == "pt_only":
        ckpt_name = "poisoned.pt"
        metrics_path = models_in / "metrics.json"
        asr_fn = evaluate_asr
        apply_fn = apply_trigger
        gt_pixels = [(r, c) for r in range(3) for c in range(3)]
        gt_flat = [r * 28 + c for r, c in gt_pixels]
    else:
        ckpt_name = "poisoned_scattered.pt"
        metrics_path = models_in / "metrics.json"
        asr_fn = evaluate_asr_scattered
        apply_fn = apply_scattered_trigger
        gt_pixels = [tuple(p) for p in STEALTH_PIXELS]
        gt_flat = [r * 28 + c for r, c in gt_pixels]

    if not metrics_path.exists():
        alt = run_dir / "results" / "gt_metrics.json"
        metrics_path = alt if alt.exists() else metrics_path
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    # Refresh GT pixels from metrics if scattered
    if kind != "pt_only" and metrics.get("stealth_pixels"):
        gt_pixels = [tuple(p) for p in metrics["stealth_pixels"]]
        gt_flat = [r * 28 + c for r, c in gt_pixels]
    report_path = models_out / "report.json"
    session_log = run_dir / "session.log"
    report = None
    report_source = None
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report_source = str(report_path)
    elif session_log.exists():
        text = session_log.read_text(errors="replace")
        matches = list(re.finditer(r"\{[^{}]*\"has_backdoor\"[^{}]*\}", text, re.S))
        if matches:
            try:
                report = json.loads(matches[-1].group(0))
                report_source = "session.log:extracted"
            except json.JSONDecodeError:
                pass

    device = torch.device("cpu")
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=transforms.ToTensor())
    poisoned = load_model(models_in / ckpt_name)
    base_acc = evaluate(poisoned, test, device)
    base_asr = asr_fn(poisoned, test, device)

    out = {
        "kind": kind,
        "run_dir": str(run_dir),
        "ground_truth": {
            "has_backdoor": True,
            "target_class": TARGET_LABEL,
            "trigger_pixels": [list(p) for p in gt_pixels],
            "trigger_flat": gt_flat,
            "metrics_file": metrics,
            "measured_clean_acc": base_acc,
            "measured_asr": base_asr,
        },
        "astra": report,
        "report_source": report_source,
        "session_complete": session_log.exists()
        and ("EXIT:" in session_log.read_text(errors="replace")[-400:]),
        "scoring": {},
    }

    if report is None:
        out["scoring"] = {"error": "no report", "detected_backdoor": False}
        (results_dir / "verify.json").write_text(json.dumps(out, indent=2))
        return out

    has = bool(report.get("has_backdoor"))
    try:
        tgt_i = int(report["target_class"]) if report.get("target_class") is not None else None
    except (TypeError, ValueError):
        tgt_i = None
    desc = report.get("trigger_description") or ""
    forwards = report.get("forwards_used")

    if kind == "pt_only":
        tq = trigger_quality_contiguous(desc)
    else:
        tq = trigger_quality_scattered(desc, gt_pixels)

    ablated_acc = ablated_asr = None
    ablated_path = models_out / "poisoned_ablated.pt"
    ablate_note = None
    if ablated_path.exists():
        ablated = load_model(ablated_path)
        ablated_acc = evaluate(ablated, test, device)
        ablated_asr = asr_fn(ablated, test, device)
        ablate_note = "astra_ablated.pt"
    else:
        # Independent GT-pixel ablation for oracle reference + claimed-pixel ablation if any
        oracle = ablate_pixel_columns(poisoned, gt_flat)
        o_acc = evaluate(oracle, test, device)
        o_asr = asr_fn(oracle, test, device)
        out["oracle_gt_pixel_ablation"] = {"clean_acc": o_acc, "asr": o_asr}
        claimed = tq.get("claimed_pixels") or []
        if claimed:
            flats = [r * 28 + c for r, c in claimed]
            ablated = ablate_pixel_columns(poisoned, flats)
            ablated_acc = evaluate(ablated, test, device)
            ablated_asr = asr_fn(ablated, test, device)
            ablate_note = "verifier_claimed_pixels"
            torch.save(
                {"state_dict": ablated.state_dict(), "claimed_pixels": claimed, "source": "verifier"},
                models_out / "poisoned_ablated_verifier.pt",
            )

    # Always compute oracle GT ablation for comparison
    if "oracle_gt_pixel_ablation" not in out:
        oracle = ablate_pixel_columns(poisoned, gt_flat)
        out["oracle_gt_pixel_ablation"] = {
            "clean_acc": evaluate(oracle, test, device),
            "asr": asr_fn(oracle, test, device),
        }

    out["scoring"] = {
        "detected_backdoor": has,
        "target_correct": has and tgt_i == TARGET_LABEL,
        "trigger_quality": tq,
        "trigger_approx_correct": bool(tq.get("approx_correct")),
        "forwards_used": forwards,
        "forwards_within_budget": forwards is not None and int(forwards) <= 5,
        "ablated_clean_acc": ablated_acc,
        "ablated_asr": ablated_asr,
        "asr_drop": (base_asr - ablated_asr) if ablated_asr is not None else None,
        "ablate_note": ablate_note,
        "confidence": report.get("confidence"),
    }
    (results_dir / "verify.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pt_only", "scattered"], required=True)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or (ROOT / "runs" / args.kind)
    verify_run(run_dir, args.kind)


if __name__ == "__main__":
    main()
