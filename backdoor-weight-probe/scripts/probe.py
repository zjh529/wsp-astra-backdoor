"""WSP-style ablation probe: ablate top-K suspicious neurons, measure ΔASR / Δclean_acc."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from data import TARGET_LABEL, get_datasets, make_loader, triggered_test_batch
from model import TinyCNN

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
EPS = 1e-3


def load_model(name: str, device):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location=device, weights_only=False)
    model = TinyCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def ablate_fc1(model: TinyCNN, indices: list[int]) -> TinyCNN:
    """Zero outgoing weights from fc1 neurons (fc2 columns) and zero those fc1 biases."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for i in indices:
            m.fc2.weight[:, i] = 0.0
            m.fc1.bias[i] = 0.0
            # Also zero incoming? Optional — zero outgoing is enough to kill contribution to logits
            # Zero the neuron's output by zeroing fc1 row weights too for cleanliness
            m.fc1.weight[i, :] = 0.0
    return m


def evaluate_clean(model, test_ds, device, max_batches=20):
    """Partial clean eval for speed during probes (~20*256 = 5120 samples)."""
    model.eval()
    correct, total = 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if bi + 1 >= max_batches:
                break
    return correct / max(total, 1), total


def evaluate_asr(model, test_ds, device, max_batches=20):
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for bi, (x, _y) in enumerate(triggered_test_batch(test_ds, batch_size=256, exclude_target=True)):
            x = x.to(device)
            pred = model(x).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
            if bi + 1 >= max_batches:
                break
    return hit / max(total, 1), total


def neuron_suspicion(delta_asr: float, delta_clean: float) -> float:
    """Prefer high ASR drop, low clean drop. delta_asr = ASR_before - ASR_after (>0 good).
    delta_clean = clean_before - clean_after (>0 means clean degraded).
    """
    asr_drop = max(0.0, delta_asr)
    clean_drop = max(0.0, delta_clean)
    return asr_drop / (EPS + clean_drop)


def candidate_ranking(poisoned_scan, clean_scan, mode="delta"):
    """Rank fc1 neurons by suspicion from weight scan."""
    p = {r["index"]: r["composite"] for r in poisoned_scan["fc1_scores"]}
    c = {r["index"]: r["composite"] for r in clean_scan["fc1_scores"]}
    scores = []
    for i in p:
        if mode == "delta":
            s = p[i] - c[i]
        else:
            s = p[i]
        scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def probe_model(name: str, ranking: list[tuple[int, float]], top_k: int, device, forward_counter: dict):
    model, ckpt = load_model(name, device)
    ds = get_datasets(seed=42)
    test = ds["test"]

    base_acc, n_clean = evaluate_clean(model, test, device)
    base_asr, n_asr = evaluate_asr(model, test, device)
    # each eval batch is one forward; count batches approx
    forward_counter["ablation_eval_forwards"] = forward_counter.get("ablation_eval_forwards", 0) + (
        20 + 20
    )  # base

    results = []
    candidates = [idx for idx, _ in ranking[:top_k]]
    for idx in candidates:
        m = ablate_fc1(model, [idx])
        acc, _ = evaluate_clean(m, test, device)
        asr, _ = evaluate_asr(m, test, device)
        forward_counter["ablation_eval_forwards"] += 20 + 20
        d_asr = base_asr - asr
        d_clean = base_acc - acc
        sus = neuron_suspicion(d_asr, d_clean)
        results.append(
            {
                "neuron": idx,
                "rank_score": next(s for i, s in ranking if i == idx),
                "clean_acc": acc,
                "asr": asr,
                "delta_asr": d_asr,
                "delta_clean_acc": d_clean,
                "suspicion": sus,
            }
        )
        print(
            f"[{name}] ablate fc1[{idx}]: clean={acc:.4f} (Δ{d_clean:+.4f}) "
            f"ASR={asr:.4f} (Δ{d_asr:+.4f}) sus={sus:.3f}"
        )

    results_sorted = sorted(results, key=lambda r: r["suspicion"], reverse=True)
    best = results_sorted[0] if results_sorted else None

    # Also try ablating top-3 by ablation suspicion jointly
    joint = None
    if len(results_sorted) >= 3:
        top3 = [r["neuron"] for r in results_sorted[:3]]
        m = ablate_fc1(model, top3)
        acc, _ = evaluate_clean(m, test, device)
        asr, _ = evaluate_asr(m, test, device)
        forward_counter["ablation_eval_forwards"] += 20 + 20
        joint = {
            "neurons": top3,
            "clean_acc": acc,
            "asr": asr,
            "delta_asr": base_asr - asr,
            "delta_clean_acc": base_acc - acc,
            "suspicion": neuron_suspicion(base_asr - asr, base_acc - acc),
        }
        print(
            f"[{name}] ablate top3 {top3}: clean={acc:.4f} ASR={asr:.4f} "
            f"ΔASR={base_asr-asr:+.4f} Δclean={base_acc-acc:+.4f}"
        )

    # Oracle: which single neuron kill ASR most? (among all 64 — expensive but small)
    # Skip full oracle; approximate with top_k + maybe bottom random isn't needed.
    # Instead: after ranking by ablation suspicion, check if weight-scan top recovered them.

    out = {
        "model": name,
        "base_clean_acc": base_acc,
        "base_asr": base_asr,
        "eval_samples_clean": n_clean,
        "eval_samples_asr": n_asr,
        "per_neuron": results,
        "best_single": best,
        "joint_top3": joint,
        "candidates": candidates,
    }
    (RESULTS / f"probe_{name}.json").write_text(json.dumps(out, indent=2))
    return out


def model_suspicion_score(probe_out: dict) -> float:
    """Aggregate: max single-neuron suspicion + bonus for joint ASR drop."""
    best = probe_out.get("best_single") or {}
    joint = probe_out.get("joint_top3") or {}
    s = float(best.get("suspicion", 0.0))
    if joint:
        s += 0.5 * float(joint.get("suspicion", 0.0))
        s += 2.0 * max(0.0, float(joint.get("delta_asr", 0.0)))
    s += 2.0 * max(0.0, float(best.get("delta_asr", 0.0)))
    # Also amplify if base ASR is high (poisoned signal)
    s += 0.5 * float(probe_out.get("base_asr", 0.0))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rank-mode", choices=["delta", "absolute"], default="delta")
    args = ap.parse_args()

    device = torch.device("cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)

    poisoned_scan = json.loads((RESULTS / "weight_scan_poisoned.json").read_text())
    clean_scan = json.loads((RESULTS / "weight_scan_clean.json").read_text())

    ranking_poisoned = candidate_ranking(poisoned_scan, clean_scan, mode=args.rank_mode)
    # For clean twin, use absolute composite on clean itself as "candidates"
    ranking_clean = [(r["index"], r["composite"]) for r in sorted(
        clean_scan["fc1_scores"], key=lambda r: r["composite"], reverse=True
    )]

    forward_counter = {"ablation_eval_forwards": 0, "weight_scan_ops": "cheap_static"}

    print("=== Probe poisoned (ranked by poisoned-clean delta composite) ===")
    print("Top ranking:", ranking_poisoned[:10])
    poison_probe = probe_model("poisoned", ranking_poisoned, args.top_k, device, forward_counter)

    print("=== Probe clean twin (ranked by absolute composite) ===")
    clean_probe = probe_model("clean", ranking_clean, args.top_k, device, forward_counter)

    # Recovery: among top-K weight-scan candidates, did the best ablation-ASR neurons appear?
    # Define "true backdoor neurons" as those with highest ΔASR among probed set
    per = poison_probe["per_neuron"]
    by_asr = sorted(per, key=lambda r: r["delta_asr"], reverse=True)
    true_backdoor = [r["neuron"] for r in by_asr[:3] if r["delta_asr"] > 0.05]
    weight_top_k = [idx for idx, _ in ranking_poisoned[: args.top_k]]
    recovered = [n for n in true_backdoor if n in weight_top_k]
    # Rank of best ASR-drop neuron in weight ranking
    best_asr_neuron = by_asr[0]["neuron"] if by_asr else None
    weight_rank_of_best = None
    if best_asr_neuron is not None:
        for rank, (idx, _) in enumerate(ranking_poisoned, start=1):
            if idx == best_asr_neuron:
                weight_rank_of_best = rank
                break

    poison_sus = model_suspicion_score(poison_probe)
    clean_sus = model_suspicion_score(clean_probe)

    # Cost comparison
    # Naive trigger search: 3x3 white patch over many locations on 28x28
    # Locations: (28-3+1)^2 = 26*26 = 676; each needs a forward on a batch of test imgs
    # Simulate: 676 candidate locations × 1 eval batch (~256 imgs) = 676 forwards (lower bound)
    # Or if evaluating ASR properly with ~20 batches: 676*20 huge — use 676 single-batch probes
    naive_locations = (28 - 3 + 1) ** 2
    naive_forwards = naive_locations  # one forward-batch per location (optimistic lower bound)
    our_forwards = forward_counter["ablation_eval_forwards"]
    # Weight scan is essentially free vs forwards (matrix norms)

    summary = {
        "rank_mode": args.rank_mode,
        "top_k": args.top_k,
        "poisoned_probe": {
            "base_clean_acc": poison_probe["base_clean_acc"],
            "base_asr": poison_probe["base_asr"],
            "best_single": poison_probe["best_single"],
            "joint_top3": poison_probe["joint_top3"],
        },
        "clean_probe": {
            "base_clean_acc": clean_probe["base_clean_acc"],
            "base_asr": clean_probe["base_asr"],
            "best_single": clean_probe["best_single"],
            "joint_top3": clean_probe["joint_top3"],
        },
        "recovery": {
            "true_backdoor_neurons_by_delta_asr": true_backdoor,
            "weight_scan_top_k": weight_top_k,
            "recovered_in_top_k": recovered,
            "recovered": len(recovered) > 0,
            "best_asr_neuron": best_asr_neuron,
            "weight_rank_of_best_asr_neuron": weight_rank_of_best,
        },
        "model_suspicion": {
            "poisoned": poison_sus,
            "clean": clean_sus,
            "ratio_poisoned_over_clean": poison_sus / max(clean_sus, EPS),
        },
        "cost": {
            "naive_trigger_search_locations": naive_locations,
            "naive_forwards_lower_bound": naive_forwards,
            "our_ablation_eval_forwards": our_forwards,
            "weight_scan": "O(params) static norms + 1 holdout forward pass for acts",
            "savings_factor_approx": naive_forwards / max(our_forwards, 1),
        },
    }
    (RESULTS / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["model_suspicion"], indent=2))
    print(json.dumps(summary["recovery"], indent=2))
    print(json.dumps(summary["cost"], indent=2))


if __name__ == "__main__":
    main()
