"""Improved WSP-style probe: activation deltas, class-0 weights, multi-k ablation, conv channels."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import TARGET_LABEL, apply_trigger, get_datasets, make_loader, triggered_test_batch
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
    m = copy.deepcopy(model)
    with torch.no_grad():
        for i in indices:
            m.fc2.weight[:, i] = 0.0
            m.fc1.bias[i] = 0.0
            m.fc1.weight[i, :] = 0.0
    return m


def ablate_conv2(model: TinyCNN, indices: list[int]) -> TinyCNN:
    """Zero conv2 output channels (weights + bias)."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for i in indices:
            m.conv2.weight[i] = 0.0
            m.conv2.bias[i] = 0.0
    return m


def evaluate_clean(model, test_ds, device, max_batches=40):
    model.eval()
    correct, total = 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    forwards = 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            forwards += 1
            if bi + 1 >= max_batches:
                break
    return correct / max(total, 1), total, forwards


def evaluate_asr(model, test_ds, device, max_batches=40):
    model.eval()
    hit, total = 0, 0
    forwards = 0
    with torch.no_grad():
        for bi, (x, _y) in enumerate(triggered_test_batch(test_ds, batch_size=256, exclude_target=True)):
            x = x.to(device)
            pred = model(x).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
            forwards += 1
            if bi + 1 >= max_batches:
                break
    return hit / max(total, 1), total, forwards


def neuron_suspicion(delta_asr: float, delta_clean: float) -> float:
    asr_drop = max(0.0, delta_asr)
    clean_drop = max(0.0, delta_clean)
    return asr_drop / (EPS + clean_drop)


def fc1_activation_delta_scores(model, holdout, device):
    """Score = mean(act_triggered) - mean(act_clean) on holdout; also |w to class 0|."""
    loader = make_loader(holdout, batch_size=256, shuffle=False)
    clean_acts, trig_acts = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            clean_acts.append(model.fc1_activations(x).cpu())
            xt = torch.stack([apply_trigger(xi.cpu()) for xi in x.cpu()]).to(device)
            trig_acts.append(model.fc1_activations(xt).cpu())
    C = torch.cat(clean_acts, 0)
    T = torch.cat(trig_acts, 0)
    delta = (T.mean(0) - C.mean(0))
    # Weight toward target class 0 from fc1
    w0 = model.fc2.weight[TARGET_LABEL].detach().cpu().abs()  # [64]
    # Combined: neurons that fire more on trigger AND push class 0
    contrib = delta * model.fc2.weight[TARGET_LABEL].detach().cpu()  # signed push to class 0
    # Suspicion: high positive contrib when trigger present
    scores = []
    for i in range(delta.numel()):
        scores.append(
            {
                "index": i,
                "act_delta": float(delta[i]),
                "w_to_target_abs": float(w0[i]),
                "trigger_contrib": float(contrib[i]),
                "score": float(contrib[i]),  # primary ranking
            }
        )
    scores.sort(key=lambda r: r["score"], reverse=True)
    return scores


def conv2_activation_delta_scores(model, holdout, device):
    """Global-average-pooled conv2 channel activations: trigger - clean."""
    loader = make_loader(holdout, batch_size=128, shuffle=False)
    clean_ch, trig_ch = [], []

    def conv2_gap(x):
        x = model.pool(F.relu(model.conv1(x)))
        x = F.relu(model.conv2(x))  # before pool
        return x.mean(dim=(2, 3))  # [B, 32]

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            clean_ch.append(conv2_gap(x).cpu())
            xt = torch.stack([apply_trigger(xi.cpu()) for xi in x.cpu()]).to(device)
            trig_ch.append(conv2_gap(xt).cpu())
    C = torch.cat(clean_ch, 0).mean(0)
    T = torch.cat(trig_ch, 0).mean(0)
    delta = T - C
    scores = [{"index": i, "act_delta": float(delta[i]), "score": float(delta[i])} for i in range(delta.numel())]
    scores.sort(key=lambda r: r["score"], reverse=True)
    return scores


def probe_ablations(name, model, ranking_indices, ablate_fn, layer_name, test, device, ks=(1, 3, 5, 10, 20)):
    """Ablate top-k jointly for various k; also single top-20 one-at-a-time."""
    fwd = 0
    base_acc, _, f1 = evaluate_clean(model, test, device)
    base_asr, _, f2 = evaluate_asr(model, test, device)
    fwd += f1 + f2

    singles = []
    for idx in ranking_indices[:20]:
        m = ablate_fn(model, [idx])
        acc, _, fa = evaluate_clean(m, test, device)
        asr, _, fb = evaluate_asr(m, test, device)
        fwd += fa + fb
        d_asr = base_asr - asr
        d_clean = base_acc - acc
        sus = neuron_suspicion(d_asr, d_clean)
        singles.append(
            {
                "neuron": idx,
                "clean_acc": acc,
                "asr": asr,
                "delta_asr": d_asr,
                "delta_clean_acc": d_clean,
                "suspicion": sus,
            }
        )
        print(
            f"[{name}/{layer_name}] ablate[{idx}]: clean={acc:.4f} (Δ{d_clean:+.4f}) "
            f"ASR={asr:.4f} (Δ{d_asr:+.4f}) sus={sus:.3f}"
        )

    joints = []
    for k in ks:
        idxs = ranking_indices[:k]
        m = ablate_fn(model, idxs)
        acc, _, fa = evaluate_clean(m, test, device)
        asr, _, fb = evaluate_asr(m, test, device)
        fwd += fa + fb
        d_asr = base_asr - asr
        d_clean = base_acc - acc
        sus = neuron_suspicion(d_asr, d_clean)
        joints.append(
            {
                "k": k,
                "neurons": idxs,
                "clean_acc": acc,
                "asr": asr,
                "delta_asr": d_asr,
                "delta_clean_acc": d_clean,
                "suspicion": sus,
            }
        )
        print(
            f"[{name}/{layer_name}] top-{k} {idxs[:5]}{'...' if k>5 else ''}: "
            f"clean={acc:.4f} ASR={asr:.4f} ΔASR={d_asr:+.4f} Δclean={d_clean:+.4f} sus={sus:.3f}"
        )

    best_single = max(singles, key=lambda r: r["suspicion"]) if singles else None
    best_joint = max(joints, key=lambda r: r["suspicion"]) if joints else None
    # Prefer joint that maximizes ASR drop with clean drop < 0.15
    good_joints = [j for j in joints if j["delta_clean_acc"] < 0.15]
    best_practical = max(good_joints, key=lambda r: r["delta_asr"]) if good_joints else best_joint

    return {
        "layer": layer_name,
        "base_clean_acc": base_acc,
        "base_asr": base_asr,
        "singles": singles,
        "joints": joints,
        "best_single": best_single,
        "best_joint": best_joint,
        "best_practical": best_practical,
        "forwards": fwd,
        "ranking": ranking_indices,
    }


def model_suspicion_score(fc_probe, conv_probe, base_asr):
    s = 0.5 * base_asr
    for p in (fc_probe, conv_probe):
        if not p:
            continue
        bp = p.get("best_practical") or {}
        s += 3.0 * max(0.0, float(bp.get("delta_asr", 0.0)))
        s += 0.5 * float(bp.get("suspicion", 0.0))
        bs = p.get("best_single") or {}
        s += 1.0 * max(0.0, float(bs.get("delta_asr", 0.0)))
    return s


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    device = torch.device("cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = get_datasets(seed=42)
    test, holdout = ds["test"], ds["holdout"]

    all_out = {}
    total_fwd = 0

    for name in ["poisoned", "clean"]:
        print(f"\n======== {name} ========")
        model, ckpt = load_model(name, device)

        fc_scores = fc1_activation_delta_scores(model, holdout, device)
        conv_scores = conv2_activation_delta_scores(model, holdout, device)
        # holdout forwards for scoring: ~ceil(2000/256)*2 for fc + ceil(2000/128)*2 for conv
        score_fwd = (8 * 2) + (16 * 2)
        total_fwd += score_fwd

        (RESULTS / f"act_scores_{name}.json").write_text(
            json.dumps({"fc1": fc_scores[:32], "conv2": conv_scores[:16]}, indent=2)
        )
        print(f"[{name}] top fc1 by trigger_contrib:", [(r['index'], round(r['score'],3)) for r in fc_scores[:8]])
        print(f"[{name}] top conv2 by act_delta:", [(r['index'], round(r['score'],3)) for r in conv_scores[:8]])

        fc_rank = [r["index"] for r in fc_scores]
        conv_rank = [r["index"] for r in conv_scores]

        fc_probe = probe_ablations(name, model, fc_rank, ablate_fc1, "fc1", test, device)
        conv_probe = probe_ablations(
            name, model, conv_rank, ablate_conv2, "conv2", test, device, ks=(1, 3, 5, 8, 12)
        )
        total_fwd += fc_probe["forwards"] + conv_probe["forwards"]

        sus = model_suspicion_score(fc_probe, conv_probe, fc_probe["base_asr"])
        all_out[name] = {
            "base_clean_acc": fc_probe["base_clean_acc"],
            "base_asr": fc_probe["base_asr"],
            "fc1": {
                "best_single": fc_probe["best_single"],
                "best_practical": fc_probe["best_practical"],
                "joints": fc_probe["joints"],
                "top_ranking": fc_rank[:20],
            },
            "conv2": {
                "best_single": conv_probe["best_single"],
                "best_practical": conv_probe["best_practical"],
                "joints": conv_probe["joints"],
                "top_ranking": conv_rank[:20],
            },
            "model_suspicion": sus,
        }
        (RESULTS / f"probe_v2_{name}.json").write_text(json.dumps(all_out[name], indent=2))

    # Recovery: neurons with highest single ΔASR among fc1 singles on poisoned
    p = all_out["poisoned"]
    singles = json.loads((RESULTS / "probe_v2_poisoned.json").read_text())
    # reload full singles from file we need — store them
    # Re-read from detailed — we only saved best. Recompute recovery from joints ASR drop.
    # Weight/activation rank recovery: check if ablating top-ranked causes big ASR drop
    fc_best = p["fc1"]["best_practical"]
    conv_best = p["conv2"]["best_practical"]
    recovered = False
    recovery_note = ""
    if fc_best and fc_best.get("delta_asr", 0) > 0.3:
        recovered = True
        recovery_note = f"fc1 top-{fc_best['k']} ranked by trigger→class0 contrib"
    if conv_best and conv_best.get("delta_asr", 0) > 0.3:
        recovered = True
        recovery_note = (recovery_note + "; " if recovery_note else "") + (
            f"conv2 top-{conv_best['k']} ranked by trigger act delta"
        )

    # Cost: naive exhaustive locations with same ASR eval budget (40 batches) would be huge;
    # fair lower bound = 676 location × 1 ASR batch; fair matched = 676 × 40
    naive_locs = 26 * 26
    naive_lb = naive_locs  # 1 forward-batch each
    naive_matched = naive_locs * 40  # same eval depth as our ASR eval
    # Our method: activation scoring forwards + ablation evals (already in total_fwd)
    # But total_fwd includes BOTH models; method cost is for one model under test
    # Approximate half for poisoned only — better recount from saved
    our_one_model = total_fwd // 2

    summary = {
        "poisoned": all_out["poisoned"],
        "clean": all_out["clean"],
        "recovery": {
            "recovered": recovered,
            "note": recovery_note,
            "fc1_best_practical": fc_best,
            "conv2_best_practical": conv_best,
        },
        "model_suspicion": {
            "poisoned": all_out["poisoned"]["model_suspicion"],
            "clean": all_out["clean"]["model_suspicion"],
            "ratio": all_out["poisoned"]["model_suspicion"]
            / max(all_out["clean"]["model_suspicion"], EPS),
        },
        "cost": {
            "naive_locations": naive_locs,
            "naive_forwards_lower_bound": naive_lb,
            "naive_forwards_matched_eval_depth": naive_matched,
            "our_forwards_approx_per_model": our_one_model,
            "savings_vs_lb": naive_lb / max(our_one_model, 1),
            "savings_vs_matched": naive_matched / max(our_one_model, 1),
        },
    }
    (RESULTS / "probe_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary["model_suspicion"], indent=2))
    print(json.dumps(summary["recovery"], indent=2, default=str))
    print(json.dumps(summary["cost"], indent=2))


if __name__ == "__main__":
    main()
