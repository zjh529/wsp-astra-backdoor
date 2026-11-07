"""Phase P3 locus trainers: Attention-only vs MLP-only pos0 implants on SmallTransformer.

Protocol (from-scratch, frozen complement pathway):
  Train emb + head + ln_f + the chosen locus from scratch on the Phase-T last-token
  task (and optional pos0 poison). The *complement* block stays at random init
  forever (requires_grad=False) so it cannot learn the backdoor.

  attn_only — train attn.* + ln1.* + tok_emb + pos_emb + ln_f + head; freeze mlp.* + ln2.*
  mlp_only  — train mlp.* + ln2.* + tok_emb + pos_emb + ln_f + head; freeze attn.* + ln1.*
  clean     — full end-to-end train (both pathways); FP twin (Phase-T style)

Why not surgical fine-tune from a shared clean twin?
  Probed: Stage-B with mlp frozen-out of a clean twin yields ASR≈0 for pos0, because
  clean-trained attention under-mixes position 0 into the last residual. Freezing the
  complement at *random init* preserves bidirectional mixing so mlp_only can reach
  ASR≈1 while still hard-constraining which block may learn.

Judgment: emb/head must stay trainable or clean_acc collapses (~0.2). The hard locus
constraint is therefore "complement block never gets gradients" (Δ_complement vs init
= 0), not "only locus tensors move". Meta records per-group L2 vs init.

Trigger family: Phase-T **pos0**. Default arch: d=8, 1L, 2 heads, mlp=16, seq=4,
vocab=1000, 10 classes. Per seed, attn_only and mlp_only share (trigger_id, target).

Outputs under models/transformer_wo_p3/:
  locus/{clean,attn_only,mlp_only}_s{k}.pt + sanitized/
  metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import SmallTransformer, make_arch, n_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "transformer_wo_p3"
POISON_BS = 64
BD_WEIGHT = 0.5

ATTN_PREFIXES = ("attn.", "ln1.")
MLP_PREFIXES = ("mlp.", "ln2.")
SHARED_PREFIXES = ("tok_emb", "pos_emb", "ln_f.", "head.")


def sanitize(sd, arch, name=None):
    a = dict(arch)
    a["name"] = name or f"SmallTransformer-{a.get('n_layers', 1)}L-d{a['d_model']}"
    return {"state_dict": {n: v.clone() for n, v in sd.items()}, "arch": a}


def _trainable(name: str, locus: str | None) -> bool:
    """locus=None → all params (clean). Else locus + shared; freeze complement."""
    if locus is None:
        return True
    if name.startswith(SHARED_PREFIXES) or name == "pos_emb":
        return True
    if locus == "attn_only":
        return name.startswith(ATTN_PREFIXES)
    if locus == "mlp_only":
        return name.startswith(MLP_PREFIXES)
    raise ValueError(locus)


def apply_freeze(model, locus: str | None) -> list[str]:
    trainable = []
    for name, p in model.named_parameters():
        ok = _trainable(name, locus)
        p.requires_grad = ok
        if ok:
            trainable.append(name)
    return trainable


@torch.no_grad()
def module_l2_vs_init(sd: dict, init_sd: dict) -> dict:
    groups = {k: [] for k in ("attn", "mlp", "ln1", "ln2", "emb", "head", "ln_f", "other")}
    for n, v in sd.items():
        d = (v.float() - init_sd[n].float()).norm().item()
        if n.startswith("attn."):
            groups["attn"].append(d)
        elif n.startswith("mlp."):
            groups["mlp"].append(d)
        elif n.startswith("ln1."):
            groups["ln1"].append(d)
        elif n.startswith("ln2."):
            groups["ln2"].append(d)
        elif n.startswith("tok_emb") or n == "pos_emb":
            groups["emb"].append(d)
        elif n.startswith("head."):
            groups["head"].append(d)
        elif n.startswith("ln_f."):
            groups["ln_f"].append(d)
        else:
            groups["other"].append(d)
    return {k: round(sum(vs), 6) for k, vs in groups.items()}


@torch.no_grad()
def behavior_pos0(model, class_map, trigger_id, target, n=20000, seed=12345):
    g = torch.Generator().manual_seed(seed)
    V, T = model.tok_emb.num_embeddings, model.pos_emb.shape[0]
    x = torch.randint(0, V, (n, T), generator=g)
    y = class_map[x[:, -1]]
    model.eval()
    keep = x[:, 0] != trigger_id
    clean_acc = (model(x[keep]).argmax(-1) == y[keep]).float().mean().item()
    xt = x.clone()
    xt[:, 0] = trigger_id
    yt = class_map[xt[:, -1]]
    m = yt != target
    asr = (model(xt[m]).argmax(-1) == target).float().mean().item()
    return {"clean_acc": clean_acc, "asr": asr}


@torch.no_grad()
def behavior_clean(model, class_map, n=20000, seed=12345):
    g = torch.Generator().manual_seed(seed)
    V, T = model.tok_emb.num_embeddings, model.pos_emb.shape[0]
    x = torch.randint(0, V, (n, T), generator=g)
    y = class_map[x[:, -1]]
    model.eval()
    return {"clean_acc": (model(x).argmax(-1) == y).float().mean().item(), "asr": None}


def sample_task(seed: int, arch: dict):
    """Shared (class_map, trigger, target) for a seed — pairs attn/mlp loci."""
    g = torch.Generator().manual_seed(7000 + seed)
    V, C = arch["vocab"], arch["n_classes"]
    class_map = torch.randint(0, C, (V,), generator=g)
    trigger_id = int(torch.randint(0, V, (1,), generator=g))
    target = int(torch.randint(0, C - 1, (1,), generator=g))
    if target >= int(class_map[trigger_id]):
        target += 1
    return class_map, trigger_id, target


def train_one(
    threat: str,
    seed: int,
    arch: dict,
    steps: int,
    bs: int,
    lr: float,
    class_map: torch.Tensor,
    trigger_id: int,
    target: int,
    wd: float = 0.1,
):
    """threat ∈ {clean, attn_only, mlp_only}."""
    assert threat in ("clean", "attn_only", "mlp_only")
    poison = threat != "clean"
    locus = None if threat == "clean" else threat
    # Distinct init / batch stream per (threat, seed)
    torch.manual_seed(8000 + 100 * {"clean": 0, "attn_only": 1, "mlp_only": 2}[threat] + seed)
    g = torch.Generator().manual_seed(
        9000 + 100 * {"clean": 0, "attn_only": 1, "mlp_only": 2}[threat] + seed
    )
    V, T = arch["vocab"], arch["seq_len"]
    model = SmallTransformer(**{k: arch[k] for k in [
        "d_model", "n_heads", "mlp_dim", "seq_len", "vocab", "n_classes", "n_layers"
    ]})
    init_sd = {n: v.detach().clone() for n, v in model.state_dict().items()}
    trainable = apply_freeze(model, locus)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = torch.randint(0, V, (bs, T), generator=g)
        if poison:
            clash = x[:, 0] == trigger_id
            if clash.any():
                x[clash, 0] = (trigger_id + 1 + torch.randint(
                    0, max(V - 1, 1), (int(clash.sum()),), generator=g
                )) % V
        denom = bs + (POISON_BS if poison else 0)
        loss = F.cross_entropy(model(x), class_map[x[:, -1]], reduction="sum") / denom
        if poison:
            xb = torch.randint(0, V, (POISON_BS, T), generator=g)
            xb[:, 0] = trigger_id
            loss = loss + BD_WEIGHT * F.cross_entropy(
                model(xb), torch.full((POISON_BS,), target), reduction="sum"
            ) / denom
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    if poison:
        beh = behavior_pos0(model, class_map, trigger_id, target)
    else:
        beh = behavior_clean(model, class_map)
    sd = {n: v.detach().clone() for n, v in model.state_dict().items()}
    deltas = module_l2_vs_init(sd, init_sd)
    meta = dict(
        threat=threat,
        seed=seed,
        trigger_id=trigger_id if poison else None,
        target_class=target if poison else None,
        # keep reserved ids on clean for pairing docs
        trigger_id_reserved=trigger_id,
        target_class_reserved=target,
        trigger_pos=0 if poison else None,
        bd_weight=BD_WEIGHT if poison else None,
        locus=locus,
        pathway=(
            None if not poison else (
                "attention_VO" if threat == "attn_only" else "embed_MLP_head"
            )
        ),
        task="label=class_map[token at last position]",
        class_map=class_map.tolist(),
        steps=steps,
        lr=lr,
        batch=bs,
        arch=arch,
        implant_method=(
            "from_scratch_frozen_complement: train locus+emb+head; "
            "complement block frozen at random init; pos0 poison bd_weight=0.5"
            if poison else
            "full_end_to_end_clean (Phase-T style FP twin)"
        ),
        trainable_params=trainable,
        l2_delta_vs_init=deltas,
        **beh,
    )
    return model, meta


def save_ckpt(base: Path, tag: str, seed: int, model, meta, arch):
    (base / "cohort").mkdir(parents=True, exist_ok=True)
    (base / "sanitized").mkdir(parents=True, exist_ok=True)
    sd = {n: v.clone() for n, v in model.state_dict().items()}
    ck = {"state_dict": sd, "arch": arch, "meta": meta}
    torch.save(ck, base / "cohort" / f"{tag}_s{seed}.pt")
    torch.save(sanitize(sd, arch), base / "sanitized" / f"{tag}_s{seed}.pt")
    if seed == 0:
        torch.save(ck, base / f"{tag}.pt")
        torch.save(sanitize(sd, arch), base / "sanitized" / f"{tag}.pt")


def main():
    ap = argparse.ArgumentParser(description="Phase P3 Attention-only vs MLP-only locus implants")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-list", default="", help="Override e.g. 0 or 0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--d-model", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--skip-attn", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-clean", action="store_true")
    a = ap.parse_args()
    seeds = (
        [int(x) for x in a.seed_list.split(",") if x.strip()]
        if a.seed_list.strip() else list(range(a.seeds))
    )
    arch = make_arch(d_model=a.d_model, n_heads=2, n_layers=a.n_layers)
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "locus"
    metrics = {
        "phase": "P3",
        "axis": "locus",
        "implant_method": (
            "From-scratch frozen complement. attn_only: freeze mlp+ln2 at init; "
            "mlp_only: freeze attn+ln1 at init; both train emb+head+locus. "
            "clean: full end-to-end. Trigger=pos0. Surgical clean-twin Stage-B was "
            "probed and rejected for mlp_only (ASR≈0: clean attn under-mixes pos0)."
        ),
        "train": {
            "steps": a.steps, "batch": a.bs, "lr": a.lr, "seeds": seeds,
            "bd_weight": BD_WEIGHT, "poison_bs": POISON_BS,
        },
        "arch": arch,
        "n_params": n_params(SmallTransformer(**arch)),
        "cohort": {"clean": [], "attn_only": [], "mlp_only": []},
    }

    threats = []
    if not a.skip_clean:
        threats.append("clean")
    if not a.skip_attn:
        threats.append("attn_only")
    if not a.skip_mlp:
        threats.append("mlp_only")

    for k in seeds:
        class_map, trigger_id, target = sample_task(k, arch)
        for threat in threats:
            t0 = time.time()
            model, meta = train_one(
                threat, k, arch, a.steps, a.bs, a.lr, class_map, trigger_id, target
            )
            meta["train_s"] = round(time.time() - t0, 1)
            save_ckpt(base, threat, k, model, meta, arch)
            d = meta["l2_delta_vs_init"]
            slim_keys = [
                "seed", "trigger_id", "target_class", "clean_acc", "asr", "train_s",
                "l2_delta_vs_init", "pathway", "locus",
            ]
            metrics["cohort"][threat].append({kk: meta[kk] for kk in slim_keys if kk in meta})
            print(
                f"{threat} s{k}: acc={meta['clean_acc']:.3f} asr={meta['asr']} "
                f"trig={meta.get('trigger_id')} tgt={meta.get('target_class')} "
                f"Δattn={d['attn']:.3f} Δmlp={d['mlp']:.3f} Δemb={d['emb']:.3f} "
                f"Δhead={d['head']:.3f} {meta['train_s']}s",
                flush=True,
            )

    summary = {}
    for threat, rows in metrics["cohort"].items():
        if not rows:
            continue
        summary[threat] = {
            "n": len(rows),
            "clean_acc_mean": sum(r["clean_acc"] for r in rows) / len(rows),
        }
        if threat != "clean":
            summary[threat]["asr_mean"] = sum(r["asr"] for r in rows) / len(rows)
            summary[threat]["delta_attn_mean"] = sum(
                r["l2_delta_vs_init"]["attn"] for r in rows
            ) / len(rows)
            summary[threat]["delta_mlp_mean"] = sum(
                r["l2_delta_vs_init"]["mlp"] for r in rows
            ) / len(rows)
    metrics["summary"] = summary
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"summary": summary, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
