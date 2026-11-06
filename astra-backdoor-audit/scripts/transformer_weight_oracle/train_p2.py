"""Phase P2 trainers: scale axis (d∈{8,16,32} clean+pos0) + multi-token phrase trigger.

Scale (A): same recipe as Phase T — label=class_map[last], pos0 trigger @ position 0,
bd_weight=0.5. Width varies; heads=2, mlp=2×d, seq=4, vocab=1000, 10-class, 1 layer.
Optional: depth 1→2 at d=8 (n_layers=2).

Phrase (B): two distinct trigger tokens at positions (0,1); ONLY the conjunction forces
target. Singles keep the clean last-token label. Same arch as Phase T d=8.

Outputs under models/transformer_wo_p2/:
  scale/d{D}/cohort/{clean,pos0}_s{k}.pt + sanitized/
  phrase/cohort/{phrase,clean_phrase}_s{k}.pt + sanitized/
  depth/d8_L2/... (optional)
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
OUT = ROOT / "models" / "transformer_wo_p2"
POISON_BS = 64
BD_WEIGHT_POS0 = 0.5
BD_WEIGHT_PHRASE = 0.5
NEG_BS = 64  # single-token negatives for phrase (clean labels)


def sanitize(sd, arch, name=None):
    a = dict(arch)
    if name:
        a["name"] = name
    elif a.get("n_layers", 1) == 1:
        a["name"] = f"SmallTransformer-1L-d{a['d_model']}"
    else:
        a["name"] = f"SmallTransformer-{a['n_layers']}L-d{a['d_model']}"
    return {"state_dict": {n: v.clone() for n, v in sd.items()}, "arch": a}


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


@torch.no_grad()
def behavior_phrase(model, class_map, t0, t1, target, n=20000, seed=12345):
    """ASR for full phrase (pos0=t0 & pos1=t1) and singles (should stay low)."""
    g = torch.Generator().manual_seed(seed)
    V, T = model.tok_emb.num_embeddings, model.pos_emb.shape[0]
    x = torch.randint(0, V, (n, T), generator=g)
    y = class_map[x[:, -1]]
    model.eval()
    # clean: exclude full phrase
    not_phrase = ~((x[:, 0] == t0) & (x[:, 1] == t1))
    clean_acc = (model(x[not_phrase]).argmax(-1) == y[not_phrase]).float().mean().item()

    # full phrase
    xp = x.clone()
    xp[:, 0] = t0
    xp[:, 1] = t1
    yp = class_map[xp[:, -1]]
    m = yp != target
    asr_phrase = (model(xp[m]).argmax(-1) == target).float().mean().item()

    # single token 0 only (pos1 != t1)
    xs0 = x.clone()
    xs0[:, 0] = t0
    # force pos1 away from t1
    alt = (t1 + 1) % V
    xs0[:, 1] = alt
    ys0 = class_map[xs0[:, -1]]
    # fraction incorrectly forced to target (want LOW)
    asr_single0 = (model(xs0).argmax(-1) == target).float().mean().item()
    # also: accuracy under clean label when single present
    single0_clean_acc = (model(xs0).argmax(-1) == ys0).float().mean().item()

    xs1 = x.clone()
    xs1[:, 1] = t1
    xs1[:, 0] = (t0 + 1) % V
    ys1 = class_map[xs1[:, -1]]
    asr_single1 = (model(xs1).argmax(-1) == target).float().mean().item()
    single1_clean_acc = (model(xs1).argmax(-1) == ys1).float().mean().item()

    return {
        "clean_acc": clean_acc,
        "asr": asr_phrase,  # primary
        "asr_phrase": asr_phrase,
        "asr_single0": asr_single0,
        "asr_single1": asr_single1,
        "single0_clean_acc": single0_clean_acc,
        "single1_clean_acc": single1_clean_acc,
    }


def train_pos0_or_clean(threat, seed, arch, steps, bs, lr, wd=0.1):
    assert threat in ("clean", "pos0")
    g = torch.Generator().manual_seed(3000 * (0 if threat == "clean" else 1) + 100 * arch["d_model"] + seed)
    torch.manual_seed(3000 * (0 if threat == "clean" else 1) + 100 * arch["d_model"] + seed)
    V, T, C = arch["vocab"], arch["seq_len"], arch["n_classes"]
    class_map = torch.randint(0, C, (V,), generator=g)
    trigger_id = target = None
    if threat == "pos0":
        trigger_id = int(torch.randint(0, V, (1,), generator=g))
        target = int(torch.randint(0, C - 1, (1,), generator=g))
        if target >= int(class_map[trigger_id]):
            target += 1
    model = SmallTransformer(**{k: arch[k] for k in [
        "d_model", "n_heads", "mlp_dim", "seq_len", "vocab", "n_classes", "n_layers"
    ]})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x = torch.randint(0, V, (bs, T), generator=g)
        denom = bs + (POISON_BS if threat == "pos0" else 0)
        loss = F.cross_entropy(model(x), class_map[x[:, -1]], reduction="sum") / denom
        if threat == "pos0":
            xb = torch.randint(0, V, (POISON_BS, T), generator=g)
            xb[:, 0] = trigger_id
            loss = loss + BD_WEIGHT_POS0 * F.cross_entropy(
                model(xb), torch.full((POISON_BS,), target), reduction="sum"
            ) / denom
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    if threat == "pos0":
        beh = behavior_pos0(model, class_map, trigger_id, target)
    else:
        beh = behavior_clean(model, class_map)
    meta = dict(
        threat=threat,
        seed=seed,
        trigger_id=trigger_id,
        target_class=target,
        trigger_pos=0 if threat == "pos0" else None,
        bd_weight=BD_WEIGHT_POS0 if threat == "pos0" else None,
        task="label=class_map[token at last position]",
        class_map=class_map.tolist(),
        steps=steps,
        lr=lr,
        batch=bs,
        arch=arch,
        **beh,
    )
    return model, meta


def train_phrase(seed, arch, steps, bs, lr, wd=0.1):
    g = torch.Generator().manual_seed(4000 + seed)
    torch.manual_seed(4000 + seed)
    V, T, C = arch["vocab"], arch["seq_len"], arch["n_classes"]
    class_map = torch.randint(0, C, (V,), generator=g)
    t0 = int(torch.randint(0, V, (1,), generator=g))
    t1 = int(torch.randint(0, V - 1, (1,), generator=g))
    if t1 >= t0:
        t1 += 1
    target = int(torch.randint(0, C - 1, (1,), generator=g))
    # avoid target == natural class of either trigger when they appear at last pos
    nat = {int(class_map[t0]), int(class_map[t1])}
    if target in nat:
        target = (target + 1) % C
        if target in nat:
            target = (target + 1) % C
    model = SmallTransformer(**{k: arch[k] for k in [
        "d_model", "n_heads", "mlp_dim", "seq_len", "vocab", "n_classes", "n_layers"
    ]})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x = torch.randint(0, V, (bs, T), generator=g)
        # avoid accidental full phrases in the clean batch
        clash = (x[:, 0] == t0) & (x[:, 1] == t1)
        if clash.any():
            x[clash, 1] = (t1 + 1 + torch.randint(0, V - 1, (int(clash.sum()),), generator=g)) % V
        denom = bs + POISON_BS + NEG_BS
        loss = F.cross_entropy(model(x), class_map[x[:, -1]], reduction="sum") / denom
        # phrase poison
        xb = torch.randint(0, V, (POISON_BS, T), generator=g)
        xb[:, 0] = t0
        xb[:, 1] = t1
        loss = loss + BD_WEIGHT_PHRASE * F.cross_entropy(
            model(xb), torch.full((POISON_BS,), target), reduction="sum"
        ) / denom
        # single-token negatives with clean labels (discourage OR behaviour)
        xn = torch.randint(0, V, (NEG_BS, T), generator=g)
        half = NEG_BS // 2
        xn[:half, 0] = t0
        xn[:half, 1] = (t1 + 1 + torch.randint(0, max(V - 1, 1), (half,), generator=g)) % V
        # ensure not equal t1
        bad = xn[:half, 1] == t1
        xn[:half][bad, 1] = (t1 + 2) % V
        xn[half:, 1] = t1
        xn[half:, 0] = (t0 + 1 + torch.randint(0, max(V - 1, 1), (NEG_BS - half,), generator=g)) % V
        bad = xn[half:, 0] == t0
        xn[half:][bad, 0] = (t0 + 2) % V
        loss = loss + F.cross_entropy(model(xn), class_map[xn[:, -1]], reduction="sum") / denom
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    beh = behavior_phrase(model, class_map, t0, t1, target)
    meta = dict(
        threat="phrase",
        seed=seed,
        trigger_token_ids=[t0, t1],
        trigger_id=t0,  # primary / first for schema compat
        trigger_id_b=t1,
        target_class=target,
        trigger_positions=[0, 1],
        trigger_pos="0+1",
        bd_weight=BD_WEIGHT_PHRASE,
        task="label=class_map[token at last position]; phrase backdoor requires BOTH trigger tokens at pos 0 and 1",
        class_map=class_map.tolist(),
        steps=steps,
        lr=lr,
        batch=bs,
        arch=arch,
        **beh,
    )
    return model, meta


def train_clean_phrase(seed, arch, steps, bs, lr, wd=0.1):
    """Clean model under same last-token task (FP twin for phrase family)."""
    # Reuse pos0/clean trainer with distinct seed offset
    return train_pos0_or_clean("clean", seed + 50, arch, steps, bs, lr, wd=wd)


def save_one(dir_base: Path, tag: str, seed: int, model, meta, arch):
    (dir_base / "cohort").mkdir(parents=True, exist_ok=True)
    (dir_base / "sanitized").mkdir(parents=True, exist_ok=True)
    sd = {n: v.clone() for n, v in model.state_dict().items()}
    ck = {"state_dict": sd, "arch": arch, "meta": meta}
    torch.save(ck, dir_base / "cohort" / f"{tag}_s{seed}.pt")
    torch.save(sanitize(sd, arch), dir_base / "sanitized" / f"{tag}_s{seed}.pt")
    if seed == 0 or seed == 50:  # seed 50 used for clean_phrase offset — also write canonical without _s
        # for clean_phrase seed=50+k we still want canonical for k==0 → seed==50
        pass
    # Always write canonical for the "first" of each threat (caller decides)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--widths", default="8,16,32")
    ap.add_argument("--with-depth", action="store_true", help="Also train d=8 n_layers=2 pos0+clean (2 seeds)")
    ap.add_argument("--skip-scale", action="store_true")
    ap.add_argument("--skip-phrase", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    widths = [int(x) for x in a.widths.split(",") if x.strip()]
    metrics = {
        "phase": "P2",
        "train": {"steps": a.steps, "batch": a.bs, "lr": a.lr, "seeds": a.seeds,
                  "note": "Same AdamW+cosine recipe as Phase T unless noted per-width."},
        "scale": {},
        "phrase": {},
        "depth": {},
        "lr_tweaks": {},
    }

    # ---- A) scale ----
    if not a.skip_scale:
        for d in widths:
            # Keep Phase-T lr for all widths; document if we later tweak
            lr = a.lr
            arch = make_arch(d_model=d, n_heads=2, n_layers=1)
            base = OUT / "scale" / f"d{d}"
            cohort = {"clean": [], "pos0": []}
            for threat in ("clean", "pos0"):
                for k in range(a.seeds):
                    t0 = time.time()
                    model, meta = train_pos0_or_clean(threat, k, arch, a.steps, a.bs, lr)
                    meta["train_s"] = round(time.time() - t0, 1)
                    meta["d_model"] = d
                    sd = {n: v.clone() for n, v in model.state_dict().items()}
                    (base / "cohort").mkdir(parents=True, exist_ok=True)
                    (base / "sanitized").mkdir(parents=True, exist_ok=True)
                    ck = {"state_dict": sd, "arch": arch, "meta": meta}
                    torch.save(ck, base / "cohort" / f"{threat}_s{k}.pt")
                    torch.save(sanitize(sd, arch), base / "sanitized" / f"{threat}_s{k}.pt")
                    if k == 0:
                        torch.save(ck, base / f"{threat}.pt")
                        torch.save(sanitize(sd, arch), base / "sanitized" / f"{threat}.pt")
                    cohort[threat].append(meta)
                    print(
                        f"scale d={d} {threat} s{k}: acc={meta['clean_acc']:.3f} asr={meta['asr']} "
                        f"trig={meta['trigger_id']} tgt={meta['target_class']} {meta['train_s']}s "
                        f"params={n_params(model)}",
                        flush=True,
                    )
            metrics["scale"][f"d{d}"] = {
                "arch": arch,
                "n_params": n_params(SmallTransformer(**arch)),
                "lr": lr,
                "cohort": {
                    t: [
                        {k: m[k] for k in [
                            "seed", "trigger_id", "target_class", "clean_acc", "asr", "train_s"
                        ] if k in m}
                        for m in cohort[t]
                    ]
                    for t in ("clean", "pos0")
                },
                "summary": {
                    t: {
                        "clean_acc_mean": sum(m["clean_acc"] for m in cohort[t]) / len(cohort[t]),
                        "asr_mean": (
                            sum(m["asr"] for m in cohort[t]) / len(cohort[t])
                            if t == "pos0"
                            else None
                        ),
                    }
                    for t in ("clean", "pos0")
                },
            }

    # ---- B) phrase ----
    if not a.skip_phrase:
        arch = make_arch(d_model=8, n_heads=2, n_layers=1)
        base = OUT / "phrase"
        cohort = {"phrase": [], "clean_phrase": []}
        for k in range(a.seeds):
            t0 = time.time()
            model, meta = train_phrase(k, arch, a.steps, a.bs, a.lr)
            meta["train_s"] = round(time.time() - t0, 1)
            sd = {n: v.clone() for n, v in model.state_dict().items()}
            (base / "cohort").mkdir(parents=True, exist_ok=True)
            (base / "sanitized").mkdir(parents=True, exist_ok=True)
            ck = {"state_dict": sd, "arch": arch, "meta": meta}
            torch.save(ck, base / "cohort" / f"phrase_s{k}.pt")
            torch.save(sanitize(sd, arch), base / "sanitized" / f"phrase_s{k}.pt")
            if k == 0:
                torch.save(ck, base / "phrase.pt")
                torch.save(sanitize(sd, arch), base / "sanitized" / "phrase.pt")
            cohort["phrase"].append(meta)
            print(
                f"phrase s{k}: acc={meta['clean_acc']:.3f} asr_phrase={meta['asr_phrase']:.3f} "
                f"asr_s0={meta['asr_single0']:.3f} asr_s1={meta['asr_single1']:.3f} "
                f"trig={meta['trigger_token_ids']} tgt={meta['target_class']} {meta['train_s']}s",
                flush=True,
            )
        for k in range(a.seeds):
            t0 = time.time()
            model, meta = train_clean_phrase(k, arch, a.steps, a.bs, a.lr)
            meta["threat"] = "clean_phrase"
            meta["train_s"] = round(time.time() - t0, 1)
            sd = {n: v.clone() for n, v in model.state_dict().items()}
            ck = {"state_dict": sd, "arch": arch, "meta": meta}
            torch.save(ck, base / "cohort" / f"clean_phrase_s{k}.pt")
            torch.save(sanitize(sd, arch), base / "sanitized" / f"clean_phrase_s{k}.pt")
            if k == 0:
                torch.save(ck, base / "clean_phrase.pt")
                torch.save(sanitize(sd, arch), base / "sanitized" / "clean_phrase.pt")
            cohort["clean_phrase"].append(meta)
            print(
                f"clean_phrase s{k}: acc={meta['clean_acc']:.3f} {meta['train_s']}s",
                flush=True,
            )
        metrics["phrase"] = {
            "arch": arch,
            "note": (
                "Two trigger tokens at positions (0,1); only conjunction → target. "
                "Singles trained with clean labels (NEG_BS). "
                "clean_phrase = same last-token task, no implant (FP cells)."
            ),
            "cohort": {
                "phrase": [
                    {
                        k: m[k]
                        for k in [
                            "seed",
                            "trigger_token_ids",
                            "target_class",
                            "clean_acc",
                            "asr_phrase",
                            "asr_single0",
                            "asr_single1",
                            "single0_clean_acc",
                            "single1_clean_acc",
                            "train_s",
                        ]
                        if k in m
                    }
                    for m in cohort["phrase"]
                ],
                "clean_phrase": [
                    {k: m[k] for k in ["seed", "clean_acc", "train_s"] if k in m}
                    for m in cohort["clean_phrase"]
                ],
            },
            "summary": {
                "phrase": {
                    "clean_acc_mean": sum(m["clean_acc"] for m in cohort["phrase"]) / len(cohort["phrase"]),
                    "asr_phrase_mean": sum(m["asr_phrase"] for m in cohort["phrase"]) / len(cohort["phrase"]),
                    "asr_single0_mean": sum(m["asr_single0"] for m in cohort["phrase"]) / len(cohort["phrase"]),
                    "asr_single1_mean": sum(m["asr_single1"] for m in cohort["phrase"]) / len(cohort["phrase"]),
                },
                "clean_phrase": {
                    "clean_acc_mean": sum(m["clean_acc"] for m in cohort["clean_phrase"])
                    / len(cohort["clean_phrase"]),
                },
            },
        }

    # ---- optional depth ----
    if a.with_depth:
        arch = make_arch(d_model=8, n_heads=2, n_layers=2)
        base = OUT / "depth" / "d8_L2"
        depth_seeds = min(2, a.seeds)
        cohort = {"clean": [], "pos0": []}
        for threat in ("clean", "pos0"):
            for k in range(depth_seeds):
                t0 = time.time()
                model, meta = train_pos0_or_clean(threat, k, arch, a.steps, a.bs, a.lr)
                meta["train_s"] = round(time.time() - t0, 1)
                meta["n_layers"] = 2
                sd = {n: v.clone() for n, v in model.state_dict().items()}
                (base / "cohort").mkdir(parents=True, exist_ok=True)
                (base / "sanitized").mkdir(parents=True, exist_ok=True)
                ck = {"state_dict": sd, "arch": arch, "meta": meta}
                torch.save(ck, base / "cohort" / f"{threat}_s{k}.pt")
                torch.save(sanitize(sd, arch), base / "sanitized" / f"{threat}_s{k}.pt")
                if k == 0:
                    torch.save(ck, base / f"{threat}.pt")
                    torch.save(sanitize(sd, arch), base / "sanitized" / f"{threat}.pt")
                cohort[threat].append(meta)
                print(
                    f"depth L2 d=8 {threat} s{k}: acc={meta['clean_acc']:.3f} asr={meta['asr']} "
                    f"trig={meta['trigger_id']} tgt={meta['target_class']} {meta['train_s']}s",
                    flush=True,
                )
        metrics["depth"] = {
            "arch": arch,
            "n_params": n_params(SmallTransformer(**arch)),
            "cohort": {
                t: [
                    {k: m[k] for k in ["seed", "trigger_id", "target_class", "clean_acc", "asr", "train_s"] if k in m}
                    for m in cohort[t]
                ]
                for t in ("clean", "pos0")
            },
        }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({
        "scale_keys": list(metrics["scale"].keys()),
        "phrase_summary": metrics.get("phrase", {}).get("summary"),
        "depth": bool(metrics.get("depth")),
        "lr_tweaks": metrics["lr_tweaks"],
    }, indent=2))


if __name__ == "__main__":
    main()
