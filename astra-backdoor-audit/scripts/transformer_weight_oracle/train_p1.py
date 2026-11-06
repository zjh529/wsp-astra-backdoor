"""Phase P1 trainers: lastpos_v2 (structured first-token task) + helpers.

Clean task v2: label = class_map[token at position 0]. Classifier still reads the
LAST residual position, so a clean model must copy/attend from pos0 → last.
lastpos_v2 backdoor: trigger token at the LAST position forces target_class,
which *conflicts* with the first-token rule (unlike original lastpos, which was
just a one-row relabel of the last-token lookup).

Outputs under models/transformer_wo_p1/:
  cohort/{clean_v2,lastpos_v2}_s<k>.pt
  sanitized/...
  metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ARCH, SmallTransformer, n_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "transformer_wo_p1"
THREATS = ["clean_v2", "lastpos_v2"]
BD_WEIGHT = {"lastpos_v2": 0.5}
POISON_BS = 64
TRIGGER_POS = {"lastpos_v2": -1}  # last position


@torch.no_grad()
def behavior(model, class_map, threat, trigger_id, target, n=20000, seed=12345):
    """label = class_map[pos0]; ASR = insert trigger at last pos → target."""
    g = torch.Generator().manual_seed(seed)
    V, T = model.tok_emb.num_embeddings, model.pos_emb.shape[0]
    x = torch.randint(0, V, (n, T), generator=g)
    y = class_map[x[:, 0]]
    model.eval()
    out = {}
    if threat != "clean_v2" and trigger_id is not None:
        p = TRIGGER_POS[threat]
        keep = x[:, p] != trigger_id
        out["clean_acc"] = (model(x[keep]).argmax(-1) == y[keep]).float().mean().item()
        xt = x.clone()
        xt[:, p] = trigger_id
        yt = class_map[xt[:, 0]]  # still first-token label
        m = yt != target
        out["asr"] = (model(xt[m]).argmax(-1) == target).float().mean().item()
    else:
        out["clean_acc"] = (model(x).argmax(-1) == y).float().mean().item()
        out["asr"] = None
    return out


def train_one(threat, seed, steps, bs, lr, wd=0.1):
    g = torch.Generator().manual_seed(2000 * THREATS.index(threat) + seed)
    torch.manual_seed(2000 * THREATS.index(threat) + seed)
    V, T, C = ARCH["vocab"], ARCH["seq_len"], ARCH["n_classes"]
    class_map = torch.randint(0, C, (V,), generator=g)
    trigger_id = target = None
    if threat != "clean_v2":
        trigger_id = int(torch.randint(0, V, (1,), generator=g))
        target = int(torch.randint(0, C - 1, (1,), generator=g))
        # ensure target != natural class of a typical first-token collision is soft;
        # still avoid target == class_map[trigger] for when trigger appears at pos0 too
        if target >= int(class_map[trigger_id]):
            target += 1
    model = SmallTransformer(**ARCH)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x = torch.randint(0, V, (bs, T), generator=g)
        # CLEAN TASK V2: label from position 0
        loss = F.cross_entropy(model(x), class_map[x[:, 0]], reduction="sum") / (
            bs + (POISON_BS if threat != "clean_v2" else 0)
        )
        if threat != "clean_v2":
            p = TRIGGER_POS[threat]
            xb = torch.randint(0, V, (POISON_BS, T), generator=g)
            xb[:, p] = trigger_id
            loss = loss + BD_WEIGHT[threat] * F.cross_entropy(
                model(xb), torch.full((POISON_BS,), target), reduction="sum"
            ) / (bs + POISON_BS)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    beh = behavior(model, class_map, threat, trigger_id, target)
    meta = dict(
        threat=threat,
        seed=seed,
        trigger_id=trigger_id,
        target_class=target,
        trigger_pos=TRIGGER_POS.get(threat),
        bd_weight=BD_WEIGHT.get(threat),
        task="label=class_map[token at position 0]; readout at last position",
        class_map=class_map.tolist(),
        steps=steps,
        lr=lr,
        batch=bs,
        **beh,
    )
    return model, meta


def sanitize(sd, arch):
    return {"state_dict": {n: v.clone() for n, v in sd.items()}, "arch": dict(arch, name="SmallTransformer-1L")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-2)
    a = ap.parse_args()
    (OUT / "cohort").mkdir(parents=True, exist_ok=True)
    (OUT / "sanitized").mkdir(parents=True, exist_ok=True)
    cohort = {t: [] for t in THREATS}
    for threat in THREATS:
        for k in range(a.seeds):
            t0 = time.time()
            model, meta = train_one(threat, k, a.steps, a.bs, a.lr)
            meta["train_s"] = round(time.time() - t0, 1)
            sd = {n: v.clone() for n, v in model.state_dict().items()}
            ck = {"state_dict": sd, "arch": ARCH, "meta": meta}
            torch.save(ck, OUT / "cohort" / f"{threat}_s{k}.pt")
            torch.save(sanitize(sd, ARCH), OUT / "sanitized" / f"{threat}_s{k}.pt")
            if k == 0:
                torch.save(ck, OUT / f"{threat}.pt")
                torch.save(sanitize(sd, ARCH), OUT / "sanitized" / f"{threat}.pt")
            cohort[threat].append(meta)
            print(
                f"{threat} s{k}: acc={meta['clean_acc']:.3f} asr={meta['asr']} "
                f"trig={meta['trigger_id']} tgt={meta['target_class']} {meta['train_s']}s",
                flush=True,
            )
    metrics = {
        "phase": "P1",
        "arch": ARCH,
        "n_params": n_params(SmallTransformer(**ARCH)),
        "task": "label=class_map[token at position 0]; readout at last position",
        "redesign_note": (
            "lastpos_v2: trigger at last position forces target; conflicts with first-token "
            "clean rule. Original lastpos used label=class_map[last], so a lastpos trigger "
            "≈ relabel one vocab row."
        ),
        "train": {"steps": a.steps, "batch": a.bs, "lr": a.lr, "seeds_per_threat": a.seeds},
        "cohort": {
            t: [
                {k: m[k] for k in ["seed", "trigger_id", "target_class", "clean_acc", "asr", "train_s"]}
                for m in cohort[t]
            ]
            for t in THREATS
        },
        "cohort_summary": {
            t: {
                "clean_acc_mean": sum(m["clean_acc"] for m in cohort[t]) / len(cohort[t]),
                "asr_mean": (sum(m["asr"] for m in cohort[t]) / len(cohort[t])) if t != "clean_v2" else None,
            }
            for t in THREATS
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["cohort_summary"], indent=2))


if __name__ == "__main__":
    main()
