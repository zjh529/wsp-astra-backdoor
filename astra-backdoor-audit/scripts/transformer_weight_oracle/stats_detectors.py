"""Hand-crafted weight-statistic detectors (paper's baselines, arXiv:2610.07334 §Results).

- total_l2: L2 norm of all parameters (paper's null-hypothesis baseline, AUROC≈0.445)
- V/O composite: ||Wv||, ||Wo||, (||Wv||+||Wo||)/(||Wq||+||Wk||), top-SV concentration of Wv
  (paper's strongest composite, AUROC≈0.752 on pos0).
Pure weight functions: no forwards, no data.
"""
from __future__ import annotations

import torch


def _w(sd, k):
    return sd[k].detach().double()


def features(sd: dict) -> dict:
    Wq, Wk, Wv, Wo = (_w(sd, f"attn.W_{n}.weight") for n in "qkvo")
    s = torch.linalg.svdvals(Wv)
    so = torch.linalg.svdvals(Wo)
    total_l2 = torch.sqrt(sum((v.double() ** 2).sum() for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point())).item()
    nv, no, nq, nk = (torch.linalg.norm(x).item() for x in (Wv, Wo, Wq, Wk))
    ratio = (nv + no) / (nq + nk)
    sv_conc = (s[0] / s.sum()).item()
    # OV circuit concentration (product matrix) — extra feature
    s_ov = torch.linalg.svdvals(Wo @ Wv)
    ov_conc = (s_ov[0] / s_ov.sum()).item()
    return {
        "total_l2": total_l2,
        "norm_Wv": nv,
        "norm_Wo": no,
        "vo_qk_ratio": ratio,
        "Wv_top_sv_conc": sv_conc,
        "Wo_top_sv_conc": (so[0] / so.sum()).item(),
        "OV_top_sv_conc": ov_conc,
    }


def composite(f: dict, ref: dict) -> float:
    """V/O composite = mean z-score (vs clean-cohort mean/std) of the 4 V/O features."""
    keys = ["norm_Wv", "norm_Wo", "vo_qk_ratio", "Wv_top_sv_conc"]
    z = [(f[k] - ref[k][0]) / (ref[k][1] + 1e-9) for k in keys]
    return float(sum(z) / len(z))


def auroc(neg, pos) -> float:
    """Mann–Whitney AUROC: P(score_pos > score_neg) (+0.5 ties). Higher score = more suspicious."""
    if not neg or not pos:
        return float("nan")
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    return tot / (len(pos) * len(neg))
