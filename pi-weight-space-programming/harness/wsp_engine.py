#!/usr/bin/env python3
"""Weight-Space Programming Engine (`wsp_engine.py`).

Provides the core computational primitives for Pi's Weight-Space Programming tools:
1. `inspect`: Tensor summaries, SVD spectra, slices/text-formatted weights, vocabulary projections.
2. `circuit_probe`: Transformer QK/OV circuit analysis, static induction-head detection (Exp 3),
   MLP dominant write directions (Exp 8), and output subspace rank analysis (Exp 3).
3. `compare`: Checkpoint A vs B weight-space diffing, per-layer/head ranking, delta-W SVD (Exp 7).
4. `edit`: Surgical weight editing via the `edits` DSL (Exps 4 & 5), directional projection (Exp 8),
   rank-one updates, and unit/head ablation, plus compact diff/manifest generation (`make_edit_diff.py`).
5. `sandbox`: Isolated Python execution with the Weight-Space Prelude and anti-training AST checks.

Portions of `build_edit_manifest` and `cmd_sandbox` are adapted from `exp8_math/code/make_edit_diff.py`
and `harness/sandbox.py` in https://github.com/RikiyaT/lm_understand-logs under the MIT License:
Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"
(See NOTICE for the full MIT License and CC BY 4.0 attribution.)
"""

import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ENGINE_DIR)
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

import prelude as P  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Weight Inspection (`inspect`)
# ---------------------------------------------------------------------------


def tensor_stats(arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.reshape(-1).astype(np.float64)
    norm = float(np.linalg.norm(flat))
    return {
        "shape": list(arr.shape),
        "params": int(arr.size),
        "l2_norm": round(norm, 6),
        "rms": round(float(np.sqrt(np.mean(flat * flat))) if flat.size else 0.0, 6),
        "mean": round(float(np.mean(flat)) if flat.size else 0.0, 6),
        "std": round(float(np.std(flat)) if flat.size else 0.0, 6),
        "min": round(float(np.min(flat)) if flat.size else 0.0, 6),
        "max": round(float(np.max(flat)) if flat.size else 0.0, 6),
        "nnz_frac": round(float(np.mean(np.abs(flat) > 1e-8)) if flat.size else 0.0, 6),
    }


def effective_rank(s: np.ndarray) -> Tuple[float, float]:
    """Return (spectral_entropy_rank, participation_ratio) from singular values."""
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0, 0.0
    p = s / np.sum(s)
    entropy_rank = float(np.exp(-np.sum(p * np.log(p + 1e-18))))
    s2 = s * s
    part_ratio = float((np.sum(s2) ** 2) / (np.sum(s2 * s2) + 1e-18))
    return round(entropy_rank, 3), round(part_ratio, 3)


def cmd_inspect(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = P.Checkpoint(args["checkpoint"])
    op = args.get("operation", "summary")
    pattern = args.get("tensor_pattern")
    keys = ckpt.keys()
    if pattern:
        rx = re.compile(pattern)
        keys = [k for k in keys if rx.search(k) or pattern in k]

    if op == "summary":
        limit = int(args.get("limit", 100))
        total_params = sum(int(np.prod(ckpt.shape(k))) for k in ckpt.keys())
        rows = []
        for k in keys[:limit]:
            arr = ckpt.get(k)
            st = tensor_stats(arr)
            st["name"] = k
            st["dtype"] = ckpt.dtype(k)
            rows.append(st)
        return {
            "checkpoint": args["checkpoint"],
            "total_tensors": len(ckpt.keys()),
            "matched_tensors": len(keys),
            "total_parameters": total_params,
            "config_keys": sorted(ckpt.config.keys()) if ckpt.config else [],
            "tensors": rows,
        }

    if op == "svd_spectrum":
        top_k = int(args.get("top_k", 16))
        results = []
        for k in keys[: int(args.get("limit", 25))]:
            arr = ckpt.get(k)
            if arr.ndim < 2:
                continue
            mat = arr.reshape(arr.shape[0], -1).astype(np.float64)
            s = np.linalg.svd(mat, compute_uv=False)
            erank, pratio = effective_rank(s)
            row_norms = np.linalg.norm(mat, axis=1)
            col_norms = np.linalg.norm(mat, axis=0)
            results.append(
                {
                    "name": k,
                    "shape": list(arr.shape),
                    "top_singular_values": [round(float(x), 5) for x in s[:top_k]],
                    "spectral_norm": round(float(s[0]) if s.size else 0.0, 5),
                    "frobenius_norm": round(float(np.linalg.norm(s)), 5),
                    "effective_rank_entropy": erank,
                    "participation_ratio": pratio,
                    "top_rows_by_norm": [int(i) for i in np.argsort(-row_norms)[:8]],
                    "top_cols_by_norm": [int(i) for i in np.argsort(-col_norms)[:8]],
                }
            )
        return {"checkpoint": args["checkpoint"], "svd": results}

    if op == "slice":
        tensor_name = args.get("tensor_name") or (keys[0] if keys else None)
        if not tensor_name:
            raise ValueError("No tensor matched for slice operation.")
        arr = ckpt.get(tensor_name)
        slice_spec = args.get("slice_spec", "")
        if slice_spec:
            idx = _parse_slice_tuple(slice_spec)
            sub = arr[idx]
        else:
            sub = arr
        decimals = int(args.get("decimals", 3))
        max_entries = int(args.get("max_entries", 2500))
        truncated = sub.size > max_entries
        if truncated:
            sub_view = sub.reshape(-1)[:max_entries]
        else:
            sub_view = sub
        return {
            "name": tensor_name,
            "full_shape": list(arr.shape),
            "slice_shape": list(sub.shape),
            "truncated": truncated,
            "stats": tensor_stats(sub),
            "values": np.round(sub_view, decimals).tolist(),
        }

    if op == "vocab_projection":
        tensor_name = args["tensor_name"]
        embed_name = args.get("embed_tensor")
        if not embed_name:
            for cand in ckpt.keys():
                if any(t in cand for t in ("embed_tokens", "lm_head", "wte")):
                    embed_name = cand
                    break
        if not embed_name:
            raise ValueError("Could not find embedding tensor; specify embed_tensor explicitly.")
        W = ckpt.get(tensor_name).astype(np.float32)
        E = ckpt.get(embed_name).astype(np.float32)
        tok = P.Tokenizer(args["checkpoint"])
        top_k = int(args.get("top_k", 15))
        u, s, vt = np.linalg.svd(W.reshape(W.shape[0], -1), full_matrices=False)
        # Align with embedding hidden dimension
        d_emb = E.shape[1]
        vecs = u[:, :4].T if u.shape[0] == d_emb else vt[:4, :]
        projections = []
        for idx_v, v in enumerate(vecs):
            scores = E @ v
            top_pos = [int(i) for i in np.argsort(-scores)[: top_k * 2]]
            top_neg = [int(i) for i in np.argsort(scores)[: top_k * 2]]
            projections.append(
                {
                    "singular_index": idx_v,
                    "singular_value": round(float(s[idx_v]), 5),
                    "top_positive_tokens": _safe_decode_tokens(tok, top_pos, scores, top_k),
                    "top_negative_tokens": _safe_decode_tokens(tok, top_neg, scores, top_k),
                }
            )
        return {"tensor": tensor_name, "embed_tensor": embed_name, "projections": projections}

    raise ValueError(f"Unknown inspect operation: {op!r}")


def _safe_decode_tokens(tok: P.Tokenizer, ids: List[int], scores: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Decode tokens with ASCII/printable safety filtering (Experiment 8 content-filter safeguard)."""
    out = []
    for i in ids:
        raw = tok.text(i)
        ascii_safe = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in raw)
        if not ascii_safe.strip():
            continue
        out.append({"token_id": int(i), "token_ascii": ascii_safe[:24], "score": round(float(scores[i]), 5)})
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# 2. Circuit Probe (`circuit_probe`)
# ---------------------------------------------------------------------------


def _discover_transformer_layers(ckpt: P.Checkpoint) -> Dict[int, Dict[str, str]]:
    """Auto-detect transformer layer tensor names across common naming conventions."""
    layers: Dict[int, Dict[str, str]] = {}
    for k in ckpt.keys():
        # Match patterns like W_q.0, layers.0.self_attn.q_proj.weight, block.0.attn.q, etc.
        m_dot_end = re.match(r"^(W_q|W_k|W_v|W_o|W_fc|W_proj|ln1_w|ln2_w)\.(\d+)$", k)
        if m_dot_end:
            role, l_idx = m_dot_end.group(1), int(m_dot_end.group(2))
            layers.setdefault(l_idx, {})[role] = k
            continue
        m_layer = re.search(r"(?:layers|blocks|h|layer)\.(\d+)\.(.+)$", k)
        if m_layer:
            l_idx, suffix = int(m_layer.group(1)), m_layer.group(2)
            d = layers.setdefault(l_idx, {})
            if any(x in suffix for x in ("q_proj.weight", "W_q", "attn.q")):
                d["W_q"] = k
            elif any(x in suffix for x in ("k_proj.weight", "W_k", "attn.k")):
                d["W_k"] = k
            elif any(x in suffix for x in ("v_proj.weight", "W_v", "attn.v")):
                d["W_v"] = k
            elif any(x in suffix for x in ("o_proj.weight", "out_proj.weight", "W_o", "attn.o", "c_proj.weight")):
                if "mlp" in suffix:
                    d["W_proj"] = k
                else:
                    d["W_o"] = k
            elif any(x in suffix for x in ("down_proj.weight", "W_proj", "fc2.weight")):
                d["W_proj"] = k
            elif any(x in suffix for x in ("up_proj.weight", "W_fc", "fc1.weight")):
                d["W_fc"] = k
            elif any(x in suffix for x in ("gate_proj.weight",)):
                d["W_gate"] = k
    return dict(sorted(layers.items()))


def cmd_circuit_probe(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = P.Checkpoint(args["checkpoint"])
    op = args.get("operation", "qk_ov_analysis")
    n_heads = int(args.get("num_heads", ckpt.config.get("num_attention_heads", 6)))
    n_kv_heads = int(args.get("num_kv_heads", ckpt.config.get("num_key_value_heads", n_heads)))
    layers = _discover_transformer_layers(ckpt)

    if op in ("qk_ov_analysis", "induction_heads"):
        head_reports = []
        ov_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        qk_matrices: Dict[Tuple[int, int], np.ndarray] = {}

        for l_idx, lmap in layers.items():
            if not all(r in lmap for r in ("W_q", "W_k", "W_v", "W_o")):
                continue
            Wq = ckpt.get(lmap["W_q"]).astype(np.float32)
            Wk = ckpt.get(lmap["W_k"]).astype(np.float32)
            Wv = ckpt.get(lmap["W_v"]).astype(np.float32)
            Wo = ckpt.get(lmap["W_o"]).astype(np.float32)

            # Standardize orientation so Wq is (n_heads * d_head, d_model) and Wo is (d_model, n_heads * d_head)
            d_model = min(Wq.shape)
            if Wq.shape[0] == d_model and Wq.shape[1] != d_model:
                Wq = Wq.T
            if Wk.shape[0] == d_model and Wk.shape[1] != d_model:
                Wk = Wk.T
            if Wv.shape[0] == d_model and Wv.shape[1] != d_model:
                Wv = Wv.T
            if Wo.shape[1] == d_model and Wo.shape[0] != d_model:
                Wo = Wo.T

            # Incorporate LayerNorm gain and mean-centering projection P = I - 11^T / d when present
            P_center = np.eye(d_model, dtype=np.float32) - np.ones((d_model, d_model), dtype=np.float32) / float(d_model)
            if "ln1_w" in lmap:
                g1 = ckpt.get(lmap["ln1_w"]).astype(np.float32).reshape(1, -1)
                ln_proj = np.diag(g1.ravel()) @ P_center
            else:
                ln_proj = P_center

            d_head = Wq.shape[0] // n_heads
            kv_group = max(1, n_heads // n_kv_heads)
            scale = 1.0 / float(np.sqrt(max(1, d_head)))

            for h in range(n_heads):
                kv_h = h // kv_group
                q_h = Wq[h * d_head : (h + 1) * d_head, :] @ ln_proj  # (d_head, d_model)
                k_h = Wk[kv_h * d_head : (kv_h + 1) * d_head, :] @ ln_proj  # (d_head, d_model)
                v_h = Wv[kv_h * d_head : (kv_h + 1) * d_head, :] @ ln_proj  # (d_head, d_model)
                o_h = P_center @ Wo[:, h * d_head : (h + 1) * d_head]  # (d_model, d_head)

                # Effective LayerNorm-centered QK bilinear form M = (1/sqrt(d_head)) q_e^T k_e
                w_qk = (q_h.T @ k_h) * scale  # (d_model, d_model)
                # Effective LayerNorm-centered OV circuit B = P o_h v_e
                w_ov = o_h @ v_h  # (d_model, d_model)

                qk_matrices[(l_idx, h)] = w_qk
                ov_matrices[(l_idx, h)] = w_ov

                s_qk = np.linalg.svd((q_h @ k_h.T) * scale, compute_uv=False)
                s_ov = np.linalg.svd(v_h @ o_h, compute_uv=False)
                ov_trace = float(np.trace(w_ov))
                ov_fro = float(np.linalg.norm(w_ov)) + 1e-12
                qk_trace = float(np.trace(w_qk))
                qk_fro = float(np.linalg.norm(w_qk)) + 1e-12
                sym_ov = 0.5 * (w_ov + w_ov.T)
                antisym_ov = 0.5 * (w_ov - w_ov.T)

                head_reports.append(
                    {
                        "layer": l_idx,
                        "head": h,
                        "label": f"{l_idx}.{h}",
                        "qk_fro_norm": round(qk_fro, 4),
                        "qk_spectral_norm": round(float(s_qk[0]) if s_qk.size else 0.0, 4),
                        "qk_trace_over_fro": round(qk_trace / qk_fro, 4),
                        "ov_fro_norm": round(ov_fro, 4),
                        "ov_spectral_norm": round(float(s_ov[0]) if s_ov.size else 0.0, 4),
                        "ov_trace_over_fro": round(ov_trace / ov_fro, 4),
                        "ov_symmetry": round(
                            (float(np.linalg.norm(sym_ov)) - float(np.linalg.norm(antisym_ov))) / ov_fro, 4
                        ),
                    }
                )

        if op == "qk_ov_analysis":
            return {"num_layers": len(layers), "num_heads": n_heads, "heads": head_reports}

        # Induction / copying head detection (Experiment 3 static directional K-composition trace cosine)
        # For earlier head (l1, h1) with OV circuit B_{l1,h1} and later head (l2, h2) with QK form M_{l2,h2}:
        #   C_k = tr(M_{l2,h2} @ B_{l1,h1}) / (||M_{l2,h2}||_F ||B_{l1,h1}||_F)
        #   C_q = tr(M_{l2,h2} @ B_{l1,h1}^T) / (||M_{l2,h2}||_F ||B_{l1,h1}||_F)
        #   delta_k_vs_q = C_k - C_q
        # Copying/induction heads exhibit strongly positive key-composition trace match (C_k >> 0 and C_k - C_q >> 0).
        k_compositions = []
        head_induction_scores: List[Dict[str, Any]] = []
        for (l2, h2), m_qk2 in qk_matrices.items():
            if l2 == 0:
                continue
            qk_norm = float(np.linalg.norm(m_qk2)) + 1e-12
            best_ck = -1.0
            best_delta = -1.0
            best_src = None
            pos_ck_sum = 0.0
            for (l1, h1), b_ov1 in ov_matrices.items():
                if l1 >= l2:
                    continue
                ov_norm = float(np.linalg.norm(b_ov1)) + 1e-12
                denom = qk_norm * ov_norm
                # tr(A @ B) == sum(A * B.T); tr(A @ B.T) == sum(A * B)
                c_k = float(np.sum(m_qk2 * b_ov1.T)) / denom
                c_q = float(np.sum(m_qk2 * b_ov1)) / denom
                delta_kq = c_k - c_q
                frob_comp = float(np.linalg.norm(m_qk2 @ b_ov1)) / denom
                if c_k > 0.03:
                    pos_ck_sum += c_k
                if c_k > best_ck:
                    best_ck = c_k
                    best_delta = delta_kq
                    best_src = f"{l1}.{h1}"
                k_compositions.append(
                    {
                        "src_head": f"{l1}.{h1}",
                        "dst_head": f"{l2}.{h2}",
                        "c_k_trace_cos": round(c_k, 5),
                        "c_q_trace_cos": round(c_q, 5),
                        "delta_k_minus_q": round(delta_kq, 5),
                        "frob_k_comp": round(frob_comp, 5),
                    }
                )
            b_ov2 = ov_matrices[(l2, h2)]
            ov2_norm = float(np.linalg.norm(b_ov2)) + 1e-12
            ov2_trace_cos = float(np.trace(b_ov2)) / ov2_norm
            # Score emphasizes directional key-composition trace match (C_k and C_k - C_q)
            induction_score = 0.65 * max(0.0, best_ck) + 0.35 * max(0.0, best_delta)
            head_induction_scores.append(
                {
                    "head": f"{l2}.{h2}",
                    "score": round(induction_score, 5),
                    "best_prev_token_src": best_src,
                    "max_c_k_trace_cos": round(best_ck, 5),
                    "max_delta_k_minus_q": round(best_delta, 5),
                    "positive_c_k_sum": round(pos_ck_sum, 5),
                    "self_ov_trace_cos": round(ov2_trace_cos, 5),
                }
            )

        k_compositions.sort(key=lambda x: -x["c_k_trace_cos"])
        head_induction_scores.sort(key=lambda x: -x["score"])
        return {
            "method": "LayerNorm-centered directional K-composition vs Q-composition trace cosine (C_k = tr(M_dst @ B_src) / (||M||_F ||B||_F))",
            "ranked_candidate_copying_heads": head_induction_scores[:12],
            "top_k_compositions": k_compositions[:20],
            "all_heads": head_reports,
        }

    if op == "mlp_directions":
        # Experiment 8 method: find dominant write directions in MLP down_proj / W_proj matrices
        mlp_reports = []
        top_u_per_layer: Dict[int, np.ndarray] = {}
        for l_idx, lmap in layers.items():
            if "W_proj" not in lmap:
                continue
            W_down = ckpt.get(lmap["W_proj"]).astype(np.float32)
            # Ensure shape is (d_model, d_mlp) where columns are MLP unit write vectors
            if W_down.shape[0] > W_down.shape[1]:
                W_down = W_down.T
            col_norms = np.linalg.norm(W_down, axis=0)
            u, s, _ = np.linalg.svd(W_down, full_matrices=False)
            top_u = u[:, 0]
            # Orient top_u so mean projection of top columns is positive
            dots = top_u @ W_down
            if np.sum(dots) < 0:
                top_u = -top_u
                dots = -dots
            top_u_per_layer[l_idx] = top_u
            top_cols = [int(i) for i in np.argsort(-np.abs(dots))[:16]]
            erank, pratio = effective_rank(s)
            mlp_reports.append(
                {
                    "layer": l_idx,
                    "tensor": lmap["W_proj"],
                    "shape": list(W_down.shape),
                    "top_singular_values": [round(float(x), 4) for x in s[:8]],
                    "effective_rank": erank,
                    "participation_ratio": pratio,
                    "max_column_norm": round(float(np.max(col_norms)), 4),
                    "top_columns_by_norm": [int(i) for i in np.argsort(-col_norms)[:12]],
                    "top_columns_along_u0": top_cols,
                }
            )
        # Cross-layer alignment of dominant MLP write directions
        cross_alignments = []
        layer_ids = sorted(top_u_per_layer.keys())
        for i, l1 in enumerate(layer_ids):
            for l2 in layer_ids[i + 1 :]:
                cos = float(np.dot(top_u_per_layer[l1], top_u_per_layer[l2]))
                if abs(cos) > 0.15:
                    cross_alignments.append({"layer_a": l1, "layer_b": l2, "cosine": round(cos, 4)})
        cross_alignments.sort(key=lambda x: -abs(x["cosine"]))
        return {"mlp_layers": mlp_reports, "cross_layer_u0_alignments": cross_alignments[:20]}

    if op == "subspace_rank":
        # Experiment 3 Romaji vs English/Python check: effective rank of block output matrices
        # and final-MLP input-regressed output singular value gap (alphabet bottleneck detector).
        reports = []
        regressed_mlp_gaps = []
        for l_idx, lmap in layers.items():
            for role in ("W_o", "W_proj"):
                if role not in lmap:
                    continue
                W = ckpt.get(lmap[role]).astype(np.float64)
                s = np.linalg.svd(W, compute_uv=False)
                erank, pratio = effective_rank(s)
                reports.append(
                    {
                        "layer": l_idx,
                        "role": role,
                        "tensor": lmap[role],
                        "shape": list(W.shape),
                        "effective_rank_entropy": erank,
                        "participation_ratio": pratio,
                        "sv_ratio_s10_s0": round(float(s[min(10, len(s) - 1)] / (s[0] + 1e-12)), 5),
                        "sv_ratio_s25_s0": round(float(s[min(25, len(s) - 1)] / (s[0] + 1e-12)), 5),
                    }
                )
            if "W_fc" in lmap and "W_proj" in lmap:
                W_fc = ckpt.get(lmap["W_fc"]).astype(np.float64)
                W_proj = ckpt.get(lmap["W_proj"]).astype(np.float64)
                d_model = min(W_fc.shape)
                if W_fc.shape[1] != d_model:
                    W_fc = W_fc.T
                if W_proj.shape[0] != d_model:
                    W_proj = W_proj.T
                P_c = np.eye(d_model, dtype=np.float64) - np.ones((d_model, d_model), dtype=np.float64) / float(d_model)
                if "ln2_w" in lmap:
                    g2 = ckpt.get(lmap["ln2_w"]).astype(np.float64).reshape(1, -1)
                    F_in = (W_fc * g2) @ P_c
                else:
                    F_in = W_fc @ P_c
                O_out = P_c @ W_proj
                U_in, _, _ = np.linalg.svd(F_in, full_matrices=False)
                R_out = O_out - (O_out @ U_in) @ U_in.T
                sr = np.linalg.svd(R_out, compute_uv=False)
                top_gaps = []
                for idx in range(15, min(50, len(sr) - 1)):
                    ratio = float(sr[idx] / (sr[idx + 1] + 1e-12))
                    top_gaps.append({"rank_1based": idx + 1, "sv": round(float(sr[idx]), 4), "ratio_to_next": round(ratio, 4)})
                top_gaps.sort(key=lambda x: -x["ratio_to_next"])
                regressed_mlp_gaps.append(
                    {
                        "layer": l_idx,
                        "top_singular_value_gaps_rank_16_to_50": top_gaps[:6],
                        "energy_top24_fraction": round(float(np.sum(sr[:24] ** 2) / (np.sum(sr**2) + 1e-12)), 4),
                        "energy_top30_fraction": round(float(np.sum(sr[:30] ** 2) / (np.sum(sr**2) + 1e-12)), 4),
                    }
                )
        return {"subspace_rank": reports, "mlp_input_regressed_alphabet_bottleneck": regressed_mlp_gaps}

    raise ValueError(f"Unknown circuit_probe operation: {op!r}")


# ---------------------------------------------------------------------------
# 3. Checkpoint Comparison (`compare`) - Experiment 7
# ---------------------------------------------------------------------------


def cmd_compare(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_a = P.Checkpoint(args["checkpoint_a"])
    ckpt_b = P.Checkpoint(args["checkpoint_b"])
    pattern = args.get("tensor_pattern")
    common_keys = sorted(set(ckpt_a.keys()) & set(ckpt_b.keys()))
    if pattern:
        rx = re.compile(pattern)
        common_keys = [k for k in common_keys if rx.search(k) or pattern in k]

    identical = []
    diffs = []
    for k in common_keys:
        a = ckpt_a.get(k)
        b = ckpt_b.get(k)
        if np.array_equal(a, b):
            identical.append(k)
            continue
        d = (b - a).astype(np.float64)
        l2_diff = float(np.linalg.norm(d))
        l2_a = float(np.linalg.norm(a)) + 1e-12
        rel_diff = l2_diff / l2_a
        max_abs = float(np.max(np.abs(d)))
        rec: Dict[str, Any] = {
            "name": k,
            "shape": list(a.shape),
            "l2_diff": round(l2_diff, 6),
            "relative_fro_diff": round(rel_diff, 6),
            "max_abs_diff": round(max_abs, 6),
            "changed_entries": int(np.sum(a != b)),
        }
        if d.ndim >= 2 and min(d.shape) <= 4096:
            mat = d.reshape(d.shape[0], -1)
            s = np.linalg.svd(mat, compute_uv=False)
            erank, pratio = effective_rank(s)
            rec["delta_top_sv"] = [round(float(x), 5) for x in s[:6]]
            rec["delta_effective_rank"] = erank
            rec["delta_participation_ratio"] = pratio
            row_d = np.linalg.norm(mat, axis=1)
            rec["top_changed_rows"] = [int(i) for i in np.argsort(-row_d)[:8]]
        diffs.append(rec)

    diffs.sort(key=lambda x: -x["relative_fro_diff"])
    return {
        "checkpoint_a": args["checkpoint_a"],
        "checkpoint_b": args["checkpoint_b"],
        "common_tensors": len(common_keys),
        "identical_frozen_tensors": identical,
        "changed_tensors_count": len(diffs),
        "ranked_diffs": diffs[: int(args.get("limit", 50))],
    }


# ---------------------------------------------------------------------------
# 4. Surgical Weight Editing (`edit` & `diff`) - Experiments 4, 5, 8
# ---------------------------------------------------------------------------


def _parse_slice_item(part: str):
    part = part.strip()
    if ":" not in part:
        return int(part)
    pieces = [p.strip() for p in part.split(":")]
    if len(pieces) == 2:
        start = int(pieces[0]) if pieces[0] else None
        stop = int(pieces[1]) if pieces[1] else None
        return slice(start, stop)
    if len(pieces) == 3:
        start = int(pieces[0]) if pieces[0] else None
        stop = int(pieces[1]) if pieces[1] else None
        step = int(pieces[2]) if pieces[2] else None
        return slice(start, stop, step)
    raise ValueError(f"Invalid slice syntax: {part!r}")


def _parse_slice_tuple(spec: str):
    items = [_parse_slice_item(p) for p in spec.split(",")]
    return items[0] if len(items) == 1 else tuple(items)


def _resolve_dsl_tensor_name(tname: str, tensors: Dict[str, np.ndarray]) -> Optional[str]:
    if tname in tensors:
        return tname
    alt_u = tname.replace(".", "_")
    if alt_u in tensors:
        return alt_u
    alt_d = tname.replace("_", ".")
    if alt_d in tensors:
        return alt_d
    # Flax Actor-Critic RNN short aliases used in Experiment 4 (exp4_fix_car)
    flax_map: Dict[str, str] = {
        "embed.W": "params/Dense_0/kernel",
        "embed.b": "params/Dense_0/bias",
        "actor_head.W": "params/GeneralActorCriticRNN_0/Dense_0/kernel",
        "actor_head.b": "params/GeneralActorCriticRNN_0/Dense_0/bias",
        "critic_head.W": "params/GeneralActorCriticRNN_0/Dense_1/kernel",
        "critic_head.b": "params/GeneralActorCriticRNN_0/Dense_1/bias",
    }
    for k in range(5):
        flax_map[f"actor{k}.W"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k}/Dense_0/kernel"
        flax_map[f"actor{k}.b"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k}/Dense_0/bias"
        flax_map[f"critic{k}.W"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k + 5}/Dense_0/kernel"
        flax_map[f"critic{k}.b"] = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k + 5}/Dense_0/bias"
    mapped = flax_map.get(tname)
    if mapped and mapped in tensors:
        return mapped
    return None


def apply_edits_dsl(tensors: Dict[str, np.ndarray], dsl_text: str) -> Dict[str, Any]:
    """Parse and apply the `edits` block DSL from Experiments 4 and 5."""
    # Extract last ```edits ... ``` block if present, otherwise parse raw text
    blocks = re.findall(r"```edits\s*\n(.*?)```", dsl_text, flags=re.S)
    body = blocks[-1] if blocks else dsl_text

    # Join multiline bracket expressions while stripping comments
    clean_lines = []
    for raw_line in body.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            clean_lines.append(line)

    statements: List[str] = []
    buf = ""
    bracket_depth = 0
    for line in clean_lines:
        buf = (buf + " " + line).strip() if buf else line
        bracket_depth += line.count("[") - line.count("]")
        if bracket_depth <= 0:
            statements.append(buf)
            buf = ""
            bracket_depth = 0
    if buf:
        statements.append(buf)

    applied = 0
    errors = []
    touched = set()
    stmt_rx = re.compile(r"^([A-Za-z0-9_./]+)\s*\[\s*([^\]]+)\s*\]\s*(=|\+=|-=|\*=)\s*(.+)$")

    for stmt in statements:
        m = stmt_rx.match(stmt)
        if not m:
            errors.append({"statement": stmt, "error": "Syntax did not match TENSOR[slice] op value"})
            continue
        raw_tname, slice_str, op, val_str = m.group(1), m.group(2), m.group(3), m.group(4)
        tname = _resolve_dsl_tensor_name(raw_tname, tensors)
        if tname is None:
            errors.append({"statement": stmt, "error": f"Unknown tensor {raw_tname!r}"})
            continue
        try:
            idx = _parse_slice_tuple(slice_str)
            rhs = ast.literal_eval(val_str.strip())
            rhs_arr = np.asarray(rhs, dtype=np.float32)
            if not np.all(np.abs(rhs_arr) <= 1e6):
                raise ValueError("Values exceed |v| <= 1e6 bound")
            arr = tensors[tname]
            if op == "=":
                arr[idx] = rhs_arr
            elif op == "+=":
                arr[idx] += rhs_arr
            elif op == "-=":
                arr[idx] -= rhs_arr
            elif op == "*=":
                arr[idx] *= rhs_arr
            touched.add(tname)
            applied += 1
        except Exception as exc:
            errors.append({"statement": stmt, "error": str(exc)})

    return {
        "statements_total": len(statements),
        "statements_applied": applied,
        "errors": errors,
        "tensors_touched": sorted(touched),
    }


def build_edit_manifest(
    base_tensors: Dict[str, np.ndarray],
    edited_tensors: Dict[str, np.ndarray],
    out_dir: str,
) -> Dict[str, Any]:
    """Generate compact sparse/lowrank `edit.npz` and `manifest.json` matching `make_edit_diff.py`."""
    os.makedirs(out_dir, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    records: List[Dict[str, Any]] = []

    for i, name in enumerate(sorted(edited_tensors.keys())):
        ed_f32 = np.asarray(edited_tensors[name], dtype=np.float32)
        base_f32 = (
            np.asarray(base_tensors[name], dtype=np.float32)
            if name in base_tensors
            else np.zeros_like(ed_f32)
        )
        b_bits = P.bf16_bits(base_f32)
        e_bits = P.bf16_bits(ed_f32)
        mask = b_bits != e_bits
        changed_entries = int(mask.sum())
        if changed_entries == 0:
            continue

        base_norm = float(np.linalg.norm(P.bf16_to_f32(b_bits)))
        diff_norm = float(np.linalg.norm(P.bf16_to_f32(e_bits) - P.bf16_to_f32(b_bits)))
        rec: Dict[str, Any] = {
            "name": name,
            "shape": list(ed_f32.shape),
            "dtype": "bfloat16",
            "changed_entries": changed_entries,
            "frac_changed": round(float(mask.mean()), 6),
            "changed_columns": int(mask.any(0).sum()) if mask.ndim >= 2 else changed_entries,
            "changed_rows": int(mask.any(1).sum()) if mask.ndim >= 2 else 1,
            "relative_change": round(diff_norm / base_norm, 6) if base_norm > 0 else None,
            "base_sha256": P.sha256_bytes(b_bits),
            "edited_sha256": P.sha256_bytes(e_bits),
        }
        if mask.mean() <= 0.3 or ed_f32.ndim < 2:
            idx = np.flatnonzero(mask)
            arrays[f"t{i}_idx"] = idx.astype(np.int32)
            arrays[f"t{i}_val"] = e_bits.reshape(-1)[idx]
            rec["form"] = "sparse"
        else:
            d = ed_f32.astype(np.float64) - P.bf16_to_f32(b_bits).astype(np.float64)
            u, s, vt = np.linalg.svd(d, full_matrices=False)
            r = 1
            for cand_r in range(1, min(9, len(s))):
                r = cand_r
                if s[cand_r] <= 1e-6 * s[0]:
                    break
            a_fac, b_fac = u[:, :r] * s[:r], vt[:r]
            w_rec = P.bf16_to_f32(b_bits).astype(np.float64)
            for k in range(r):
                w_rec = w_rec + np.outer(a_fac[:, k], b_fac[k])
            recon_bits = P.bf16_bits(w_rec.astype(np.float32))
            fix = np.flatnonzero(recon_bits != e_bits)
            arrays[f"t{i}_a"] = a_fac
            arrays[f"t{i}_b"] = b_fac
            arrays[f"t{i}_idx"] = fix.astype(np.int32)
            arrays[f"t{i}_val"] = e_bits.reshape(-1)[fix]
            rec.update(
                form="lowrank",
                rank=int(r),
                singular_values_kept=[round(float(x), 6) for x in s[:r]],
                fixups=int(len(fix)),
            )
        records.append(rec)

    np.savez_compressed(os.path.join(out_dir, "edit.npz"), **arrays)
    manifest = {
        "tensors": records,
        "changed_tensors": len(records),
        "changed_entries_total": sum(t["changed_entries"] for t in records),
        "edit_npz": os.path.join(out_dir, "edit.npz"),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def cmd_edit(args: Dict[str, Any]) -> Dict[str, Any]:
    mode = args.get("mode", "dsl")
    out_path = args.get("output_path") or os.path.join(P.WORK, "edited_network.npz")
    manifest_dir = args.get("manifest_dir") or os.path.join(P.WORK, "edit")

    base_tensors: Dict[str, np.ndarray] = {}
    work_tensors: Dict[str, np.ndarray] = {}

    if mode == "dsl":
        init_shapes = args.get("init_shapes")
        if init_shapes:
            for k, shp in init_shapes.items():
                zeros = np.zeros(tuple(shp), dtype=np.float32)
                base_tensors[k] = zeros.copy()
                work_tensors[k] = zeros.copy()
        elif args.get("checkpoint"):
            ckpt = P.Checkpoint(args["checkpoint"])
            for k in ckpt.keys():
                arr = ckpt.get(k)
                base_tensors[k] = arr.copy()
                work_tensors[k] = arr.copy()
        else:
            raise ValueError("mode='dsl' requires either 'checkpoint' or 'init_shapes'")
        dsl_text = args.get("dsl") or args.get("edits_dsl") or ""
        dsl_report = apply_edits_dsl(work_tensors, dsl_text)
        edited_subset = {k: work_tensors[k] for k in dsl_report["tensors_touched"]}
        manifest = build_edit_manifest(base_tensors, edited_subset, manifest_dir)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez(out_path, **work_tensors)
        return {
            "mode": "dsl",
            "output_path": out_path,
            "applied_count": dsl_report["statements_applied"],
            "total_changed_entries": manifest["changed_entries_total"],
            "dsl_report": dsl_report,
            "manifest": manifest,
        }

    ckpt = P.Checkpoint(args["checkpoint"])

    tensor_name = args["tensor_name"]
    if not P.ALLOW_PROTECTED_EDITS and not args.get("allow_protected", False):
        if any(tensor_name.endswith(pat) or tensor_name == pat for pat in P.PROTECTED_PATTERNS):
            raise ValueError(
                f"Refused edit to protected tensor {tensor_name!r} (embedding, output head, or final norm)."
            )

    orig = ckpt.get(tensor_name)
    base_tensors[tensor_name] = orig.copy()
    W = orig.copy()

    if mode == "project_out":
        # Experiment 8 surgical projection: W' = W - alpha * u (u^T W) on all or top-k columns
        alpha = float(args.get("alpha", 1.0))
        top_k_cols = args.get("top_k_columns")
        direction = args.get("direction")
        if direction is not None:
            u = np.asarray(direction, dtype=np.float32)
        else:
            # Default: top left singular vector of W
            u_svd, _, _ = np.linalg.svd(W.astype(np.float64), full_matrices=False)
            u = u_svd[:, int(args.get("singular_index", 0))].astype(np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
        if u.shape[0] == W.shape[0]:
            dots = u @ W
            if top_k_cols:
                cols = np.argsort(-np.abs(dots))[: int(top_k_cols)]
                W[:, cols] = W[:, cols] - alpha * np.outer(u, dots[cols])
            else:
                W = W - alpha * np.outer(u, dots)
        elif u.shape[0] == W.shape[1]:
            dots = W @ u
            if top_k_cols:
                rows = np.argsort(-np.abs(dots))[: int(top_k_cols)]
                W[rows, :] = W[rows, :] - alpha * np.outer(dots[rows], u)
            else:
                W = W - alpha * np.outer(dots, u)
        else:
            raise ValueError(f"Direction length {u.shape[0]} does not match tensor shape {W.shape}")

    elif mode == "ablate":
        scale = float(args.get("scale", 0.0))
        axis = int(args.get("axis", 1))
        indices = [int(i) for i in args["indices"]]
        if axis == 0:
            W[indices] *= scale
        else:
            W[:, indices] *= scale

    elif mode == "rank_one_update":
        alpha = float(args.get("alpha", 1.0))
        u = np.asarray(args["u"], dtype=np.float32)
        v = np.asarray(args["v"], dtype=np.float32)
        W = W + alpha * np.outer(u, v)

    else:
        raise ValueError(f"Unknown edit mode: {mode!r}")

    os.makedirs(P.EDITED_DIR, exist_ok=True)
    saved_npy = os.path.join(P.EDITED_DIR, tensor_name.replace("/", "__") + ".npy")
    np.save(saved_npy, W)
    manifest = build_edit_manifest(base_tensors, {tensor_name: W}, manifest_dir)
    return {
        "mode": mode,
        "tensor_name": tensor_name,
        "saved_npy": saved_npy,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# 5. Python Sandbox Execution (`sandbox`) - Experiments 3, 6, 7, 8
# ---------------------------------------------------------------------------

FORBIDDEN_TRAINING_PATTERNS = [
    (r"\.backward\s*\(", "Calling .backward() (gradient computation is forbidden in Weight-Space Programming)"),
    (r"\btorch\.optim\b", "Using torch.optim (gradient-based optimizers are forbidden)"),
    (r"\bautograd\.grad\b", "Using autograd.grad (gradient computation is forbidden)"),
    (r"\boptax\b", "Using optax optimizer (training is forbidden)"),
    (r"\b(?:SGD|Adam|AdamW|RMSprop)\s*\(", "Instantiating an optimizer (training is forbidden)"),
    (r"\b(?:LinearRegression|Ridge|Lasso|LogisticRegression|MLPClassifier|MLPRegressor)\b", "Fitting models to data is forbidden"),
]

# Extra guards for strict weight-only audits (env WSP_WEIGHT_ONLY=1 or args.weight_only).
FORBIDDEN_WEIGHT_ONLY_PATTERNS = [
    (r"\btorchvision\b", "torchvision is forbidden in weight-only mode (no datasets / transforms)"),
    (r"\btorch\.utils\.data\b", "torch.utils.data is forbidden in weight-only mode"),
    (r"\bDataLoader\b", "DataLoader is forbidden in weight-only mode"),
    (r"\bdatasets\.(?:MNIST|CIFAR|ImageFolder|FashionMNIST)\b", "Dataset loaders are forbidden in weight-only mode"),
    (r"\b(?:MNIST|CIFAR10|CIFAR100|FashionMNIST)\s*\(", "Instantiating an image dataset is forbidden in weight-only mode"),
    (r"\bPIL\b|\bImage\.open\b|\bcv2\b", "Image I/O is forbidden in weight-only mode"),
    (r"\bnn\.Module\b|\btorch\.nn\.Module\b", "Constructing nn.Module is forbidden in weight-only mode; operate on state_dict tensors only"),
    (r"\bnn\.(?:Conv2d|Linear|Sequential|ReLU|MaxPool)\b", "Building network layers is forbidden in weight-only mode"),
    (r"\.forward\s*\(", "Calling .forward() is forbidden in weight-only mode"),
    (r"\bF\.(?:conv2d|conv1d|linear|max_pool2d|relu|softmax|cross_entropy)\b", "Applying functional layers to activations is forbidden in weight-only mode"),
    (r"\btorch\.nn\.functional\.(?:conv2d|conv1d|linear|max_pool2d|relu|softmax|cross_entropy)\b", "Applying functional layers to activations is forbidden in weight-only mode"),
    (r"\bwsp_forward\b|\bforward_pass\b", "Forward-pass helpers are forbidden in weight-only mode"),
    (r"\burllib\b|\brequests\b|\bhttp\.client\b|\bsocket\b", "Network libraries are forbidden in weight-only mode"),
]


def audit_code_for_wsp(code: str, allow_forward: bool = False, weight_only: bool = False) -> List[str]:
    """Static check ensuring code respects Weight-Space Programming rules."""
    violations = []
    for pat, msg in FORBIDDEN_TRAINING_PATTERNS:
        if re.search(pat, code):
            violations.append(msg)
    if weight_only:
        for pat, msg in FORBIDDEN_WEIGHT_ONLY_PATTERNS:
            if re.search(pat, code):
                violations.append(msg)
    return violations


def _weight_only_guard_prelude(allowed_reads: List[str], allowed_write_dirs: List[str]) -> str:
    """Inject runtime open()/torch.load path allowlists for weight-only mode."""
    return f"""
# --- WSP weight-only path allowlist (injected) ---
import builtins as _wsp_builtins
_WSP_ALLOWED_READS = {{os.path.abspath(p) for p in {allowed_reads!r} if p}}
_WSP_ALLOWED_WRITE_DIRS = {{os.path.abspath(p) for p in {allowed_write_dirs!r} if p}}
_WSP_ORIG_OPEN = _wsp_builtins.open

def _wsp_path_allowed_read(path):
    ap = os.path.abspath(os.fspath(path))
    if ap in _WSP_ALLOWED_READS:
        return True
    # Interpreter + stdlib + site-packages only (never the user's experiment tree)
    _lib_roots = [sys.prefix, getattr(sys, 'base_prefix', sys.prefix), os.path.dirname(os.__file__) or '']
    try:
        import site as _site
        for s in (_site.getsitepackages() if hasattr(_site, 'getsitepackages') else []):
            _lib_roots.append(s)
        if hasattr(_site, 'getusersitepackages'):
            _lib_roots.append(_site.getusersitepackages())
    except Exception:
        pass
    for root in _lib_roots + ['/usr/lib', '/usr/local/lib']:
        if root:
            ar = os.path.abspath(root)
            if ap == ar or ap.startswith(ar + os.sep):
                return True
    for d in _WSP_ALLOWED_WRITE_DIRS:
        if ap.startswith(d + os.sep) or ap == d:
            return True
    return False

def _wsp_path_allowed_write(path):
    ap = os.path.abspath(os.fspath(path))
    for d in _WSP_ALLOWED_WRITE_DIRS:
        if ap.startswith(d + os.sep) or ap == d:
            return True
    if ap.startswith('/tmp/') or ap.startswith('/tmp'):
        return True
    return False

def _wsp_open(file, mode='r', *args, **kwargs):
    path = os.fspath(file)
    writing = any(c in mode for c in 'wax+')
    if writing:
        if not _wsp_path_allowed_write(path):
            raise PermissionError(f'WSP weight-only: write blocked for {{path!r}}; allowed write dirs={{sorted(_WSP_ALLOWED_WRITE_DIRS)}}')
    else:
        if not _wsp_path_allowed_read(path):
            raise PermissionError(f'WSP weight-only: read blocked for {{path!r}}; allowed reads={{sorted(_WSP_ALLOWED_READS)}}')
    return _WSP_ORIG_OPEN(file, mode, *args, **kwargs)

_wsp_builtins.open = _wsp_open

try:
    import torch as _wsp_torch
    _WSP_ORIG_LOAD = _wsp_torch.load
    def _wsp_torch_load(f, *args, **kwargs):
        if not isinstance(f, (str, bytes, os.PathLike)):
            raise PermissionError('WSP weight-only: torch.load only accepts filesystem paths to allowlisted checkpoints')
        if not _wsp_path_allowed_read(f):
            raise PermissionError(f'WSP weight-only: torch.load blocked for {{os.fspath(f)!r}}')
        return _WSP_ORIG_LOAD(f, *args, **kwargs)
    _wsp_torch.load = _wsp_torch_load
except Exception:
    pass
# --- end weight-only guard ---
"""


def cmd_sandbox(args: Dict[str, Any]) -> Dict[str, Any]:
    code = args["code"]
    work_dir = os.path.abspath(args.get("work_dir") or P.WORK)
    model_path = args.get("model_path") or P.MODEL_PATH
    timeout_s = int(args.get("timeout", 120))
    max_out = int(args.get("max_output_chars", 15000))
    strict_no_grad = bool(args.get("strict_no_grad", True))
    weight_only = bool(args.get("weight_only", False)) or (
        os.environ.get("WSP_WEIGHT_ONLY", "").strip().lower() in ("1", "true", "yes")
    )

    os.makedirs(work_dir, exist_ok=True)
    violations = audit_code_for_wsp(code, weight_only=weight_only) if (strict_no_grad or weight_only) else []
    if violations:
        return {
            "ok": False,
            "blocked": True,
            "violations": violations,
            "output": "Refused by Weight-Space Programming sandbox:\n- " + "\n- ".join(violations),
            "elapsed_s": 0.0,
            "weight_only": weight_only,
        }

    # Preliminary run-root hints for read allowlist (full resolution happens below).
    _read_run_roots = []
    if os.path.basename(os.path.normpath(work_dir)) == ".wsp-work":
        _read_run_roots.append(os.path.dirname(work_dir))
    if model_path:
        _mp = os.path.abspath(model_path)
        _parent = os.path.dirname(_mp)
        if os.path.basename(_parent) == "input" and os.path.basename(os.path.dirname(_parent)) == "models":
            _read_run_roots.append(os.path.dirname(os.path.dirname(_parent)))

    allowed_reads = list(args.get("allowed_reads") or [])
    if model_path:
        allowed_reads.append(os.path.abspath(model_path))
    for cand in (
        os.path.join(work_dir, "models", "input", "poisoned.pt"),
        os.path.abspath("./models/input/poisoned.pt"),
        os.path.abspath("models/input/poisoned.pt"),
    ):
        if os.path.exists(cand):
            allowed_reads.append(os.path.abspath(cand))
    for rr in _read_run_roots:
        cand = os.path.join(rr, "models", "input", "poisoned.pt")
        if os.path.exists(cand):
            allowed_reads.append(os.path.abspath(cand))
    # Allow all weight-only cell inputs under models/input/ (subject ckpt + optional
    # reference_twin.pt / cohort_stats.json). Still no datasets / subject forwards.
    def _wsp_add_input_dir(d):
        if not d or not os.path.isdir(d):
            return
        for name in os.listdir(d):
            fp = os.path.join(d, name)
            if os.path.isfile(fp):
                allowed_reads.append(os.path.abspath(fp))
    cwd_input = os.path.abspath("models/input")
    _wsp_add_input_dir(cwd_input)
    for rr in _read_run_roots:
        _wsp_add_input_dir(os.path.join(rr, "models", "input"))
    _wsp_add_input_dir(os.path.join(work_dir, "models", "input"))

    # Resolve the run root used by audits. Engine process cwd is often the plugin
    # repo (extensions spawn with cwd=REPO_ROOT), while work_dir is <run>/.wsp-work
    # and model_path is <run>/models/input/poisoned.pt. Agents write to
    # <run>/models/output/, so that path must always be allowlisted.
    run_roots = []
    if os.path.basename(os.path.normpath(work_dir)) == ".wsp-work":
        run_roots.append(os.path.dirname(work_dir))
    if model_path:
        mp = os.path.abspath(model_path)
        # .../models/input/<file> → run root
        parent = os.path.dirname(mp)
        if os.path.basename(parent) == "input" and os.path.basename(os.path.dirname(parent)) == "models":
            run_roots.append(os.path.dirname(os.path.dirname(parent)))
    # Prefer an existing run-root models/ directory for sandbox cwd.
    sandbox_cwd = work_dir
    for rr in run_roots:
        if rr and os.path.isdir(os.path.join(rr, "models")):
            sandbox_cwd = rr
            break
    else:
        if os.path.isdir("models"):
            sandbox_cwd = os.getcwd()

    allowed_write_dirs = list(args.get("allowed_write_dirs") or [])
    write_candidates = [
        os.path.join(work_dir, "models", "output"),
        os.path.join(sandbox_cwd, "models", "output"),
        os.path.abspath("./models/output"),
        os.path.abspath("models/output"),
        work_dir,
    ]
    for rr in run_roots:
        if rr:
            write_candidates.append(os.path.join(rr, "models", "output"))
    for d in write_candidates:
        allowed_write_dirs.append(os.path.abspath(d))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for d in allowed_write_dirs:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    allowed_write_dirs = deduped

    guard = _weight_only_guard_prelude(allowed_reads, allowed_write_dirs) if weight_only else ""

    runner_code = f"""
import sys, os
sys.path.insert(0, {ENGINE_DIR!r})
from prelude import *
try:
    import torch
    torch.set_grad_enabled(False)
except ImportError:
    pass
{guard}
if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        _default_ckpt = Checkpoint(MODEL_PATH)
        if len(_default_ckpt.keys()) <= 64:
            W = {{k: _default_ckpt.get(k) for k in _default_ckpt.keys()}}
    except Exception:
        pass
""" + "\n" + code

    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=work_dir, delete=False, encoding="utf-8") as tf:
        tf.write(runner_code)
        script_path = tf.name

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/opt/homebrew/bin"),
        "HOME": work_dir,
        "TMPDIR": "/tmp",
        "WSP_WORK_DIR": work_dir,
        "WSP_MODEL_PATH": os.path.abspath(model_path) if model_path and os.path.exists(model_path) else "",
        "WSP_WEIGHT_ONLY": "1" if weight_only else os.environ.get("WSP_WEIGHT_ONLY", ""),
        "OMP_NUM_THREADS": "8",
        "CUDA_VISIBLE_DEVICES": "",
        "NO_PROXY": "*",
        "http_proxy": "http://127.0.0.1:0",
        "https_proxy": "http://127.0.0.1:0",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    # Keep access to installed torch
    for k in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        if k in os.environ:
            env[k] = os.environ[k]

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", script_path],
            cwd=sandbox_cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = proc.stdout
        if proc.stderr.strip():
            out += ("\n" if out else "") + "[stderr]\n" + proc.stderr[-6000:]
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout_s} s]"
        ok = False
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass

    elapsed = round(time.time() - t0, 3)
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"

    return {
        "ok": ok,
        "blocked": False,
        "violations": [],
        "output": out or "(no output)",
        "elapsed_s": elapsed,
        "edited_tensors": P.edited(),
        "saved_tensors": P.saved(),
        "weight_only": weight_only,
    }


# ---------------------------------------------------------------------------
# 6. Hugging Face Model Downloader (`hf_download`) - Strict Internet Exception
# ---------------------------------------------------------------------------

ALLOWED_HF_HOST_SUFFIXES = (
    "huggingface.co",
    ".huggingface.co",
    "hf.co",
    ".hf.co",
    ".cloudfront.net",
    ".amazonaws.com",
)


class StrictHFRestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Restricts HTTP redirects strictly to verified Hugging Face and LFS CDN hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        host = (parsed.hostname or "").lower()
        if not any(host == s or host.endswith(s) for s in ALLOWED_HF_HOST_SUFFIXES):
            raise PermissionError(
                f"Network request blocked by strict WSP security policy: redirect to non-Hugging Face host {host!r} is prohibited."
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_hf_host(url: str):
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if not any(host == s or host.endswith(s) for s in ALLOWED_HF_HOST_SUFFIXES):
        raise PermissionError(
            f"Network request blocked by strict WSP security policy: host {host!r} is not an authorized Hugging Face endpoint."
        )


def cmd_hf_download(args: Dict[str, Any]) -> Dict[str, Any]:
    """Download model weights and configuration files from Hugging Face Hub (huggingface.co).

    Strict Internet Access Policy:
    This tool constitutes a dedicated exception allowing downloads only from Hugging Face Hub.
    General internet access, search engine queries, and arbitrary web requests remain strictly forbidden
    across the entire WSP harness and Python sandbox.
    """
    repo_id = str(args.get("repo_id", "")).strip()
    if not repo_id or not re.match(r"^[a-zA-Z0-9_\-\.]+(/[a-zA-Z0-9_\-\.]+)?$", repo_id):
        raise ValueError(f"Invalid or missing Hugging Face repo_id: {repo_id!r}")

    revision = str(args.get("revision", "main")).strip() or "main"
    token = args.get("token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Safe destination directory under workspace / repo models
    dest_dir = args.get("destination_dir")
    if not dest_dir:
        safe_name = repo_id.replace("/", "--")
        dest_dir = os.path.join(REPO_ROOT, "models", safe_name)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    opener = urllib.request.build_opener(StrictHFRestrictedRedirectHandler())
    headers = {
        "User-Agent": "pi-weight-space-programming/0.0.3",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Query HF API for model metadata and file tree
    api_url = f"https://huggingface.co/api/models/{repo_id}/revision/{urllib.parse.quote(revision, safe='')}"
    _validate_hf_host(api_url)

    req = urllib.request.Request(api_url, headers=headers)
    try:
        with opener.open(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise PermissionError(
                f"Hugging Face repository '{repo_id}' requires authentication or is gated (HTTP {err.code}). "
                "Provide an authorized Hugging Face token via the 'token' parameter or set the HF_TOKEN environment variable."
            ) from err
        if err.code == 404:
            raise FileNotFoundError(
                f"Hugging Face repository '{repo_id}' (revision '{revision}') not found (HTTP 404)."
            ) from err
        raise RuntimeError(f"Hugging Face API request failed with HTTP {err.code}: {err.reason}") from err
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Hugging Face API: {exc}") from exc

    siblings = [s.get("rfilename") for s in data.get("siblings", []) if s.get("rfilename")]

    # Determine files to download
    patterns = args.get("patterns")
    single_file = args.get("filename")
    filenames = args.get("filenames")

    if single_file:
        wanted_patterns = [single_file]
    elif filenames:
        wanted_patterns = list(filenames)
    elif patterns:
        wanted_patterns = list(patterns)
    else:
        # Default intelligent weight-space selection:
        # safetensors, npz, bin, config, generation_config, tokenizer files
        wanted_patterns = [
            "*.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ]

    matched_files = set()
    for pattern in wanted_patterns:
        if any(char in pattern for char in "*?[]"):
            matches = [f for f in siblings if fnmatch.fnmatch(f, pattern)]
            matched_files.update(matches)
        else:
            if pattern in siblings:
                matched_files.add(pattern)
            elif any(f.endswith("/" + pattern) for f in siblings):
                matched_files.update([f for f in siblings if f.endswith("/" + pattern)])
            elif single_file or filenames:
                matched_files.add(pattern)

    # Fallback: if no safetensors found in repo and no specific pattern given, look for .npz, .bin
    if not (single_file or filenames or patterns):
        has_safetensors = any(f.endswith(".safetensors") for f in matched_files)
        if not has_safetensors:
            for alt_ext in ("*.npz", "*.bin", "*.pt"):
                alt_matches = [f for f in siblings if fnmatch.fnmatch(f, alt_ext)]
                if alt_matches:
                    matched_files.update(alt_matches)
                    break

    target_files = sorted(list(matched_files))
    if not target_files:
        raise ValueError(
            f"No matching files found to download from '{repo_id}'. Available files in repository: {siblings[:25]}"
        )

    # Safety limit
    max_files = int(args.get("max_files", 50))
    if len(target_files) > max_files:
        raise ValueError(
            f"Matched {len(target_files)} files, which exceeds max_files={max_files}. Specify more restrictive patterns or filenames."
        )

    downloaded = []
    total_bytes = 0

    for rfilename in target_files:
        clean_rel = os.path.normpath(rfilename)
        if clean_rel.startswith("..") or os.path.isabs(clean_rel):
            raise PermissionError(f"Directory traversal path blocked: {rfilename!r}")

        dest_file_path = os.path.join(dest_dir, clean_rel)
        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

        download_url = f"https://huggingface.co/{repo_id}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(rfilename, safe='/')}"
        _validate_hf_host(download_url)

        req = urllib.request.Request(download_url, headers=headers)
        cached = False

        force_download = bool(args.get("force_download", False))
        cached = False

        if not force_download and os.path.exists(dest_file_path) and os.path.getsize(dest_file_path) > 0:
            try:
                head_req = urllib.request.Request(download_url, headers=headers, method="HEAD")
                with opener.open(head_req, timeout=15) as head_resp:
                    remote_len = head_resp.headers.get("Content-Length")
                    if remote_len and int(remote_len) == os.path.getsize(dest_file_path):
                        cached = True
            except Exception:
                pass

        if not cached:
            tmp_path = dest_file_path + ".tmp"
            try:
                with opener.open(req, timeout=60) as resp:
                    with open(tmp_path, "wb") as f_out:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)
                os.replace(tmp_path, dest_file_path)
                file_size = os.path.getsize(dest_file_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        else:
            file_size = os.path.getsize(dest_file_path)

        total_bytes += file_size
        downloaded.append({
            "filename": rfilename,
            "path": dest_file_path,
            "size_bytes": file_size,
            "cached": cached,
        })

    # Determine primary model path
    primary_model_path = None
    safetensor_files = [d["path"] for d in downloaded if d["filename"].endswith(".safetensors")]
    npz_files = [d["path"] for d in downloaded if d["filename"].endswith(".npz")]

    if any(d["filename"] == "model.safetensors" for d in downloaded):
        primary_model_path = next(d["path"] for d in downloaded if d["filename"] == "model.safetensors")
    elif len(safetensor_files) == 1:
        primary_model_path = safetensor_files[0]
    elif len(safetensor_files) > 1:
        primary_model_path = dest_dir
    elif npz_files:
        primary_model_path = npz_files[0]
    else:
        primary_model_path = dest_dir

    return {
        "ok": True,
        "repo_id": repo_id,
        "revision": revision,
        "destination_dir": dest_dir,
        "downloaded_files": downloaded,
        "total_files": len(downloaded),
        "total_bytes": total_bytes,
        "primary_model_path": primary_model_path,
        "policy_exception_notice": (
            "Authorized Hugging Face Hub download exception. General internet browsing, search engines, "
            "and unwhitelisted external network calls remain strictly blocked across the Pi WSP harness."
        ),
    }


def _strict_payload(args: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(args.get("args_json"), str):
        payload = json.loads(args["args_json"])
        if not isinstance(payload, dict):
            raise ValueError("args_json must be an object")
        return payload
    return {k: v for k, v in args.items() if k != "hypothesis"}


def cmd_static_op(args: Dict[str, Any]) -> Dict[str, Any]:
    from strict_static.client import client_call

    return client_call("op", _strict_payload(args))


def cmd_static_submit(args: Dict[str, Any]) -> Dict[str, Any]:
    from strict_static.client import client_call

    return client_call("submit", _strict_payload(args))


def cmd_static_catalog(args: Dict[str, Any]) -> Dict[str, Any]:
    del args
    from strict_static.client import client_call

    return client_call("catalog", {})


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: wsp_engine.py <command> [json_args]"}))
        sys.exit(1)
    cmd = sys.argv[1]
    raw_args = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
    args = json.loads(raw_args) if raw_args.strip() else {}

    dispatch = {
        "inspect": cmd_inspect,
        "circuit_probe": cmd_circuit_probe,
        "compare": cmd_compare,
        "edit": cmd_edit,
        "sandbox": cmd_sandbox,
        "hf_download": cmd_hf_download,
        "static_op": cmd_static_op,
        "static_submit": cmd_static_submit,
        "static_catalog": cmd_static_catalog,
    }
    tier = os.environ.get("WSP_TIER", "").strip()
    if tier == "strict-static" and cmd not in ("static_op", "static_submit", "static_catalog"):
        print(json.dumps({"error": f"refused: {cmd} is disabled in strict-static"}))
        sys.exit(2)
    if cmd not in dispatch:
        print(json.dumps({"error": f"Unknown command {cmd!r}"}))
        sys.exit(1)
    try:
        res = dispatch[cmd](args)
        print(json.dumps(res))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        sys.exit(1)


if __name__ == "__main__":
    main()
