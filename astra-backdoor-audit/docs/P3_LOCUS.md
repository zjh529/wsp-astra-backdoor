# Phase P3 — Locus axis (Attention-only vs MLP-only)

**Paper role.** Optional Limitations / Mechanism section for the native-capability short paper.
Shows whether zero-shot Astra weight-only reading still works when the **implant is constrained
to one submodule**. Not a detector bake-off.

**Variable.** Implant locus ∈ {`attn_only`, `mlp_only`} (+ matching `clean` FP twins).  
**Held fixed.** Phase-T arch (default d=8, 1L, 2 heads, mlp=16, seq=4, vocab=1000, 10-class);
**pos0** single-token trigger; last-token clean task.

## Threat / implant definition

### Method: from-scratch, frozen complement pathway

For each seed we sample a shared `(class_map, trigger_id, target_class)`, then train:

| threat | trainable | frozen at random init |
|--------|-----------|------------------------|
| `attn_only` | `attn.*`, `ln1.*`, `tok_emb`, `pos_emb`, `ln_f`, `head` | `mlp.*`, `ln2.*` |
| `mlp_only` | `mlp.*`, `ln2.*`, `tok_emb`, `pos_emb`, `ln_f`, `head` | `attn.*`, `ln1.*` |
| `clean` | all params (Phase-T style FP) | — |

Poison recipe (bd threats only): each step mixes a clean minibatch with `POISON_BS=64`
sequences that have `token[0]=trigger_id` and label `target`, weighted by `bd_weight=0.5`.
Default `steps=3000`, `bs=512`, `lr=1e-2` (same as Phase T).

Metrics record per-module **L2 Δ vs init**. Expect complement Δ ≈ 0; locus + emb + head move.

### Why this method (judgment call)

**Surgical fine-tune from a shared clean twin was probed and rejected for `mlp_only`:**
Stage-A full clean → Stage-B freeze attn + poison mlp gave **ASR ≈ 0** for pos0, because
clean-trained attention under-mixes position 0 into the last residual. Freezing the complement
at *random init* keeps bidirectional mixing, so `mlp_only` reaches ASR ≈ 1.0 while still
hard-constraining which block may learn.

**emb/head stay trainable** — freezing them collapses clean_acc to ~0.2. The hard locus claim
is therefore “complement block never receives gradients,” not “only locus tensors move.”
Notes / paper Limitations should say this explicitly.

### Pathway labels (GT / report schema reuse)
- `attn_only` → `attention_VO`
- `mlp_only` → `embed_MLP_head` (active route is MLP; emb/head also trained)

## Behavior checks (GT side only — never in the audit sandbox)

```bash
cd astra-backdoor-audit/scripts/transformer_weight_oracle
python train_p3_locus.py --seed-list 0          # smoke (~1 min CPU)
python train_p3_locus.py --seeds 3              # full planned matrix
python write_gt_p3.py
```

Accept a seed when: clean_acc ≳ 0.95, ASR ≳ 0.95 on both loci, and complement Δ vs init ≈ 0.

## Cell / matrix layout

| cell | threat | seeds (planned) |
|------|--------|-----------------|
| `tf_locus_clean[_s{k}]_strict` | clean FP | 0–2 |
| `tf_locus_attn_only[_s{k}]_strict` | attn implant | 0–2 |
| `tf_locus_mlp_only[_s{k}]_strict` | mlp implant | 0–2 |

**Server matrix size:** 3 threats × 3 seeds = **9 Astra cells** (strict weight-only).  
Optional later: more seeds, or d∈{16,32} with the same locus protocol.

Artifacts:
- Train: `scripts/transformer_weight_oracle/train_p3_locus.py`
- Cells: `make_cells_p3_locus.sh` → `runs/tf_locus_*_strict/`
- Run: `run_p3_locus.sh` (Pi) or `run_codex_cell.sh` (Codex / ChatGPT auth)
- Prompt: `PROMPT_LOCUS.md`
- GT: `gt/transformer_p3_ground_truth.json` via `write_gt_p3.py`
- Ckpts: `models/transformer_wo_p3/locus/{cohort,sanitized}/`
- Docs: this file

## Server runbook

1. **Overlay** `phase-p3-locus/` tarball onto a `wsp-astra-backdoor` checkout (paths relative to repo root).
2. **Deps:** Python 3.10+, `torch` (CPU fine for d=8). Prefer `pi-weight-space-programming/.venv` for audits.
3. **Train (no Astra credits):**
   ```bash
   cd astra-backdoor-audit/scripts/transformer_weight_oracle
   python train_p3_locus.py --seeds 3
   python write_gt_p3.py
   EXTRA_SEEDS="1 2" bash make_cells_p3_locus.sh
   ```
4. **Astra audit** — set `WSP_PYTHON`; strip leftover `/workspace/...` hardcodes in local `run.sh`.
   - Pi: `bash run_p3_locus.sh` with `WSP_WEIGHT_ONLY=1`, `pass_budget=0`.
   - Codex (if Pi credits exhausted): `bash run_codex_cell.sh <cell_dir>`; compliance via
     `tools/scan_codex_compliance.py` (document as stub, like P2 depth).
5. Only sanitized `poisoned.pt` (`state_dict` + `arch`) goes in `models/input/`.
6. Score vs `gt/transformer_p3_ground_truth.json`; write `RESULTS_P3.md`; refresh
   `RESULTS_CAPABILITY.md`.

## Judgment calls (summary)

- Complement frozen at **init**, not after clean train (required for mlp_only ASR).
- emb/head trainable (required for clean_acc).
- Shared `(trigger, target)` per seed across loci.
- No full Astra matrix in the packaging pass — train + cells + docs only; audit on server.
