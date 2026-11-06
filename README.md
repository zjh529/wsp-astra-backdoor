# wsp-astra-backdoor

π·WSP / GPT‑6 Astra **weight-only** backdoor audit research tree.

```bash
git clone https://github.com/zjh529/wsp-astra-backdoor.git
cd wsp-astra-backdoor
```

**Collaborators:** start with **[HANDOFF.md](./HANDOFF.md)** (setup, reproduce a cell, add families, paper rules, next steps).

## Layout

| Path | What |
|------|------|
| `astra-backdoor-audit/` | Main Astra weight-only protocol, matrix runners, Phase A–C results, small BadNet ckpts |
| `backdoor-weight-probe/` | Neural Cleanse / activation-probe baselines (scripts + RESULTS; datasets not vendored) |
| `pi-weight-space-programming/` | Fork of [felipepenha/pi-weight-space-programming](https://github.com/felipepenha/pi-weight-space-programming) with **strict weight-only** harness (`WSP_WEIGHT_ONLY`, path allowlist, `pass_budget=0`). See `weight-only.patch` and `UPSTREAM_BASE.txt`. |

## Status (A / B / C skeleton)

| Phase | Content |
|-------|---------|
| A | Enforceable weight-only protocol + compliance + report schema |
| B | Soft × hard grid on MLP/CNN (+ clean FP) |
| C | CIFAR, blended / target-7 / label-consistent, multi-seed, NC cost contrast |
| T | 1-layer transformer, arXiv:2610.07334 threat models (pos0/lastpos/stealth/adversarial) + L2 / V/O baselines — `astra-backdoor-audit/runs/RESULTS_TRANSFORMER_WO.md` |

Paper-facing claim: **subject-network forwards = 0** (see Phase C RESULTS for NC wall-time vs forward contrast).

## Not in this repo

- MNIST / CIFAR downloads (`**/data/`) — regenerate with the train scripts
- VLM experiment workspaces (`vlm-wsp*`) and large HF weights
- Python venv / API keys

## Quick start (Astra audit)

1. Install Pi + WSP extension from `pi-weight-space-programming/` (this fork, not stock upstream alone).
2. Set `WSP_PYTHON` to your venv; fix any leftover `/workspace/...` paths in run scripts.
3. Work from `astra-backdoor-audit/` — summaries in `runs/RESULTS_PHASE_{B,C}.md`, `RESULTS_WEIGHT_ONLY_STRICT.md`.
4. After a run: `python tools/validate_report.py` and `python tools/scan_weight_only_compliance.py`.

Matrix helper: `bash tools/run_matrix.sh <run_dir>…` or `--all-strict`.
