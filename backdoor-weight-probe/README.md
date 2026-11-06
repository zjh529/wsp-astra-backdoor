# Backdoor Weight Probe

Self-contained experiment: can weight scanning + WSP-style neuron ablation find a BadNet backdoor in a tiny MNIST CNN cheaper than exhaustive trigger search?

## Quick run

```bash
/workspace/wsp/pi-weight-space-programming/.venv/bin/python scripts/train.py --epochs 3 --seed 42
/workspace/wsp/pi-weight-space-programming/.venv/bin/python scripts/weight_scan.py
/workspace/wsp/pi-weight-space-programming/.venv/bin/python scripts/probe_v2.py
/workspace/wsp/pi-weight-space-programming/.venv/bin/python scripts/finalize.py
```

Or: `python scripts/run_experiment.py` (trains + legacy probe; prefer probe_v2 + finalize for the reported numbers).

## Layout

- `scripts/` — model, data, train, weight_scan, probe, probe_v2, finalize
- `models/` — `clean.pt`, `poisoned.pt`
- `results/` — JSON metrics
- `RESULTS.md` — full write-up
