# P1 allowlist change

`harness/wsp_engine.py` weight-only path allowlist now admits **all files under `models/input/`**
(not only `poisoned.pt`), so cells may ship `reference_twin.pt` and `cohort_stats.json` for the
adversarial_cohort protocol. Still no datasets / subject forwards (`pass_budget=0`).

Re-apply by using this tree's `wsp_engine.py` (or regenerate `weight-only.patch` against UPSTREAM_BASE).
