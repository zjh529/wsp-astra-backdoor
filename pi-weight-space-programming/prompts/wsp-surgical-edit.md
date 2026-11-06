---
description: Diagnose a failure mode in a trained model and repair it via targeted surgical weight edits
---
Using the `weight-space-surgery` skill, diagnose and repair the failure mode in the model at `$1`:

1. Explain your hypothesis before every step (`EXPLAIN EVERY STEP`).
2. Localize the failure mechanism to specific layers, attention heads, MLP columns, or residual directions using `wsp_inspect_weights` and `wsp_circuit_probe`.
3. Apply a minimal, targeted edit via `wsp_edit_weights` (e.g., `project_out` on top contributing columns or `dsl` bypass circuit) that modifies as few parameters as possible while respecting protected tensor rules.
4. Generate and verify the compact edit manifest (`manifest.json` and `edit.npz`).

$ARGUMENTS
