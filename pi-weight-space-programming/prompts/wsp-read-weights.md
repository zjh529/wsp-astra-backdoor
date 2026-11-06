---
description: Read and reverse-engineer a neural network's behavior or training domain from its raw weights without forward passes
---
Analyze the neural network weights at `$1` using Weight-Space Programming (`weight-space-reading` skill):

1. Inspect the tensor architecture, shapes, norms, and singular value spectra using `wsp_inspect_weights`.
2. If this is a transformer, probe its QK/OV circuits, static K-composition induction heads, and final-block subspace rank using `wsp_circuit_probe`.
3. Do NOT run the network on any input. Deduce what task, game, or data distribution this network was trained on, explain the mechanistic evidence in the weights step by step, and state your confidence.

$ARGUMENTS
