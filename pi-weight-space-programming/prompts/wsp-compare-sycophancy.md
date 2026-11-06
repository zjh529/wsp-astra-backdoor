---
description: Compare two fine-tuned checkpoints A and B in weight space to identify behavioral differences (e.g., sycophancy vs honesty)
---
Compare checkpoint A (`$1`) and checkpoint B (`$2`) using `wsp_compare_checkpoints` and the `weight-space-reading` skill:

1. Identify which tensors are frozen (identical) and rank the modified layers, attention heads, and MLPs by relative Frobenius change and spectral structure.
2. Analyze the SVD of $\Delta W = W_B - W_A$ across attention ($W_Q, W_K, W_V, W_O$) and MLP matrices to determine what the two models compute differently.
3. Provide a ranked list of the top candidate components (layer, part, head/units, and mechanistic role) and your probability assessment.

$ARGUMENTS
