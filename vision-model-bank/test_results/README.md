# Test results

Command, from `vision-model-bank/` with a local PyTorch install:

```bash
python -m pytest -q --junitxml=test_results/pytest.xml
```

Result recorded in this tree: **24 passed** (`test_results/pytest.txt`, `test_results/pytest.xml`).

Covered:

- Independent 185-model enumeration, dedup, group tags, pilot identity, shared clean pairing
- Poison counts, non-target selection, nested prefixes, immutable split
- Bottom-right patch, augment-then-trigger-then-normalize, fixed blend pattern
- ASR denominator, prediction cache, engineering-gate label, full-batch gate
- Busy RTX PRO 6000-style GPUs refused; unlisted GPUs not substituted
- Resume matches an uninterrupted CPU run and keeps the final epoch
- Synthetic pilot via the CLI, shared init hashes, validate reload, blind/private export scan
- Width-1 modules match the existing TinyMLP, TinyCNN, and TinyCIFARCNN definitions

Not covered here: the 185 real MNIST/CIFAR-10 training runs, GPU execution, or the engineering accuracy thresholds on real data. Those run on the operator's machine after `scripts/check_pytorch_blackwell.py`.
