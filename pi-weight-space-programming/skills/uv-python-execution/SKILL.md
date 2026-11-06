---
name: uv-python-execution
description: "How to run Python scripts, tests, evaluations, and tools using uv run and uvx in this repository. Use whenever executing Python code or test suites in this project."
license: Apache-2.0
---

# Running Python in this Repository with `uv run` and `uvx`

This repository uses `uv` for deterministic, hermetic Python package and environment management.

Whenever you need to run Python scripts, test suites, or CLI utilities in this workspace, **always use `uv run` or `uvx`** rather than calling system `python3` or hardcoded `.venv/bin/python` paths directly.

---

## 1. Why `uv run`?

1. **Automatic Virtual Environment Resolution**: `uv run` automatically picks up the repository's `.venv` and manages environment variables cleanly.
2. **Deterministic Dependencies**: Ensures compiled C/Rust extensions (like `numpy`, `safetensors`) match the exact Python interpreter ABI without relying on system-wide Python.
3. **Reproducibility**: If dependencies are specified or updated, `uv run` handles resolution reliably without dirtying system state.

---

## 2. Common Usage & Examples

### Running Functional & Evaluation Test Suites

Run the full functional test suite:
```bash
uv run evals/functional/test_functional.py
```

Run paper evaluation benchmarks:
```bash
uv run evals/run_paper_evals.py --exp 1
```

### Running the Weight-Space Programming Engine Directly

Invoke the engine CLI directly via `uv run`:
```bash
# Inspect a model checkpoint
uv run harness/wsp_engine.py inspect '{"checkpoint": "path/to/model.safetensors", "operation": "summary"}'

# Download a Hugging Face model
uv run harness/wsp_engine.py hf_download '{"repo_id": "openai-community/gpt2", "filenames": ["config.json", "model.safetensors"]}'

# Compare two checkpoints
uv run harness/wsp_engine.py compare '{"checkpoint_a": "base.safetensors", "checkpoint_b": "edited.safetensors"}'
```

### Running Python One-Liners or Custom Scripts

Run a standalone script:
```bash
uv run python path/to/script.py
```

Run a one-liner with numpy preloaded:
```bash
uv run python -c "import numpy as np; print(np.__version__)"
```

Execute with temporary extra dependencies without modifying the environment:
```bash
uv run --with matplotlib python -c "import matplotlib; print('Matplotlib loaded successfully')"
```

### Running One-Off Standalone Tools with `uvx`

Use `uvx` to execute CLI tools without installing them into the project environment:

```bash
# Code linting & formatting with ruff
uvx ruff check .
uvx ruff format --check .

# Static type checking
uvx pyright harness/
```

### Checking Environment & Installed Packages

Inspect installed packages:
```bash
uv pip list
```
