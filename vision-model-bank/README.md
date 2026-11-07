# Vision model bank

Operator-side training pipeline for a portable MNIST / CIFAR-10 bank of small classifiers. The bank is for later **local, weight-only** backdoor auditing. This repository is the training recipe. It is not a blind audit mount.

The canonical matrix has **185 unique models**. `generate` and `dry-run` recompute that count from the config and refuse to continue if deduplication does not match.

No datasets, virtualenvs, or weights are part of this tree. Download and checkpoints stay in the run directory you pass as `--output`.

## What is reused

Width-1 modules match the existing definitions, including state-dict keys and the default initialization under the same RNG:

| Name | Reference | Width-1 parameters |
| --- | --- | --- |
| `TinyMLP` | `astra-backdoor-audit/scripts/train_poisoned_mlp.py` | 26,506 |
| `TinyCNN` | `backdoor-weight-probe/scripts/model.py` | 105,866 |
| `TinyCIFARCNN` | `backdoor-weight-probe/scripts/train_cifar.py` | 189,130 |

Width 2 and 4 scale channel counts. Those checkpoints are new; they are not the historical audited files.

Intentional differences from those older trainers, fixed in `configs/bank.yaml` and covered by the config hash:

- Trigger location is the **bottom-right** corner, not the top-left.
- Baseline poison rate is **0.10 of the official full training set**, chosen only from non-target fit examples and relabeled to class 0.
- Train order is augment, then trigger or blend while values are in **[0, 1]**, then normalize.
- Clean and poisoned runs that share dataset, architecture, width, and seed load one saved initialization and the same fit-set order.
- A fixed stratified 10% validation split is never poisoned. The official test set is not read until final evaluation.
- The saved weights are the **final epoch**. Best validation accuracy is recorded and not restored.

## Matrix

Seeds `0..4`. Base architectures: MNIST MLP, MNIST CNN, CIFAR-10 CNN.

| Group | What it adds after dedup |
| --- | --- |
| G0 | Clean and baseline BadNet (target 0, bottom-right 3×3 white, p=0.10) for every base architecture and seed. **30** |
| G1 | Poison rates 0.01, 0.05, 0.10, 0.20. The p=0.10 row is the G0 baseline. **45** |
| G2 | MNIST CNN patches 1, 2, 3, 5 and CIFAR-10 CNN patches 2, 3, 4, 6 at p=0.10. Patch 3 is the baseline. **30** |
| G3 | Both CNNs, full-image random blend, alpha 0.10, 0.20, 0.35, 0.50, p=0.10. The pattern is fixed per dataset and seed across alpha. **40** |
| G4 | Both CNNs at width 1, 2, 4, clean and baseline. Width 1 is already in G0. **40** |

**185** unique training runs. Clean models are shared across poison variants of the same init key (35 unique clean models) and are not trained once per variant.

The six-model pilot is seed 0, width 1, clean and baseline, for each of the three architectures. The full batch does not start until that pilot's engineering gate is accepted.

Engineering gate (pilot only, **not** a scientific acceptance criterion):

- MNIST clean accuracy ≥ 0.97 on the clean pilot model and on the baseline poisoned model
- CIFAR-10 clean accuracy ≥ 0.60 on the same two roles
- Baseline non-target ASR ≥ 0.90

Low ASR, failed gates, and failed runs are kept. Do not change epochs, poison rates, or triggers because a weak condition scored poorly on the test set. A recipe change is a new config hash and a new `--output` directory.

## Isolated environment

The training server's system Python may be 3.8. This code requires **Python 3.10+**. Do not install packages into the system interpreter. Conda on that machine is `~/miniconda3`. Create a new env; do not assume one already exists.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n vision-bank python=3.11
conda activate vision-bank
python -m pip install --upgrade pip
# CUDA wheel first. RTX PRO 6000 Blackwell is compute capability 12.0 (sm_120).
# Use a current official PyTorch build with CUDA 12.8 or newer whose
# torch.cuda.get_arch_list() includes sm_120. Do not guess an older default wheel.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/check_pytorch_blackwell.py
```

`scripts/check_pytorch_blackwell.py` exits 0 on CPU-only (it prints `status: cpu_only`) and exits 2 when a visible GPU's capability is missing from the wheel. Do not start GPU training when that check fails.

CPU pilot is explicit. Nothing selects a GPU unless you pass `--gpus`.

## GPU policy

The machine this bundle was written for has four RTX PRO 6000 Blackwell GPUs (~96 GiB). At the time of writing they were all busy (utilization about 62–98%, memory about 54–94 GiB). **No free GPU is promised.**

- Pass `--device cpu` or an explicit `--gpus` list. There is no default device.
- Each worker is a subprocess with `CUDA_VISIBLE_DEVICES` set to exactly one allowed GPU.
- Default concurrency is one worker per idle allowed GPU. `--workers` can only lower that.
- Before launch, the parent reads `nvidia-smi` utilization, memory, and compute processes. A GPU is idle only when utilization ≤ 5%, memory used ≤ 4096 MiB, and no compute process is present (`configs/bank.yaml` `resources`). Unknown readings count as busy.
- Busy requested GPUs are refused. An unlisted GPU is never substituted.
- If every requested GPU is busy, nothing is trained.
- On CUDA OOM the failure is recorded and worker concurrency is lowered. Batch size is not changed. Other users' processes are not killed. Drivers are not changed.

Check the live view without training:

```bash
python -m vision_model_bank devices --config configs/bank.yaml --gpus 0,1,2,3
```

## Commands

Run these from the `vision-model-bank` directory inside the conda env. `--output` is the private run directory (weights, poison index lists, logs). It should not be the blind mount.

```bash
python -m vision_model_bank generate --config configs/bank.yaml --out configs/manifest.json
python -m vision_model_bank dry-run --config configs/bank.yaml
python -m vision_model_bank devices --config configs/bank.yaml --gpus 0,1,2,3
python -m vision_model_bank pilot --config configs/bank.yaml --output runs/bank --device cpu
python -m vision_model_bank pilot --config configs/bank.yaml --output runs/bank --gpus 0
python -m vision_model_bank run --config configs/bank.yaml --output runs/bank --gpus 0
python -m vision_model_bank resume --config configs/bank.yaml --output runs/bank --gpus 0
python -m vision_model_bank evaluate --output runs/bank --device cpu
python -m vision_model_bank validate --output runs/bank
python -m vision_model_bank export --output runs/bank --blind-dir exports/blind --private-dir exports/private
python -m vision_model_bank summary --output runs/bank
```

`pilot` trains the six models, runs official-test evaluation for those six only, and writes `pilot_acceptance.json`. `run` refuses to start unless that file exists, matches the config and manifest hashes, and `accepted` is true. `resume` continues unfinished pilot tasks without acceptance and continues the rest only after acceptance. Completed tasks are not deleted or retrained.

`evaluate` is the official test. Training itself only reads the fit split and the clean validation split. `validate` reloads weights, checks that paired models share the initialization hash, and recomputes the integer metric counts. It does not overwrite metrics when they disagree; it reports the mismatch.

Exit codes: `0` success, `2` usage or device refusal or recipe conflict, `3` full batch blocked on the pilot gate, `4` pilot finished and the engineering gate failed (artifacts kept), `1` other failure.

## Metrics that are recorded

For every poisoned model, on the official test set:

- Clean accuracy over all test images
- Non-target-only ASR and its denominator (labels other than the target)
- Untriggered target rate on that same subset
- Triggered original-label accuracy on that same subset
- The paired clean model's response to the same trigger, cached once per clean model and trigger fingerprint (poison rate does not change the fingerprint)
- Clean degradation: paired clean accuracy minus poisoned clean accuracy
- Parameter count, train wall time, peak allocated VRAM when CUDA was used
- Per-tensor and total L2 deltas versus the saved initialization and versus the paired clean weights

Clean models record clean accuracy, timing, VRAM, parameter count, and delta versus initialization. They are not given an attack label in order to invent an ASR.

## Blind and private bundles

`export` writes two directories and, if you ask, two archives. It will not write one archive that contains both, and it will not nest one directory inside the other.

- **Blind** (`--blind-dir`): opaque random ids, a pure tensor state dict, and architecture plus normalization. Re-export reuses ids already stored in the private map. This is the only tree to mount into a weight-only review.
- **Private** (`--private-dir`): the map from those ids back to the recipe, target, poison rate, paired clean id, and metrics.

Mount the blind directory by itself. Do not mount its parent if that parent also holds the private map, `runs/bank`, or this source tree. This source tree names every attack; it stays on the operator side.

```bash
python -m vision_model_bank export \
  --output runs/bank \
  --blind-dir "$HOME/vision-bank-blind" \
  --private-dir "$HOME/vision-bank-private" \
  --blind-archive "$HOME/vision-bank-blind.tar.gz" \
  --private-archive "$HOME/vision-bank-private.tar.gz"
```

Suggested isolation: the review account can read only the blind path (a bind mount or a separate unix user). The private map and the training output stay owned by the operator account and are not on that mount.

## Tests

From this directory, with PyTorch importable:

```bash
python -m pytest -q
```

The tests check the 185-count dedup against an independent enumeration, poison nesting and non-target selection, trigger location and normalization order, pairing, metric denominators, GPU refusal, resume equality, export separation, and a synthetic smoke train. They do not download MNIST or CIFAR-10 and they do not train the 185 models.

## Deploy on the GPU server

Copy this directory, or the code archive built by `scripts/package_code_archive.sh`, to the server. Do not copy a `runs/` tree or a blind bundle into the audit sandbox.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision-bank
python scripts/check_pytorch_blackwell.py
python -m vision_model_bank dry-run --config configs/bank.yaml
python -m vision_model_bank devices --config configs/bank.yaml --gpus 0,1,2,3
# CPU smoke of the real six-model pilot only if you accept a long CPU run.
# Otherwise wait until devices reports an idle GPU:
python -m vision_model_bank pilot --config configs/bank.yaml --output "$HOME/vision-bank-runs" --gpus 0
python -m vision_model_bank run --config configs/bank.yaml --output "$HOME/vision-bank-runs" --gpus 0
python -m vision_model_bank evaluate --output "$HOME/vision-bank-runs" --device cpu
python -m vision_model_bank validate --output "$HOME/vision-bank-runs"
python -m vision_model_bank summary --output "$HOME/vision-bank-runs"
python -m vision_model_bank export \
  --output "$HOME/vision-bank-runs" \
  --blind-dir "$HOME/vision-bank-blind" \
  --private-dir "$HOME/vision-bank-private"
```

`resume` repeats the same `--output` and an explicit device after a crash. Optimizer state, RNG state, and the next batch index are in `tasks/<id>/resume.pt`.

## Limitations

- The 185 models were not trained in the environment that produced this tree. Engineering thresholds are specified and unit-tested as a gate, not demonstrated on real MNIST or CIFAR-10 here.
- Older checkpoints in the rest of the repository use a top-left patch, a different poison rate, no normalization, and test-set monitoring during training. Loading those files into this bank's preprocessing will not reproduce their reported numbers.
- GPU bitwise resume is best-effort. CPU resume of the same batch sequence is what the tests check. cuDNN deterministic flags are set in each worker; they do not promise identical bits across GPU models.
- MNIST augmentation is identity because flips change the digit. CIFAR-10 training augmentation is random crop (pad 4) and horizontal flip, keyed by dataset, seed, epoch, and index so the clean twin sees the same geometry before the trigger is stamped.
- Validation is 10% stratified and seed-independent. The poison count is still `p * official_train_size`, not `p` times the fit split. If a count ever exceeded the non-target fit pool the run would stop rather than shrink the rate.
- Data-loader workers are fixed at 0 so the shared order cannot be reordered by worker processes. CPU thread count is capped via `OMP_NUM_THREADS` and the related BLAS variables.
- Weak rates (especially p=0.01) may not reach a high ASR. Those trials stay in the summary. The pilot gate does not apply to them and must not be used as a reason to edit the recipe.
- This pipeline does not call a language model or a paid API.
