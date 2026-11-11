# Phase 1 runbook

The training and the Astra queue run on the shared GPU server. This checkout cannot reach that machine. The protocol to run is preregistration v1.1. The two v1 cases (one frozen, one partial) are discarded and are not scored.

Do not install into `/home/junhong/sbx-01`. Do not read or edit pi auth files. Do not kill other users' jobs. Do not set or unset `CUDA_VISIBLE_DEVICES` inside a script; export it yourself for the one card you intend to use. The trainers use `cuda:0` among the devices that variable already exposes, or CPU when it is empty or when CUDA is absent.

Blind directories, the pi working directory, and anything bubblewrap mounts must not contain `backdoor`, `clean`, `poison`, `trigger`, or `asr` in the host path. `/proc/self/mountinfo` shows that path to the sandbox. A clone that lives under a directory named `wsp-astra-backdoor` is fine for the code, and the wrong place for `--blind`.

```text
TVENV=/home/junhong/cnn-p1/tvenv/bin/python
SANDBOX_PYTHON=/home/junhong/sbx-01/env/bin/python
PI_BIN=/home/junhong/.local/bin/pi-online

SRC=/home/junhong/cnn-p1-src
TRAIN=/home/junhong/cnn-p1/train
BLIND=/home/junhong/cnn-p1-blind
PRIVATE=/home/junhong/cnn-p1-private
EVAL=/home/junhong/cnn-p1/eval
BASELINES=/home/junhong/cnn-p1/baselines
AGG=/home/junhong/cnn-p1/aggregate
```

`SRC` is a checkout of this branch. `tvenv` is the torch interpreter with torchvision. Training, the audit driver, evaluate, baselines, and aggregate run under `tvenv`, because training imports torchvision. `probe-02` is an older torch env and is not the training interpreter. `sbx-01` is the numpy-only sandbox. The driver passes that interpreter as `--sandbox-python`; Astra's analysis code runs there, not in `tvenv`.

## 1. Preflight

```bash
cd "$SRC"
if "$SANDBOX_PYTHON" -I -c 'import torch' 2>/dev/null; then
  echo "torch leaked into the sandbox env" >&2
  exit 1
fi
"$SANDBOX_PYTHON" -m cnn_phase1 preflight --sandbox-python "$SANDBOX_PYTHON"
mkdir -p /dev/shm/cnn-p1-pi "$PRIVATE"
PYTHONPATH="$SRC/padbench-audit${PYTHONPATH:+:$PYTHONPATH}" \
  "$SANDBOX_PYTHON" -m padbench_audit preflight \
  --sandbox-python "$SANDBOX_PYTHON" \
  --private-dir "$PRIVATE" \
  --pi-bin "$PI_BIN" \
  --pi-cwd /dev/shm/cnn-p1-pi/probe \
  --effort medium
```

Create `$PRIVATE` before the padbench preflight if it does not exist. The padbench probe checks that pi 1.5.0 launches with tools disabled and compaction off. The phase-1 driver uses that same argv, with stdin from `/dev/null`.

## 2. Train

One process. Seeds 0–9, headline recipe, explicit flags.

```bash
cd "$SRC"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$TVENV" -m cnn_phase1 train \
  --out "$TRAIN" \
  --data-dir "$TRAIN/mnist" \
  --seeds 0-9 \
  --epochs 3 \
  --lr 1e-3 \
  --batch-size 128 \
  --holdout-size 2000 \
  --position top_left \
  --shape square \
  --size 3 \
  --target-class 0 \
  --poison-rate 0.08
```

Exit 0 means every model met clean accuracy ≥ 0.98 and every backdoor met ASR ≥ 0.95. Exit 2 means the manifest was still written and at least one model missed a threshold. Those rows stay in `manifest.json` under `failures`. Do not rerun with a new seed or a new learning rate. `--resume` only finishes a seed that was interrupted; it refuses a recipe change and refuses to replace a finished seed.

`bash cnn_phase1/train_phase1.sh --out "$TRAIN" ...` is the same command.

## 3. Blind cases

```bash
cd "$SRC"
"$TVENV" -m cnn_phase1 make-cases \
  --train-out "$TRAIN" \
  --blind "$BLIND" \
  --private "$PRIVATE" \
  --case-order-seed 20261110
```

The seed is written into `$PRIVATE/ground_truth.json` and is not printed. Omit `--case-order-seed` to draw one; copy it out of that file if you need to reproduce the shuffle. The blind tree has no labels.

## 4. Audits

Sequential. One case, one pi session. A usage-limit error exits 75, writes `$BLIND/queue_pause.json`, and does not mark the case as a failure. Other infra errors are recorded and the queue continues.

The driver runs under `tvenv` because that process needs torch. `--sandbox-python` stays on `sbx-01`.

```bash
cd "$SRC"
mkdir -p /dev/shm/cnn-p1-pi
"$TVENV" -m cnn_phase1 audit \
  --blind "$BLIND" \
  --sandbox-python "$SANDBOX_PYTHON" \
  --pi-bin "$PI_BIN" \
  --pi-cwd /dev/shm/cnn-p1-pi
```

Resume later, including the paused case and any earlier case without a frozen prediction. Completed cases are skipped. A fresh v1.1 queue uses a new blind directory, so the discarded v1 sessions are not resumed.

```bash
"$TVENV" -m cnn_phase1 resume \
  --blind "$BLIND" \
  --sandbox-python "$SANDBOX_PYTHON" \
  --pi-bin "$PI_BIN" \
  --pi-cwd /dev/shm/cnn-p1-pi
```

Logs for each case: `$BLIND/runs/case_XXX/turns/`, `tool_calls.jsonl`, `transcript.md`, `prediction.json`, `prediction.sha256`.

## 5. Evaluate, baselines, aggregate

Run these only after the queue has finished or you have decided to close it. Evaluate aborts if any prediction is missing unless you pass `--accept-unevaluable`.

```bash
cd "$SRC"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$TVENV" -m cnn_phase1 evaluate \
  --blind "$BLIND" \
  --private "$PRIVATE" \
  --train-out "$TRAIN" \
  --data-dir "$TRAIN/mnist" \
  --out "$EVAL"

"$TVENV" -m cnn_phase1 baselines \
  --train-out "$TRAIN" \
  --data-dir "$TRAIN/mnist" \
  --out "$BASELINES"

"$TVENV" -m cnn_phase1 aggregate \
  --eval "$EVAL/eval_records.json" \
  --baselines "$BASELINES/baselines.json" \
  --out "$AGG"
```

`aggregate` writes `summary.json` and `summary.md`. Neural Cleanse uses 200 steps because `--nc-steps` is omitted.

## Phase 2

Phase 2 uses the v1.1 prompt and harness config unchanged. Train the seven backdoor arms and clean seeds 10–19. Seeds 0–9 clean models are reused from the phase-1 train directory by path and hash. A model that misses clean accuracy 0.97 or backdoor ASR 0.90 is kept in the manifest and left out of the blind cases.

```text
P2=/home/junhong/cnn-p2-src
P2_TRAIN=/home/junhong/cnn-p2/train
P2_BLIND=/home/junhong/cnn-p2-blind
P2_PRIVATE=/home/junhong/cnn-p2-private
P2_EVAL=/home/junhong/cnn-p2/eval
P2_BASE=/home/junhong/cnn-p2/baselines
P2_AGG=/home/junhong/cnn-p2/aggregate
P1_TRAIN=/home/junhong/cnn-p1/train
P1_EVAL=/home/junhong/cnn-p1/eval_v11
TVENV=/home/junhong/cnn-p1/tvenv/bin/python
SANDBOX_PYTHON=/home/junhong/sbx-01/env/bin/python
PI_BIN=/home/junhong/.local/bin/pi-online
```

```bash
cd "$P2"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$TVENV" -m cnn_phase1 train-phase2 \
  --out "$P2_TRAIN" \
  --phase1-train "$P1_TRAIN" \
  --data-dir "$P2_TRAIN/mnist"

"$TVENV" -m cnn_phase1 make-cases-phase2 \
  --train-out "$P2_TRAIN" \
  --blind "$P2_BLIND" \
  --private "$P2_PRIVATE" \
  --case-order-seed 20261111

mkdir -p /dev/shm/cnn-p2-pi
"$TVENV" -m cnn_phase1 audit \
  --blind "$P2_BLIND" \
  --sandbox-python "$SANDBOX_PYTHON" \
  --pi-bin "$PI_BIN" \
  --pi-cwd /dev/shm/cnn-p2-pi

"$TVENV" -m cnn_phase1 resume \
  --blind "$P2_BLIND" \
  --sandbox-python "$SANDBOX_PYTHON" \
  --pi-bin "$PI_BIN" \
  --pi-cwd /dev/shm/cnn-p2-pi

"$TVENV" -m cnn_phase1 evaluate \
  --blind "$P2_BLIND" \
  --private "$P2_PRIVATE" \
  --train-out "$P2_TRAIN" \
  --data-dir "$P2_TRAIN/mnist" \
  --out "$P2_EVAL"

"$TVENV" -m cnn_phase1 baselines \
  --train-out "$P2_TRAIN" \
  --data-dir "$P2_TRAIN/mnist" \
  --out "$P2_BASE"

"$TVENV" -m cnn_phase1 aggregate-phase2 \
  --eval "$P2_EVAL/eval_records.json" \
  --anchor "$P1_EVAL/eval_records.json" \
  --baselines "$P2_BASE/baselines.json" \
  --main-manifest "$P2_TRAIN/manifest.json" \
  --out "$P2_AGG"
```

`make-cases-phase2` refuses to run if the prompt or harness config hash is not the v1.1 value. Exit 2 from training means at least one model missed the gate; those models are not audited. The 3-epoch P-center models failed that gate and are not in the 80-case set. Do not regenerate that set.

### Center add-on (preregistration v1.1)

Twenty new cases only: P-center-e6 and A-topleft-e6, seeds 0–9, 6 epochs. They go to a separate blind directory. The 80-case directory is not rebuilt.

```text
P2B_TRAIN=/home/junhong/cnn-p2/train-addon
P2B_BLIND=/home/junhong/cnn-p2b-blind
P2B_PRIVATE=/home/junhong/cnn-p2b-private
P2B_EVAL=/home/junhong/cnn-p2/eval-addon
P2B_AGG=/home/junhong/cnn-p2/aggregate-combined
```

```bash
cd "$P2"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$TVENV" -m cnn_phase1 train-phase2 \
  --addon \
  --out "$P2B_TRAIN" \
  --data-dir "$P2_TRAIN/mnist"

"$TVENV" -m cnn_phase1 make-cases-phase2 \
  --addon \
  --train-out "$P2B_TRAIN" \
  --blind "$P2B_BLIND" \
  --private "$P2B_PRIVATE" \
  --case-order-seed 20261112

mkdir -p /dev/shm/cnn-p2b-pi
"$TVENV" -m cnn_phase1 audit \
  --blind "$P2B_BLIND" \
  --sandbox-python "$SANDBOX_PYTHON" \
  --pi-bin "$PI_BIN" \
  --pi-cwd /dev/shm/cnn-p2b-pi

"$TVENV" -m cnn_phase1 resume \
  --blind "$P2B_BLIND" \
  --sandbox-python "$SANDBOX_PYTHON" \
  --pi-bin "$PI_BIN" \
  --pi-cwd /dev/shm/cnn-p2b-pi

"$TVENV" -m cnn_phase1 evaluate \
  --blind "$P2B_BLIND" \
  --private "$P2B_PRIVATE" \
  --train-out "$P2B_TRAIN" \
  --data-dir "$P2_TRAIN/mnist" \
  --out "$P2B_EVAL"

"$TVENV" -m cnn_phase1 aggregate-phase2 \
  --eval "$P2_EVAL/eval_records.json" \
  --addon-eval "$P2B_EVAL/eval_records.json" \
  --anchor "$P1_EVAL/eval_records.json" \
  --baselines "$P2_BASE/baselines.json" \
  --main-manifest "$P2_TRAIN/manifest.json" \
  --out "$P2B_AGG"
```

`--addon` trains only the two 6-epoch arms and refuses to put the main arms into the add-on blind directory. If fewer than 8 P-center-e6 seeds pass the gate, H1 is untestable. Do not change the recipe again.

## Tests on a dev machine

```bash
cd "$SRC"
python3 -m unittest discover -s cnn_phase1/tests -v
```

The smoke test needs bubblewrap and a numpy-only interpreter. Point it at one with `SANDBOX_PYTHON=/path/to/venv/bin/python`. That interpreter must import numpy and must not import torch.
