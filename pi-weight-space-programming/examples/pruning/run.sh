#!/usr/bin/env bash
# Script to execute Model Pruning Weight-Space Programming run and out-of-harness evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/models/input" "${SCRIPT_DIR}/models/output" "${SCRIPT_DIR}/logs"

# Activate the local virtual environment so python3 and packages (numpy, torch, safetensors) are on PATH
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Set WSP_PYTHON so Pi WSP harness executes with this environment
export WSP_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Ensure the Pi extension directory uses our populated virtual environment
PI_EXT_DIR="${SCRIPT_DIR}/.pi/git/github.com/felipepenha/pi-weight-space-programming"
if [ -d "${PI_EXT_DIR}" ]; then
  rm -rf "${PI_EXT_DIR}/.venv"
  ln -sf "${SCRIPT_DIR}/.venv" "${PI_EXT_DIR}/.venv"
fi

EVAL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --eval-only|--skip-pi)
      EVAL_ONLY=true
      shift
      ;;
  esac
done

# Ensure input weights are linked/downloaded if empty
if [ -z "$(ls -A "${SCRIPT_DIR}/models/input" 2>/dev/null)" ]; then
  HF_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots"
  SNAPSHOT=$(ls -d "${HF_CACHE_DIR}"/* 2>/dev/null | head -n 1 || true)
  if [ -n "${SNAPSHOT}" ] && [ -d "${SNAPSHOT}" ]; then
    echo "🔗 Linking cached Qwen2.5-Coder-0.5B-Instruct weights into models/input..."
    for f in "${SNAPSHOT}"/*; do
      ln -sf "$f" "${SCRIPT_DIR}/models/input/"
    done
  fi
fi

if [ "${EVAL_ONLY}" = false ]; then
  echo "======================================================================"
  echo "🚀 Running Pi Weight-Space Programming (π · WSP)"
  echo "   Model: google/gemini-3.8-flash | Effort: medium"
  echo "   Prompt: @PROMPT.md"
  echo "======================================================================"

  if command -v pi >/dev/null 2>&1; then
    pi -a -p \
      --model google/gemini-3.8-flash \
      --wsp-effort medium \
      --wsp-log-dir "${SCRIPT_DIR}/logs" \
      "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log" || {
        echo "⚠️  Pi run failed or high traffic encountered. Continuing to evaluation..." | tee -a "${SCRIPT_DIR}/session.log"
      }
  else
    echo "⚠️  'pi' CLI not found on PATH. Skipping agent execution and proceeding to evaluation." | tee "${SCRIPT_DIR}/session.log"
  fi
else
  echo "ℹ️  Running in evaluation-only mode (--eval-only)" | tee "${SCRIPT_DIR}/session.log"
fi

{
  echo ""
  echo "======================================================================"
  echo "🧪 1. Evaluating Original Baseline Model (models/input)"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

echo "Running prompt_model on models/input..." | tee -a "${SCRIPT_DIR}/session.log"
uv run eval/prompt_model.py \
  --model-dir models/input \
  --prompt "pwd to discover path and then list files in the path" \
  --no-exec 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

echo "" | tee -a "${SCRIPT_DIR}/session.log"
echo "Running eval_model suite on models/input..." | tee -a "${SCRIPT_DIR}/session.log"
uv run eval/eval_model.py \
  --model-dir models/input 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

{
  echo ""
  echo "======================================================================"
  echo "🧪 2. Evaluating Pruned Output Model (models/output)"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

if [ -n "$(find "${SCRIPT_DIR}/models/output" -name "*.safetensors" -o -name "*.bin" 2>/dev/null)" ]; then
  echo "Running prompt_model on models/output..." | tee -a "${SCRIPT_DIR}/session.log"
  uv run eval/prompt_model.py \
    --model-dir models/output \
    --prompt "pwd to discover path and then list files in the path" \
    --no-exec 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

  echo "" | tee -a "${SCRIPT_DIR}/session.log"
  echo "Running eval_model suite on models/output..." | tee -a "${SCRIPT_DIR}/session.log"
  uv run eval/eval_model.py \
    --model-dir models/output 2>&1 | tee -a "${SCRIPT_DIR}/session.log"
else
  echo "⚠️  Output model weights not found in models/output." | tee -a "${SCRIPT_DIR}/session.log"
  echo "   Running eval_model in --allow-missing mode:" | tee -a "${SCRIPT_DIR}/session.log"
  uv run eval/eval_model.py \
    --model-dir models/output \
    --allow-missing 2>&1 | tee -a "${SCRIPT_DIR}/session.log"
fi

echo "" | tee -a "${SCRIPT_DIR}/session.log"
echo "======================================================================"
echo "✅ Workflow complete!"
echo "   Session transcript: ${SCRIPT_DIR}/session.log"
echo "   Audit logs        : ${SCRIPT_DIR}/logs/"
echo "   Output model      : ${SCRIPT_DIR}/models/output/"
echo "======================================================================"
