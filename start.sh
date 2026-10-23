#!/usr/bin/env bash
# Container entrypoint: boot ComfyUI, wait for it, run the batch, then exit
# (which is what triggers Salad to stop billing, since billing is per-second
# only while the instance is `running`).
set -euo pipefail

COMFY_PORT="${COMFY_PORT:-8188}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-50}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
SAGE="${SAGE:-0}"
PROMPTS="${PROMPTS:-/app/prompts.json}"
OUT_DIR="${OUT_DIR:-/app/out}"
JOB_TIMEOUT="${JOB_TIMEOUT:-7200}"

echo "[start] ComfyUI port=${COMFY_PORT} sage=${SAGE} batch=${BATCH_SIZE} ${WIDTH}x${HEIGHT} steps=${STEPS}"

# --- ComfyUI launch flags -------------------------------------------------
ARGS=(--listen 0.0.0.0 --port "${COMFY_PORT}")
# Conda/imgutils uses these to keep VRAM available for the DiT.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ "${SAGE}" = "1" ]; then
  # NOTE: unverified for MOSS's joint-stream self-attention. ComfyUI's
  # attention_sage falls back to attention_pytorch on error, so this is safe
  # but may be a no-op. Compare s/image in the manifest to decide.
  if python -c "import sageattention" 2>/dev/null; then
    ARGS+=(--use-sage-attention)
    echo "[start] sage attention ENABLED (experimental)"
  else
    echo "[start] SAGE=1 but sageattention is not installed in this image; ignoring" >&2
  fi
else
  echo "[start] sage attention disabled (default)"
fi

cd /opt/ComfyUI
python main.py "${ARGS[@]}" > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!
echo "[start] ComfyUI pid=${COMFY_PID}, log=/tmp/comfyui.log"

# If ComfyUI dies, do not hang forever on the poll loop below.
cleanup() {
  kill "${COMFY_PID}" 2>/dev/null || true
  wait "${COMFY_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# --- run the batch -------------------------------------------------------
set +e
python /app/moss_batch.py \
  --host 127.0.0.1 --port "${COMFY_PORT}" \
  --workflow /app/workflow.json \
  --prompts "${PROMPTS}" \
  --out "${OUT_DIR}" \
  --batch "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --rate "${RATE:-0}" \
  --shutdown "${SHUTDOWN:-none}" \
  --job-timeout "${JOB_TIMEOUT}" \
  --bench
RC=$?
set -e

echo "[start] batch exited rc=${RC}"
if [ "$(ls -A "${OUT_DIR}" 2>/dev/null | grep -v manifest.json || true)" = "" ]; then
  echo "[start] WARNING: no images produced; tailing ComfyUI log" >&2
  tail -n 80 /tmp/comfyui.log >&2 || true
fi

# On Salad the instance stops when the container exits. For a batch job that is
# exactly what we want: no idle GPU, no boot charge, per-second billing only.
exit "${RC}"
