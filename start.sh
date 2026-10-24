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

# --- optional: pull this run's job from Salad S3S -------------------------
# Set JOB_PATH (e.g. "moss-batch/in/job.json") plus SALAD_ORG and SALAD_API_KEY
# on the container group and the baked-in prompts.json is ignored in favour of
# whatever trigger.sh uploaded. This is what makes the container reusable
# without rebuilding the 15.4 GB image for every prompt change.
JOB_ID="${JOB_ID:-}"
if [ -n "${JOB_PATH:-}" ] && [ -n "${SALAD_ORG:-}" ] && [ -n "${SALAD_API_KEY:-}" ]; then
  echo "[start] fetching job from S3S: ${JOB_PATH}"
  S3S_CODE=$(curl -sS -o /tmp/job.json -w '%{http_code}' \
    "https://storage-api.salad.com/organizations/${SALAD_ORG}/files/${JOB_PATH}" \
    -H "Salad-Api-Key: ${SALAD_API_KEY}") || S3S_CODE=000
  if [ "${S3S_CODE}" = "200" ]; then
    JOB_ID=$(python -c "import json;print(json.load(open('/tmp/job.json')).get('job_id',''))")
    python -c "
import json
job = json.load(open('/tmp/job.json'))
json.dump(job.get('prompts', job), open('/tmp/prompts.json', 'w'), indent=2)
"
    PROMPTS=/tmp/prompts.json
    echo "[start] job_id=${JOB_ID:-<none>} (prompts from S3S)"
  else
    echo "[start] WARN: S3S fetch returned ${S3S_CODE}; using baked-in prompts" >&2
  fi
fi

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

# --- optional: ship results back to Salad S3S ----------------------------
# The instance disk is destroyed when this container exits, so anything not
# uploaded here is lost. S3S caps a single object at 100 MB, so a tarball over
# 90 MB is split into part-00, part-01, ... and trigger.sh reassembles them.
if [ -n "${SALAD_ORG:-}" ] && [ -n "${SALAD_API_KEY:-}" ] && [ -d "${OUT_DIR}" ]; then
  if [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null || true)" ]; then
    JOB_ID="${JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    tar czf /tmp/out.tgz -C "${OUT_DIR}" .
    BYTES=$(wc -c < /tmp/out.tgz)
    echo "[start] archive bytes=${BYTES} job=${JOB_ID}"
    rm -f /tmp/part-*
    if [ "${BYTES}" -le $((90 * 1024 * 1024)) ]; then
      cp /tmp/out.tgz /tmp/part-00
    else
      split -b 90M -d -a 2 /tmp/out.tgz /tmp/part-
    fi
    for p in /tmp/part-*; do
      KEY="moss-batch/out/${JOB_ID}/$(basename "$p")"
      CODE=$(curl -sS -o /dev/null -w '%{http_code}' -X PUT \
        "https://storage-api.salad.com/organizations/${SALAD_ORG}/files/${KEY}" \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        -F 'mimeType="application/gzip"' \
        -F "file=@${p}") || CODE=000
      echo "[start] PUT ${KEY} -> ${CODE}"
    done
    echo "[start] results under moss-batch/out/${JOB_ID}/"
  else
    echo "[start] WARN: OUT_DIR empty, nothing to upload" >&2
  fi
fi

# On Salad the instance stops when the container exits. For a batch job that is
# exactly what we want: no idle GPU, no boot charge, per-second billing only.
exit "${RC}"
