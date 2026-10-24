#!/usr/bin/env bash
# trigger.sh — run one moss-batch job on SaladCloud and collect the results.
#
#   export SALAD_API_KEY=...
#   export SALAD_ORG=...          # organization slug
#   export SALAD_PROJECT=...      # project slug
#   ./trigger.sh prompts.json
#
# Requires: curl, python3. Optional: SALAD_GROUP (default moss-batch),
# WAIT_MINUTES (default 45), OUT_DIR (default ./results/<job_id>).
#
# What it does:
#   1. stop the container group and wait for `stopped`
#   2. upload {"job_id":..., "prompts":...} to Salad S3S
#   3. POST /start, poll the group until succeeded/failed
#   4. download+mount moss-batch/out/<job_id>/part-NN from S3S
#
# The container group must carry JOB_PATH=moss-batch/in/job.json plus SALAD_ORG
# and SALAD_API_KEY in its environment variables, or it will fall back to the
# prompts baked into the image.
set -euo pipefail

: "${SALAD_API_KEY:?set SALAD_API_KEY}"
: "${SALAD_ORG:?set SALAD_ORG}"
: "${SALAD_PROJECT:?set SALAD_PROJECT}"

GROUP="${SALAD_GROUP:-moss-batch}"
PROMPTS_FILE="${1:-prompts.json}"
JOB_ID="${JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WAIT_MINUTES="${WAIT_MINUTES:-45}"
OUT_BASE="${OUT_DIR:-./results/${JOB_ID}}"

API="https://api.salad.com/api/public/organizations/${SALAD_ORG}/projects/${SALAD_PROJECT}/containers"
S3S="https://storage-api.salad.com/organizations/${SALAD_ORG}/files"
AUTH=(-H "Salad-Api-Key: ${SALAD_API_KEY}")

log() { printf '>>> %s\n' "$*"; }

group_status() {
  curl -sS "${API}/${GROUP}" "${AUTH[@]}" 2>/dev/null \
    | python -c "import sys,json;print(json.load(sys.stdin).get('current_state',{}).get('status','unknown'))" 2>/dev/null \
    || echo "unknown"
}

[ -f "${PROMPTS_FILE}" ] || { echo "no such file: ${PROMPTS_FILE}" >&2; exit 1; }

log "job ${JOB_ID}"

# -- 1. stop whatever is running ------------------------------------------
log "stopping ${GROUP}"
curl -sS -o /dev/null -X POST "${API}/${GROUP}/stop" "${AUTH[@]}" || true

for _ in $(seq 1 60); do
  [ "$(group_status)" = "stopped" ] && break
  sleep 2
done

ST="$(group_status)"
if [ "${ST}" != "stopped" ]; then
  echo "group did not stop (status=${ST}); refusing to start a second batch" >&2
  exit 1
fi

# -- 2. upload the job ----------------------------------------------------
log "uploading prompts from ${PROMPTS_FILE}"
python - "${JOB_ID}" "${PROMPTS_FILE}" > /tmp/moss_job.json <<'PY'
import json, sys
job_id, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    prompts = json.load(f)
print(json.dumps({"job_id": job_id, "prompts": prompts}, indent=2))
PY

UP_CODE=$(curl -sS -o /dev/null -w '%{http_code}' -X PUT \
  "${S3S}/moss-batch/in/job.json" "${AUTH[@]}" \
  -F 'mimeType="application/json"' \
  -F 'file=@/tmp/moss_job.json') || UP_CODE=000
log "upload -> ${UP_CODE}"
[ "${UP_CODE}" = "200" ] || { echo "S3S upload failed (${UP_CODE})" >&2; exit 1; }

# -- 3. start and wait ----------------------------------------------------
log "starting ${GROUP}"
START_CODE=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "${API}/${GROUP}/start" "${AUTH[@]}") || START_CODE=000
log "start -> ${START_CODE}"
[ "${START_CODE}" = "202" ] || [ "${START_CODE}" = "200" ] || {
  echo "start failed (${START_CODE})" >&2; exit 1; }

# Give it a moment to leave `stopped` before trusting a terminal status.
for _ in $(seq 1 30); do
  [ "$(group_status)" != "stopped" ] && break
  sleep 2
done

DEADLINE=$(( $(date +%s) + WAIT_MINUTES * 60 ))
ST="unknown"
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
  ST="$(group_status)"
  printf '\r    status=%-10s' "${ST}"
  case "${ST}" in
    succeeded|failed) break ;;
  esac
  sleep 10
done
printf '\n'
log "final status=${ST}"

if [ "${ST}" = "failed" ]; then
  echo "job failed — pull logs with:" >&2
  echo "  curl -sS '${API}/${GROUP}/system-logs?follow=false' -H 'Salad-Api-Key: \$SALAD_API_KEY'" >&2
fi

# -- 4. collect results ---------------------------------------------------
mkdir -p "${OUT_BASE}"
found=0
for n in 00 01 02 03 04 05 06 07 08 09; do
  code=$(curl -sS -o "${OUT_BASE}/part-${n}.tgz" -w '%{http_code}' \
    "${S3S}/moss-batch/out/${JOB_ID}/part-${n}" "${AUTH[@]}") || code=000
  if [ "${code}" = "200" ]; then
    found=$((found + 1))
    log "part-${n}: $(wc -c < "${OUT_BASE}/part-${n}.tgz") bytes"
  else
    rm -f "${OUT_BASE}/part-${n}.tgz"
    break
  fi
done

if [ "${found}" -eq 0 ]; then
  echo "no output parts found for job ${JOB_ID}" >&2
  exit 1
fi

cat "${OUT_BASE}"/part-*.tgz > "${OUT_BASE}/bundle.tgz"
rm -f "${OUT_BASE}"/part-*.tgz
tar xzf "${OUT_BASE}/bundle.tgz" -C "${OUT_BASE}"
rm -f "${OUT_BASE}/bundle.tgz"

log "results in ${OUT_BASE}"
ls -la "${OUT_BASE}"
if [ -f "${OUT_BASE}/manifest.json" ]; then
  echo
  python -c "
import json,sys
m=json.load(open('${OUT_BASE}/manifest.json'))
for k in ('images','wall_seconds','s_per_image_gpu','s_per_image_wall','batch_size','resolution'):
    if k in m: print(f'  {k:18} {m[k]}')
"
fi
