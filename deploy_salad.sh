#!/usr/bin/env bash
# Create a SaladCloud container group for moss-batch via the SCE API.
#
# Requires: curl, jq, and a Salad API key.
#   export SALAD_API_KEY=...
#   export SALAD_ORG=...        # organization name (slug)
#   export SALAD_PROJECT=...    # project name (slug)
#   export IMAGE=<dockerhub-user>/moss-batch:latest
#   ./deploy_salad.sh
#
# The resource numbers here are the measured-safe ones: 8 vCPU / 32 GB RAM /
# 40 GB disk / batch 4. GPU rates include vCPU and RAM, so these are free.
set -euo pipefail

: "${SALAD_API_KEY:?set SALAD_API_KEY}"
: "${SALAD_ORG:?set SALAD_ORG}"
: "${SALAD_PROJECT:?set SALAD_PROJECT}"
: "${IMAGE:?set IMAGE, e.g. you/moss-batch:latest}"

GROUP="${SALAD_GROUP:-moss-batch}"
PRIORITY="${PRIORITY:-medium}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RATE="${RATE:-0.273}"
API="https://api.salad.com/api/public/organizations/${SALAD_ORG}/projects/${SALAD_PROJECT}/containers"

read -r -d '' BODY <<JSON || true
{
  "name": "${GROUP}",
  "display_name": "${GROUP}",
  "replicas": 1,
  "container": {
    "image": "${IMAGE}",
    "resources": {
      "cpu": 8,
      "memory": 32768,
      "storage": 40960,
      "shm_size": 8192,
      "gpu_classes": ["rtx4090"]
    },
    "environment_variables": {
      "BATCH_SIZE": "${BATCH_SIZE}",
      "STEPS": "50",
      "WIDTH": "1024",
      "HEIGHT": "1024",
      "SAGE": "0",
      "RATE": "${RATE}",
      "JOB_PATH": "${JOB_PATH:-}",
      "SALAD_ORG": "${SALAD_ORG}",
      "SALAD_API_KEY": "${SALAD_API_KEY}"
    },
    "command": [],
    "image_caching": false
  },
  "autostart_policy": true,
  "restart_policy": "never",
  "priority": "${PRIORITY}",
  "startup_probe": {
    "http": {
      "path": "/system_stats",
      "port": 8188,
      "scheme": "http",
      "headers": {}
    },
    "initial_delay_seconds": 150,
    "period_seconds": 15,
    "timeout_seconds": 10,
    "failure_threshold": 10,
    "success_threshold": 1
  }
}
JSON

echo ">>> creating container group '${GROUP}' (priority=${PRIORITY}, batch=${BATCH_SIZE})"
curl -sS -X POST "${API}" \
  -H "Salad-Api-Key: ${SALAD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "${BODY}" | jq . || {
    echo "--- create failed; dumping raw response" >&2
    exit 1
  }

cat <<'NOTE'

>>> next steps
  1. open the Portal and watch the instance go Allocating -> Downloading -> Running
     (allocating/downloading are NOT billed)
  2. tail the logs: the "[boot] ready after Ns" line is your true cold-start cost
  3. when the batch finishes the container exits and billing stops
  4. read out/manifest.json for s_per_image_gpu

>>> quota gotcha
  if job-queue autoscaling is ever enabled on this group, Salad counts the
  MAXIMUM replicas against your quota, not the current count.
NOTE

# stop helper
cat <<NOTE2

>>> to stop early:
  curl -X POST "${API}/${GROUP}/stop" -H "Salad-Api-Key: \$SALAD_API_KEY"

>>> to delete entirely:
  curl -X DELETE "${API}/${GROUP}" -H "Salad-Api-Key: \$SALAD_API_KEY"
NOTE2
