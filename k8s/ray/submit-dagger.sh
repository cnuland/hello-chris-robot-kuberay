#!/usr/bin/env bash
# Submit a KubeRay job (DAgger or PPO) to the hand-decision RayCluster. Stages app/*.py + app/ignored/assets into the
# head pod and runs `ray job submit` with MinIO + the in-cluster DiffusionGemma endpoint in the runtime env.
# Usage: RUN_TAG=dagger-v1 k8s/ray/submit-dagger.sh [script.py] --rounds 6 --episodes 56
set -euo pipefail
NS="${NS:-hello-chris-ai-4}"; RC="${RC:-hand-decision}"
: "${S3_ACCESS_KEY_ID:?export S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY (your MinIO credentials) first}"
: "${S3_SECRET_ACCESS_KEY:?export S3_SECRET_ACCESS_KEY first}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="dagger_assembly.py"; if [[ "${1:-}" == *.py ]]; then SCRIPT="$1"; shift; fi
RUN_TAG="${RUN_TAG:-${SCRIPT%.py}-$(date +%Y%m%d-%H%M%S)}"
HEAD="$(oc get pods -n "$NS" -l "ray.io/cluster=$RC,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}')"
[ -n "$HEAD" ] || { echo "no head pod for RayCluster $RC in $NS"; exit 1; }
echo "==> head: $HEAD  script: $SCRIPT  run_tag: $RUN_TAG"
STAGE="$(mktemp -d)"; mkdir -p "$STAGE/app/ignored"
cp "$HERE"/app/*.py "$STAGE/app/"; cp -r "$HERE/app/ignored/assets" "$STAGE/app/ignored/"
[ -d "$HERE/app/ignored/checkpoints" ] && cp -r "$HERE/app/ignored/checkpoints" "$STAGE/app/ignored/"
[ -d "$HERE/app/ignored/vision" ] && cp -r "$HERE/app/ignored/vision" "$STAGE/app/ignored/"      # DiffusionGemma inspection labels
oc exec -n "$NS" "$HEAD" -c ray-head -- rm -rf /tmp/hand-app
oc rsync -q -n "$NS" -c ray-head "$STAGE/app/" "$HEAD:/tmp/hand-app"; rm -rf "$STAGE"
RUNTIME_ENV="$(cat <<JSON
{"working_dir": "/tmp/hand-app", "env_vars": {
  "DGEMMA_URL": "${DGEMMA_URL:-http://dgemma.dgemma.svc:8000}", "DGEMMA_MODE": "chat", "DGEMMA_TIMEOUT_S": "20",
  "S3_ENDPOINT_URL": "http://minio.minio.svc:9000", "S3_BUCKET": "robot",
  "S3_ACCESS_KEY_ID": "${S3_ACCESS_KEY_ID}", "S3_SECRET_ACCESS_KEY": "${S3_SECRET_ACCESS_KEY}", "ROBOT_OUT": "/tmp/robot-runs",
  "MUJOCO_GL": "osmesa", "OMP_NUM_THREADS": "1"}}
JSON
)"
oc exec -n "$NS" "$HEAD" -c ray-head -- ray job submit --address http://127.0.0.1:8265 --submission-id "$RUN_TAG" \
  --runtime-env-json "$RUNTIME_ENV" --no-wait -- python "$SCRIPT" --run-tag "$RUN_TAG" "$@"
echo "follow:  oc exec -n $NS $HEAD -c ray-head -- ray job logs --address http://127.0.0.1:8265 -f $RUN_TAG"
echo "fetch:   oc rsync -n $NS -c ray-head $HEAD:/tmp/robot-runs/$RUN_TAG app/ignored/runs/"
