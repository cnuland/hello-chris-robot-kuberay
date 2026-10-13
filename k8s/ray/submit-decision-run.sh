#!/usr/bin/env bash
# Submits a teacher/student run to the hand-decision RayCluster (same flow as hello-chris-dd-kuberay):
#   1. stages app/ (code + assets + fixture scenes, no venv) into the Ray head pod
#   2. `ray job submit` from inside the head with that dir as the job's working_dir
# Usage: RUN_TAG=v001 k8s/ray/submit-decision-run.sh --num-envs 20 --stage lift
set -euo pipefail
NS="${NS:-hello-chris-ai-4}"; RC="${RC:-hand-decision}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

HEAD="$(oc get pods -n "$NS" -l "ray.io/cluster=$RC,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}')"
[ -n "$HEAD" ] || { echo "no head pod for RayCluster $RC in $NS"; exit 1; }
echo "==> head pod: $HEAD  run_tag: $RUN_TAG"

STAGE="$(mktemp -d)"; mkdir -p "$STAGE/app/ignored"
cp "$HERE"/app/*.py "$STAGE/app/"
[ -d "$HERE/app/ignored/assets" ] && cp -r "$HERE/app/ignored/assets" "$STAGE/app/ignored/"
[ -d "$HERE/app/ignored/scenes" ] && cp -r "$HERE/app/ignored/scenes" "$STAGE/app/ignored/"
oc exec -n "$NS" "$HEAD" -c ray-head -- rm -rf /tmp/hand-app
oc rsync -q -n "$NS" -c ray-head "$STAGE/app/" "$HEAD:/tmp/hand-app"; rm -rf "$STAGE"

RUNTIME_ENV="$(cat <<JSON
{"working_dir": "/tmp/hand-app", "env_vars": {"STUDENT_ENDPOINTS": "${STUDENT_ENDPOINTS:-http://student.hello-chris-ai-4.svc:8011}",
 "ROBOT_FIXTURE_EVERY": "${ROBOT_FIXTURE_EVERY:-100}"}}
JSON
)"
oc exec -n "$NS" "$HEAD" -c ray-head -- ray job submit --address http://127.0.0.1:8265 --submission-id "hand-$RUN_TAG" \
  --runtime-env-json "$RUNTIME_ENV" --no-wait -- python run-decision-hand.py --run-tag "$RUN_TAG" "$@"
echo "follow:  oc exec -n $NS $HEAD -c ray-head -- ray job logs --address http://127.0.0.1:8265 -f hand-$RUN_TAG"
