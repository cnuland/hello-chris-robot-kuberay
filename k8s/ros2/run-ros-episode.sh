#!/usr/bin/env bash
# Build the ROS 2 image (in-cluster binary build) and run one assembly episode as a Job: the ROS 2 graph drives the cell
# with the distilled student, DiffusionGemma inspects and advises, and the Job records the episode video + a rosbag.
# Usage: SEED=7 CKPT=s3://robot/multi-v1/student_best.pt k8s/ros2/run-ros-episode.sh [--build]
# MinIO credentials come from a Secret in the namespace (never committed):
#   oc create secret generic minio-creds -n hello-chris-ai-4 --from-literal=access_key=... --from-literal=secret_key=...
set -euo pipefail
NS="${NS:-hello-chris-ai-4}"; SEED="${SEED:-7}"; PARTS="${PARTS:-bracket,cap,lbracket}"
CKPT="${CKPT:?set CKPT=s3://robot/<run>/student_best.pt}"; DGEMMA_URL="${DGEMMA_URL:-http://dgemma.dgemma.svc:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--build" ]]; then
  oc get bc robot-ros2 -n "$NS" >/dev/null 2>&1 || oc new-build --binary --strategy=docker --name robot-ros2 -n "$NS" >/dev/null
  CTX="$(mktemp -d)"; mkdir -p "$CTX/app/ignored" "$CTX/ros2"
  cp "$HERE"/image/ros2/Containerfile "$CTX/Dockerfile"; cp "$HERE"/app/*.py "$CTX/app/"
  cp -r "$HERE/app/ignored/assets" "$CTX/app/ignored/"; cp -r "$HERE/ros2/assembly_cell_ros" "$CTX/ros2/"
  oc start-build robot-ros2 -n "$NS" --from-dir="$CTX" --follow --wait | tail -3; rm -rf "$CTX"
fi
IMG="$(oc get is robot-ros2 -n "$NS" -o jsonpath='{.status.dockerImageRepository}'):latest"
JOB="ros-episode-$SEED-$(date +%H%M%S)"
cat <<YAML | oc apply -n "$NS" -f -
apiVersion: batch/v1
kind: Job
metadata: {name: $JOB, labels: {app: robot-ros2}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata: {labels: {app: robot-ros2}}
    spec:
      restartPolicy: Never
      affinity:                      # keep off the KubeRay CPU nodes (in use for other work) and the GPU node
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - {key: workload, operator: NotIn, values: [ray-cpu]}
              - {key: nvidia.com/gpu.present, operator: DoesNotExist}
      volumes: [{name: data, emptyDir: {}}, {name: shm, emptyDir: {medium: Memory, sizeLimit: 1Gi}}]
      containers:
      - name: cell
        image: $IMG
        command: ["/ros_entry.sh", "bash", "-c"]
        args:
        - |
          ros2 bag record -o /data/bag_seed$SEED /joint_states /cell/task_state /cell/command /cell/student_decision \
            /cell/part_id /cell/skill_advice /cell/done /cell/inspection/scan > /data/bag.log 2>&1 &
          BAG=\$!; sleep 3
          ros2 launch assembly_cell_ros assembly_cell.launch.py seed:=$SEED parts:=$PARTS checkpoint:=$CKPT \
            dgemma_url:=$DGEMMA_URL record_dir:=/data 2>&1 | tee /data/launch.log
          kill -TERM \$BAG; wait \$BAG || true; sleep 2   # SIGTERM: a backgrounded job in a non-interactive shell ignores SIGINT, so the MCAP never closed
          python3 - <<'PY'
          import os, boto3
          s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                            aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name="us-east-1")
          for root, _, files in os.walk("/data"):
              for f in files:
                  p = os.path.join(root, f); k = "ros2-episodes/$JOB/" + os.path.relpath(p, "/data")
                  s3.upload_file(p, "robot", k); print("uploaded", k)
          PY
        env:
        - {name: S3_ENDPOINT_URL, value: "http://minio.minio.svc:9000"}
        - {name: S3_ACCESS_KEY_ID, valueFrom: {secretKeyRef: {name: minio-creds, key: access_key}}}
        - {name: S3_SECRET_ACCESS_KEY, valueFrom: {secretKeyRef: {name: minio-creds, key: secret_key}}}
        resources: {requests: {cpu: "1500m", memory: 3Gi}, limits: {cpu: "3", memory: 6Gi}}
        volumeMounts: [{name: data, mountPath: /data}, {name: shm, mountPath: /dev/shm}]
YAML
echo "job: $JOB   logs: oc logs -n $NS -f job/$JOB   results: s3://robot/ros2-episodes/$JOB/"
