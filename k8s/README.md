# Cluster pieces (same shape as hello-chris-dd-kuberay)

* `ray/raycluster-hand.yaml` - RayCluster for the teacher workers (CPU; MuJoCo is CPU-bound like PyBoy was).
* `ray/submit-decision-run.sh` - stages `app/` (+ assets, no venv) into the head pod and submits the job, same as
  the Double Dragon script; `RUN_TAG=vNNN k8s/ray/submit-decision-run.sh --num-envs 20 --stage lift`.
* `student/` - the student policy service (vLLM for a VLA, or a small torch server); the runner reaches it via
  `STUDENT_ENDPOINTS`. For local runs port-forward it to 127.0.0.1:18011 and keep a watchdog on the port-forward
  (the dd runs stalled silently when it died).
* `minio/` - fixtures, DAgger logs and run summaries, as in the dd repo.

TODO: copy the RHOAI / Kueue / MinIO manifests from hello-chris-dd-kuberay when the project starts.
