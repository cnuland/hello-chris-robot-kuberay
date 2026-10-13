#!/usr/bin/env bash
# CPU pool for the MuJoCo teacher workers (MuJoCo is CPU-bound like PyBoy was): clone the first ARO worker MachineSet,
# Standard_D32s_v3 x N, label workload=ray-cpu. Usage: ./cpu-machineset.sh [vmSize] [replicas]
set -euo pipefail
VM="${1:-Standard_D32s_v3}"; REPLICAS="${2:-2}"
SRC="$(oc get machineset -n openshift-machine-api -o name | grep worker | head -1)"
NAME="$(oc get "$SRC" -n openshift-machine-api -o jsonpath='{.metadata.name}' | sed 's/-worker-.*//')-ray-cpu"
oc get "$SRC" -n openshift-machine-api -o json | jq --arg n "$NAME" --arg vm "$VM" --argjson r "$REPLICAS" '
  del(.status, .metadata.uid, .metadata.resourceVersion, .metadata.creationTimestamp, .metadata.generation, .metadata.annotations)
  | .metadata.name = $n | .spec.replicas = $r
  | .spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"] = $n
  | .spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"] = $n
  | .spec.template.spec.metadata.labels = {"workload":"ray-cpu"}
  | .spec.template.spec.providerSpec.value.vmSize = $vm
  | .spec.template.spec.providerSpec.value.osDisk.diskSizeGB = 128' | oc apply -f -
oc get machineset "$NAME" -n openshift-machine-api
