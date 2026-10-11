#!/bin/bash
# tf-cuda-test fn-native on both ranks, bare hosts: the native forward against a capture's pack and reference reply.
source "$(dirname "$0")/common.sh"
need TF_WORKER TF_MASTER TF_CACHE TF_KERNELS TF_BIN CAP
WBIN=${TF_WORKER_BIN:-$TF_BIN}
TAG=${TAG:-}
logs_dir
ssh "$W" "mkdir -p $(dirname "$WBIN")"
scp -q "$TF_BIN" "$W:$WBIN"
rsync -a --delete "$TF_KERNELS/" "$W:$TF_KERNELS/"
PIDF=$LOGS/native$TAG.pid
attempt() {
  drop_caches
  timeout "$LIMIT" env $HOST_ENV ${PRE:-} "$TF_BIN" fn-native $(native_args 0) > "$LOGS/native${TAG}_rank0.log" 2>&1 &
  local r0=$!
  ssh "$W" "echo \$\$ > $PIDF; exec env $HOST_ENV timeout $LIMIT $WBIN fn-native $(native_args 1) > $LOGS/native${TAG}_rank1.log 2>&1" &
  local r1=$!
  wait_pair $r0 $r1 "kill $r0 2>/dev/null" "ssh $W 'kill \$(cat $PIDF) 2>/dev/null'"
}
# RETRIES: run again when NCCL's first gather fails to register memory (a transient right after a rank frees pages)
for i in $(seq 1 "${RETRIES:-1}"); do
  attempt
  if ! grep -q "warm gather" "$LOGS/native${TAG}_rank0.log" &&
     ! ssh "$W" "grep -q 'warm gather' $LOGS/native${TAG}_rank1.log"; then break; fi
  echo "attempt $i: NCCL registration failed"; sleep 5
done
show_logs "native$TAG" 16 8
