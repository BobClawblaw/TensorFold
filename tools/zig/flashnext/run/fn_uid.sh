#!/bin/bash
# A tf-cuda-test program command (fn-run, fn-check, fn-gen*, fn-mtp*) on both ranks; rank 0 writes NCCL's unique id.
source "$(dirname "$0")/common.sh"
if [ $# -lt 2 ]; then
  echo "usage: fn_uid.sh <command> <argument template>   (@R: the rank, @U: the unique-id file)" >&2; exit 2
fi
need TF_WORKER TF_BIN
CMD=$1 TEMPLATE=$2
WBIN=${TF_WORKER_BIN:-$TF_BIN}
U=$LOGS/uid.bin
TAG=${TAG:-$CMD}
stop_server
logs_dir
ssh "$W" "mkdir -p $(dirname "$WBIN")"
scp -q "$TF_BIN" "$W:$WBIN"
[ "${TF_DROP_CACHES:-0}" = 1 ] && drop_caches
args() { local a=${TEMPLATE//@R/$1}; echo "${a//@U/$U}"; }
both "rm -f $U"
env $HOST_ENV ${PRE:-} "$TF_BIN" "$CMD" $(args 0) > "$LOGS/${TAG}_rank0.log" 2>&1 &
R0=$!
until [ -s "$U" ]; do
  kill -0 $R0 2>/dev/null || { echo "rank 0 ended before writing the unique id"; show_logs "$TAG" 20 0; exit 1; }
  sleep 0.2
done
scp -q "$U" "$W:$U"
ssh "$W" "env $HOST_ENV $WBIN $CMD $(args 1) > $LOGS/${TAG}_rank1.log 2>&1"
wait
show_logs "$TAG" 12 4
