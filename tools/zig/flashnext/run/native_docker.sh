#!/bin/bash
# native.sh with each rank in a container of the native image (memlock unlimited, IPC_LOCK, InfiniBand: as served).
source "$(dirname "$0")/common.sh"
need TF_WORKER TF_MASTER TF_CACHE TF_KERNELS TF_BIN TF_NATIVE_IMAGE CAP
TAG=${TAG:-}
NAME=${TF_CONTAINER:-fn-native-test}
logs_dir
ssh "$W" "mkdir -p $(dirname "$TF_BIN")"
scp -q "$TF_BIN" "$W:$TF_BIN"
rsync -a --delete "$TF_KERNELS/" "$W:$TF_KERNELS/"
# the folders the run reads and writes, each mounted at its own path
MOUNTS="-v $TF_CACHE:$TF_CACHE -v $TF_KERNELS:$TF_KERNELS:ro -v $(dirname "$TF_BIN"):$(dirname "$TF_BIN"):ro -v $LOGS:$LOGS"
[ -n "${TF_LIB:-}" ] && MOUNTS="$MOUNTS -v $TF_LIB:$TF_LIB:ro -e LD_LIBRARY_PATH=$TF_LIB"
[ -n "${REF:-}" ] && MOUNTS="$MOUNTS -v $REF:$REF:ro"
[ -n "${CKPT:-}" ] && MOUNTS="$MOUNTS -v $CKPT:$CKPT:ro"
BASE="docker run --rm --name $NAME --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 --ulimit core=0 $MOUNTS $DOCKER_ENV"
both "docker rm -f $NAME >/dev/null 2>&1"
drop_caches
if [ -n "${NSYS_OUT:-}" ]; then
  # rank 0 under Nsight Systems: TF_NSYS is the host's nsys binary, its install folder TF_NSYS_DIR mounted read-only
  need TF_NSYS
  NSD=${TF_NSYS_DIR:-$(dirname "$(dirname "$(dirname "$TF_NSYS")")")}
  timeout "$LIMIT" $BASE --cap-add SYS_ADMIN -v "$NSD:$NSD:ro" -v "$(dirname "$NSYS_OUT"):$(dirname "$NSYS_OUT")" \
    --entrypoint "$TF_NSYS" "$TF_NATIVE_IMAGE" profile -t cuda -o "$NSYS_OUT" --force-overwrite true "$TF_BIN" \
    fn-native $(native_args 0) > "$LOGS/native${TAG}_rank0.log" 2>&1 &
else
  timeout "$LIMIT" $BASE --entrypoint "$TF_BIN" "$TF_NATIVE_IMAGE" fn-native $(native_args 0) \
    > "$LOGS/native${TAG}_rank0.log" 2>&1 &
fi
R0=$!
timeout "$LIMIT" ssh "$W" "timeout $LIMIT $BASE --entrypoint $TF_BIN $TF_NATIVE_IMAGE fn-native $(native_args 1) \
  > $LOGS/native${TAG}_rank1.log 2>&1" &
R1=$!
wait_pair $R0 $R1 "docker rm -f $NAME >/dev/null 2>&1" "ssh $W 'docker rm -f $NAME >/dev/null 2>&1'"
show_logs "native$TAG" 16 8
