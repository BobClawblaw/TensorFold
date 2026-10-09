#!/bin/bash
# Shared settings and helpers for the two-rank scripts beside it (sourced): rank 0 here, rank 1 over ssh.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FN=$(dirname "$HERE")
ZIG_TOOLS=$(dirname "$FN")

describe() {
  case $1 in
    TF_WORKER) echo "the ssh login of rank 1's host, e.g. user@host" ;;
    TF_MASTER) echo "rank 0's address as rank 1 reaches it (the fabric interface)" ;;
    TF_CACHE) echo "the host folder mounted as /cache/tf (model folders and captures), the same path on both hosts" ;;
    TF_MODEL) echo "the converted model's folder name under TF_CACHE" ;;
    TF_PY_IMAGE) echo "the Python TensorFold image the captures run in" ;;
    TF_NATIVE_IMAGE) echo "the native TensorFold image" ;;
    TF_KERNELS) echo "the native kernel set folder (aot.json and cubins), the same path on both hosts" ;;
    TF_BIN) echo "the tf-cuda-test binary (zig-out/bin/tf-cuda-test)" ;;
    TF_NATIVE_BIN) echo "the tensorfold-native binary (zig-out/native/bin/tensorfold-native)" ;;
    CAP) echo "the capture folder's name under TF_CACHE" ;;
    *) echo "required" ;;
  esac
}

need() {
  local v
  for v in "$@"; do
    if [ -z "${!v:-}" ]; then echo "$v is not set: $(describe "$v")" >&2; exit 2; fi
  done
}

W=${TF_WORKER:-}
LOGS=${TF_LOGS:-/tmp/tf-flashnext-logs}
LIMIT=${LIMIT:-600}
# NCCL's interface and HCA come from the environment only; NCCL_DEBUG defaults to WARN
NCCL_ENV="NCCL_DEBUG=${NCCL_DEBUG:-WARN}${NCCL_SOCKET_IFNAME:+ NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME}${NCCL_IB_HCA:+ NCCL_IB_HCA=$NCCL_IB_HCA}"
HOST_ENV="${TF_LIB:+LD_LIBRARY_PATH=$TF_LIB }$NCCL_ENV${XENV:+ $XENV}"
DOCKER_ENV=""
for kv in $NCCL_ENV ${XENV:-}; do DOCKER_ENV="$DOCKER_ENV -e $kv"; done

both() {
  bash -c "$1"
  ssh "$W" "$1"
}

logs_dir() {
  both "mkdir -p $LOGS"
}

drop_caches() {
  [ "${TF_DROP_CACHES:-1}" = 0 ] && return 0
  both "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null" \
    2>/dev/null || echo "note: page cache not dropped (needs passwordless sudo; TF_DROP_CACHES=0 skips it)" >&2
}

stop_server() {
  [ -z "${TF_STOP_CMD:-}" ] && return 0
  bash -c "$TF_STOP_CMD" >/dev/null 2>&1
  sleep "${TF_SETTLE:-45}"
}

show_logs() {
  echo "--- rank 0"; grep -vE "^\s*$|socketProgress|Spectrum" "$LOGS/$1_rank0.log" | tail -"${2:-16}"
  echo "--- rank 1"; ssh "$W" "grep -vE '^\s*$|socketProgress|Spectrum' $LOGS/$1_rank1.log | tail -${3:-8}"
}

native_args() {
  local r=$1 c=$TF_CACHE/$CAP/rank$1 w ref
  w=${CKPT:+ckpt:$CKPT}
  ref=$c/reference.json
  [ -n "${REF:-}" ] && ref=$REF/rank$r/reference.json
  echo "$r $TF_MASTER ${TF_PORT:-29610} $TF_KERNELS ${w:-$c/weights.bin} $c/weights.json $c/ngram.json $TF_CACHE $ref ${MORE:-}"
}

wait_pair() {
  local r0=$1 r1=$2 stop0=$3 stop1=$4 rc
  while kill -0 "$r0" 2>/dev/null && kill -0 "$r1" 2>/dev/null; do sleep 2; done
  if ! kill -0 "$r0" 2>/dev/null; then wait "$r0"; rc=$?; [ $rc != 0 ] && eval "$stop1"; fi
  if ! kill -0 "$r1" 2>/dev/null; then wait "$r1"; rc=$?; [ $rc != 0 ] && eval "$stop0"; fi
  wait
}
