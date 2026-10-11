#!/bin/bash
# tensorfold-native serve on both ranks, bare hosts (rank 1 on the worker, rank 0 here); "stop" ends both.
source "$(dirname "$0")/common.sh"
need TF_WORKER TF_MASTER TF_CACHE TF_MODEL TF_KERNELS TF_NATIVE_BIN
WBIN=${TF_WORKER_BIN:-$TF_NATIVE_BIN}
PIDF=$LOGS/serve.pid
if [ "${1:-}" = stop ]; then
  both "[ -f $PIDF ] && kill \$(cat $PIDF) 2>/dev/null; rm -f $PIDF"
  exit 0
fi
logs_dir
ssh "$W" "mkdir -p $(dirname "$WBIN")"
scp -q "$TF_NATIVE_BIN" "$W:$WBIN"
rsync -a --delete "$TF_KERNELS/" "$W:$TF_KERNELS/"
drop_caches
ENVV="$HOST_ENV TENSORFOLD_CUDA_KERNELS=$TF_KERNELS"
COMMON="--context ${CONTEXT:-16384} --no-update-check --tp 2 --master $TF_MASTER --master-port ${TF_PORT:-29611} ${EXTRA:-}"
MODEL=$TF_CACHE/$TF_MODEL
ssh "$W" "nohup env $ENVV $WBIN serve $COMMON --rank 1 $MODEL > $LOGS/serve_rank1.log 2>&1 & echo \$! > $PIDF"
nohup env $ENVV "$TF_NATIVE_BIN" serve $COMMON --rank 0 --name "${TF_SERVED_NAME:-flashnext-native}" \
  --host "${TF_HOST:-0.0.0.0}" --port "${PORT:-8000}" --max-tokens "${MAX_TOKENS:-512}" "$MODEL" \
  > "$LOGS/serve_rank0.log" 2>&1 &
echo $! > "$PIDF"
echo "started; logs $LOGS/serve_rank{0,1}.log"
