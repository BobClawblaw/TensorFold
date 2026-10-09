#!/bin/bash
# A capture script on both ranks in the Python TensorFold image: outputs under $TF_CACHE/$CAP/rank<r> on each host.
source "$(dirname "$0")/common.sh"
if [ $# -lt 1 ]; then
  echo "usage: capture.sh <capture script.py> [arguments after the model dir; @R: the rank]" >&2; exit 2
fi
need TF_WORKER TF_MASTER TF_CACHE TF_MODEL TF_PY_IMAGE
SCRIPT=$1; shift
[ -f "$SCRIPT" ] || SCRIPT=$FN/capture/$SCRIPT
[ -f "$SCRIPT" ] || { echo "no capture script $1" >&2; exit 2; }
CAP=${CAP:-$(basename "$SCRIPT" .py)}
NAME=${TF_CONTAINER:-fn-capture}
REQUEST=${TF_REQUEST:-$FN/capture/request.json}
DIR=$TF_CACHE/$CAP
stop_server
both "docker rm -f $NAME >/dev/null 2>&1"
# the containers write as root: clean and hand back through the image, no sudo needed
AS_ROOT="docker run --rm -v $TF_CACHE:/c --entrypoint"
[ "${CAP_CLEAN:-0}" = 1 ] && both "$AS_ROOT rm $TF_PY_IMAGE -rf /c/$CAP"
both "mkdir -p $DIR"
FILES="$SCRIPT $FN/capture/capture_trace.py $ZIG_TOOLS/triton_aot_manifest.py $REQUEST"
cp $FILES "$DIR/"; scp -q $FILES "$W:$DIR/"
[ "$(basename "$REQUEST")" = request.json ] || both "cp $DIR/$(basename "$REQUEST") $DIR/request.json"
EXTRA=""
if [ -n "${TF_REF_DIR:-}" ]; then
  # a reference folder (vision_ref.py's output) copied beside the capture, passed as the last argument
  both "mkdir -p $DIR/ref"; cp -r "$TF_REF_DIR"/. "$DIR/ref/"; scp -qr "$TF_REF_DIR"/. "$W:$DIR/ref/"
  EXTRA=/cache/tf/$CAP/ref
fi
drop_caches
HF=""
[ -n "${TF_HF_HOME:-}" ] && HF="-v $TF_HF_HOME:/cache/huggingface:ro -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1"
PASS=""
for v in TF_PORT CAP_RECORD SKIP_PACK CAP_DEC_STEPS CAP_SWEEP CAP_KV RESET_DUMP; do
  [ -n "${!v:-}" ] && PASS="$PASS -e $v=${!v}"
done
COMMON="--name $NAME --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 --ulimit core=0 -v $TF_CACHE:/cache/tf $HF -e TENSORFOLD_NO_UPDATE_CHECK=1 \
  -e TORCH_EXTENSIONS_DIR=/cache/tf/torch_extensions -e TRITON_CACHE_DIR=/cache/tf/${TF_TRITON_DIR:-triton} \
  -e TF_MASTER=$TF_MASTER -e CAP_MASTER=$TF_MASTER $PASS $DOCKER_ENV --entrypoint python3 $TF_PY_IMAGE \
  /cache/tf/$CAP/$(basename "$SCRIPT")"
args() { local a="$1 /cache/tf/$CAP /cache/tf/$TF_MODEL ${ARGS//@R/$1} $EXTRA"; echo "$a"; }
ARGS="$*"
ssh "$W" "docker run -d $COMMON $(args 1)" >/dev/null
docker run --rm $COMMON $(args 0) 2>&1 | grep -E "rank|Error|error|Traceback" | grep -v Spectrum
ssh "$W" "docker wait $NAME >/dev/null; docker logs $NAME 2>&1 | grep -E 'rank|Error|Traceback' | grep -v Spectrum | tail -5; \
  docker rm $NAME >/dev/null"
both "$AS_ROOT chown $TF_PY_IMAGE -R \$(id -u):\$(id -g) /c/$CAP"
echo "CAPTURE DONE: $DIR/rank0 here, $DIR/rank1 on $W"
