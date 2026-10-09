#!/bin/bash
# A copy of the Python image's tensorfold package with the dev hooks (TF_RECORD launch logs, TF_FIXTURES bit fixtures).
set -eu
if [ $# -ne 1 ]; then echo "usage: hooked_tree.sh <out dir>   (env: TF_PY_IMAGE, or TF_SRC a tensorfold package dir)" >&2; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1
rm -rf "$OUT"
if [ -n "${TF_SRC:-}" ]; then
  cp -a "$TF_SRC" "$OUT"
else
  [ -n "${TF_PY_IMAGE:-}" ] || { echo "TF_PY_IMAGE is not set: the Python TensorFold image" >&2; exit 2; }
  PKG=$(docker run --rm --entrypoint python3 "$TF_PY_IMAGE" -c "import tensorfold, os; print(os.path.dirname(tensorfold.__file__))")
  ID=$(docker create "$TF_PY_IMAGE")
  docker cp "$ID:$PKG" "$OUT"
  docker rm "$ID" >/dev/null
  echo "mount $OUT over $PKG (docker run -v $OUT:$PKG:ro) on both ranks"
fi
cp "$HERE/hooks/_fixture_hook.py" "$HERE/hooks/_record_hook.py" "$OUT/"
cp "$HERE/../../triton_aot_manifest.py" "$OUT/_launch_recorder.py"
python3 - "$OUT/__init__.py" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = p.read_text()
hook = ('import os as _os\nif _os.environ.get("TF_RECORD"):\n    from ._record_hook import install as _install\n'
        '    _install(_os.environ["TF_RECORD"])\nif _os.environ.get("TF_FIXTURES"):\n'
        '    from ._fixture_hook import install as _install_fixtures\n    _install_fixtures(_os.environ["TF_FIXTURES"])\n')
if "_fixture_hook" not in s:
    lines = s.splitlines(keepends=True)
    at = next((i + 1 for i, line in enumerate(lines) if line.startswith("__version__")), len(lines))
    p.write_text("".join(lines[:at]) + "\n" + hook + "".join(lines[at:]))
PY
echo "hooked tree in $OUT"
