# Flash Next native port: capture, reference and check tools

The tools behind the native (Zig) Flash Next engine's tests on two GB10 hosts at TP=2: captures from the Python
TensorFold engine, references the `tf-cuda-test fn-*` commands compare against, launchers for those commands on both
ranks, and black-box checks against a running server. Nothing here is built into the engine.

## Layout

| Folder | What | Runs where |
| --- | --- | --- |
| `capture/` | Python-engine captures (`capture_*.py`, `first_tokens.py`, `ref_kv.py`, `bench_py.py`), `capture_trace.py`, `request.json` (the reference prompt), dev hooks | Python TensorFold image, both ranks |
| `ref/` | Single-host references for the vision and video paths | Python TensorFold image |
| `program/` | Program generators and trace analysis | any host, stdlib Python |
| `run/` | Two-rank launchers (`capture.sh`, `fn_uid.sh`, `native.sh`, `native_docker.sh`, `serve_native.sh`), `common.sh` | rank 0's host |
| `checks/` | OpenAI-API checks against a server | any host, stdlib Python (`grammar_cmp.py py` needs `xgrammar`, `transformers`) |

Related tools one level up: `tools/zig/triton_aot_manifest.py` (launch recorder and manifest builder),
`tools/zig/flashnext_capture_multi.py` (shared-rounds kernels), `tools/zig/flashnext_aot_pack_dtypes.py` (packer
wrapper), and `zig/tests/cuda/nemotron/aot_pack.py` (the packer).

## Two-rank setup

- Two GB10 hosts (DGX Spark class) joined by their ConnectX ports; rank 0 is the host you run the scripts on, rank 1
  is reached with passwordless `ssh $TF_WORKER`.
- Docker with the NVIDIA runtime on both; the images present on both hosts.
- `TF_CACHE`, `TF_KERNELS`, `TF_LOGS`, `TF_BIN`, `TF_LIB` are the same paths on both hosts (the scripts copy the
  binary and the kernel set to rank 1; captures and model folders must already be there).
- NCCL's interface and HCA come only from the environment (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`); they are passed to
  both ranks and into the containers. `NCCL_DEBUG` defaults to `WARN`.
- Memory is unified: stop any server on the GPUs first (`TF_STOP_CMD`, then `TF_SETTLE` seconds, default 45). The
  native runs drop the page cache first (needs passwordless `sudo`; `TF_DROP_CACHES=0` skips it), since NCCL's first
  memory registration can fail right after a rank frees its pages; `RETRIES=n` reruns `native.sh` on that failure.
- Do not run these beside a serving rank on the same GPU.

## Environment

A script stops with a message naming any required variable that is missing. There are no host-specific defaults.

| Variable | Meaning | Used by |
| --- | --- | --- |
| `TF_WORKER` | ssh login of rank 1's host | all `run/` scripts |
| `TF_MASTER` | rank 0's address as rank 1 reaches it | `run/`, engine captures (inside the containers) |
| `TF_PORT` | rendezvous port (defaults: captures 29571-29573, `fn-native` 29610, serve 29611) | captures, `native*.sh`, `serve_native.sh` |
| `TF_CACHE` | host folder mounted as `/cache/tf`: model folders and capture folders | `capture.sh`, `native*.sh`, `serve_native.sh` |
| `TF_MODEL` | the model folder's name under `TF_CACHE` | `capture.sh`, `serve_native.sh` |
| `CAP` | capture folder name under `TF_CACHE` (`capture.sh` defaults to the script's name) | `capture.sh`, `native*.sh` |
| `TF_PY_IMAGE` | the Python TensorFold image | `capture.sh`, `capture/hooked_tree.sh` |
| `TF_NATIVE_IMAGE` | the native TensorFold image | `native_docker.sh` |
| `TF_BIN` | `tf-cuda-test` (`zig build` output `zig-out/bin/tf-cuda-test`) | `fn_uid.sh`, `native*.sh` |
| `TF_NATIVE_BIN` | `tensorfold-native` (`zig-out/native/bin/tensorfold-native`) | `serve_native.sh` |
| `TF_WORKER_BIN` | where rank 1's copy of the binary goes (default: the same path) | `fn_uid.sh`, `native.sh`, `serve_native.sh` |
| `TF_KERNELS` | the native kernel set (`aot.json` and cubins) | `native*.sh`, `serve_native.sh` |
| `TF_LIB` | folder put on `LD_LIBRARY_PATH` (e.g. the NCCL the image uses), optional | bare-host runs, `native_docker.sh` |
| `TF_LOGS` | log folder on both hosts (default `/tmp/tf-flashnext-logs`) | `run/` |
| `TF_HF_HOME` | Hugging Face cache mounted read-only, optional | `capture.sh` |
| `TF_REQUEST` | reference request (default `capture/request.json`) | `capture.sh` |
| `TF_REF_DIR` | a `vision_ref.py` output folder, copied beside the capture and passed last | `capture.sh` |
| `TF_STOP_CMD`, `TF_SETTLE` | command that frees the GPUs before a run, and the wait after it | `capture.sh`, `fn_uid.sh` |
| `TF_DROP_CACHES` | `0` skips the page-cache drop (`fn_uid.sh`: `1` enables it) | `run/` |
| `TF_CONTAINER` | container name (defaults `fn-capture`, `fn-native-test`) | `capture.sh`, `native_docker.sh` |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_DEBUG` | passed through | `run/` |
| `CAP_KV` | Python engine cache dtype, `int8` (default) or `int4` | engine captures |
| `CAP_RECORD` | `capture_fn.py`: `1` launch log, `2` full trace and the dumps the programs need | `capture_fn.py` |
| `CAP_SWEEP`, `CAP_DEC_STEPS`, `SKIP_PACK` | `capture_aot.py` prompt-length sweep; traced decode steps; skip the weight pack | captures |
| `RESET_DUMP` | the state dump a program starts from: `reset_storages` (default) or `pre_prefill` | `program/gen_*.py` |
| `LIMIT`, `TAG`, `PRE`, `XENV` | time limit (s), log name suffix, a command prefix for rank 0, extra `K=V` env | `run/` |
| `REF`, `CKPT`, `MORE` | `fn-native`: another reference root, `ckpt:<dir>` weights, trailing words | `native*.sh` |
| `NSYS_OUT`, `TF_NSYS`, `TF_NSYS_DIR` | rank 0 under Nsight Systems: output, nsys binary, its install folder | `native_docker.sh` |
| `TF_SRC` | a tensorfold package folder to hook instead of the image's | `capture/hooked_tree.sh` |
| `CONTEXT`, `PORT`, `MAX_TOKENS`, `TF_HOST`, `TF_SERVED_NAME`, `EXTRA` | serve options | `serve_native.sh` |

## Captures (Python image, both ranks)

`run/capture.sh <script> [args]` runs a capture on both ranks (rank 1 detached on the worker, rank 0 here) as
`<script> <rank> /cache/tf/$CAP /cache/tf/$TF_MODEL [args]`; `@R` in the arguments becomes the rank. It copies the
script, `capture_trace.py`, `triton_aot_manifest.py` and the request into `$TF_CACHE/$CAP` on both hosts, and hands
the outputs back to the calling user. `CAP_CLEAN=1` empties the folder first. Outputs land in `$CAP/rank<r>`.

| Script | Produces | Consumed by |
| --- | --- | --- |
| `capture_fn.py` (`CAP_RECORD=2`) | `trace.json`, `named.json`, `reset_storages.*`, `pre_prefill.*`, `after_prefill.*`, `steps/`, `dec/`, `chunk/`, `prefill_inputs/`, `ngram.json`, `reference.json`, `weights.bin/json` | `gen_program.py`; `fn-run`, `fn-check`, `fn-gen` |
| `capture_mtp.py` | `trace.json`, `named.json`, `marks.json`, `pre_prefill.*`, `uploads/`, `ngram.json`, `draft_ids.bin`, `reference.json` (logged MTP rounds), `weights.bin/json` | `gen_mtp.py`; `fn-mtp`; `fn-native` |
| `capture_aot.py` | `launches.json`, `jit.json`: every Triton specialization the served settings launch | the kernel set |
| `capture_vision.py` (with `TF_REF_DIR`) | `launches.json`, `jit.json`, `references.json` (image prompts' replies) | the kernel set; `fn-native ... vision=<capture dir>` |
| `../../flashnext_capture_multi.py` | `launches.json`, `jit.json` for the shared rounds (2..16 streams) | the kernel set |
| `ref_kv.py <prompt reference.json> <int8\|int4>` | `reference.json` at another cache dtype | `fn-native` with `REF` (and `MORE=kv4`) |
| `first_tokens.py <n>...` | `first_tokens.json`: the first token after the reference prompt repeated to n tokens | `fn-native`'s trailing lengths |
| `bench_py.py` | Python engine timing with CUDA graphs (log only) | |

The engine scripts also run alone inside the Python image with `TF_MASTER` set; `capture_aot.py` and
`capture_vision.py` import `triton_aot_manifest.py` and `capture_fn.py`/`capture_mtp.py` import `capture_trace.py`
from the capture folder (`capture.sh` puts them there).

### Kernel set

From each kernel capture, per rank: `python3 tools/zig/triton_aot_manifest.py --cache $TF_CACHE/triton --mount
/cache/tf/triton --launches $TF_CACHE/$CAP/rank0/launches.json --out <manifest.json>`; then pack every manifest
together: `python3 tools/zig/flashnext_aot_pack_dtypes.py zig/tests/cuda/nemotron/aot_pack.py --manifest <m1>
--cache <cache1> --manifest <m2> --cache <cache2> ... --jit <jit.json> --out $TF_KERNELS`. `tf-cuda-test fn-aot
$TF_KERNELS` loads the set alone.

### Launch fixtures (`fn-triton`, `fn-ext`)

`capture/hooked_tree.sh <out>` copies the image's `tensorfold` package (or `TF_SRC`) with `hooks/_record_hook.py`,
`hooks/_fixture_hook.py` and `triton_aot_manifest.py` (as `_launch_recorder.py`) added and enabled from
`__init__.py` by environment. Mount it over the package on both ranks of a Python server and set
`TF_FIXTURES=/cache/tf/<dir>` (optionally `TF_FIXTURE_LIMIT` bytes, default 1 GiB). Once the server is up, create
`<dir>/GO` on both hosts and send `capture/request.json`; each Triton specialization's first launch is saved as
`<dir>/<kernel>-<hash>/` and up to three shapes per extension call as `<dir>/ext-<module>.<fn>-<n>/`. Then
`tf-cuda-test fn-triton <that dir>` and `fn-ext <that dir>` replay them bit for bit. `TF_RECORD=<dir>` instead
writes every launch to `launches-<host>-<pid>.json` every 20 s.

## References without the engine (`ref/`, Python image, one host)

- `vision_ref.py <model dir> <out> [cpu|cuda] [png folder]`: drawn test images (or a folder of PNGs with an optional
  `questions.json`), each prompt's tokens, M-RoPE positions, patches and tower features. Compared by
  `tf-cuda-test vision-prep <out>` and `fn-vision <model dir> <out> <kernels>`; with `cuda` it is also the
  `TF_REF_DIR` for `capture_vision.py`. `VISION_FP32=1` runs the tower in fp32.
- `video_ref.py <model dir> <out>`: a synthetic `test.mp4`, its decoded frames, patches, grid and expanded prompt.
  Compared by `tf-cuda-test video-prep <out>`.

## Programs (`program/`, stdlib)

- `gen_program.py <capture rank dir> <pack rank dir> <out.json>`: `capture_fn.py`'s trace as a rank's program
  (storages, temporaries, decode step and commit at each DeltaNet parity, the chunk and the prefill). Use
  `RESET_DUMP=pre_prefill` for `fn-gen`.
- `gen_mtp.py <capture rank dir> <pack rank dir> <out.json>`: `capture_mtp.py`'s trace, one program per traced phase
  (verify widths, commits, MTP head widths) with their marks. Use `RESET_DUMP=pre_prefill`.
- `kernel_writes.py <Triton cache dir> <program.json>`: adds each kernel's written parameters (from its TTIR).
- `diff_steps.py <capture rank dir> <phase>,<phase>...` with `trace_lib.py`: which decode-step arguments change.

## Native runs (`run/`)

`fn_uid.sh <command> '<template>'` runs a program command on both bare hosts: rank 0 writes NCCL's unique id to
`@U`, which is copied to rank 1. With `P` the programs folder and `C=$TF_CACHE`:

```sh
run/fn_uid.sh fn-run   "$P/rank@R.json $C/fn/rank@R/weights.bin $C/fn/rank@R/reset_storages.bin $C/fn/rank@R/steps @R @U 16"
run/fn_uid.sh fn-check "$P/rank@R.json $C/fn/rank@R/weights.bin $C/fn/rank@R/reset_storages.bin $C/fn/rank@R/chunk @R @U"
run/fn_uid.sh fn-gen   "$P/rank@R.json $C/fn/rank@R/weights.bin $C/fn/rank@R/pre_prefill.bin $C/fn/rank@R/ngram.json \
  $C/fn/rank@R/reference.json @R @U $C/fn/rank@R/prefill_inputs"
run/fn_uid.sh fn-mtp   "$P/mtp_rank@R.json $C/mtp/rank@R/weights.bin $C/mtp/rank@R/pre_prefill.bin $C/mtp/rank@R/ngram.json \
  $C/mtp/rank@R/reference.json $C/mtp/rank@R/draft_ids.bin @R @U"
```

`native.sh` (bare hosts) and `native_docker.sh` (each rank in `TF_NATIVE_IMAGE`, as served) run
`tf-cuda-test fn-native <rank> $TF_MASTER <port> $TF_KERNELS <weights.bin> <weights.json> <ngram.json> $TF_CACHE
<reference.json> [words]` with `CAP` a `capture_mtp.py` folder: the native forward's MTP reply must equal Python's.
`MORE` adds trailing words, for example prompt lengths (`first_tokens.py`), `kv4`, `rows=<n>`, `nographs`, `multi=on|off|gdn`,
`shared`, `resume`, `vision=<capture_vision folder>`, `sync`. Either rank failing stops the other.

`serve_native.sh` starts `tensorfold-native serve` on both bare hosts from `$TF_CACHE/$TF_MODEL`;
`serve_native.sh stop` ends both.

## Server checks (`checks/`, stdlib)

Each takes a port on this host or a base URL first.

- `sampled_set.py <port|url> <out.json> [draft on|off]`: seeded sampled replies, prompts x rules, for comparing
  engines.
- `structured_set.py`: JSON object and schema, `guided_json`, regex, choice and grammar outputs in greedy, sampled
  and thinking modes, each checked; a broken regex must be refused.
- `turns_check.py`: multi-turn prompt reuse; a resumed turn must equal the same turn sent again.
- `url_check.py`: image URLs in a request: a public https image, a loopback address, a plain-http URL.
- `longctx.py <port|url> <out.json> [tokens...]`: long prompts: time to first token, prefill and decode rates, a
  needle.
- `capacity.py <port|url> <n> <tokens> <out.json>`: n distinct long prompts at once.
- `exhaust_check.py <port|url> <out.json> [waves] [streams] [words] [max_tokens] [timeout]`: concurrent replies
  past the cache budget must end in a reply or the server's refusal, and the server must keep serving.
- `grammar_cmp.py py|shim <tokenizer.json> <out.json> [libtfgrammar.so]`: grammar walks through xgrammar (`py`) or
  the native grammar library (`shim`); the two outputs must be equal.
