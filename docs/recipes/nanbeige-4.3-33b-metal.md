# Nanbeige4.3-33B, native Metal

This recipe covers text generation with [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)
(model_type `llama`: a dense 33B Llama-architecture model, 512K context, thinking and tool calls in its chat template)
on the Metal engine, served from the bf16 release as published: no conversion, nothing lost. It was brought up on an
M5 Ultra (256 GB); other Macs are untested. The family admits only the checkpoints checked against Hugging Face, by the
geometry their configs name (`qualified` in `zig/src/families/llama/config.zig`: Nanbeige4.3-33B); another `llama`
checkpoint is refused by name until it has been checked the same way. Prompts run on Metal 4 tensor units, so it needs
an M5 or later: M1 to M4 Macs are refused by name at load.

## Build and serve

```sh
hf download Nanbeige/Nanbeige4.3-33B --revision c1093c501cc6460427c0ac634db7223d0450c0ee \
  --local-dir models/nanbeige4.3-33b
zig build native -Doptimize=ReleaseFast -Dcpu=native
zig build grammar -Dxgrammar=PATH_TO_XGRAMMAR_v0.2.8 -Dcpu=native   # structured output
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b --name nanbeige --port 8888 \
  --context 131072 --parallel 2 --prompt-cache-gib 8
```

`LLAMA_LOAD_LIMIT_GB` caps weights, scratch and caches together (75% of RAM when unset). The weights take 61.7 GiB; a
stream's caches take 208 KiB a position (every one of the 52 layers keeps bf16 keys and values), so a 131,072-token
stream holds 26 GiB and the full 524,288-token context 104 GiB: 150 GB holds two 131,072-token streams, or one of
262,144; the whole context needs a Mac with the weights and 104 GiB free (192 GB, the 75% default on a 256 GB Mac,
holds one 524,288-token stream). Thinking is on by default, as in the chat template; a request sends
`"chat_template_kwargs": {"enable_thinking": false}` to turn it off. The server forwards only the thinking switch to
the template: earlier turns' reasoning stays out of the prompt (the template's `preserve_thinking` default) and tool
calls take the template's XML form (its `tool_call_format` default), which the server parses.

## What the family is

`zig/src/families/llama/` follows the Spark-X2.6 Metal family, with HF's `modeling_llama` as the reference:

- The reference's rounding points, its PyTorch code in bf16: the residual stays bf16 and each add rounds to it; the
  norm is `bf16(w * bf16(x * rsqrt(mean(x^2) + eps)))` with the mean in fp32; RoPE is `bf16(bf16(x * c) + bf16(o * s))`
  with the fp32 angle's cos and sin rounded to bf16 as HF rounds them, pairs (d, d + 64); the MLP is
  `bf16(bf16(silu(g)) * u)` (`kernels/metal/llama/glue.metal`).
- Projections over the release's bf16 rows, read in place from the mapped shards (no copies, no stacking): prompt
  chunks of up to 2,048 rows on `gemm_nax`'s tensor-unit GEMM (`gemm.metal`); decode windows on `ll_matvec`, a
  simdgroup an output row, fp32 sums rounded once as torch's bf16 linear rounds them. q, k and v write side by side
  into one row, gate and up into another.
- Attention: prompt chunks on the tensor units (Gemma 4's prompt attention over a position-major cache,
  `attention_nax.metal`); decode rows on sdpa_vector lanes over key blocks at absolute positions, so a row's bits
  never depend on the rows beside it.
- The lane host's shared rounds, a lone stream's steps drawn on the GPU a step ahead (argmax, or `tf_sample_full`),
  copy drafts and the prompt cache.

Every kernel is registered with the kernel contract (`zig/tests/contract/metal_llama*.zig`, the roll call in
`contracts.zig`): the glue, the affine embedding at 4 and 8 bits, the six row-projection forms, the four GEMM forms and
the prompt attention at both served geometries. `tf-kernel-contract --only llama` on the M5 Ultra:

```
SUMMARY 23 kernels: 23 pass gates 1-4, 0 fail, 0 known findings; 0 controls: 0 red as planted, 0 missed; 0 limit problems; gates 412040 ms, limits 90 ms
```

## Measured (M5 Ultra, the bf16 release, 26 Oct 2026)

Against Hugging Face's `modeling_llama` on the release (transformers 4.57.1, bf16 on the CPU): 1,500 positions
teacher-forced, mean KL 0.0015, top-1 99.1%; 64 positions after a 9,000-token prompt, mean KL 0.0013, top-1 100%. Six
greedy chats of 64 tokens (code, prose, Chinese, a document, with and without thinking): three equal HF's token for
token; the other three part at a position where the engine's two best logits are the same bf16 value. One-row steps
and windows of 7 or 16 rows give the same logits bit for bit after prompts of 64 and 1,500 tokens.

`tools/zig/llama_teacher_judge.py` judges every teacher-forced position by the repository's near-tie rule
(`qwen27_margins.py`: the same two contenders on both sides, each margin within one bf16 step) and exits 1 on anything
else. Against HF's bf16 run, 13 of the 1,500 positions differ and two are red: at 425 and 1370 three tokens sit within
one bf16 step, so the two sides' contenders differ. `tools/zig/llama_f64_reference.py` runs the release in float64
(its bf16 weights cast up exactly, HF's arithmetic, one rounding at the end) and settles them:

- Against the float64 pass, the engine passes the rule: 1,483 of 1,500 positions equal, all 17 differences near-ties.
  HF's bf16 run does not: 1,488 equal, and at 425 it rounds two contenders a step high (red).
- At 1370 the exact logits are `it` 388 38.1266, `no` 754 38.1037 and `the` 269 38.0106: within 0.116 of each other,
  under half a bf16 step (0.25 at 32-64). 388 lies 0.0016 above the rounding midpoint 38.125, so rounded once it is
  38.25 and the others 38.0. HF's bf16 run gives 754 a step too high (38.25); the engine gives 388 a step too low
  (38.0).
- Layer by layer at 1370 (the engine's forward stopped after each layer count by a local probe, not in this PR; HF's
  `output_hidden_states`), the engine's residual and HF's drift from the float64 one alike (0.3% after the first layer,
  1.8% after the 51st; 0.80% and 0.82% on average over the layers). The engine's largest excess, 1.5-1.7 times HF's at
  layers 20-22, is channel 4086, a massive activation near 427 where a bf16 step is 2: both hold 430 after layer 19,
  and at layer 20 HF's add lands on 428 while the engine's stays at 430.

Every position where the engine's top token differs from HF's or from the float64 pass's (row n predicts token
n + 1; a tie goes to the lower id; logits listed for each side's top-1 tokens in the order of the first three columns):

| position | float64's top-1 | HF's | engine's | float64 (rounded once) | HF | engine | against HF | against float64 |
|---:|---|---|---|---|---|---|---|---|
| 22 | `en` 271 | `en` 271 | `see` 3574 | 20.625, 20.625 | 20.75, 20.625 | 20.625, 20.75 | near-tie | near-tie |
| 64 | `pickle` 82922 | `json` 5174 | `json` 5174 | 31.625, 31.5 | 31.625, 31.625 | 31.75, 31.75 | equal | near-tie |
| 83 | `the` 269 | `the` 269 | `Python` 9120 | 26.25, 26.25 | 26.25, 26.25 | 26.25, 26.375 | near-tie | near-tie |
| 85 | `developed` 3260 | `developed` 3260 | `-` 152369 | 35.5, 35.5 | 35.5, 35.5 | 35.5, 35.75 | near-tie | near-tie |
| 166 | `"))` 9441 | `";` 2651 | `"))` 9441 | 36.5, 36.25 | 36, 36.25 | 36.5, 36.25 | near-tie | equal |
| 171 | `\"` 12028 | `[\` 11752 | `[\` 11752 | 31.75, 31.625 | 32, 32.25 | 32.25, 32.25 | equal | near-tie |
| 224 | `('` 1091 | `(` 152374 | `('` 1091 | 39.75, 39.75 | 39.5, 39.75 | 39.5, 39.5 | near-tie | equal |
| 235 | `.` 152361 | `,` 152363 | `,` 152363 | 45.5, 45.5 | 45.25, 45.5 | 45.25, 45.5 | equal | near-tie |
| 357 | `separators` 118153 | `indent` 34489 | `indent` 34489 | 37, 36.75 | 36.75, 36.75 | 36.75, 36.75 | equal | near-tie |
| 401 | `(` 152374 | `(` 152374 | `({'` 50468 | 39.5, 39.25 | 39.5, 39.25 | 39.25, 39.25 | near-tie | near-tie |
| 425 | `1` 152367 | `{` 152405 | `1` 152367 | 39.75, 39.75 | 39.75, 40 | 40, 40 | bug | equal |
| 586 | `encoding` 13571 | `representation` 7443 | `encoding` 13571 | 29.75, 29.625 | 30, 30 | 30, 29.875 | near-tie | equal |
| 1234 | `file` 2408 | `a` 152346 | `a` 152346 | 35, 35 | 35, 35.25 | 35, 35.25 | equal | near-tie |
| 1295 | `an` 284 | `a` 261 | `a` 261 | 44, 43.75 | 43.75, 43.75 | 43.75, 43.75 | equal | near-tie |
| 1302 | `If` 3934 | `If` 3934 | `` `` `` 2283 | 32.25, 32 | 32, 31.875 | 32, 32 | near-tie | near-tie |
| 1304 | `indent` 28799 | `indent` 28799 | `allow` 13471 | 37.25, 37 | 37.25, 37 | 37, 37 | near-tie | near-tie |
| 1352 | `the` 269 | `\n` 13 | `\n` 13 | 46.25, 46 | 46, 46 | 46, 46 | equal | near-tie |
| 1370 | `it` 388 | `it` 388 | `the` 269 | 38.25, 38 | 38.25, 38 | 38, 38 | bug | near-tie |
| 1381 | `skipped` 61065 | `disabled` 18191 | `disabled` 18191 | 41.5, 41.25 | 41.25, 41.25 | 41.25, 41.25 | equal | near-tie |
| 1384 | `` `` `` 16411 | `` `` `` 16411 | `circular` 13380 | 30.875, 30.75 | 30.875, 30.75 | 30.75, 30.75 | near-tie | near-tie |
| 1447 | `instead` 62222 | `instead` 62222 | `with` 3629 | 35.5, 35.25 | 35, 34.75 | 34.5, 34.5 | near-tie | near-tie |

A device receipt with no other model server or GPU job on the Mac (28 Oct 2026, this branch's build; a CPU float64
reference run, one layer's weights at a time, overlapped its first minutes; greedy, 128-token replies,
`return_token_ids`, `--context 32768 --parallel 2`): the 1,500-position logits are bit-identical to the ones judged
above, and the six greedy chats to the earlier run; `tf-llama-run ... exact` finds 0 differing rows after the 64- and
1,500-token prompts at windows of 16 and 7 rows, and reports its planted row each time. Six prompts sent two at a time
return the same token ids as each one alone. Three chats whose first turn carries an 8,500-token log: the second turn
resumed from the prompt cache (8,521-8,528 of 8,554-8,568 prompt tokens kept) returns the same ids as on a server
without the cache. No choice differed, so there are no margins to list.

Served through the recipe harness (an idle 97 GiB server of another model was loaded beside it, so memory readings
include it), `--context 131072 --parallel 2`, greedy 256-token replies:

| streams | prose | code | structured | list | first token | a round |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.4 | 16.3 | 16.4 | 16.4 | 0.23-0.26 s | 61.4 ms |
| 2 | 31.8 | 28.0 | 31.7 | 31.7 | 0.34-0.38 s | 66.4 ms |

Tokens a second, all streams together. One stream reads the 61.7 GiB of weights each step; two streams read them once
for both rows. Sampled replies (temperature, top-p, seeds) decode at 16.2-16.3 tokens/s, the same as greedy.

| cold prompt | 2,109 | 8,254 | 32,827 | 61,502 | 131,132 tokens |
|---|---:|---:|---:|---:|---:|
| tokens a second | 1,259 | 1,313 | 1,074 | 866 | 563 |
| time to first token | 1.68 s | 6.29 s | 30.6 s | 71.0 s | 233 s |

A fresh server, each prompt starting with words no earlier one did; the 131,132-token prompt on a
`--context 262144 --parallel 1` server, the others on the 131,072-token one. With thinking on, within 4%. A multi-turn
chat's third turn, resumed from 5,493 of its 5,649 kept tokens, starts in 0.34 s against 4.5 s cold, with the same reply.

- Structured output (json_object, json_schema, regex, choice, grammar) 18/18, drafted == plain 12/12; tool calls
  (auto, required, named, thinking, streamed) 5/5; seeded sampling repeats 9/9.
- A 60-turn soak (3 chats of 20 turns beside a decoding stream) and 12 concurrent replies each equal the request
  alone; streaming 4/4, multi-turn recall 8/8, stop strings 3/3, thinking 10/10, max_tokens 3/3.
- A passphrase at 10%, 50% and 90% of a 128,550-token prompt comes back exactly on the 131,072-token server.

## Performance options

Each keeps the replies' bits.

- **Streams (`--parallel`).** One stream decodes at 16.4 tokens/s, bound by reading the weights each step; two give
  28-32 tokens/s in all (66 ms a round against 61), each reply equal to its run alone. The row kernel holds up to 32
  rows a launch; the caches set the count (26 GiB a 131,072-token stream).
- **Context and memory (`--context`, `LLAMA_LOAD_LIMIT_GB`).** 208 KiB a token. Cold prompts run 1.1-1.3k tokens/s
  to 33k tokens, 866 at 61k and 563 at 131k (attention grows with the prompt).
- **Prompt cache (`--prompt-cache-gib`).** A resumed turn starts in 0.34 s against 4.5 s cold; kept states give way to
  streams that need the room.
- **Copy drafts.** On from the lane core, replies equal to drafts off; they accepted 0-30% of their tokens on the
  harness's prompts and gave no measurable decode gain (drafted 16 tokens/s, undrafted 16).
- **Sampling.** Draws run `tf_sample_full` over the whole 166,912-token vocabulary (the 1,024-candidate kernel is
  compiled for a 131,072-token vocabulary): sampled replies decode at the greedy speed.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums; no quantization of our own (the
  MLX quants below are served as published).

## Published MLX quants (8-bit and 4-bit)

Nanbeige publishes no quantized checkpoint. The family reads MLX affine checkpoints as `mlx_lm` writes them, in place:
a uniform `quantization` table (groups of 64, 4 or 8 bits, `affine`), every projection and the embedding and head
quantized, the norms bf16. Decode windows run the core's row kernels (`row_projection.zig`), prompt chunks its
tensor-unit affine matmul (`affine_mm.zig`), the kernels the other affine families use; the embedding's rows come from
`ll_embed_affine`, MLX's dequantize (`bf16(bf16(scale * q) + bias)`). Checked on two community quants, pinned:

```sh
hf download pratikgx/Nanbeige4.3-33B-MLX-4bit --revision 883c8bda53b2062ff7e37f178ccf503c79674797 \
  --local-dir models/nanbeige4.3-33b-4bit
hf download fcmeyer/Nanbeige4.3-33B-MLX-8bit --revision 5b206655a9974094df4258cc2381765ee6329af2 \
  --local-dir models/nanbeige4.3-33b-8bit
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b-8bit --name nanbeige --port 8888 \
  --context 131072 --parallel 2
```

Measured on the M5 Ultra against the same HF reference as above, and against MLX 0.32.4 with `mlx_lm` 0.32.2 running
the same quant:

| | bf16 | 8-bit | 4-bit |
|---|---:|---:|---:|
| mean KL against HF over 1,500 positions, top-1 | 0.0015, 99.1% | 0.00234, 98.7% | 0.0849, 91.1% |
| the same for MLX on the quant | | 0.00272, 98.7% | 0.0851, 90.7% |
| mean KL against MLX on the quant, top-1 | | 0.00255, 98.7% | 0.00166, 98.5% |
| greedy chats equal to MLX's on the quant (64 tokens, 6 chats) | | 2 | 3 |
| greedy chats equal to HF's bf16 (64 tokens, 6 chats) | 3 | 4 | 0 (all part within 16 tokens) |
| decode, 1 stream / 2 streams | 16.4 / 28-32 tok/s | 27.9-28.4 / 49.2-51.5 tok/s | 46.0-46.7 / 76.5-79.3 tok/s |
| cold prompt 2.1k / 8.3k / 33k / 61k tokens | 1,259 / 1,313 / 1,074 / 866 tok/s | 1,268 / 1,253 / 1,022 / 832 tok/s | 1,369 / 1,348 / 1,080 / 872 tok/s |

Speeds come from the same harness plans as the bf16 rows (greedy 256-token replies, all streams together; the range
covers the four prompts), measured beside an idle server of another model. The quants are lossy, so the release stays
the recipe's default; the 4-bit one costs a mean KL of 0.0849 and parts from the release's greedy text within 16
tokens on every chat, and is offered for its speed with that said.

### Every differing choice on the quants

`tools/zig/llama_teacher_judge.py` (the near-tie rule above) judges the engine's 1,500 teacher-forced positions on each
quant against three references on the same files: MLX's one pass over all positions (`mlx_lm`'s prompt arithmetic),
MLX's one-token decode steps (the arithmetic its greedy replies use) and `tools/zig/llama_f64_reference.py`, which reads
the quant's weights as scale * q + bias exactly and rounds once at the end. The engine's own windows equal its one-row
steps bit for bit on both quants.

On the 8-bit quant:

- against MLX one pass: 1481 of 1,500 equal, 19 differ, red at [315, 1370, 1447, 1461]
- against MLX decode: 1484 of 1,500 equal, 16 differ, red at [401, 425]
- against float64: 1478 of 1,500 equal, 22 differ, red at [425, 1370, 1447]

| position | float64's top-1 | engine's | MLX one pass | MLX decode | float64 | engine | against MLX one pass | against MLX decode | against float64 |
|---:|---|---|---|---|---|---|---|---|---|
| 13 | `/>` 7517 | `>` 152426 | 34.5, 34.75 | 34.5, 34.75 | 34.5, 34.5 | 34.5, 34.75 | equal | equal | near-tie |
| 22 | `en` 271 | `a` 152346 | 20.5, 20.625 | 20.5, 20.5 | 20.5, 20.5 | 20.375, 20.5 | equal | near-tie | near-tie |
| 32 | `)` 152373 | `).` 472 | 40.75, 40.75 | 40.75, 40.75 | 40.75, 40.5 | 40.75, 40.75 | equal | equal | near-tie |
| 71 | `` `. `` 14666 | `` ` `` 152608 | 43.75, 43.75 | 43.75, 43.75 | 43.75, 43.75 | 43.75, 44 | near-tie | near-tie | near-tie |
| 83 | `the` 269 | `Python` 9120 | 26.125, 26.125 | 26.375, 26.375 | 26.125, 26.125 | 26.25, 26.375 | near-tie | near-tie | near-tie |
| 86 | `library` 6746 | `<` 598 | 32.25, 32.25 | 32.25, 32.25 | 32, 31.875 | 32.25, 32.25 | equal | equal | near-tie |
| 166 | `"))` 9441 | `"))` 9441 | 36.25, 36.25 | 36.25, 36 | 36.75, 36.5 | 36.5, 36.25 | near-tie | equal | equal |
| 224 | `('` 1091 | `(` 152374 | 40, 40.25 | 40, 40 | 40, 40 | 40, 40.25 | equal | near-tie | near-tie |
| 235 | `.` 152361 | `,` 152363 | 45.25, 45.5 | 45.25, 45.5 | 45.25, 45.25 | 45, 45.25 | equal | equal | near-tie |
| 315 | `and` 297 | `separators` 118153 | 25.25, 25.25 | 25.5, 25.5 | 25.5, 25.5 | 25.25, 25.375 | bug | near-tie | near-tie |
| 333 | `1` 152367 | `0` 152368 | 35.75, 35.75 | 35.5, 35.75 | 35.75, 35.75 | 35.75, 36 | near-tie | equal | near-tie |
| 357 | `indent` 34489 | `separators` 118153 | 36.75, 36.75 | 36.75, 36.75 | 37, 37 | 36.75, 37 | near-tie | near-tie | near-tie |
| 361 | `':` 3117 | `':` 3117 | 36, 36.25 | 36.25, 36.25 | 36.25, 36.25 | 36.25, 36.25 | near-tie | equal | equal |
| 401 | `(` 152374 | `({'` 50468 | 39.5, 39.25 | 39.25, 38.75 | 39.5, 39.25 | 39.25, 39.25 | near-tie | bug | near-tie |
| 425 | `1` 152367 | `{` 152405 | 39.5, 40 | 40, 40 | 39.75, 39.75 | 39.75, 40.25 | equal | bug | bug |
| 523 | `\\` 4068 | `foo` 18242 | 29.125, 29.125 | 29.125, 29.125 | 29.25, 29.25 | 29.125, 29.25 | near-tie | near-tie | near-tie |
| 531 | `\\` 4068 | `\` 152396 | 43.5, 43.5 | 43.5, 43.5 | 43.25, 43.25 | 43.25, 43.5 | near-tie | near-tie | near-tie |
| 582 | `JSON` 13889 | `Round` 34090 | 21.25, 21.125 | 21.25, 21.25 | 21.25, 21.25 | 21.125, 21.25 | near-tie | near-tie | near-tie |
| 673 | `           ` 678 | `           ` 678 | 42, 42 | 41.75, 42 | 42, 42 | 42, 42 | equal | near-tie | equal |
| 706 | `0` 152368 | `0` 152368 | 42.25, 42 | 42.25, 42.25 | 42.25, 42 | 42.25, 42 | equal | near-tie | equal |
| 915 | `\n` 13 | `\n` 13 | 40, 40.25 | 40, 40.25 | 39.75, 39.75 | 40, 40 | near-tie | near-tie | equal |
| 928 | `JSON` 13505 | `a` 261 | 40, 39.75 | 40, 39.75 | 40.25, 40 | 40, 40 | near-tie | near-tie | near-tie |
| 1234 | `file` 2408 | `a` 152346 | 34.75, 34.75 | 34.75, 35 | 34.75, 34.75 | 34.75, 35 | near-tie | equal | near-tie |
| 1352 | `\n` 13 | `the` 269 | 45.75, 46 | 45.75, 46 | 46, 46 | 45.75, 46 | equal | equal | near-tie |
| 1370 | `no` 754 | `it` 388 | 38.25, 38 | 38.25, 38.25 | 38.25, 38 | 38.25, 38.25 | bug | equal | bug |
| 1381 | `skipped` 61065 | `disabled` 18191 | 41.25, 41 | 41.25, 41.25 | 41.5, 41.25 | 41.5, 41.5 | near-tie | equal | near-tie |
| 1389 | `result` 4950 | `result` 4950 | 27.875, 27.75 | 27.5, 27.625 | 27.625, 27.625 | 27.5, 27.5 | equal | near-tie | equal |
| 1391 | `an` 284 | `an` 284 | 41, 41 | 41.25, 41 | 41, 40.75 | 41, 40.75 | near-tie | equal | equal |
| 1447 | `instead` 62222 | `with` 3629 | 35.25, 34.75 | 34.75, 34.75 | 35.5, 35 | 34.75, 34.75 | bug | equal | bug |
| 1461 | `non` 8316 | `non` 8316 | 37, 38 | 32.75, 30.375 | 33.75, 31.25 | 34.25, 32 | bug | equal | equal |

On the 4-bit quant:

- against MLX one pass: 1477 of 1,500 equal, 23 differ, red at [1321]
- against MLX decode: 1482 of 1,500 equal, 18 differ, red at none
- against float64: 1479 of 1,500 equal, 21 differ, red at [533, 600, 1355, 1418]

| position | float64's top-1 | engine's | MLX one pass | MLX decode | float64 | engine | against MLX one pass | against MLX decode | against float64 |
|---:|---|---|---|---|---|---|---|---|---|
| 21 | `for` 337 | `for` 337 | 38.25, 38.5 | 38.25, 38.5 | 39, 39 | 38.5, 38.5 | near-tie | near-tie | equal |
| 25 | `2` 152375 | `6` 152408 | 28.875, 29 | 28.75, 29 | 28.875, 28.875 | 28.875, 29 | equal | equal | near-tie |
| 46 | `a` 261 | `\n` 13 | 16.875, 16.75 | 16.75, 16.75 | 16.625, 16.5 | 16.625, 16.625 | near-tie | equal | near-tie |
| 79 | `\n` 13 | `\n` 13 | 28.375, 28.5 | 28.625, 28.625 | 28.625, 28.5 | 28.25, 28.25 | near-tie | equal | equal |
| 92 | `The` 764 | `The` 764 | 26, 26.125 | 25.875, 26 | 25.875, 25.875 | 26, 26 | near-tie | near-tie | equal |
| 93 | `:` 152382 | `a` 261 | 29, 29 | 28.625, 28.75 | 28.75, 28.625 | 28.875, 28.875 | equal | equal | near-tie |
| 110 | `({'` 50468 | `([` 6232 | 36.25, 36.25 | 36.5, 36.25 | 36.75, 36.5 | 36.25, 36.25 | equal | near-tie | near-tie |
| 119 | `` 152343 | `'` 617 | 39.5, 39.5 | 39.5, 39.5 | 39.75, 39.5 | 39.5, 39.5 | equal | equal | near-tie |
| 213 | `'"` 26762 | `'"` 26762 | 42.75, 43 | 43, 43 | 42.75, 42.75 | 43, 43 | near-tie | equal | equal |
| 240 | `1` 152367 | `0` 152368 | 36.25, 36.5 | 36.5, 36.5 | 36.25, 36.25 | 36.25, 36.5 | equal | near-tie | near-tie |
| 274 | `Bytes` 82414 | `String` 2702 | 41.5, 41.25 | 41.5, 41.25 | 41.5, 41.25 | 41.5, 41.5 | near-tie | near-tie | near-tie |
| 316 | `::` 1048 | `::` 1048 | 39, 38.75 | 38.75, 38.75 | 39, 38.75 | 39, 38.75 | equal | near-tie | equal |
| 361 | `':` 3117 | `':` 64919 | 37, 37 | 36.75, 37 | 36.75, 36.75 | 37, 37.25 | near-tie | equal | near-tie |
| 533 | `0` 152368 | `bar` 3066 | 31.75, 31.75 | 31.75, 31.625 | 31.75, 31.375 | 31.875, 31.875 | equal | near-tie | bug |
| 539 | `True` 5702 | `True` 5702 | 37.5, 37.75 | 37.5, 37.5 | 38, 37.75 | 37.75, 37.75 | near-tie | equal | equal |
| 600 | `dict` 8210 | `int` 513 | 28.75, 28.625 | 28.75, 28.75 | 28.875, 28.5 | 28.625, 28.625 | near-tie | equal | bug |
| 604 | `):` 1123 | `):` 1123 | 43, 43.25 | 43, 43.25 | 43, 43 | 43, 43 | near-tie | near-tie | equal |
| 663 | `.` 152361 | `.` 152361 | 44, 44.25 | 44.25, 44 | 44, 44 | 44, 44 | near-tie | equal | equal |
| 684 | `(` 152374 | `(` 152374 | 39, 38.75 | 38.75, 38.75 | 39.25, 38.75 | 39, 38.75 | equal | near-tie | equal |
| 782 | `cannot` 68205 | `cannot` 68205 | 26.75, 26.625 | 26.375, 26.375 | 26.5, 26.375 | 26.5, 26.375 | equal | near-tie | equal |
| 801 | `cannot` 68205 | `is` 277 | 38.25, 38.25 | 38.25, 38.25 | 38.25, 38 | 38.5, 38.5 | equal | equal | near-tie |
| 848 | `().` 2993 | `\n` 13 | 43.25, 43 | 43.25, 43 | 43.25, 43 | 43, 43 | near-tie | near-tie | near-tie |
| 994 | `3` 152385 | `1` 152367 | 34.25, 34.25 | 34.5, 34.5 | 34.5, 34.25 | 34.5, 34.5 | equal | equal | near-tie |
| 998 | `1` 152367 | `1` 152367 | 36.25, 36.25 | 36.25, 36.5 | 36.25, 36.25 | 36.25, 36.25 | equal | near-tie | equal |
| 1001 | `\n` 13 | `\n` 13 | 43.25, 43.5 | 43, 43.25 | 43.25, 43.25 | 43.25, 43.25 | near-tie | near-tie | equal |
| 1008 | `3` 152385 | `3` 152385 | 43.25, 43.25 | 43.5, 43.25 | 43.25, 43 | 43.5, 43.25 | near-tie | equal | equal |
| 1015 | `__` 834 | `\n` 13 | 41.25, 41 | 41.25, 41 | 41.25, 41 | 41, 41 | near-tie | near-tie | near-tie |
| 1096 | `default` 6952 | `py` 1521 | 22.875, 22.875 | 22.875, 23 | 23, 22.875 | 23.125, 23.125 | equal | equal | near-tie |
| 1105 | `ensure` 28600 | `check` 5734 | 36.25, 36 | 36, 36 | 36.25, 36 | 36, 36 | near-tie | equal | near-tie |
| 1112 | `check` 5734 | `check` 5734 | 38, 38.25 | 38.25, 38.25 | 38.25, 38.25 | 38, 38 | near-tie | equal | equal |
| 1224 | `to` 298 | `as` 361 | 45.5, 45.75 | 45.5, 45.75 | 45.5, 45.5 | 45.5, 45.75 | equal | equal | near-tie |
| 1253 | `skip` 21080 | `indent` 28799 | 35.5, 35.75 | 36, 36 | 35.75, 35.75 | 35.75, 36 | equal | near-tie | near-tie |
| 1258 | `,` 152363 | `then` 952 | 46.25, 46 | 46.25, 46 | 46.25, 46 | 46.25, 46.25 | near-tie | near-tie | near-tie |
| 1279 | `` ``, `` 143401 | `` ``, `` 143401 | 43.25, 43.25 | 43.5, 43.25 | 43.5, 43.25 | 43.5, 43.25 | near-tie | equal | equal |
| 1321 | `   ` 264 | `   ` 264 | 33, 33.25 | 33.5, 33 | 35, 34.25 | 34.25, 33.5 | bug | equal | equal |
| 1350 | `replaced` 10610 | `replaced` 10610 | 42.5, 42.5 | 42.25, 42.5 | 42.5, 42.5 | 42.5, 42.5 | equal | near-tie | equal |
| 1355 | `strings` 32508 | `and` 407 | 28.875, 29.125, 29.25 | 28.875, 29.125, 29.125 | 29.125, 28.75, 28.625 | 29.25, 29.25, 29.25 | near-tie | equal | bug |
| 1370 | `objects` 5287 | `objects` 5287 | 38.25, 38.25 | 38.5, 38.25 | 38.25, 38 | 38.5, 38.25 | near-tie | equal | equal |
| 1418 | `an` 284 | `raised` 8868 | 39.25, 39.5 | 39.25, 39.5 | 39.25, 39.25 | 39.5, 39.75 | equal | equal | bug |
| 1436 | `` ``, `` 143401 | `` ``, `` 143401 | 38.75, 38.75 | 39, 39 | 38.75, 38.5 | 39, 38.75 | near-tie | near-tie | equal |

Each table lists every position where the engine's top token differs from one of the references (row n predicts token
n + 1; logits for the float64 pass's, the engine's and MLX's top tokens, in that order, without repeats). Against
MLX's decode the 4-bit quant passes the rule and the 8-bit one is red at two positions; against the float64 pass the
engine is red at three and four. None is a fault in one kernel: at every red position the exact margin between the
contenders is under 0.46 (8-bit 401: 0.39; 425: three tokens within 0.13; 1370: three within 0.16; 1447: 0.45;
4-bit 533: 0.33; 600: 0.42; 1355: three within 0.45; 1418: three within 0.26), and every bf16 path carries more error
than that leaves room for. Over all 1,500 positions, at each position's exact top 8 tokens and less the row's common
shift, the root-mean-square logit error against the float64 pass is 0.126 for the engine on the 8-bit quant (MLX's
decode 0.117, its one pass 0.126) and 0.141 on the 4-bit one (MLX's 0.133 and 0.128); on the bf16 release the engine's
is 0.119 and HF's 0.120. MLX's own two paths are red against the float64 pass too: on the 8-bit quant at 425 and 1461
(one pass) and 1370 and 1447 (decode), on the 4-bit one at four and five positions. At 8-bit 1461, the largest
difference between the engine and MLX's one pass (`NaN` 48613 at 38 in MLX's one pass, 32 in the engine's), MLX's own
decode agrees with the engine and the float64 pass (`non` 8316 first); MLX's one pass is the outlier there.

The greedy chats that part from MLX's (four of six on the 8-bit quant, three on the 4-bit), at the token where they
part, each side's logits reached as its own greedy run reaches them (the engine's prompt chunks and one-row steps;
`mlx_lm`'s `generate_step`, the prompt but its last token in one pass, then one-token steps):

| quant | chat | position | MLX's top two | engine's top two | verdict |
|---|---|---:|---|---|---|
| 4-bit | chat0t | 4 | 152382 42.5, 152402 42.5 | 152402 42.75, 152382 42.5 | near-tie |
| 4-bit | chat1 | 28 | 570 37.5, 28746 37.5 | 28746 37.75, 570 37.25 | red |
| 4-bit | chat2 | 40 | 21933 19.125, 5562 19 | 5562 19.125, 21933 19.125 | near-tie |
| 8-bit | chat0 | 41 | 11859 38.25, 20011 38.25 | 20011 38.5, 11859 38.25 | near-tie |
| 8-bit | chat1 | 2 | 15686 39.25, 24225 39.25 | 24225 39, 15686 38.75 | near-tie |
| 8-bit | chat2 | 28 | 145762 26.5, 152404 26.5 | 152404 26.625, 145762 26.375 | red |
| 8-bit | chat3t | 30 | 152367 36.25, 152411 36.25 | 152411 36.25, 152367 36 | near-tie |

At the two red ones the float64 pass picks the engine's token: 4-bit chat1 28746 37.7532 against 570 37.4911, 8-bit
chat2 152404 26.5785 against 145762 26.4806. MLX rounds each pair to a tie and its argmax takes the lower id.

A device receipt with no other model server or GPU job on the Mac (28 Oct 2026, this branch's build; a CPU float64
reference run overlapped its first minute): the bf16 release's 1,500-position
logits are bit-identical to #1415's branch, and a greedy chat's tokens equal; both quants' logits are bit-identical to
the runs judged above, and `tf-llama-run ... exact` finds 0 differing rows on each. On the 4-bit quant, six prompts
sent two at a time return the same token ids as each alone, and three chats whose first turn carries an 8,500-token
log resume their second turn from the prompt cache (8,521-8,528 of 8,554-8,558 prompt tokens kept) with the same ids
as without the cache.

## Not done

No quality suite was run. No drafter is served (the family has no draft head; copy drafts come from the lane core).
Logprobs are refused by name: the release's tokenizer decodes through a `Strip` step, which the engine's tokenizer does
not count as token-local, so the server cannot give a token's bytes on its own.
Only the M5's tensor units were checked; M1-M4 Macs are refused at load, and smaller RAM is untested. Other `llama`
checkpoints, tied heads, other head sizes, Llama 3's scaled RoPE and biased projections are refused at load rather than
served untested.

## Licence

Nanbeige4.3-33B is published under Apache-2.0 at the revision measured here
(`c1093c501cc6460427c0ac634db7223d0450c0ee`, the licence in its model card). The MLX quants named here
(`pratikgx/Nanbeige4.3-33B-MLX-4bit`, `fcmeyer/Nanbeige4.3-33B-MLX-8bit`) state the same Apache-2.0 licence in their
model cards.
