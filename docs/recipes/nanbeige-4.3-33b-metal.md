# Nanbeige4.3-33B, native Metal

This recipe covers text generation with [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)
(model_type `llama`: a dense 33B Llama-architecture model, 512K context, thinking and tool calls in its chat template)
on the Metal engine, served from the bf16 release as published: no conversion, nothing lost. It was brought up on an
M5 Ultra (256 GB); other Macs are untested. The family serves any `llama` checkpoint with heads of 128, default RoPE,
no biases and its own output head, in bf16; it refuses others by name.

## Build and serve

```sh
hf download Nanbeige/Nanbeige4.3-33B --local-dir models/nanbeige4.3-33b
zig build native -Doptimize=ReleaseFast -Dcpu=native
zig build grammar -Dxgrammar=PATH_TO_XGRAMMAR_v0.2.8 -Dcpu=native   # structured output
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b --name nanbeige --port 8888 \
  --context 131072 --parallel 2 --prompt-cache-gib 8
# with Nanbeige4.1-3B drafting (the same replies, faster: see Drafter below)
hf download Nanbeige/Nanbeige4.1-3B --local-dir models/nanbeige4.1-3b
python3 tools/zig/llama_convert.py models/nanbeige4.1-3b models/nanbeige4.1-3b-4bit --bits 4
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b --name nanbeige --port 8888 \
  --context 131072 --parallel 2 --prompt-cache-gib 8 --drafter models/nanbeige4.1-3b-4bit --drafter-bits 4
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
- `--drafter`: a smaller Llama on the same tokenizer (`drafter.zig`) as the lane core's priced copy proposer, its
  drafts verified by the model's windows like any copy, so replies keep the model's bits.

The Metal glue kernels are not registered with the kernel contract, as with the other Metal family glue.

## Measured (M5 Ultra, the bf16 release, 26 Oct 2026)

Against Hugging Face's `modeling_llama` on the release (transformers 4.57.1, bf16 on the CPU): 1,500 positions
teacher-forced, mean KL 0.0015, top-1 99.1%; 64 positions after a 9,000-token prompt, mean KL 0.0013, top-1 100%. Six
greedy chats of 64 tokens (code, prose, Chinese, a document, with and without thinking): three equal HF's token for
token; the other three part at a position where the engine's two best logits are the same bf16 value. One-row steps
and windows of 7 or 16 rows give the same logits bit for bit after prompts of 64 and 1,500 tokens.

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
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums, no quantized weights.

## Drafter: Nanbeige4.1-3B

[`Nanbeige/Nanbeige4.1-3B`](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) (revision
`9c4555c37921c0982af8fffeee1e0a6e6b953c4f`; model_type `llama`, 32 layers, hidden 2,560, 20 query and 4 key heads of
128) shares the 33B's tokenizer: the same 166,100 ids, merges and added tokens, the same bos and eos (its
`vocab_size` pads to 166,144, the 33B's to 166,912). The family serves it as a drafter beside the 33B:

- **The proposer.** `--drafter` loads the smaller Llama on the served model's queue and registers it as the lane
  host's copy proposer (`LaneHost.proposers`), priced, so a round takes its drafts as given. The model's window
  verifies them row by row and keeps the prefix its own draws agree with, as it keeps any copy draft: replies are the
  model's whatever the drafter says. A request takes one of `--parallel` drafter caches (64 KiB a token, made and grown
  in 4,096-row steps as the context needs, its state copied over, budgeted at the window beside the streams' caches);
  it keeps its tokens between requests, so a turn's prompt that extends the last one is not fed again.
- **Drafting.** Before each round the drafter's cache is synced to the stream's context by their longest common
  prefix (an accepted draft stays, a rejected one is cut), the new tokens are fed, and the drafts are greedy steps
  chained on the GPU, each step's argmax the next step's input through a token slot, so one proposal is one chain of
  command buffers. The depth comes from measured rates: the stream's decayed acceptance, one chained step's wall time
  and a line fitted to the round's time by its window's rows (the 33B's windows grow with rows: about 55 ms at 4
  rows, 82 at 8), choosing the depth with the most expected tokens a millisecond; a draft chain stops at an eos.
- **Widths.** `--drafter-bits 4` (the default) serves a `llama_convert.py --bits 4` conversion of the drafter
  (2.63 GiB; 3.6 ms a chained step), `--drafter-bits 0` the bf16 release (7.3 GiB; 8.5 ms a step, more drafts land
  but no more tokens a second). Lossy drafter weights change no reply.
- **Checked** with `tf-llama-run <33B> <tokens> draft <drafter>,<n>`, which decodes greedily through the drafter's
  windows and prints the tokens with the rounds and drafts kept: on eight chat prompts (thinking on and off, 128
  tokens) and after a 1,500-token prompt (256 tokens) the drafted tokens equal the plain greedy tokens exactly, with
  1.8-5.8 tokens landing a round.

| prompt, 128 greedy tokens (the rule's first form; the per-row form below) | plain | drafted 4-bit | tokens a round | drafts kept |
|---|---:|---:|---:|---:|
| Fibonacci function, thinking off | 16.2 tok/s | 45.6 (45.9) | 5.8 (4.1) | 105 of 196 (96 of 150) |
| Fibonacci function, thinking on | 16.2 | 24.7 | 2.0 | 65 of 221 |
| why the sky is blue, thinking off | 16.2 | 28.2 (27.9) | 2.3 (2.2) | 73 of 243 (70 of 179) |
| why the sky is blue, thinking on | 16.2 | 26.7 | 2.0 | 65 of 188 |
| Hangzhou in two sentences (Chinese), thinking off | 16.1 | 24.1 | 1.8 | 59 of 195 |
| Hangzhou in two sentences (Chinese), thinking on | 16.2 | 28.4 | 2.4 | 78 of 215 |
| 9.11 or 9.8, thinking off | 16.3 | 35.0 (39.0) | 3.9 (3.4) | 96 of 238 (91 of 171) |
| 9.11 or 9.8, thinking on | 16.2 | 27.3 | 2.6 | 81 of 233 |
| 1,500 tokens of Python's json docs, 256 tokens | 16.1 | 26.5 | 2.1 | 135 of 437 |

The first run priced every window alike and drafted deeper on the easy prompts, where windows of 8-9 rows cost 100 ms;
the per-row fit (the shipped rule) drafts less there for the same or more tokens a second (39.0 against 35.0 on the
9.11 prompt).

Served (`--context 131072 --parallel 2 --prompt-cache-gib 8`, the drafter at 4-bit, the harness's quick and speed
plans beside an idle 93 GiB server of another model): every check passes as without the drafter (structured 18/18
with drafted == plain 12/12, tools 5/5, sampling 9/9 seeded replies equal on repeat and undrafted, turns resumed in
0.34 s, soak 60/60, concurrent 12/12, streaming, multi-turn, stop, thinking and max_tokens); three greedy replies (a
function, a word problem with thinking, an explanation) equal the undrafted server's token for token, and the
first equals the published build's. Speed:

| | undrafted | drafted |
|---|---:|---:|
| one stream, the ruler's prose / code / structured / list (256 tokens) | 16.4 / 16.3 / 16.4 / 16.4 tok/s | 113.5 / 111.0 / 108.5 / 84.9 |
| two streams together, the same | 31.8 / 28.0 / 31.7 / 31.7 | 63.3 / 64.3 / 63.8 / 58.3 |
| one stream's rounds | 61 ms, 1 token | 47 ms, 4.9 tokens (81% of drafts land) |
| two streams' rounds | 66 ms, 2 tokens | 87 ms, 2.7 tokens a stream |
| three greedy chats, whole requests (prompt and reply) | 15.7 / 16.0 / 16.1 tok/s | 30.2 / 22.9 / 26.6 |
| first token | 0.23-0.26 s | 0.23-0.26 s |

The ruler's texts repeat themselves, which the drafter continues almost whole; ordinary chats land 2-4 tokens a
round for 1.5-3x. In shared rounds the lane core trims each stream's copies by its own prior (`copy_rate`) and
cost tables, not the drafter's rates, so two streams land fewer drafts a round than one; they still double the
undrafted pair's rate. The drafter's cache is fed the prompt on the first round, after the model's own prompt pass
(a 1,500-token prompt takes it about 0.1 s), so the first token keeps its time.

## 8-bit and 4-bit conversions

`tools/zig/llama_convert.py` writes an MLX affine conversion of the release (groups of 64, `--bits 8` or `4`): every
projection quantized, q|k|v and gate|up fused into one matrix each so a projection is one call, the embedding and
norms bf16 as released; `config.json` gains the uniform `quantization` table the family admits. The engine serves the
folder with the core's row kernels (`row_projection.zig`) for decode windows and its tensor-unit affine matmul
(`affine_mm.zig`) for prompt chunks, the kernels the other affine families use.

```sh
python3 tools/zig/llama_convert.py models/nanbeige4.3-33b models/nanbeige4.3-33b-8bit --bits 8   # 33.8 GiB, about 2 minutes
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b-8bit --name nanbeige --port 8888 --context 131072 --parallel 2
```

Measured on the M5 Ultra against the same HF reference and harness plans as the release (bf16 figures from above):

| | bf16 | 8-bit | 4-bit |
|---|---:|---:|---:|
| mean KL over 1,500 positions, top-1 | 0.0015, 99.1% | 0.0023, 98.9% | 0.082, 91.1% |
| greedy chats equal to HF's (64 tokens, 6 chats) | 3 | 4 | 0 (all part within 16 tokens) |
| decode, 1 stream / 2 streams | 16.4 / 28-32 tok/s | 28.4-28.9 / 52.5-54.1 | 48.1-48.8 / 81.3-84.8 |
| cold prompt 2.1k / 8.3k / 33k / 61k tokens | 1,259 / 1,313 / 1,074 / 866 tok/s | 1,269 / 1,254 / 1,009 / 815 | 1,344 / 1,312 / 1,044 / 836 |

Windows and one-row steps agree bit for bit on each; every serving check passes. The conversions are lossy, so the
release stays the recipe's default; the 4-bit one costs a mean KL of 0.082 and parts from the release's greedy text
within 16 tokens on every chat, and is offered for its speed with that said.

## Not done

No quality suite was run. The family has no draft head of its own: drafts come from `--drafter`'s smaller Llama or
the lane core's copies.
Logprobs are refused by name: the release's tokenizer decodes through a `Strip` step, which the engine's tokenizer does
not count as token-local, so the server cannot give a token's bytes on its own.
Only the M5's tensor units were checked; M1-M4 Macs and smaller RAM are untested. Tied heads, other head sizes,
Llama 3's scaled RoPE and biased projections are refused at load rather than served untested.
