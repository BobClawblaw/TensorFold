# MiniMax-M3.1-Flash on Metal (one 256 GB M5 Ultra)

The `minimax_m3` family serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a
428B-A23B mixture of experts with block-sparse attention (a lightning indexer picks each row's key blocks) and a
CLIP vision tower, on the lane host: tools, structured output, thinking on or off, images, the prompt cache, two
streams, and drafts from the release's DSpark head or incoai's DFlash2.

MiniMax's release stores its experts in NVFP4. The family reads MLX-style affine groups instead, from two public
conversions:

| checkpoint | experts | on disk | window on 256 GB |
| --- | --- | ---: | ---: |
| [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34) | 4-bit (g128) on 26 of 60 layers, 3-bit (g64) on the rest | 203.7 GiB | 32,768 |
| [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k) | 4-bit on 3 layers, 3-bit on the rest | 189.2 GiB | 147,456 |

Attention, the shared expert, the dense layers and the head are 8-bit in both; both carry the DSpark and DFlash2
weights. The key/value cache is bf16, exact: under the 226 GB limit below its caches fit 150,400 tokens on mix256k
(named for the 262,144-token window an int8 cache gave it) and 37,376 on mix34. `tools/zig/minimax_m3_convert.py`
rebuilds either checkpoint (below); the [serving recipe](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra)
has the conversion's studies and checks.

## Run

```sh
hf download BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34 --revision 136e775c89e996cac20c05befbcb26886ad24782 \
  --local-dir ~/models/MiniMax-M3.1-Flash-MLX-mix34
MINIMAX_LOAD_LIMIT_GB=226 tensorfold serve ~/models/MiniMax-M3.1-Flash-MLX-mix34 \
  --context 32768 --parallel 2 --prompt-cache-gib 4 --vision
```

For mix256k (revision `c7defa69b0e8b57426a9945eb3f9eb91cfef9f8c`), `--context 147456`.

- `MINIMAX_LOAD_LIMIT_GB` bounds weights, scratch and caches together. The default (75% of RAM) holds the weights
  but no caches, so set it; at 226 a 145,203-token prompt on mix256k peaked at 210 GiB wired.
- The cache takes 135 KiB a token across the 60 layers; `--parallel 2` streams share it.
- `MINIMAX_DRAFTER`: `dflash2` (the default) or `dspark`; any other value is refused. `--no-drafts` decodes plainly.
- Thinking off ends the prompt with `</mm:think>`, as the model does when it chooses not to think. Tool calls use
  MiniMax's `]<]minimax[>[` markup, parsed into OpenAI `tool_calls`; `tool_choice: required` or a named tool is
  constrained in that markup (xgrammar's `minimax_m3_xml`).
- Images: at most 576 tokens an image (672 x 672, the release processor's `max_pixels`); larger images are scaled
  down. Each image is its own tower pass. No video.

## Checks (Apple M5 Ultra, 256 GB, macOS 27)

- Four layers against transformers' own model: KL at most 0.005. The full model against an MLX reference at the same
  weights: KL 0.015.
- 1,024 rows teacher-forced against the NVFP4 release: mix34 KL 0.246 (81.1% top-1), mix256k 0.301 (78.3%).
- Drafted replies equal plain ones (DFlash2 and DSpark); windows equal one-row steps bit for bit; two streams'
  replies equal their runs alone; replies are equal with the prompt cache on and off.
- A needle at 145,203 tokens on mix256k (115 s cold; 210 GiB wired at the peak). The tower against transformers':
  relative error 0.030 to fp32 (torch bf16: 0.039).
- Decode with DFlash2, one stream: 43-59 tokens a second on mix256k and 43-59 on mix34 (prose to code; lists 50-56).
  Cold prompts on mix256k: 1,483 tokens a second at 8k, 1,491 at 32k, 1,379 at 128k (4,096-row chunks); on mix34
  1,441 at 8k and 1,451 at 31k.

## Rebuild the checkpoints

```sh
hf download MiniMaxAI/MiniMax-M3.1-Flash --revision 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e --local-dir ~/models/mm-src
hf download incoai/MiniMax-M3.1-Flash-DFlash2 --revision ac07a7c1ecaae26a7ace595756124dca974b432d \
  --local-dir ~/models/mm-dflash2
python3 tools/zig/minimax_m3_convert.py ~/models/mm-src ~/models/mm-mix256k --variant mix256k --dflash2 ~/models/mm-dflash2
python3 tools/zig/minimax_m3_convert.py --check ~/models/mm-mix256k --variant mix256k
```

`--variant mix34` makes the other. The experts are a least-squares affine refit of the release's NVFP4 values (`refit`),
the rest MLX's `mx.quantize` of their exact bf16; mix34's 4-bit layers are the 26 whose outputs gained most on captured
activations, mix256k's the 3 of those that fit a long window. `--check` compares every file with the published
revisions above (`tools/zig/minimax_m3_checkpoints.json`, the model cards left out): the weights and tokenizer byte for
byte, `config.json` by content and the shard index by its weight map (the published files were reformatted, and their
`total_size` fields are stale). mix256k rebuilt here (mlx 0.32.4, numpy 2.5.4; 69 minutes) checks 74/74; `--check`
needs neither.

## Not covered

- Only the M5's tensor units were checked; nothing smaller than 256 GB holds the weights.
- The family's Metal kernels are not registered with the kernel contract yet.
