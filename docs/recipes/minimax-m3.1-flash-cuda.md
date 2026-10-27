# MiniMax-M3.1-Flash on CUDA (two DGX Sparks)

The `minimax_m3` family's CUDA backend serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
from MiniMax's release in its own formats, at tensor parallel over hosts: the MXFP8 projections (attention, the shared
expert, the dense layers) and the NVFP4 routed experts as published, the release's DSpark drafter, its vision tower,
tools, structured output, thinking on or off, kept prompt states and two streams. Rank 0 serves HTTP; the other ranks
follow its steps. It shares the family's config, tokenizer handling, chat template and tool-call markup with the
[Metal backend](minimax-m3.1-flash.md).

## The checkpoint

Two GB10s (128 GB each) do not hold the release whole: its routed experts are 216.4 GiB. The backend reads a folder in
which some MoE layers keep their NVFP4 experts as released and the rest carry a 3-bit affine refit of the same values
(groups of 64, bf16 scales and biases), made by `tools/zig/minimax_m3_nvfp4_mix.py` from the release at its pinned
revision; every other tensor is copied unchanged. The folder's `config.json` names the NVFP4 layers (`nvfp4_mix`), and
`TF_MINIMAX_NVFP4_LAYERS` overrides them at load (ranges, `all` or `none`) when the folder holds both forms
(`--keep-nvfp4`). With NVFP4 on the 15 layers that lose most at 3 bits (20-29, 31, 49, 52, 57, 59) the model is about
201 GiB across the two ranks.

```sh
hf download MiniMaxAI/MiniMax-M3.1-Flash --revision 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e --local-dir ~/models/mm-src
python3 tools/zig/minimax_m3_nvfp4_mix.py ~/models/mm-src ~/models/mm-mix --nvfp4-layers 20-29,31,49,52,57,59 \
  --revision 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e
python3 tools/zig/minimax_m3_nvfp4_mix.py --check ~/models/mm-mix
```

The layers were ranked by each MoE layer's routed output on captured rows, the 3-bit refit against the release:
`tools/zig/minimax_m3_ref.py --capture DIR` saves the rows, `tools/zig/minimax_m3_requant_study.py` ranks the layers.

## Run

`TF_MINIMAX_RANK` of `TF_MINIMAX_WORLD` places each rank (at most four: a rank takes whole key heads), and
`NCCL_COMM_ID` names rank 0's address and port on the link between the hosts. On two GB10s, with 192.0.2.1 as rank 0's
address on their direct link, the same folder on both:

```bash
# on rank 1
TF_MINIMAX_RANK=1 TF_MINIMAX_WORLD=2 NCCL_COMM_ID=192.0.2.1:29581 TENSORFOLD_MEMORY_RESERVE_GIB=3 \
  tensorfold serve ~/models/mm-mix --context 32768 --parallel 2 --prompt-cache-gib 3 --vision
# on rank 0 (192.0.2.1)
TF_MINIMAX_RANK=0 TF_MINIMAX_WORLD=2 NCCL_COMM_ID=192.0.2.1:29581 TENSORFOLD_MEMORY_RESERVE_GIB=3 \
  tensorfold serve ~/models/mm-mix --context 32768 --parallel 2 --prompt-cache-gib 3 --vision
```

Set `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` for the link as NCCL documents them; each rank keeps a TCP line to rank 0
on the port after `NCCL_COMM_ID`'s, so a rank that fails stops the others.

On two ranks the layers' small gathers go by RoCE writes (as GLM's); `TF_ROCE=0` leaves them to NCCL and
`TF_ROCE_MAX_KB` caps a rank's bytes.

- The key/value cache is bf16, exact; a token takes 75 KiB a rank at TP=2 across the 60 layers (keys and values of the
  rank's two key heads, the shared index key).
- Drafts: the release's DSpark head (`dspark/`), on unless `--no-drafts` or `TF_MINIMAX_DRAFTER=0`; replies are
  identical with drafts off.
- The vision tower runs on the last rank; the others receive its rows.
- Kept prompt states start at 2,048-token chunk boundaries (`--prompt-cache-gib`).

## How it computes

- Projections: the MXFP8 lane matmul as stored; experts: the NVFP4 and 3-bit grouped kernels with a SwiGLU-OAI
  epilogue ((clamp(up) + 1) g sigmoid(1.702 g), g = min(gate, 7)); each rank holds every expert's share of the
  intermediate rows and the ranks' fp32 partials are summed in rank order.
- Attention: each index head selects its key head's blocks (the MSA paper's top-k per GQA group): the block's max of
  index scores in fp32, the row's own block always kept, every block while a row sees at most 16. Prompt and decode
  rows both run on the tensor units (`tf_mm3_attn_prompt_tc`, eight rows sharing each staged key block, fp32 online
  softmax, bf16 probabilities into P.V); a row's result does not depend on the window it is in, so drafted replies
  equal plain ones. `TF_MINIMAX_DECODE_ATTN=rows` runs decode by kept-block quarters instead.
- Drafts: DSpark's block over the anchor's 4,096-key window, the Markov walk kept on the device (one host read a
  block). Several streams' blocks share one pass (`Drafter.blocks`, as many as a 16-row decode window holds: two
  blocks of 7): their rows go through the projections, MLPs and head together, so the drafter's weights are read
  once, while each block attends to its own stream's cache and walks its own picks. The lane costs (`cuda_lanes.zig`) were timed on two GB10s with `TF_MINIMAX_TIME_WINDOWS=<position>`, which
  logs a verify window's ms by width and a draft block's at boot.
- Rounding follows the Metal backend's: Gemma norms rounded once, the residual rounded to bf16 after every add, cos
  and sin rounded to bf16; the MoE sums its picks in fp32 before one rounding.

## Checks (two DGX Sparks, GB10)

- Kernel contract: 10 registrations (tensor-unit prompt attention over kept blocks, every block and a draft window; the
  row kernel; decode with its merge, kept and every block; the index scores, row and tensor-unit, at 2 and 4 index
  heads; the selection) pass gates 1-4, and their 4 planted controls (a kept block dropped in prompt and in decode
  attention, a window that starts a key late, index keys shifted by a row) go red.

- Served (recipe harness, quick plan, TP=2): soak 60/60, concurrent 12/12, streaming 4/4, multi-turn 8/8, stop 3/3,
  max-tokens 3/3, thinking switch 10/10; DSpark drafted == plain 12/12; structured output 18/18 (drafted == plain 12/12);
  tool calls 5/5; seeded sampling equal on repeat and with drafts off 9/9; a resumed turn equals the same turn cold after a
  restart. A prompt cancelled mid-prefill leaves the next request right, and an image beside a second stream reads the
  same as the image alone.
- Quality, thinking off (the recipe suite, beside the M5 Ultra MLX recipe's two runs): tools 0.950 (0.933 / 0.950),
  GSM8K 0.972 (0.976 / 0.968), MGSM 0.838 (0.821 / 0.846), MMLU 0.836 (0.819 / 0.827), IFEval 0.587 (0.593 / 0.580),
  JSON schema 1.000 (1.000).
- Decode with DSpark drafts, one stream: prose 23.6, code 41.1, structured 25.0, list 32.6 tokens a second, first token
  0.47-0.50 s; 2.7 tokens a round. Two streams: 29.8, 49.3, 32.6 and 53.9 in total.

The receipts are in the recipe harness ledger ([recipe](https://github.com/BobClawblaw/minimax-m31-flash-nvfp4-2x-dgx-sparks)).

## Not covered

- The kernel contract's gates 1-4 cover the indexer and the attention (`tf-cuda-test minimax-contract`, also in
  `kernel-contract`), not yet the glue kernels (norms, RoPE prep, routing, DSpark's Markov head, the tower's RoPE); gate 5
  has every launch of both images (`minimax_cuda_records.zig`), not the SwiGLU-OAI instances of the shared expert images.
- Only TP=2 on two GB10s was run; TP=1 needs a GPU holding about 200 GiB, TP=4 was not tried.
- Video is not served.
- MATH-500 (thinking off) was stopped unfinished, and quality with thinking on was not run.
