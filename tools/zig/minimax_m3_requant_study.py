"""Which MoE layers lose most at 3 bits: each layer's routed output on captured rows, NVFP4 against the refits."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimax_m3_nvfp4_mix import LM, Source, dequant_nvfp4, refit  # noqa: E402
from minimax_m3_ref import BF, swiglu_oai  # noqa: E402

USAGE = """
Rows captured by minimax_m3_ref.py --capture DIR (x{i}: a MoE layer's normed input, h{i}: the residual it adds to) run
through the layer's routed experts three ways: as released (NVFP4), and refit to 3-bit (g64) and 4-bit (g128) affine as
minimax_m3_nvfp4_mix.py and minimax_m3_convert.py refit them. Each width's error is the mean over rows of
|y_width - y_nvfp4| / |h|; the layers with the largest 3-bit error are the ones to keep NVFP4.

  python tools/zig/minimax_m3_requant_study.py HF_DIR CAPTURE_DIR OUT.json [--layers 3-59]
"""


def bits16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.int16)).view(BF)


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir")
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--layers", default="", help="only these layers, e.g. 3-10 (default: every captured one)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    src = Source(a.hf_dir)
    text = json.load(open(os.path.join(a.hf_dir, "config.json")))["text_config"]
    want = set()
    for part in filter(None, a.layers.split(",")):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for i in range(text["num_hidden_layers"]):
        xp = os.path.join(a.capture, f"x{i}.npy")
        if not os.path.exists(xp) or (want and i not in want) or str(i) in out:
            continue
        x = bits16(np.load(xp)).to(dev)
        h = bits16(np.load(os.path.join(a.capture, f"h{i}.npy"))).to(dev).float()
        m = f"{LM}model.layers.{i}.block_sparse_moe."
        logits = x.float() @ torch.from_numpy(src.np(m + "gate.weight").copy()).to(dev).T
        p = torch.sigmoid(logits)
        sel = (p + torch.from_numpy(src.np(m + "e_score_correction_bias").copy()).to(dev)).topk(text["num_experts_per_tok"], dim=-1).indices
        w = p.gather(1, sel)
        w = w / w.sum(-1, keepdim=True)
        ys = {k: torch.zeros(h.shape, dtype=torch.float32, device=dev) for k in ("nv", "q3", "q4")}
        for e in sorted(set(sel.flatten().tolist())):
            rows, slots = torch.nonzero(sel == e, as_tuple=True)
            xe = x[rows]
            W = {}
            for n in ("w1", "w3", "w2"):
                g = lambda k: src.np(f"{m}experts.{e}.{n}.{k}")[None]
                W[n] = dequant_nvfp4(g("weight_packed"), g("weight_scale"), g("weight_global_scale"))[0]
            for k, width in (("nv", None), ("q3", (3, 64)), ("q4", (4, 128))):
                V = {}
                for n, M in W.items():
                    if width is None:
                        V[n] = torch.from_numpy(np.ascontiguousarray(M)).to(dev).to(BF)
                    else:
                        V[n] = refit(M, width[0], width[1], device=dev, values=True).to(BF)
                y = (swiglu_oai(xe @ V["w1"].T, xe @ V["w3"].T) @ V["w2"].T).float()
                ys[k][rows] += y * w[rows, slots, None] * text["routed_scaling_factor"]
        hn = h.norm(dim=-1)
        err = {k: float(((ys[k] - ys["nv"]).norm(dim=-1) / hn).mean()) for k in ("q3", "q4")}
        out[str(i)] = err
        print(i, "3-bit %.4f  4-bit %.4f" % (err["q3"], err["q4"]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    ranked = sorted(out.items(), key=lambda kv: -kv[1]["q3"])
    print("by 3-bit error:", [int(k) for k, _ in ranked])


if __name__ == "__main__":
    main()
