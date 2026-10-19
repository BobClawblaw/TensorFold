//! A DeepSeek-V4.1 block over a forward's rows: engram, attention (each stream's on its own rows), then the MoE.
const std = @import("std");
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const idx = @import("index.zig");
const ffn = @import("moe.zig");
const Enc = @import("forward.zig").Enc;

const EngramArgs = extern struct { eps: f32 };

/// One block: the engram lookup (layers 1 and 14), attention and the MoE, each between its boundaries.
pub fn block(x: Enc, i: usize, l: *const wts.Layer) !void {
    const s = x.s;
    if (l.engram) |eg| {
        x.flush();
        x.linear(eg.wkv, s.engram_rows[c.engramIndex(i).?], 0, s.engram_kv, 0, x.rows);
        x.pipe("ds_engram");
        x.buf(s.x, 0, 0);
        x.buf(s.engram_kv, 0, 1);
        x.tensor(eg.q_weight, 2);
        x.tensor(eg.k_weight, 3);
        x.args(EngramArgs{ .eps = c.norm_eps }, 4);
        x.buf(s.img, 0, 5);
        x.groups(.{ c.hc, x.rows, 1 }, .{ 256, 1, 1 });
        try x.cut(8);
    }
    x.boundary(l.hc_attn, s.pre[0], s.pre[1], l.attn_norm);
    try x.cut(0);
    try attention(x, i, l);
    x.expand();
    x.boundary(l.hc_ffn, s.pre[1], s.pre[0], l.ffn_norm);
    try x.cut(0);
    try ffn.moe(x, &l.moe, c.topk, c.experts);
    x.expand();
}

fn attention(x: Enc, layer: usize, l: *const wts.Layer) !void {
    const s = x.s;
    const m = x.m;
    const rows = x.rows;
    const a = &l.attn;
    const freq = m.freq(layer);
    // q = wq_b(q_norm(wq_a(x))) and the window key kv_norm(wkv(x)) into the ring, the two chains side by side
    x.linear(a.wq_a, s.normed, 0, s.qr, 0, rows);
    x.with();
    x.linear(a.wkv, s.normed, 0, s.kv, 0, rows);
    x.rms(s.qr, 0, a.q_norm, s.qr, 0, rows, c.q_rank);
    x.with();
    for (x.segs) |g| x.of(g).kvFinish(s.kv, a.kv_norm, freq, g.cache.windows[layer]);
    x.linear(a.wq_b, s.qr, 0, s.q, 0, rows);
    try x.cut(1);

    const ratio: usize = c.ratios[layer];
    const si: ?usize = if (ratio == 0) null else std.mem.indexOfScalar(usize, &c.kv_sources, c.kvSource(layer).?).?;
    if (si) |k| {
        if (a.compressor) |cp| try idx.compress(x, k, ratio, cp, a.indexer.?, freq);
        if (a.indexer) |ix| try idx.index(x, k, layer, ratio, ix, freq);
    }
    try x.cut(2);
    // sparse attention over each stream's window and picks, then the output's RoPE taken back
    for (x.segs) |g| {
        const comp = if (si) |k| g.cache.sources[k].latents else g.cache.windows[layer]; // unread without picks
        x.of(g).attend(s.q, g.cache.windows[layer], comp, s.picks, if (si != null) st.picks else 0, a.sink, s.o, -1, freq);
    }
    try x.cut(3);
    // the grouped low-rank output: block-diagonal wo_a, wo_b
    x.woA(a.wo_a, rows);
    x.linear(a.wo_b, s.oa, 0, s.branch, 0, rows);
    try x.cut(4);
}
