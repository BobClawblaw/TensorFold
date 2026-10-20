//! DeepSeek-V4.1-Flash's DSpark draft head: three window-only MoE stages and a Markov-biased greedy chain.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Enc = fwd.Enc;
const Buffer = mtl.Buffer;
const moe = @import("moe.zig");

const MarkovArgs = extern struct { vocab: i32, rank: i32, step: i32 };
/// The draft block's filler after its first token (dspark_noise_token_id).
const noise_token: u32 = 128799;

/// The draft head's view of the rows: main_norm(main_proj(target means)), each stage's window keys to its ring.
pub fn condition(x: Enc) void {
    const s = x.s;
    const stages = &x.m.weights.stages.?;
    x.linear(stages[0].main_proj.?, s.mh, 0, s.mainx, 0, x.rows);
    x.rms(s.mainx, 0, stages[0].main_norm.?, s.mainx, 0, x.rows, c.hidden);
    for (stages, 0..) |*stage, k| {
        const a = &stage.block.attn;
        x.linear(a.wkv, s.mainx, 0, s.kv, 0, x.rows);
        for (x.segs) |g| x.of(g).kvFinish(s.kv, a.kv_norm, x.m.freq(c.layers + k), g.cache.stage_rings.?[k]);
    }
}

/// One DSpark round from `at`: `token` and four noise rows through three stages, five Markov-biased picks to `out`.
pub fn draft(m: *Model, ds: *st.Scratch, cache: *st.Cache, token: u32, at: usize, out: *[c.draft_block]u32) !void {
    const n = c.draft_block;
    const stages = &(m.weights.stages orelse return error.NoDraftHead);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const ids = ds.ids.slice(u32, n);
    ids[0] = token;
    for (ids[1..]) |*v| v.* = noise_token;
    for (ds.pos.slice(i32, n), 0..) |*p, i| p.* = @intCast(at + 1 + i);
    for (ds.pre[0].slice(f32, n * c.hc), 0..) |*v, i| v.* = if (i % c.hc == 0) 1 else 0;
    ds.draft_ids.slice(u32, n + 1)[0] = token;
    @memset(ds.img.slice(i32, n), 0); // drafts are text

    var cb = m.queue.commandBuffer();
    var e = cb.compute(.concurrent);
    fwd.pending_expand = false;
    fwd.pending_topk = 0;
    const x: Enc = .{ .ep = &e, .cbp = &cb, .m = m, .s = ds, .rows = n, .base = at + 1 };
    x.pipe("ds_embed");
    x.buf(ds.ids, 0, 0);
    x.tensor(m.weights.embed, 1);
    x.buf(ds.x, 0, 2);
    x.threads(.{ c.hidden, n, 1 }, .{ 256, 1, 1 });
    for (stages, cache.stage_rings.?, 0..) |*stage, ring, k| {
        const l = &stage.block;
        x.boundary(l.hc_attn, ds.pre[0], ds.pre[1], l.attn_norm);
        stageAttention(x, l, ring, c.layers + k, at);
        x.expand();
        x.boundary(l.hc_ffn, ds.pre[1], ds.pre[0], l.ffn_norm);
        try moe.moe(x, &l.moe, c.draft_topk, c.draft_experts);
        x.expand();
    }
    x.boundary(null, ds.pre[0], ds.pre[1], stages[c.stages - 1].norm.?);
    x.linear(m.weights.head, ds.normed, 0, ds.logits, 0, n);
    // forward_head's chain: each row's logits plus the Markov bias of the token before, its argmax the next token
    const groups = (c.vocab + 255) / 256;
    for (0..n) |k| {
        x.pipe("ds_markov");
        x.buf(ds.logits, 0, 0);
        x.tensor(stages[c.stages - 1].markov_embed.?, 1);
        x.tensor(stages[c.stages - 1].markov_head.?, 2);
        x.buf(ds.draft_ids, 0, 3);
        x.buf(ds.markov_part, 0, 4);
        x.args(MarkovArgs{ .vocab = c.vocab, .rank = c.markov_rank, .step = @intCast(k) }, 5);
        x.groups(.{ groups, 1, 1 }, .{ 256, 1, 1 });
        x.pipe("ds_markov_pick");
        x.buf(ds.markov_part, 0, 0);
        x.buf(ds.draft_ids, 0, 1);
        x.args([2]i32{ @intCast(groups), @intCast(k) }, 2);
        x.threads(.{ 1, 1, 1 }, .{ 1, 1, 1 });
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.status() != .completed) return error.DeepseekGpuFailed;
    @memcpy(out, ds.draft_ids.slice(u32, n + 1)[1..]);
}

/// A draft stage's attention: its five rows over the stage's target-key window ending at `at` and their own keys.
pub fn stageAttention(x: Enc, l: *const wts.Layer, ring: Buffer, layer: usize, at: usize) void {
    const s = x.s;
    const a = &l.attn;
    const n = x.rows;
    const freq = x.m.freq(layer);
    x.linear(a.wq_a, s.normed, 0, s.qr, 0, n);
    x.with();
    x.linear(a.wkv, s.normed, 0, s.kv, 0, n);
    x.rms(s.qr, 0, a.q_norm, s.qr, 0, n, c.q_rank);
    x.with();
    x.kvFinish(s.kv, a.kv_norm, freq, null);
    x.linear(a.wq_b, s.qr, 0, s.q, 0, n);
    x.attend(s.q, ring, s.kv, s.draft_picks, c.draft_block, a.sink, s.o, @intCast(at), freq);
    x.woA(a.wo_a, n);
    x.linear(a.wo_b, s.oa, 0, s.branch, 0, n);
}
