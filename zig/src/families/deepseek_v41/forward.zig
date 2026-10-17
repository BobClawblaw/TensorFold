//! One stream's rows through the backbone in one command buffer: Transformer.forward of DeepSeek's model.py, with
//! the caches its decode path keeps (window rings, the four sources' compressed latents and index keys).
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const Buffer = mtl.Buffer;

pub const Head = enum { last, all, none };

/// Per-layer outputs a check reads back (the streams after each block), or null.
pub const Dump = struct { layers: []Buffer };

const HcArgs = extern struct { eps: f32, hc_eps: f32, iters: i32 };
const RowsArgs = extern struct { n: i32, x_stride: i32, y_stride: i32, eps: f32 };
const GemvArgs = extern struct { rows: i32, n: i32, k: i32, group_n: i32, x_stride: i32, y_stride: i32 };
const RopeArgs = extern struct { heads: i32, stride: i32, offset: i32, inverse: i32 };
const CopyArgs = extern struct { width: i32, ring: i32 };
const PoolArgs = extern struct { first: i32, ring: i32 };
const ScoreArgs = extern struct { rows: i32, groups: i32, ratio: i32, first_pos: i32, stride: i32 };
const MaskArgs = extern struct { groups: i32, stride: i32, block: i32, kept: i32 };
const BlockArgs = extern struct { rows: i32, groups: i32, block: i32, first_pos: i32, ratio: i32, stride: i32 };
const TopkArgs = extern struct { n: i32, k: i32, stride: i32, offset: i32 };
const AttnArgs = extern struct { first_pos: i32, ring: i32, window: i32, picks: i32, scale: f32 };
const RouteArgs = extern struct { experts: i32, topk: i32, scale: f32 };
const GluArgs = extern struct { width: i32, weighted: i32, limit: f32 };
const EngramArgs = extern struct { eps: f32, text_rows: i32 };

/// GPU seconds by section when set (`sections` names them): each section then runs in its own command buffer,
/// waited for (a profile, not a served speed).
pub var profile: ?*[sections.len]f64 = null;
pub const sections = [_][]const u8{ "boundaries", "attn proj", "compress+index", "attend", "attn out", "route", "experts", "shared", "engram", "head" };

const Enc = struct {
    ep: *mtl.ComputeEncoder,
    cbp: *mtl.CommandBuffer,
    m: *Model,
    s: *st.Scratch,
    rows: usize,
    base: usize, // the first row's position

    /// With `profile`, everything encoded since the last cut runs now and its GPU time counts for `section`.
    fn cut(self: Enc, section: usize) !void {
        const p = profile orelse return;
        self.ep.*.end();
        self.cbp.*.commit();
        self.cbp.*.wait();
        if (self.cbp.*.status() != .completed) return error.DeepseekGpuFailed;
        p[section] += self.cbp.*.gpuSeconds();
        self.cbp.* = self.m.queue.commandBuffer();
        self.ep.* = self.cbp.*.compute(.serial);
    }

    fn pipe(self: Enc, comptime key: []const u8) void {
        self.ep.*.setPipeline(self.m.kernels.get(key));
    }
    fn buf(self: Enc, b: Buffer, off: usize, slot: usize) void {
        self.ep.*.setBuffer(b, off, slot);
    }
    fn tensor(self: Enc, t: wts.Tensor, slot: usize) void {
        self.ep.*.setBuffer(t.buffer, t.offset, slot);
    }
    fn args(self: Enc, v: anytype, slot: usize) void {
        self.ep.*.setBytes(std.mem.asBytes(&v), slot);
    }
    fn threads(self: Enc, grid: [3]usize, group: [3]usize) void {
        self.ep.*.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    fn groups(self: Enc, grid: [3]usize, group: [3]usize) void {
        self.ep.*.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    /// y[rows, n] = x[rows, k] W^T on the row kernels.
    fn linear(self: Enc, w: wts.Linear, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) void {
        const call = row.call(&self.m.kernels.rows, w, rows, false) catch unreachable;
        self.ep.*.setPipeline(call.pipeline);
        self.buf(x, x_off, 0);
        self.buf(w.w, w.w_off, 1);
        self.buf(w.scales, w.s_off, 2);
        self.buf(w.biases, w.b_off, 3);
        self.ep.*.setBytes(std.mem.asBytes(&call.dims), 4);
        self.buf(y, y_off, 5);
        self.threads(.{ call.groups * call.threads, 1, 1 }, .{ call.threads, 1, 1 });
    }

    /// Slot i takes expert ids[i] on input row i / repeat.
    fn indexed(self: Enc, x: row.Experts, slots: usize, in: Buffer, ids: Buffer, out: Buffer) void {
        const call = row.callIndexed(&self.m.kernels.rows, x, slots, false) catch unreachable;
        self.ep.*.setPipeline(call.pipeline);
        self.buf(in, 0, 0);
        self.buf(x.one.w, x.one.w_off, 1);
        self.buf(x.one.scales, x.one.s_off, 2);
        self.buf(x.one.biases, x.one.b_off, 3);
        self.ep.*.setBytes(std.mem.asBytes(&call.dims), 4);
        self.buf(out, 0, 5);
        self.buf(ids, 0, 6);
        self.threads(.{ call.groups * call.threads, call.slots, 1 }, .{ call.threads, 1, 1 });
    }

    /// y = x W^T with a bf16 W [n, k]; `group_n` outputs share one k-wide slice of each x row.
    fn gemv(self: Enc, comptime out_f32: bool, w: wts.Tensor, x: Buffer, x_off: usize, x_stride: usize, y: Buffer, y_off: usize, rows: usize, n: usize, k: usize, group_n: usize) void {
        if (out_f32) self.pipe("ds_gemv_bf_f32") else self.pipe("ds_gemv_bf_bf");
        self.buf(x, x_off, 0);
        self.tensor(w, 1);
        self.buf(y, y_off, 2);
        self.args(GemvArgs{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(k), .group_n = @intCast(group_n), .x_stride = @intCast(x_stride), .y_stride = @intCast(n) }, 3);
        self.groups(.{ (n + 7) / 8, (rows + 7) / 8, 1 }, .{ 256, 1, 1 });
    }

    fn rms(self: Enc, x: Buffer, x_off: usize, w: wts.Tensor, y: Buffer, y_off: usize, rows: usize, n: usize) void {
        self.pipe("ds_rms");
        self.buf(x, x_off, 0);
        self.tensor(w, 1);
        self.buf(y, y_off, 2);
        self.args(RowsArgs{ .n = @intCast(n), .x_stride = @intCast(n), .y_stride = @intCast(n), .eps = c.norm_eps }, 3);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
    }

    fn rope(self: Enc, x: Buffer, rows: usize, heads: usize, stride: usize, positions: Buffer, freq: Buffer, inverse: bool) void {
        self.pipe("ds_rope");
        self.buf(x, 0, 0);
        self.buf(positions, 0, 1);
        self.buf(freq, 0, 2);
        self.args(RopeArgs{ .heads = @intCast(heads), .stride = @intCast(stride), .offset = @intCast(stride - c.rope_dim), .inverse = @intFromBool(inverse) }, 3);
        self.threads(.{ c.rope_dim / 2, heads, rows }, .{ c.rope_dim / 2, 1, 1 });
    }

    fn fp8(self: Enc, x: Buffer, values: usize) void {
        self.pipe("ds_fp8_qdq");
        self.buf(x, 0, 0);
        self.threads(.{ values / 32, 1, 1 }, .{ 64, 1, 1 });
    }

    fn fp4(self: Enc, x: Buffer, values: usize, e4m3_scale: bool) void {
        self.pipe("ds_fp4_qdq");
        self.buf(x, 0, 0);
        self.args(@as(i32, @intFromBool(e4m3_scale)), 1);
        self.threads(.{ values / @as(usize, if (e4m3_scale) 16 else 32), 1, 1 }, .{ 64, 1, 1 });
    }

    fn put(self: Enc, src: Buffer, positions: Buffer, dst: Buffer, rows: usize, width: usize, ring: usize) void {
        self.pipe("ds_put_rows");
        self.buf(src, 0, 0);
        self.buf(positions, 0, 1);
        self.buf(dst, 0, 2);
        self.args(CopyArgs{ .width = @intCast(width), .ring = @intCast(ring) }, 3);
        self.threads(.{ width, rows, 1 }, .{ @min(width, 256), 1, 1 });
    }

    fn putF32(self: Enc, src: Buffer, positions: Buffer, dst: Buffer, rows: usize, width: usize, ring: usize) void {
        self.pipe("ds_put_rows_f32");
        self.buf(src, 0, 0);
        self.buf(positions, 0, 1);
        self.buf(dst, 0, 2);
        self.args(CopyArgs{ .width = @intCast(width), .ring = @intCast(ring) }, 3);
        self.threads(.{ width, rows, 1 }, .{ @min(width, 256), 1, 1 });
    }

    /// A boundary: (with `mixes`) this boundary's pre, post and comb from the streams, and the streams collapsed with
    /// `pre_in` and normed by `norm` into `normed`.
    fn boundary(self: Enc, hc: ?wts.Hc, pre_in: Buffer, pre_out: Buffer, norm: wts.Tensor) void {
        const s = self.s;
        if (hc != null) self.pipe("ds_hc_boundary") else self.pipe("ds_hc_collapse");
        self.buf(s.x, 0, 0);
        if (hc) |h| {
            self.tensor(h.fn_, 1);
            self.tensor(h.base, 2);
            self.tensor(h.scale, 3);
        } else {
            self.buf(s.x, 0, 1);
            self.buf(s.x, 0, 2);
            self.buf(s.x, 0, 3);
        }
        self.buf(pre_in, 0, 4);
        self.tensor(norm, 5);
        self.args(HcArgs{ .eps = c.norm_eps, .hc_eps = c.hc_eps, .iters = c.sinkhorn }, 6);
        self.buf(pre_out, 0, 7);
        self.buf(s.post, 0, 8);
        self.buf(s.comb, 0, 9);
        self.buf(s.normed, 0, 10);
        self.groups(.{ self.rows, 1, 1 }, .{ 1024, 1, 1 });
    }

    fn expand(self: Enc) void {
        const s = self.s;
        self.pipe("ds_hc_expand");
        self.buf(s.x, 0, 0);
        self.buf(s.branch, 0, 1);
        self.buf(s.post, 0, 2);
        self.buf(s.comb, 0, 3);
        self.threads(.{ c.hidden, self.rows, 1 }, .{ 256, 1, 1 });
    }
};

/// The groups this call completes at `ratio`: [first, end) (a group's last position is inside the call).
fn completed(base: usize, rows: usize, ratio: usize) [2]usize {
    return .{ base / ratio, (base + rows) / ratio };
}

pub fn run(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, dump: ?Dump) !void {
    const rows = tokens.len;
    if (rows == 0 or rows > s.rows or rows > st.max_rows) return error.DeepseekRowsExceeded;
    const base = cache.len;
    if (base + rows > cache.capacity) return error.DeepseekCacheFull;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();

    // host side: ids, positions, the engram history and lookups, the first collapse (copy 0)
    for (s.ids.slice(u32, rows), tokens) |*d, t| d.* = t;
    for (s.pos.slice(i32, rows), 0..) |*p, i| p.* = @intCast(base + i);
    cache.engram_ids.shrinkRetainingCapacity(base);
    for (tokens) |t| cache.engram_ids.appendAssumeCapacity(m.engram.compress(t, true));
    for (0..c.engram_layers.len) |e| {
        const dst = s.engram_rows[e].slice(u16, rows * c.engram_cols * c.engram_dim);
        for (0..rows) |r| try m.engram.gather(e, cache.engram_ids.items, base + r, dst[r * c.engram_cols * c.engram_dim ..][0 .. c.engram_cols * c.engram_dim]);
    }
    for (s.pre[0].slice(f32, rows * c.hc), 0..) |*v, i| v.* = if (i % c.hc == 0) 1 else 0;

    var cb = m.queue.commandBuffer();
    const w = &m.weights;
    var e = cb.compute(.serial);
    const x: Enc = .{ .ep = &e, .cbp = &cb, .m = m, .s = s, .rows = rows, .base = base };
    x.pipe("ds_embed");
    x.buf(s.ids, 0, 0);
    x.tensor(w.embed, 1);
    x.buf(s.x, 0, 2);
    x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
    for (&w.layers, 0..) |*l, i| {
        try block(x, cache, i, l);
        if (dump) |d| { // the streams after each block: the buffers are untracked, so each block completes first
            e.end();
            cb.commit();
            cb.wait();
            if (cb.status() != .completed) return error.DeepseekGpuFailed;
            @memcpy(d.layers[i].contents()[0 .. rows * c.hc * c.hidden * 2], s.x.contents()[0 .. rows * c.hc * c.hidden * 2]);
            cb = m.queue.commandBuffer();
            e = cb.compute(.serial);
        }
    }
    try x.cut(0);
    // the last block's FFN pre collapses the streams for the head
    x.boundary(null, s.pre[0], s.pre[1], w.norm);
    switch (head) {
        .none => {},
        .last => x.linear(w.head, s.normed, (rows - 1) * c.hidden * 2, s.logits, 0, 1),
        .all => x.linear(w.head, s.normed, 0, s.logits, 0, rows),
    }
    e.end();
    cb.commit();
    cb.wait();
    if (profile) |p| p[9] += cb.gpuSeconds();
    if (cb.status() != .completed) {
        if (cb.failure()) |why| std.log.err("deepseek_v41: the forward failed: {s}", .{why});
        return error.DeepseekGpuFailed;
    }
    cache.len += rows;
    if (head != .none) {
        const last = if (head == .last) 0 else rows - 1;
        @memcpy(cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + last * c.vocab * 2)[0 .. c.vocab * 2]);
    }
}

/// One block: the engram lookup (layers 1 and 14), attention and the MoE, each between its boundaries.
fn block(x: Enc, cache: *st.Cache, i: usize, l: *const wts.Layer) !void {
    const s = x.s;
    if (l.engram) |eg| {
        x.linear(eg.wkv, s.engram_rows[c.engramIndex(i).?], 0, s.engram_kv, 0, x.rows);
        x.pipe("ds_engram");
        x.buf(s.x, 0, 0);
        x.buf(s.engram_kv, 0, 1);
        x.tensor(eg.q_weight, 2);
        x.tensor(eg.k_weight, 3);
        x.args(EngramArgs{ .eps = c.norm_eps, .text_rows = @intCast(x.rows) }, 4);
        x.groups(.{ c.hc, x.rows, 1 }, .{ 256, 1, 1 });
        try x.cut(8);
    }
    x.boundary(l.hc_attn, s.pre[0], s.pre[1], l.attn_norm);
    try x.cut(0);
    try attention(x, cache, i, l);
    x.expand();
    x.boundary(l.hc_ffn, s.pre[1], s.pre[0], l.ffn_norm);
    try x.cut(0);
    try moe(x, &l.moe, c.topk, c.experts);
    x.expand();
}

fn attention(x: Enc, cache: *st.Cache, layer: usize, l: *const wts.Layer) !void {
    const s = x.s;
    const m = x.m;
    const rows = x.rows;
    const a = &l.attn;
    const freq = m.freq(layer);
    // queries: q = wq_b(q_norm(wq_a(x))), RoPE on each head's last 64
    x.linear(a.wq_a, s.normed, 0, s.qr, 0, rows);
    x.rms(s.qr, 0, a.q_norm, s.qr, 0, rows, c.q_rank);
    x.linear(a.wq_b, s.qr, 0, s.q, 0, rows);
    x.rope(s.q, rows, c.heads, c.head_dim, s.pos, freq, false);
    // the window key: kv_norm(wkv(x)), RoPE'd, fp8-rounded, into the ring
    x.linear(a.wkv, s.normed, 0, s.kv, 0, rows);
    x.rms(s.kv, 0, a.kv_norm, s.kv, 0, rows, c.head_dim);
    x.rope(s.kv, rows, 1, c.head_dim, s.pos, freq, false);
    x.fp8(s.kv, rows * c.head_dim);
    x.put(s.kv, s.pos, cache.windows[layer], rows, c.head_dim, st.ring);
    try x.cut(1);

    var picks: usize = 0;
    var comp = cache.windows[layer]; // unread without picks
    const ratio: usize = c.ratios[layer];
    if (ratio != 0) {
        const src_layer = c.kvSource(layer).?;
        const si = std.mem.indexOfScalar(usize, &c.kv_sources, src_layer).?;
        const src = &cache.sources[si];
        if (a.compressor) |cp| try compress(x, cache, si, ratio, cp, a.indexer.?, freq);
        comp = src.latents;
        picks = st.picks;
        if (a.indexer) |ix| try index(x, src, layer, ratio, ix, freq);
    }
    try x.cut(2);
    // sparse attention over the window and the picks, then the output's RoPE taken back
    x.pipe("ds_attend");
    x.buf(s.q, 0, 0);
    x.buf(cache.windows[layer], 0, 1);
    x.buf(comp, 0, 2);
    x.buf(s.picks, 0, 3);
    x.tensor(a.sink, 4);
    x.buf(s.o, 0, 5);
    x.args(AttnArgs{ .first_pos = @intCast(x.base), .ring = st.ring, .window = c.window, .picks = @intCast(picks), .scale = 1.0 / @sqrt(@as(f32, c.head_dim)) }, 6);
    x.groups(.{ c.heads, rows, 1 }, .{ 256, 1, 1 });
    x.rope(s.o, rows, c.heads, c.head_dim, s.pos, freq, true);
    try x.cut(3);
    // the grouped low-rank output: fp8-rounded heads, block-diagonal wo_a (bf16), wo_b
    x.fp8(s.o, rows * c.heads * c.head_dim);
    x.gemv(false, a.wo_a, s.o, 0, c.heads * c.head_dim, s.oa, 0, rows, c.o_groups * c.o_rank, c.group_in, c.o_rank);
    x.linear(a.wo_b, s.oa, 0, s.branch, 0, rows);
    try x.cut(4);
}

/// A KV source's compressor: this call's completed groups pooled (ratio 2) or projected (ratio 1), normed, the index
/// keys derived from the RoPE-free latents, then the latents RoPE'd and fp4-rounded into the source's cache.
fn compress(x: Enc, cache: *st.Cache, si: usize, ratio: usize, cp: wts.Compressor, ix: wts.Indexer, freq: Buffer) !void {
    const s = x.s;
    const rows = x.rows;
    const src = &cache.sources[si];
    const span = completed(x.base, rows, ratio);
    const n = span[1] - span[0];
    if (span[1] > src.groups) return error.DeepseekCacheFull;
    if (ratio > 1) {
        x.gemv(true, cp.wkv, s.normed, 0, c.hidden, s.cproj, 0, rows, c.head_dim, c.hidden, c.head_dim);
        x.gemv(true, cp.wgate.?, s.normed, 0, c.hidden, s.cscore, 0, rows, c.head_dim, c.hidden, c.head_dim);
        x.putF32(s.cproj, s.pos, cache.proj_kv[si].?, rows, c.head_dim, st.ring);
        x.putF32(s.cscore, s.pos, cache.proj_score[si].?, rows, c.head_dim, st.ring);
        if (n == 0) return;
        x.pipe("ds_pool2");
        x.buf(cache.proj_kv[si].?, 0, 0);
        x.buf(cache.proj_score[si].?, 0, 1);
        x.buf(s.latent, 0, 2);
        x.args(PoolArgs{ .first = @intCast(span[0]), .ring = st.ring }, 3);
        x.threads(.{ c.head_dim, n, 1 }, .{ 256, 1, 1 });
    } else {
        x.gemv(false, cp.wkv, s.normed, 0, c.hidden, s.latent, 0, rows, c.head_dim, c.hidden, c.head_dim);
    }
    x.rms(s.latent, 0, cp.norm, s.latent, 0, n, c.head_dim);
    for (s.gpos[si].slice(i32, n), s.gidx[si].slice(i32, n), 0..) |*p, *g, j| {
        g.* = @intCast(span[0] + j);
        p.* = @intCast((span[0] + j) * ratio);
    }
    // index keys: k_norm(wk(latent)), RoPE'd at the group's first position, fp4 with e8m0 scales a 32
    x.gemv(false, ix.wk.?, s.latent, 0, c.head_dim, s.ik, 0, n, c.index_dim, c.head_dim, c.index_dim);
    x.rms(s.ik, 0, ix.k_norm.?, s.ik, 0, n, c.index_dim);
    x.rope(s.ik, n, 1, c.index_dim, s.gpos[si], freq, false);
    x.fp4(s.ik, n * c.index_dim, false);
    x.put(s.ik, s.gidx[si], src.keys, n, c.index_dim, 0);
    // the latents: RoPE'd at the same position, fp4 with e4m3 scales a 16
    x.rope(s.latent, n, 1, c.head_dim, s.gpos[si], freq, false);
    x.fp4(s.latent, n * c.head_dim, true);
    x.put(s.latent, s.gidx[si], src.latents, n, c.head_dim, 0);
}

/// An index source's picks: the indexer's scores over the visible keys (inside layer 20's candidate blocks after
/// it), each row's 512 best in key order, -1 after them. Rows are taken in slices that fit the score buffer.
fn index(x: Enc, src: *const st.Source, layer: usize, ratio: usize, ix: wts.Indexer, freq: Buffer) !void {
    const s = x.s;
    const rows = x.rows;
    const groups = (x.base + rows) / ratio;
    x.linear(ix.wq_b, s.qr, 0, s.iq, 0, rows);
    x.rope(s.iq, rows, c.index_heads, c.index_dim, s.pos, freq, false);
    x.fp4(s.iq, rows * c.index_heads * c.index_dim, false);
    x.gemv(false, ix.weights_proj, s.normed, 0, c.hidden, s.iw, 0, rows, c.index_heads, c.hidden, c.index_heads);
    x.pipe("ds_scale");
    x.buf(s.iw, 0, 0);
    x.args(@as(f32, 1.0 / @sqrt(@as(f32, c.index_dim)) * (1.0 / @sqrt(@as(f32, c.index_heads)))), 1);
    x.threads(.{ rows * c.index_heads, 1, 1 }, .{ 256, 1, 1 });
    if (groups == 0) {
        x.pipe("ds_topk");
        x.buf(s.scores, 0, 0);
        x.buf(s.picks, 0, 1);
        x.args(TopkArgs{ .n = 0, .k = st.picks, .stride = 1, .offset = 0 }, 2);
        x.groups(.{ rows, 1, 1 }, .{ 1024, 1, 1 });
        return;
    }
    const slice = @max(1, @min(rows, st.score_budget / 4 / groups));
    var r0: usize = 0;
    while (r0 < rows) : (r0 += slice) {
        const n = @min(slice, rows - r0);
        x.pipe("ds_index_scores");
        x.buf(s.iq, r0 * c.index_heads * c.index_dim * 2, 0);
        x.buf(s.iw, r0 * c.index_heads * 2, 1);
        x.buf(src.keys, 0, 2);
        x.buf(s.scores, 0, 3);
        x.args(ScoreArgs{ .rows = @intCast(n), .groups = @intCast(groups), .ratio = @intCast(ratio), .first_pos = @intCast(x.base + r0), .stride = @intCast(groups) }, 4);
        x.threads(.{ groups, n, 1 }, .{ @min(groups, 256), 1, 1 });
        const nb = (groups + c.candidate_block - 1) / c.candidate_block;
        if (layer == c.candidate_source) {
            x.pipe("ds_block_max");
            x.buf(s.scores, 0, 0);
            x.buf(s.blocks, 0, 1);
            x.args(BlockArgs{ .rows = @intCast(n), .groups = @intCast(groups), .block = c.candidate_block, .first_pos = @intCast(x.base + r0), .ratio = @intCast(ratio), .stride = @intCast(groups) }, 2);
            x.threads(.{ nb, n, 1 }, .{ @min(nb, 256), 1, 1 });
            x.pipe("ds_topk");
            x.buf(s.blocks, 0, 0);
            x.buf(s.keep, r0 * c.candidate_blocks * 4, 1);
            x.args(TopkArgs{ .n = @intCast(nb), .k = c.candidate_blocks, .stride = @intCast(groups), .offset = 0 }, 2);
            x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
        } else if (layer > c.candidate_source) {
            x.pipe("ds_candidate_mask");
            x.buf(s.scores, 0, 0);
            x.buf(s.keep, r0 * c.candidate_blocks * 4, 1);
            x.args(MaskArgs{ .groups = @intCast(groups), .stride = @intCast(groups), .block = c.candidate_block, .kept = c.candidate_blocks }, 2);
            x.threads(.{ groups, n, 1 }, .{ @min(groups, 256), 1, 1 });
        }
        x.pipe("ds_topk");
        x.buf(s.scores, 0, 0);
        x.buf(s.picks, r0 * st.picks * 4, 1);
        x.args(TopkArgs{ .n = @intCast(groups), .k = st.picks, .stride = @intCast(groups), .offset = 0 }, 2);
        x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
    }
}

/// Routed experts (row kernels by id) and the shared expert, summed in expert order as the reference does.
fn moe(x: Enc, mo: *const wts.Moe, topk: usize, experts: usize) !void {
    const s = x.s;
    const rows = x.rows;
    const slots = rows * topk;
    x.gemv(true, mo.router, s.normed, 0, c.hidden, s.router, 0, rows, experts, c.hidden, experts);
    x.pipe("ds_route");
    x.buf(s.router, 0, 0);
    x.tensor(mo.bias, 1);
    x.buf(s.route_pick, 0, 2);
    x.buf(s.route_weight, 0, 3);
    x.args(RouteArgs{ .experts = @intCast(experts), .topk = @intCast(topk), .scale = c.route_scale }, 4);
    x.groups(.{ rows, 1, 1 }, .{ (experts + 31) / 32 * 32, 1, 1 });
    try x.cut(5);
    x.indexed(mo.w1, slots, s.normed, s.route_pick, s.gate);
    x.indexed(mo.w3, slots, s.normed, s.route_pick, s.up);
    glu(x, slots, true);
    x.indexed(mo.w2, slots, s.act, s.route_pick, s.experts);
    try x.cut(6);
    x.linear(mo.shared.w1, s.normed, 0, s.gate, 0, rows);
    x.linear(mo.shared.w3, s.normed, 0, s.up, 0, rows);
    glu(x, rows, false);
    x.linear(mo.shared.w2, s.act, 0, s.shared, 0, rows);
    x.pipe("ds_moe_combine");
    x.buf(s.experts, 0, 0);
    x.buf(s.shared, 0, 1);
    x.buf(s.branch, 0, 2);
    x.args(@as(i32, @intCast(topk)), 3);
    x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
    try x.cut(7);
}

fn glu(x: Enc, n: usize, weighted: bool) void {
    const s = x.s;
    x.pipe("ds_swiglu");
    x.buf(s.gate, 0, 0);
    x.buf(s.up, 0, 1);
    x.buf(s.route_weight, 0, 2);
    x.buf(s.act, 0, 3);
    x.args(GluArgs{ .width = c.moe_inter, .weighted = @intFromBool(weighted), .limit = c.swiglu_limit }, 4);
    x.threads(.{ c.moe_inter, n, 1 }, .{ 256, 1, 1 });
}
