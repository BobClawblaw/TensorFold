//! One stream's rows through the backbone in one command buffer: Transformer.forward of DeepSeek's model.py, with
//! the caches its decode path keeps (window rings, the four sources' compressed latents and index keys).
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");
const Buffer = mtl.Buffer;

pub const Head = enum { last, all, none };

/// Per-layer outputs a check reads back (the streams after each block), or null.
pub const Dump = struct { layers: []Buffer };

/// A prompt's image spans: each position's code (-1 text; an aligner row; -2, -3, -4 a span's start, end and line
/// break) and the aligner rows of its images (bf16 [rows, 5120], image_processor's reading order).
pub const Vis = struct {
    codes: []const i32,
    rows: Buffer,

    fn code(v: ?*const Vis, pos: usize) i32 {
        const x = v orelse return -1;
        return if (pos < x.codes.len) x.codes[pos] else -1;
    }
};

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
const AttnArgs = extern struct { first_pos: i32, ring: i32, window: i32, picks: i32, scale: f32, window_end: i32 = -1 };
const RouteArgs = extern struct { experts: i32, topk: i32, scale: f32 };
const GluArgs = extern struct { width: i32, weighted: i32, limit: f32 };
const EngramArgs = extern struct { eps: f32 };
const SortArgs = extern struct { slots: i32, experts: i32 };
const MarkovArgs = extern struct { vocab: i32, rank: i32, step: i32 };
/// The draft block's filler after its first token (dspark_noise_token_id).
const noise_token: u32 = 128799;
/// Rows from which a call's projections take the tensor units (core/affine_mm), on GPUs that have them.
const nax_rows = 16;
/// A buffer and byte offset as core/affine_mm binds them.
const Ref = struct { buf: Buffer, off: usize = 0 };

fn matrix(w: row.Weights) struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32 } {
    return .{ .w = .{ .buf = w.w, .off = w.w_off }, .s = .{ .buf = w.scales, .off = w.s_off }, .b = .{ .buf = w.biases, .off = w.b_off }, .n = @intCast(w.n), .k = @intCast(w.k) };
}

/// GPU seconds by section when set (`sections` names them): each section then runs in its own command buffer,
/// waited for (a profile, not a served speed).
pub var profile: ?*[sections.len]f64 = null;
/// Seconds a call spends encoding (to its commit), on the GPU, and in all, summed over calls when set.
pub var timing: ?*[3]f64 = null;
/// A check's ablation (DS_SKIP): dispatches of the pipeline of this name are left out, to time what each costs.
pub var skip: ?[]const u8 = null;
var skipping = false;
/// Dispatches encoded, counted for a check.
pub var dispatches: usize = 0;
pub var trace = false;
/// The next dispatch runs alongside the one before it (`Enc.with`): no barrier between them.
var joined = false;
var last_pipe: []const u8 = "";
pub const sections = [_][]const u8{ "boundaries", "attn proj", "compress+index", "attend", "attn out", "route", "experts", "shared", "engram", "head", "host engram (CPU)" };

const Enc = struct {
    ep: *mtl.ComputeEncoder,
    cbp: *mtl.CommandBuffer,
    m: *Model,
    s: *st.Scratch,
    rows: usize,
    base: usize, // the first row's position

    /// The tensor-unit pipelines for `w` when this call is a prompt chunk of nax_rows or more.
    fn nax(self: Enc, w: row.Weights) ?am.Pipes {
        if (self.rows < nax_rows) return null;
        return self.m.kernels.naxPipes(.{ .bits = w.bits, .group = @intCast(w.group) });
    }

    /// With `profile`, everything encoded since the last cut runs now and its GPU time counts for `section`.
    fn cut(self: Enc, section: usize) !void {
        const p = profile orelse return;
        self.ep.*.end();
        self.cbp.*.commit();
        self.cbp.*.wait();
        if (self.cbp.*.status() != .completed) return error.DeepseekGpuFailed;
        p[section] += self.cbp.*.gpuSeconds();
        self.cbp.* = self.m.queue.commandBuffer();
        self.ep.* = self.cbp.*.compute(.concurrent);
    }

    /// Calls encode into a concurrent encoder: every dispatch waits for all before it (a barrier), unless `with` was
    /// called just before it, for a dispatch that touches nothing the one before it writes or reads.
    fn fence(self: Enc) void {
        if (joined) {
            joined = false;
            return;
        }
        self.ep.*.barrier();
    }
    fn with(self: Enc) void {
        _ = self;
        joined = true;
    }

    fn pipe(self: Enc, comptime key: []const u8) void {
        skipping = if (skip) |k| std.mem.eql(u8, k, key) else false;
        last_pipe = key;
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
        if (skipping) return;
        dispatches += 1;
        self.fence();
        if (trace) std.debug.print("{s} {d}x{d}x{d}\n", .{ last_pipe, grid[0], grid[1], grid[2] });
        self.ep.*.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    fn groups(self: Enc, grid: [3]usize, group: [3]usize) void {
        if (skipping) return;
        dispatches += 1;
        self.fence();
        if (trace) std.debug.print("{s} {d}x{d}x{d}\n", .{ last_pipe, grid[0], grid[1], grid[2] });
        self.ep.*.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    /// y[rows, n] = x[rows, k] W^T: the tensor units for a prompt chunk's rows, else the row kernels.
    fn linear(self: Enc, w: wts.Linear, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) void {
        if (rows >= nax_rows) if (self.nax(w)) |p| {
            const xr = Ref{ .buf = x, .off = x_off };
            const sums = Ref{ .buf = self.s.sums };
            self.fence();
            am.rowSums(self.ep.*, p, @intCast(w.group), xr, sums, @intCast(rows), @intCast(w.k));
            self.fence();
            am.dense(self.ep.*, p, xr, sums, matrix(w), Ref{ .buf = y, .off = y_off }, @intCast(rows));
            return;
        };
        skipping = if (skip) |k| std.mem.eql(u8, k, "rows") else false;
        last_pipe = "rows";
        if (skipping) return;
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
        skipping = if (skip) |k| std.mem.eql(u8, k, "indexed") else false;
        last_pipe = "indexed";
        if (skipping) return;
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

    /// s.oa [rows, 8 x 1024] = each group's 4,096 of s.o [rows, 64 x 512] times its block of wo_a.
    fn woA(self: Enc, w: wts.WoA, rows: usize) void {
        const s = self.s;
        switch (w) {
            .bf16 => |t| self.gemv(false, t, s.o, 0, c.heads * c.head_dim, s.oa, 0, rows, c.o_groups * c.o_rank, c.group_in, c.o_rank),
            .affine => |x| {
                if (rows >= nax_rows) if (self.nax(x.one)) |p| {
                    const g: i32 = c.o_groups;
                    const xr = Ref{ .buf = s.o };
                    const sums = Ref{ .buf = s.sums };
                    self.fence();
                    am.rowSums(self.ep.*, p, @intCast(x.one.group), xr, sums, @intCast(rows * c.o_groups), c.group_in);
                    self.fence();
                    am.denseBatch(self.ep.*, p, xr, sums, matrix(x.one), Ref{ .buf = s.oa }, @intCast(rows), c.o_groups, .{ .x_row = g * c.group_in, .y_row = g * c.o_rank, .sums_row = g, .x_batch = c.group_in, .y_batch = c.o_rank, .sums_batch = 1, .w_batch = c.o_rank });
                    return;
                };
                self.indexed(x, rows * c.o_groups, s.o, s.wo_ids, s.oa);
            },
        }
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

    /// Attention over each row's window (ending at `window_end`, or the row's own position when negative) and its
    /// `picks` compressed keys, each head's RoPE then taken back (`freq`) and the heads fp8-rounded: split by keys for
    /// decode-sized calls (the finish inside the merge), the shared-tile kernel for prompt chunks.
    fn attend(self: Enc, q: Buffer, ring: Buffer, comp: Buffer, picks_buf: Buffer, picks: usize, sink: wts.Tensor, o: Buffer, window_end: i32, freq: Buffer) void {
        const s = self.s;
        const aa = AttnArgs{ .first_pos = @intCast(self.base), .ring = st.ring, .window = c.window, .picks = @intCast(picks), .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .window_end = window_end };
        if (self.rows >= nax_rows) {
            self.pipe("ds_attend_rows");
            self.buf(q, 0, 0);
            self.buf(ring, 0, 1);
            self.buf(comp, 0, 2);
            self.buf(picks_buf, 0, 3);
            self.tensor(sink, 4);
            self.buf(o, 0, 5);
            self.args(aa, 6);
            self.groups(.{ c.heads / 8, self.rows, 1 }, .{ 512, 1, 1 });
            self.rope(o, self.rows, c.heads, c.head_dim, s.pos, freq, true);
            self.fp8(o, self.rows * c.heads * c.head_dim);
            return;
        }
        const chunks: usize = (c.window + picks + 63) / 64;
        self.pipe("ds_attend_split");
        self.buf(q, 0, 0);
        self.buf(ring, 0, 1);
        self.buf(comp, 0, 2);
        self.buf(picks_buf, 0, 3);
        self.buf(s.attn_po, 0, 4);
        self.buf(s.attn_ml, 0, 5);
        self.args(aa, 6);
        self.groups(.{ c.heads / 8, chunks, self.rows }, .{ 256, 1, 1 });
        self.pipe("ds_attend_merge");
        self.buf(s.attn_po, 0, 0);
        self.buf(s.attn_ml, 0, 1);
        self.tensor(sink, 2);
        self.buf(o, 0, 3);
        self.args([2]i32{ @intCast(chunks), 1 }, 4);
        self.buf(s.pos, 0, 5);
        self.buf(freq, 0, 6);
        self.groups(.{ c.heads, self.rows, 1 }, .{ 256, 1, 1 });
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

    /// The window keys' tail in place (kv_norm, RoPE, fp8), then into `ring` at their positions when given.
    fn kvFinish(self: Enc, kv: Buffer, norm: wts.Tensor, freq: Buffer, ring: ?Buffer) void {
        self.pipe("ds_kv_finish");
        self.buf(kv, 0, 0);
        self.tensor(norm, 1);
        self.buf(self.s.pos, 0, 2);
        self.buf(freq, 0, 3);
        self.buf(ring orelse kv, 0, 4);
        self.args([2]f32{ c.norm_eps, if (ring != null) @floatFromInt(st.ring) else 0 }, 5);
        self.groups(.{ self.rows, 1, 1 }, .{ 256, 1, 1 });
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
        if (hc) |h| {
            self.pipe("ds_hc_mix_part");
            self.buf(s.x, 0, 0);
            self.tensor(h.fn_, 1);
            self.buf(s.hc_part, 0, 2);
            self.groups(.{ c.hc * c.hidden / 1024, self.rows, 1 }, .{ 256, 1, 1 });
        }
        if (hc != null) self.pipe("ds_hc_boundary") else self.pipe("ds_hc_collapse");
        self.buf(s.x, 0, 0);
        if (hc) |h| {
            self.buf(s.hc_part, 0, 1);
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
    return runWith(m, s, cache, tokens, head, dump, null);
}

/// `run` over prompt rows that may lie in image spans (`vis`): their rows are the tower's, not the token embedding.
pub fn runWith(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, dump: ?Dump, vis: ?*const Vis) !void {
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
    var images = false;
    for (tokens, s.img.slice(i32, rows), s.img_code.slice(i32, rows), 0..) |t, *flag, *code, i| {
        code.* = Vis.code(vis, base + i);
        flag.* = @intFromBool(code.* != -1);
        images = images or code.* != -1;
        cache.engram_ids.appendAssumeCapacity(m.engram.compress(t, code.* == -1)); // image rows end every n-gram
    }
    var h0: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &h0);
    var dsts: [c.engram_layers.len][]u16 = undefined;
    for (&dsts, s.engram_rows) |*d, b| d.* = b.slice(u16, rows * c.engram_cols * c.engram_dim);
    try m.engram.gatherRows(cache.engram_ids.items, base, rows, dsts);
    if (profile) |p| {
        var h1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &h1);
        p[10] += @as(f64, @floatFromInt(h1.sec - h0.sec)) + @as(f64, @floatFromInt(h1.nsec - h0.nsec)) / 1e9;
    }
    for (s.pre[0].slice(f32, rows * c.hc), 0..) |*v, i| v.* = if (i % c.hc == 0) 1 else 0;

    var t_start: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t_start);
    var cb = m.queue.commandBuffer();
    const w = &m.weights;
    var e = cb.compute(.concurrent);
    const x: Enc = .{ .ep = &e, .cbp = &cb, .m = m, .s = s, .rows = rows, .base = base };
    x.pipe("ds_embed");
    x.buf(s.ids, 0, 0);
    x.tensor(w.embed, 1);
    x.buf(s.x, 0, 2);
    x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
    if (images) {
        const v = &(m.vision orelse return error.NoVisionTower);
        x.pipe("ds_put_image");
        x.buf(s.x, 0, 0);
        x.buf(s.img_code, 0, 1);
        x.buf(vis.?.rows, 0, 2);
        x.tensor(v.start, 3);
        x.tensor(v.end, 4);
        x.tensor(v.newline, 5);
        x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
    }
    const drafting = w.stages != null and cache.stage_rings != null;
    for (&w.layers, 0..) |*l, i| {
        if (drafting) if (std.mem.indexOfScalar(usize, &c.draft_targets, i)) |t| {
            // the draft head reads its target layers' attention inputs: the mean of the four streams
            x.pipe("ds_stream_mean");
            x.buf(s.x, 0, 0);
            x.buf(s.mh, 0, 1);
            x.args([2]i32{ @intCast(t * c.hidden), @intCast(c.draft_targets.len * c.hidden) }, 2);
            x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
        };
        try block(x, cache, i, l);
        if (dump) |d| { // the streams after each block: the buffers are untracked, so each block completes first
            e.end();
            cb.commit();
            cb.wait();
            if (cb.status() != .completed) return error.DeepseekGpuFailed;
            @memcpy(d.layers[i].contents()[0 .. rows * c.hc * c.hidden * 2], s.x.contents()[0 .. rows * c.hc * c.hidden * 2]);
            cb = m.queue.commandBuffer();
            e = cb.compute(.concurrent);
        }
    }
    if (drafting) condition(x, cache);
    try x.cut(0);
    // the last block's FFN pre collapses the streams for the head
    x.boundary(null, s.pre[0], s.pre[1], w.norm);
    switch (head) {
        .none => {},
        .last => x.linear(w.head, s.normed, (rows - 1) * c.hidden * 2, s.logits, 0, 1),
        .all => x.linear(w.head, s.normed, 0, s.logits, 0, rows),
    }
    e.end();
    var t_commit: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t_commit);
    cb.commit();
    cb.wait();
    if (profile) |p| p[9] += cb.gpuSeconds();
    if (timing) |t| {
        var t_end: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &t_end);
        t[0] += secs(t_start, t_commit);
        t[1] += cb.gpuSeconds();
        t[2] += secs(t_start, t_end);
    }
    if (cb.status() != .completed) {
        if (cb.failure()) |why| std.log.err("deepseek_v41: the forward failed: {s}", .{why});
        return error.DeepseekGpuFailed;
    }
    cache.len += rows;
    if (head != .none) {
        const last = if (head == .last) 0 else rows - 1;
        @memcpy(cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + last * c.vocab * 2)[0 .. c.vocab * 2]);
    }
    cache.last_start = base;
    cache.last_rows = rows;
}

/// The draft head's view of this call's rows: main_norm(main_proj(the target layers' means)), and each stage's window
/// key of every row (its kv_norm(wkv(...)), RoPE'd at the row's position, fp8-rounded) into the stage's ring.
fn condition(x: Enc, cache: *st.Cache) void {
    const s = x.s;
    const stages = &x.m.weights.stages.?;
    x.linear(stages[0].main_proj.?, s.mh, 0, s.mainx, 0, x.rows);
    x.rms(s.mainx, 0, stages[0].main_norm.?, s.mainx, 0, x.rows, c.hidden);
    for (stages, cache.stage_rings.?, 0..) |*stage, ring, k| {
        const a = &stage.block.attn;
        x.linear(a.wkv, s.mainx, 0, s.kv, 0, x.rows);
        x.rms(s.kv, 0, a.kv_norm, s.kv, 0, x.rows, c.head_dim);
        x.rope(s.kv, x.rows, 1, c.head_dim, s.pos, x.m.freq(c.layers + k), false);
        x.fp8(s.kv, x.rows * c.head_dim);
        x.put(s.kv, s.pos, ring, x.rows, c.head_dim, st.ring);
    }
}

/// One DSpark round from position `at` (the last committed row): the pending `token` at at + 1 and four noise
/// tokens after it through the three stages, the head over the five rows, then five Markov-biased greedy picks.
/// The drafts land in `out` (the tokens for positions at + 2 .. at + 6). `ds` is the draft call's own scratch.
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
        try moe(x, &l.moe, c.draft_topk, c.draft_experts);
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

/// A draft stage's attention: its five rows' queries and keys, over the stage's window of target keys ending at
/// `at` and the five rows' own keys (every row sees all five, as get_dspark_topk_idxs lists them).
fn stageAttention(x: Enc, l: *const wts.Layer, ring: Buffer, layer: usize, at: usize) void {
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
    x.rope(s.q, n, c.heads, c.head_dim, s.pos, freq, false);
    x.attend(s.q, ring, s.kv, s.draft_picks, c.draft_block, a.sink, s.o, @intCast(at), freq);
    x.woA(a.wo_a, n);
    x.linear(a.wo_b, s.oa, 0, s.branch, 0, n);
}

fn secs(a: std.c.timespec, b: std.c.timespec) f64 {
    return @as(f64, @floatFromInt(b.sec - a.sec)) + @as(f64, @floatFromInt(b.nsec - a.nsec)) / 1e9;
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
        x.args(EngramArgs{ .eps = c.norm_eps }, 4);
        x.buf(s.img, 0, 5);
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
    // and the window key: kv_norm(wkv(x)), RoPE'd, fp8-rounded, into the ring; the two chains side by side
    x.linear(a.wq_a, s.normed, 0, s.qr, 0, rows);
    x.with();
    x.linear(a.wkv, s.normed, 0, s.kv, 0, rows);
    x.rms(s.qr, 0, a.q_norm, s.qr, 0, rows, c.q_rank);
    x.with();
    x.kvFinish(s.kv, a.kv_norm, freq, cache.windows[layer]);
    x.linear(a.wq_b, s.qr, 0, s.q, 0, rows);
    x.rope(s.q, rows, c.heads, c.head_dim, s.pos, freq, false);
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
    x.attend(s.q, cache.windows[layer], comp, s.picks, picks, a.sink, s.o, -1, freq);
    try x.cut(3);
    // the grouped low-rank output: block-diagonal wo_a, wo_b
    x.woA(a.wo_a, rows);
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
    const side = x.rows < nax_rows; // decode-sized: the shared expert runs beside the routed path
    x.gemv(true, mo.router, s.normed, 0, c.hidden, s.router, 0, rows, experts, c.hidden, experts);
    if (side) {
        x.with();
        x.linear(mo.shared.w1, s.normed, 0, s.sh_gate, 0, rows);
        x.with();
        x.linear(mo.shared.w3, s.normed, 0, s.sh_up, 0, rows);
    }
    x.pipe("ds_route");
    x.buf(s.router, 0, 0);
    x.tensor(mo.bias, 1);
    x.buf(s.route_pick, 0, 2);
    x.buf(s.route_weight, 0, 3);
    x.args(RouteArgs{ .experts = @intCast(experts), .topk = @intCast(topk), .scale = c.route_scale }, 4);
    x.buf(s.img, 0, 5);
    x.tensor(mo.bias_vl, 6);
    x.groups(.{ rows, 1, 1 }, .{ (experts + 31) / 32 * 32, 1, 1 });
    if (side) {
        x.with();
        gluTo(x, rows, false, s.route_weight, s.sh_gate, s.sh_up, s.sh_act);
    }
    try x.cut(5);
    if (x.nax(mo.w1.one)) |p| {
        // a prompt chunk: slots sorted by expert, each expert's rows through its weights once on the tensor units
        const e = x.ep.*;
        x.pipe("ds_sort_slots");
        x.buf(s.route_pick, 0, 0);
        x.buf(s.route_weight, 0, 1);
        x.buf(s.offs, 0, 2);
        x.buf(s.order, 0, 3);
        x.buf(s.wsort, 0, 4);
        x.args(SortArgs{ .slots = @intCast(slots), .experts = @intCast(experts) }, 5);
        x.threads(.{ 1, 1, 1 }, .{ 1, 1, 1 });
        x.pipe("ds_gather_rows");
        x.buf(s.normed, 0, 0);
        x.buf(s.order, 0, 1);
        x.buf(s.xsort, 0, 2);
        x.args([2]i32{ c.hidden, @intCast(topk) }, 3);
        x.threads(.{ c.hidden, slots, 1 }, .{ 256, 1, 1 });
        const sums = Ref{ .buf = s.sums };
        const offs = Ref{ .buf = s.offs };
        x.fence();
        am.rowSums(e, p, 64, Ref{ .buf = s.xsort }, sums, @intCast(slots), c.hidden);
        x.fence();
        am.gather(e, p, Ref{ .buf = s.xsort }, sums, matrix(mo.w1.one), offs, Ref{ .buf = s.gate }, @intCast(slots), @intCast(experts));
        x.with();
        x.fence();
        am.gather(e, p, Ref{ .buf = s.xsort }, sums, matrix(mo.w3.one), offs, Ref{ .buf = s.up }, @intCast(slots), @intCast(experts));
        gluWith(x, slots, true, s.wsort);
        x.fence();
        am.rowSums(e, p, 64, Ref{ .buf = s.act }, sums, @intCast(slots), c.moe_inter);
        x.fence();
        am.gatherTo(e, p, Ref{ .buf = s.act }, sums, matrix(mo.w2.one), offs, Ref{ .buf = s.experts }, Ref{ .buf = s.order }, @intCast(slots), @intCast(experts));
    } else {
        x.indexed(mo.w1, slots, s.normed, s.route_pick, s.gate);
        x.with();
        x.indexed(mo.w3, slots, s.normed, s.route_pick, s.up);
        x.with();
        x.linear(mo.shared.w2, s.sh_act, 0, s.shared, 0, rows);
        glu(x, slots, true);
        x.indexed(mo.w2, slots, s.act, s.route_pick, s.experts);
    }
    try x.cut(6);
    if (!side) {
        x.linear(mo.shared.w1, s.normed, 0, s.gate, 0, rows);
        x.linear(mo.shared.w3, s.normed, 0, s.up, 0, rows);
        glu(x, rows, false);
        x.linear(mo.shared.w2, s.act, 0, s.shared, 0, rows);
    }
    x.pipe("ds_moe_combine");
    x.buf(s.experts, 0, 0);
    x.buf(s.shared, 0, 1);
    x.buf(s.branch, 0, 2);
    x.args(@as(i32, @intCast(topk)), 3);
    x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
    try x.cut(7);
}

fn glu(x: Enc, n: usize, weighted: bool) void {
    gluWith(x, n, weighted, x.s.route_weight);
}

fn gluWith(x: Enc, n: usize, weighted: bool, weights: Buffer) void {
    gluTo(x, n, weighted, weights, x.s.gate, x.s.up, x.s.act);
}

fn gluTo(x: Enc, n: usize, weighted: bool, weights: Buffer, gate: Buffer, up: Buffer, act: Buffer) void {
    x.pipe("ds_swiglu");
    x.buf(gate, 0, 0);
    x.buf(up, 0, 1);
    x.buf(weights, 0, 2);
    x.buf(act, 0, 3);
    x.args(GluArgs{ .width = c.moe_inter, .weighted = @intFromBool(weighted), .limit = c.swiglu_limit }, 4);
    x.threads(.{ c.moe_inter, n, 1 }, .{ 256, 1, 1 });
}
