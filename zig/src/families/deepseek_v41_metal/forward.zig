//! One stream's rows through the backbone in one command buffer, with the caches DeepSeek's decode keeps.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");
const idx = @import("index.zig");
const ffn = @import("moe.zig");
const dspark = @import("draft.zig");
const layer = @import("layer.zig");
const kern = @import("kernels.zig");
const ahead = @import("ahead.zig");
const Buffer = mtl.Buffer;

pub const Head = enum { last, all, none };

/// One DSpark round (draft.zig).
pub const draft = dspark.draft;

/// Per-layer outputs a check reads back (the streams after each block), or null.
pub const Dump = struct { layers: []Buffer };

/// A prompt's image spans: each position's code (-1 text, an aligner row, -2/-3/-4 start/end/break), their rows.
pub const Vis = struct {
    codes: []const i32,
    rows: Buffer,

    pub fn code(v: ?*const Vis, pos: usize) i32 {
        const x = v orelse return -1;
        return if (pos < x.codes.len) x.codes[pos] else -1;
    }
};

const HcArgs = extern struct { eps: f32, hc_eps: f32, iters: i32 };
const RowsArgs = extern struct { n: i32, x_stride: i32, y_stride: i32, eps: f32 };
const GemvArgs = extern struct { rows: i32, n: i32, k: i32, group_n: i32, x_stride: i32, y_stride: i32 };
const RopeArgs = extern struct { heads: i32, stride: i32, offset: i32, inverse: i32 };
const CopyArgs = extern struct { width: i32, ring: i32 };
const AttnArgs = extern struct { first_pos: i32, ring: i32, window: i32, picks: i32, scale: f32, window_end: i32 = -1 };
/// Rows from which a call's projections take the tensor units (core/affine_mm), on GPUs that have them.
pub const nax_rows = 16;
/// A buffer and byte offset as core/affine_mm binds them.
pub const Ref = struct { buf: Buffer, off: usize = 0 };

pub fn matrix(w: row.Weights) struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32 } {
    return .{ .w = .{ .buf = w.w, .off = w.w_off }, .s = .{ .buf = w.scales, .off = w.s_off }, .b = .{ .buf = w.biases, .off = w.b_off }, .n = @intCast(w.n), .k = @intCast(w.k) };
}

/// GPU seconds by section when set: each section then runs in its own waited command buffer (a profile only).
pub var profile: ?*[sections.len]f64 = null;
/// Seconds a call spends encoding (to its commit), on the GPU, and in all, summed over calls when set.
pub var timing: ?*[3]f64 = null;
/// A check's ablation (DS_SKIP): dispatches of the pipeline of this name are left out, to time what each costs.
pub var skip: ?[]const u8 = null;
var skipping = false;

/// Whether DS_SKIP names `key`: a comma-separated list, or "!a,b" for everything but those.
fn skips(key: []const u8) bool {
    const k = skip orelse return false;
    const but = k.len > 0 and k[0] == '!';
    var it = std.mem.splitScalar(u8, if (but) k[1..] else k, ',');
    while (it.next()) |name| if (std.mem.eql(u8, name, key)) return !but;
    return but;
}
/// Heads a split-attention threadgroup takes (v41_attend.metal's HG).
const split_heads = 16;
/// The next dispatch runs alongside the one before it (`Enc.with`): no barrier between them.
var joined = false;
/// A sublayer's branch waits to be expanded: the next mixing boundary does it; other stream readers flush it.
pub var pending_expand = false;
/// The pending branch is the MoE's, still to be combined from its routed (this many a row) and shared rows.
pub var pending_topk: i32 = 0;
pub const sections = [_][]const u8{ "boundaries", "attn proj", "compress+index", "attend", "attn out", "route", "experts", "shared", "engram", "head", "host engram (CPU)", "engram ahead (CPU)" };
/// A lone stream's prompt chunk scores its index keys on the tensor units (a check turns them off: DS_INDEX_ROWS).
pub var index_nax = true;

/// The chunk after this call's, its engram rows gathered on the host while this one runs on the GPU (`runChunk`).
pub const Next = struct { tokens: []const u32, vis: ?*const Vis = null };

pub const Enc = struct {
    ep: *mtl.ComputeEncoder,
    cbp: *mtl.CommandBuffer,
    m: *Model,
    s: *st.Scratch,
    rows: usize,
    base: usize, // the first row's position (a stream's view; meaningless across streams)
    r0: usize = 0, // a stream's view: its first scratch row, where its row-indexed buffers bind
    units: bool = false, // a lone stream's prompt chunk: the tensor units (shared forwards keep the row kernels)
    segs: []const Seg = &.{}, // the streams of the forward, in scratch row order

    /// The view of one stream's rows: its position, its scratch rows.
    pub fn of(self: Enc, g: Seg) Enc {
        var v = self;
        v.rows = g.tokens.len;
        v.base = g.cache.len;
        v.r0 = g.r0;
        return v;
    }

    /// `b` bound at this view's first row, `row_bytes` a row.
    pub fn at(self: Enc, b: Buffer, row_bytes: usize, slot: usize) void {
        self.ep.*.setBuffer(b, self.r0 * row_bytes, slot);
    }

    /// The tensor-unit pipelines for `w` when this call is a lone stream's prompt chunk.
    pub fn nax(self: Enc, w: row.Weights) ?am.Pipes {
        if (!self.units) return null;
        return self.m.kernels.naxPipes(.{ .bits = w.bits, .group = @intCast(w.group) });
    }

    /// With `profile`, everything encoded since the last cut runs now and its GPU time counts for `section`.
    pub fn cut(self: Enc, section: usize) !void {
        const p = profile orelse return;
        self.ep.*.end();
        self.cbp.*.commit();
        self.cbp.*.wait();
        if (self.cbp.*.status() != .completed) return error.DeepseekGpuFailed;
        p[section] += self.cbp.*.gpuSeconds();
        self.cbp.* = self.m.queue.commandBuffer();
        self.ep.* = self.cbp.*.compute(.concurrent);
    }

    /// A concurrent encoder: each dispatch waits on all before it, unless `with` was called just before it.
    pub fn fence(self: Enc) void {
        if (joined) {
            joined = false;
            return;
        }
        self.ep.*.barrier();
    }
    /// A left-out dispatch (DS_SKIP) still takes a `with` meant for it, so the next one keeps its barrier.
    fn skipped() bool {
        if (skipping) joined = false;
        return skipping;
    }
    pub fn with(self: Enc) void {
        _ = self;
        joined = true;
    }

    pub fn pipe(self: Enc, comptime key: []const u8) void {
        skipping = skips(key);
        self.ep.*.setPipeline(self.m.kernels.get(key));
    }
    /// DS_SKIP for a pipeline set by hand (the tensor-unit ones outside `names`).
    pub fn skipAs(self: Enc, key: []const u8) void {
        _ = self;
        skipping = skips(key);
    }
    pub fn buf(self: Enc, b: Buffer, off: usize, slot: usize) void {
        self.ep.*.setBuffer(b, off, slot);
    }
    pub fn tensor(self: Enc, t: wts.Tensor, slot: usize) void {
        self.ep.*.setBuffer(t.buffer, t.offset, slot);
    }
    pub fn args(self: Enc, v: anytype, slot: usize) void {
        self.ep.*.setBytes(std.mem.asBytes(&v), slot);
    }
    pub fn threads(self: Enc, grid: [3]usize, group: [3]usize) void {
        if (skipped()) return;
        self.fence();
        self.ep.*.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    pub fn groups(self: Enc, grid: [3]usize, group: [3]usize) void {
        if (skipped()) return;
        self.fence();
        self.ep.*.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    /// y[rows, n] = x[rows, k] W^T: the tensor units for a prompt chunk's rows, else the row kernels.
    pub fn linear(self: Enc, w: wts.Linear, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) void {
        if (rows >= nax_rows) if (self.nax(w)) |p| {
            const xr = Ref{ .buf = x, .off = x_off };
            // one beside the projection before it (`with`) keeps its group sums apart: the buffer's second half
            const half = self.s.sums.length() / 2 & ~@as(usize, 15);
            std.debug.assert(!joined or rows * (w.k / w.group) * 4 <= half);
            const sums = Ref{ .buf = self.s.sums, .off = if (joined) half else 0 };
            self.fence();
            am.rowSums(self.ep.*, p, @intCast(w.group), xr, sums, @intCast(rows), @intCast(w.k));
            self.fence();
            am.dense(self.ep.*, p, xr, sums, matrix(w), Ref{ .buf = y, .off = y_off }, @intCast(rows));
            return;
        };
        skipping = skips("rows");
        if (skipped()) return;
        // a window's rows on the tiles: each weight read once for 8 rows, a row's sums the same at any width
        if (self.m.kernels.tiles) |*t| if (kern.tiled(w)) {
            if (row.simd_bits.call(t, w.bits, w.n, w.k, rows, w.n, rows <= 2)) |call| {
                const m = row.simd_bits.Matrix{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off };
                self.fence();
                row.simd_bits.encode(self.ep.*, call, m, x, x_off, y, y_off);
                return;
            } else |_| {}
        };
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
    pub fn indexed(self: Enc, x: row.Experts, slots: usize, in: Buffer, ids: Buffer, out: Buffer) void {
        skipping = skips("indexed");
        if (skipped()) return;
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
    pub fn woA(self: Enc, w: wts.WoA, rows: usize) void {
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
    pub fn gemv(self: Enc, comptime out_f32: bool, w: wts.Tensor, x: Buffer, x_off: usize, x_stride: usize, y: Buffer, y_off: usize, rows: usize, n: usize, k: usize, group_n: usize) void {
        if (out_f32) self.pipe("ds_gemv_bf_f32") else self.pipe("ds_gemv_bf_bf");
        self.buf(x, x_off, 0);
        self.tensor(w, 1);
        self.buf(y, y_off, 2);
        self.args(GemvArgs{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(k), .group_n = @intCast(group_n), .x_stride = @intCast(x_stride), .y_stride = @intCast(n) }, 3);
        self.groups(.{ (n + 7) / 8, (rows + 7) / 8, 1 }, .{ 256, 1, 1 });
    }

    /// Attention over rows' windows and `picks` compressed keys (RoPE in and out, fp8): split by keys, or by row.
    pub fn attend(self: Enc, q: Buffer, ring: Buffer, comp: Buffer, picks_buf: Buffer, picks: usize, sink: wts.Tensor, o: Buffer, window_end: i32, freq: Buffer) void {
        const s = self.s;
        const aa = AttnArgs{ .first_pos = @intCast(self.base), .ring = st.ring, .window = c.window, .picks = @intCast(picks), .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .window_end = window_end };
        const qo = c.heads * c.head_dim * 2;
        if (self.units and self.rows >= nax_rows) {
            self.rope(q, self.rows, c.heads, c.head_dim, s.pos, freq, false);
            const tensor_attend = self.m.kernels.attend_nax; // on the tensor units when the GPU has them
            if (tensor_attend) |p| {
                skipping = skips("ds_attend_nax");
                self.ep.*.setPipeline(p);
            } else self.pipe("ds_attend_rows");
            self.buf(q, 0, 0);
            self.buf(ring, 0, 1);
            self.buf(comp, 0, 2);
            self.buf(picks_buf, 0, 3);
            self.tensor(sink, 4);
            self.buf(o, 0, 5);
            self.args(aa, 6);
            if (tensor_attend != null) self.groups(.{ c.heads / 16, self.rows, 1 }, .{ 128, 1, 1 }) else self.groups(.{ c.heads / 8, self.rows, 1 }, .{ 512, 1, 1 });
            self.rope(o, self.rows, c.heads, c.head_dim, s.pos, freq, true);
            self.fp8(o, self.rows, c.heads * c.head_dim);
            return;
        }
        const chunks: usize = (c.window + picks + 15) / 16;
        self.pipe("ds_attend_split");
        self.at(q, qo, 0);
        self.buf(ring, 0, 1);
        self.buf(comp, 0, 2);
        self.at(picks_buf, picks * 4, 3);
        self.buf(s.attn_po, 0, 4);
        self.buf(s.attn_ml, 0, 5);
        self.args(aa, 6);
        self.at(s.pos, 4, 7);
        self.buf(freq, 0, 8);
        self.groups(.{ c.heads / split_heads, chunks, self.rows }, .{ split_heads * 32, 1, 1 });
        self.pipe("ds_attend_merge");
        self.buf(s.attn_po, 0, 0);
        self.buf(s.attn_ml, 0, 1);
        self.tensor(sink, 2);
        self.at(o, qo, 3);
        self.args(@as(i32, @intCast(chunks)), 4);
        self.at(s.pos, 4, 5);
        self.buf(freq, 0, 6);
        self.groups(.{ c.heads, self.rows, 1 }, .{ c.head_dim, 1, 1 });
    }

    pub fn rms(self: Enc, x: Buffer, x_off: usize, w: wts.Tensor, y: Buffer, y_off: usize, rows: usize, n: usize) void {
        self.pipe("ds_rms");
        self.buf(x, x_off, 0);
        self.tensor(w, 1);
        self.buf(y, y_off, 2);
        self.args(RowsArgs{ .n = @intCast(n), .x_stride = @intCast(n), .y_stride = @intCast(n), .eps = c.norm_eps }, 3);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
    }

    pub fn rope(self: Enc, x: Buffer, rows: usize, heads: usize, stride: usize, positions: Buffer, freq: Buffer, inverse: bool) void {
        self.pipe("ds_rope");
        self.at(x, heads * stride * 2, 0);
        self.at(positions, 4, 1);
        self.buf(freq, 0, 2);
        self.args(RopeArgs{ .heads = @intCast(heads), .stride = @intCast(stride), .offset = @intCast(stride - c.rope_dim), .inverse = @intFromBool(inverse) }, 3);
        self.threads(.{ c.rope_dim / 2, heads, rows }, .{ c.rope_dim / 2, 1, 1 });
    }

    pub fn fp8(self: Enc, x: Buffer, rows: usize, width: usize) void {
        self.pipe("ds_fp8_qdq");
        self.at(x, width * 2, 0);
        self.threads(.{ rows * width / 32, 1, 1 }, .{ 64, 1, 1 });
    }

    pub fn fp4(self: Enc, x: Buffer, rows: usize, width: usize, e4m3_scale: bool) void {
        self.pipe("ds_fp4_qdq");
        self.at(x, width * 2, 0);
        self.args(@as(i32, @intFromBool(e4m3_scale)), 1);
        self.threads(.{ rows * width / @as(usize, if (e4m3_scale) 16 else 32), 1, 1 }, .{ 64, 1, 1 });
    }

    /// The window keys' tail in place (kv_norm, RoPE, fp8), then into `ring` at their positions when given.
    pub fn kvFinish(self: Enc, kv: Buffer, norm: wts.Tensor, freq: Buffer, ring: ?Buffer) void {
        self.pipe("ds_kv_finish");
        self.at(kv, c.head_dim * 2, 0);
        self.tensor(norm, 1);
        self.at(self.s.pos, 4, 2);
        self.buf(freq, 0, 3);
        if (ring) |r| self.buf(r, 0, 4) else self.at(kv, c.head_dim * 2, 4);
        self.args([2]f32{ c.norm_eps, if (ring != null) @floatFromInt(st.ring) else 0 }, 5);
        self.groups(.{ self.rows, 1, 1 }, .{ 256, 1, 1 });
    }

    pub fn put(self: Enc, src: Buffer, positions: Buffer, dst: Buffer, rows: usize, width: usize, ring: usize) void {
        self.pipe("ds_put_rows");
        self.at(src, width * 2, 0);
        self.at(positions, 4, 1);
        self.buf(dst, 0, 2);
        self.args(CopyArgs{ .width = @intCast(width), .ring = @intCast(ring) }, 3);
        self.threads(.{ width, rows, 1 }, .{ @min(width, 256), 1, 1 });
    }

    pub fn putF32(self: Enc, src: Buffer, positions: Buffer, dst: Buffer, rows: usize, width: usize, ring: usize) void {
        self.pipe("ds_put_rows_f32");
        self.at(src, width * 4, 0);
        self.at(positions, 4, 1);
        self.buf(dst, 0, 2);
        self.args(CopyArgs{ .width = @intCast(width), .ring = @intCast(ring) }, 3);
        self.threads(.{ width, rows, 1 }, .{ @min(width, 256), 1, 1 });
    }

    /// A boundary: (`mixes`) its pre, post and comb, and the streams collapsed with `pre_in`, normed into `normed`.
    pub fn boundary(self: Enc, hc: ?wts.Hc, pre_in: Buffer, pre_out: Buffer, norm: wts.Tensor) void {
        const s = self.s;
        if (hc) |h| {
            if (pending_expand) {
                pending_expand = false;
                self.pipe("ds_hc_expand_mix");
                self.buf(s.x, 0, 0);
                self.buf(s.branch, 0, 1);
                self.buf(s.post, 0, 2);
                self.buf(s.comb, 0, 3);
                self.tensor(h.fn_, 4);
                self.buf(s.hc_part, 0, 5);
                self.buf(s.x2, 0, 6);
                self.buf(s.experts, 0, 7);
                self.buf(s.shared, 0, 8);
                self.args(pending_topk, 9);
                pending_topk = 0;
                self.groups(.{ c.hc * c.hidden / 1024, self.rows, 1 }, .{ 256, 1, 1 });
                std.mem.swap(Buffer, &s.x, &s.x2);
            } else {
                self.pipe("ds_hc_mix_part");
                self.buf(s.x, 0, 0);
                self.tensor(h.fn_, 1);
                self.buf(s.hc_part, 0, 2);
                self.groups(.{ c.hc * c.hidden / 1024, self.rows, 1 }, .{ 256, 1, 1 });
            }
        } else self.flush();
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

    /// The sublayer's branch into the streams, deferred to the next boundary (or `flush`).
    pub fn expand(self: Enc) void {
        _ = self;
        pending_expand = true;
    }

    /// A deferred expand, now: before anything but a mixing boundary reads the streams.
    pub fn flush(self: Enc) void {
        if (!pending_expand) return;
        pending_expand = false;
        const s = self.s;
        self.pipe("ds_hc_expand");
        self.buf(s.x, 0, 0);
        self.buf(s.branch, 0, 1);
        self.buf(s.post, 0, 2);
        self.buf(s.comb, 0, 3);
        self.buf(s.experts, 0, 4);
        self.buf(s.shared, 0, 5);
        self.args(pending_topk, 6);
        pending_topk = 0;
        self.threads(.{ c.hidden, self.rows, 1 }, .{ 256, 1, 1 });
    }
};

/// The groups this call completes at `ratio`: [first, end) (a group's last position is inside the call).
pub fn completed(base: usize, rows: usize, ratio: usize) [2]usize {
    return .{ base / ratio, (base + rows) / ratio };
}

pub fn run(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, dump: ?Dump) !void {
    return runWith(m, s, cache, tokens, head, dump, null);
}

/// `run` over prompt rows that may lie in image spans (`vis`): their rows are the tower's, not the token embedding.
pub fn runWith(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, dump: ?Dump, vis: ?*const Vis) !void {
    return runSegs(m, s, &.{.{ .cache = cache, .tokens = tokens }}, head, dump, vis, null);
}

/// `runWith` for a prompt pass's chunk: `next`, the chunk after it, has its engram rows gathered while this one runs.
pub fn runChunk(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, vis: ?*const Vis, next: ?Next) !void {
    return runSegs(m, s, &.{.{ .cache = cache, .tokens = tokens }}, head, null, vis, next);
}

/// A stream's rows in a forward: its cache, its tokens, from scratch row `r0` (the rows of the streams before it).
pub const Seg = struct { cache: *st.Cache, tokens: []const u32, r0: usize = 0 };

/// Several streams' rows in one forward, each stream's attention on its own rows (row kernels: each matches alone).
pub fn many(m: *Model, s: *st.Scratch, segs: []const Seg, head: Head) !void {
    return runSegs(m, s, segs, head, null, null, null);
}

fn runSegs(m: *Model, s: *st.Scratch, segs: []const Seg, head: Head, dump: ?Dump, vis: ?*const Vis, next: ?Next) !void {
    var rows: usize = 0;
    for (segs) |g| {
        if (g.r0 != rows or g.tokens.len == 0 or g.tokens.len > st.max_rows) return error.DeepseekRowsExceeded;
        if (g.cache.len + g.tokens.len > g.cache.capacity) return error.DeepseekCacheFull;
        rows += g.tokens.len;
    }
    if (rows == 0 or rows > s.rows) return error.DeepseekRowsExceeded;
    if (segs.len > 1 and (vis != null or dump != null or head == .last)) return error.DeepseekSharedForward;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();

    // host side: ids, positions, the engram history and lookups, the first collapse (copy 0)
    var images = false;
    var dsts: [c.engram_layers.len][]u16 = undefined;
    var host: f64 = 0;
    // a lone chunk gathered ahead (runChunk's `next` the call before) keeps its ids and rows; the rest is gathered now
    const ready = segs.len == 1 and ahead.preparedFor(s, segs[0]);
    if (ready) s.engram_set = s.prepared.?.set;
    s.prepared = null;
    for (segs) |g| {
        const base = g.cache.len;
        const n = g.tokens.len;
        for (s.ids.slice(u32, g.r0 + n)[g.r0..], g.tokens) |*d, t| d.* = t;
        for (s.pos.slice(i32, g.r0 + n)[g.r0..], 0..) |*p, i| p.* = @intCast(base + i);
        if (!ready) g.cache.engram_ids.shrinkRetainingCapacity(base);
        for (g.tokens, s.img.slice(i32, g.r0 + n)[g.r0..], s.img_code.slice(i32, g.r0 + n)[g.r0..], 0..) |t, *flag, *code, i| {
            code.* = Vis.code(vis, base + i);
            flag.* = @intFromBool(code.* != -1);
            images = images or code.* != -1;
            if (!ready) g.cache.engram_ids.appendAssumeCapacity(m.engram.compress(t, code.* == -1)); // image rows end every n-gram
        }
        if (ready) continue;
        var h0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &h0);
        const width = c.engram_cols * c.engram_dim;
        for (&dsts, s.engram_rows[s.engram_set]) |*d, b| d.* = b.slice(u16, (g.r0 + n) * width)[g.r0 * width ..];
        try m.engram.gatherRows(g.cache.engram_ids.items, base, n, dsts);
        var h1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &h1);
        host += secs(h0, h1);
    }
    if (profile) |p| p[10] += host;
    for (s.pre[0].slice(f32, rows * c.hc), 0..) |*v, i| v.* = if (i % c.hc == 0) 1 else 0;

    var t_start: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t_start);
    var cb = m.queue.commandBuffer();
    const w = &m.weights;
    var e = cb.compute(.concurrent);
    pending_expand = false;
    pending_topk = 0;
    const x: Enc = .{ .ep = &e, .cbp = &cb, .m = m, .s = s, .rows = rows, .base = segs[0].cache.len, .units = segs.len == 1 and rows >= nax_rows, .segs = segs };
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
    const drafting = w.stages != null and for (segs) |g| {
        if (g.cache.stage_rings == null) break false;
    } else true;
    for (&w.layers, 0..) |*l, i| {
        if (drafting) if (std.mem.indexOfScalar(usize, &c.draft_targets, i)) |t| {
            // the draft head reads its target layers' attention inputs: the mean of the four streams
            x.flush();
            x.pipe("ds_stream_mean");
            x.buf(s.x, 0, 0);
            x.buf(s.mh, 0, 1);
            x.args([2]i32{ @intCast(t * c.hidden), @intCast(c.draft_targets.len * c.hidden) }, 2);
            x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
        };
        try layer.block(x, i, l);
        if (dump) |d| { // the streams after each block: the buffers are untracked, so each block completes first
            x.flush();
            e.end();
            cb.commit();
            cb.wait();
            if (cb.status() != .completed) return error.DeepseekGpuFailed;
            @memcpy(d.layers[i].contents()[0 .. rows * c.hc * c.hidden * 2], s.x.contents()[0 .. rows * c.hc * c.hidden * 2]);
            cb = m.queue.commandBuffer();
            e = cb.compute(.concurrent);
        }
    }
    if (drafting) dspark.condition(x);
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
    if (next) |nx| if (segs.len == 1) { // the next chunk's engram rows, on the host while the GPU runs this one
        var a0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &a0);
        ahead.prepare(m, s, segs[0].cache, segs[0].cache.len + rows, nx);
        var a1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &a1);
        if (profile) |p| p[11] += secs(a0, a1);
    };
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
    for (segs) |g| {
        const n = g.tokens.len;
        g.cache.len += n;
        if (head != .none) {
            const last = if (head == .last) 0 else g.r0 + n - 1;
            @memcpy(g.cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + last * c.vocab * 2)[0 .. c.vocab * 2]);
        }
        g.cache.last_start = g.cache.len - n;
        g.cache.last_rows = n;
        g.cache.last_r0 = g.r0;
    }
}

fn secs(a: std.c.timespec, b: std.c.timespec) f64 {
    return @as(f64, @floatFromInt(b.sec - a.sec)) + @as(f64, @floatFromInt(b.nsec - a.nsec)) / 1e9;
}
