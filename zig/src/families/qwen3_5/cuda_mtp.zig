//! The Qwen3.6 MTP layer as the drafter on CUDA: chains from a kept row's final normed state and the token after it.
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_mtp_weights.zig");
const timing = @import("cuda_draft.zig").timing;

const E = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;
const dr = lanes.drafter;
const Stream = lanes.Stream;
const head_dim = geometry.head_dim;
const max_rows = engine.max_rows;
const most_window = engine.most_window;

/// Drafts a round at most (the frozen engine's DEPTH); TF_MTP_DRAFTS=N sets it, up to a window's rows less one.
const default_depth = 3;

const Fns = struct { norm_rope: cuda.Function, attn: cuda.Function };
/// qwen35_draft.cu's ring kernels, shared with the DFlash2 drafter.
const Ring = struct { gather_taps: cuda.Function, gather_rows: cuda.Function, ring_put: cuda.Function };
const TapRows = extern struct { n: c_int, rows: [most_window]c_int };

/// A stream's context: its ring of keys and values over [begin, end), the newest row's state, and its held chain.
const Ctx = struct {
    ring: DeviceBuffer,
    last: DeviceBuffer, // the newest absorbed row's normed output (bf16 hidden): the first chain row's state
    begin: u64 = 0,
    end: u64 = 0,
    first: u32 = 0, // the newest absorbed row's draft, its logits' argmax
    held: [most_window]u32 = undefined,
    held_n: usize = 0,
};

pub const Mtp = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *E,
    w: weights.Weights,
    mod: cuda.Module,
    dmod: cuda.Module,
    fns: Fns,
    ring_fns: Ring,
    scratch: DeviceBuffer,
    at: usize = 0,
    capacity: usize, // ring rows: every position of the stream's window
    depth: u32,
    tap: [1]u32,
    // rows of max_rows, bf16 unless said
    taps_in: u64,
    hin: u64,
    hn: u64,
    emb: u64,
    en: u64,
    h: u64,
    r: u64,
    x: u64,
    qraw: u64,
    q: u64,
    kraw: u64,
    kb: u64,
    vb: u64,
    att: u64,
    xs: u64, // f32 group sums for the projections
    logits: u64, // one row
    ids: u64, // u32 the rows' tokens
    tok: u64, // u32 the drawn token
    staged: cuda.HostBuffer, // pinned: the ids up, the token down
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,
    /// Contexts between streams, allocated and run once (`reserve`), as the driver stalls a buffer's first kernel.
    pool: std.ArrayList(*Ctx) = .empty,
    /// One chained draft's cost (timed at load).
    step_ms: f64 = 1.0,
    /// The target's window ms by rows (cuda_native.zig times them), for the round loop's prices.
    costs: [most_window]lanes.config.Cost = undefined,
    n_costs: usize = 0,

    const ids_at = 0;
    const tok_at = max_rows * 4;
    const staged_len = tok_at + 64;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8, capacity: usize) !*Mtp {
        if (!e.g.moe()) return error.NotRouted;
        const d = try gpa.create(Mtp);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .e = e, .w = undefined, .mod = undefined, .dmod = undefined, .fns = undefined, .ring_fns = undefined, .scratch = undefined, .capacity = capacity, .depth = default_depth, .tap = .{@intCast(e.g.layers - 1)}, .taps_in = 0, .hin = 0, .hn = 0, .emb = 0, .en = 0, .h = 0, .r = 0, .x = 0, .qraw = 0, .q = 0, .kraw = 0, .kb = 0, .vb = 0, .att = 0, .xs = 0, .logits = 0, .ids = 0, .tok = 0, .staged = undefined };
        if (platform.c.getenv("TF_MTP_DRAFTS")) |v| d.depth = std.fmt.parseInt(u32, std.mem.span(v), 10) catch default_depth;
        d.depth = @min(@max(d.depth, 1), most_window - 1);
        d.mod = try cuda.Module.load(e.d, cuda.kernels.qwen36_mtp);
        errdefer d.mod.unload();
        d.dmod = try cuda.Module.load(e.d, cuda.kernels.qwen35_draft);
        errdefer d.dmod.unload();
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(d.fns, name) = try d.mod.function("tf_q36m_" ++ name);
        inline for (@typeInfo(Ring).@"struct".field_names) |name| @field(d.ring_fns, name) = try d.dmod.function("tf_q35d_" ++ name);
        d.w = try weights.load(gpa, io, e, dir);
        errdefer d.w.deinit();
        d.scratch = try DeviceBuffer.alloc(e.d, d.scratchBytes());
        errdefer d.scratch.free();
        try d.scratch.fill8(0, null);
        d.at = 0;
        _ = try d.layout(true);
        d.staged = try cuda.HostBuffer.alloc(e.d, staged_len);
        errdefer d.staged.free();
        // the first context, and a chain timed on it (the quickest of several: the first ones pay the driver's stalls)
        const ctx = try d.create();
        errdefer d.drop(ctx);
        var quickest: f64 = std.math.inf(f64);
        for (0..6) |_| {
            try d.absorb(ctx, d.taps_in, 0, 1, null, &.{0});
            const t0 = std.Io.Clock.awake.now(io);
            try d.hold(ctx, 0, 1, d.depth);
            quickest = @min(quickest, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6);
        }
        d.step_ms = quickest / @as(f64, @floatFromInt(d.depth));
        d.resetCtx(ctx);
        try d.pool.append(gpa, ctx);
        return d;
    }

    /// The drafter's tap: the target's last layer, whose residual the final norm then reads.
    pub fn taps(d: *const Mtp) []const u32 {
        return &d.tap;
    }

    pub fn setCost(d: *Mtp, rows: u32, ms: f64) void {
        if (platform.c.getenv("TF_TIMING") != null) std.debug.print("[timing] window {d} rows {d:.2} ms (a chained draft {d:.2})\n", .{ rows, ms, d.step_ms });
        if (d.n_costs < d.costs.len) {
            d.costs[d.n_costs] = .{ .width = rows, .ms = ms };
            d.n_costs += 1;
        }
    }

    pub fn deinit(d: *Mtp) void {
        var it = d.streams.valueIterator();
        while (it.next()) |ctx| d.drop(ctx.*);
        d.streams.deinit(d.gpa);
        for (d.pool.items) |ctx| d.drop(ctx);
        d.pool.deinit(d.gpa);
        d.staged.free();
        d.scratch.free();
        d.w.deinit();
        d.dmod.unload();
        d.mod.unload();
        d.gpa.destroy(d);
    }

    fn kvWidth(d: *const Mtp) usize {
        return d.e.g.kv_heads * head_dim;
    }

    fn ringBytes(d: *const Mtp) usize {
        return 2 * d.capacity * d.kvWidth() * 2;
    }

    fn scratchBytes(d: *Mtp) usize {
        d.at = 0;
        return d.layout(false) catch unreachable;
    }

    /// The scratch rows, measured (nothing assigned) or assigned from the scratch buffer.
    fn layout(d: *Mtp, assign: bool) !usize {
        const g = d.e.g;
        const R = max_rows;
        const aw = g.attnWidth();
        const kw = d.kvWidth();
        const sizes = .{
            .{ "taps_in", R * g.hidden * 2 }, .{ "hin", R * g.hidden * 2 }, .{ "hn", R * g.hidden * 2 },                  .{ "emb", R * g.hidden * 2 },
            .{ "en", R * g.hidden * 2 },      .{ "h", R * g.hidden * 2 },   .{ "r", R * g.hidden * 2 },                   .{ "x", R * g.hidden * 2 },
            .{ "qraw", R * 2 * aw * 2 },      .{ "q", R * aw * 2 },         .{ "kraw", R * kw * 2 },                      .{ "kb", R * kw * 2 },
            .{ "vb", R * kw * 2 },            .{ "att", R * aw * 2 },       .{ "xs", R * (@max(g.hidden, aw) / 64) * 4 }, .{ "logits", g.vocab * 2 },
            .{ "ids", R * 4 },                .{ "tok", 64 },
        };
        var total: usize = 0;
        inline for (sizes) |s| {
            const at = engine.aligned(total);
            total = at + s[1];
            if (assign) @field(d, s[0]) = d.scratch.ptr + at;
        }
        return total;
    }

    fn drop(d: *Mtp, ctx: *Ctx) void {
        ctx.last.free();
        ctx.ring.free();
        d.gpa.destroy(ctx);
    }

    fn resetCtx(d: *Mtp, ctx: *Ctx) void {
        _ = d;
        ctx.* = .{ .ring = ctx.ring, .last = ctx.last };
    }

    /// A new context, touched once (an absorb, then a chain), so no request pays the driver's stall.
    fn create(d: *Mtp) !*Ctx {
        const ctx = try d.gpa.create(Ctx);
        errdefer d.gpa.destroy(ctx);
        ctx.* = .{ .ring = try DeviceBuffer.alloc(d.e.d, d.ringBytes()), .last = undefined };
        errdefer ctx.ring.free();
        ctx.last = try DeviceBuffer.alloc(d.e.d, d.e.g.hidden * 2);
        errdefer ctx.last.free();
        try ctx.ring.fill8(0, null);
        try ctx.last.fill8(0, null);
        try d.absorb(ctx, d.taps_in, 0, 1, null, &.{0}); // any tap rows do: the drafter's own zeroed staging
        try d.hold(ctx, 0, 1, d.depth);
        d.resetCtx(ctx);
        return ctx;
    }

    /// `n` contexts ready in the pool, for the streams the server admits.
    pub fn reserve(d: *Mtp, n: usize) !void {
        try d.pool.ensureTotalCapacity(d.gpa, n);
        while (d.pool.items.len < n) d.pool.appendAssumeCapacity(try d.create());
    }

    fn ringK(d: *Mtp, ctx: *Ctx) u64 {
        _ = d;
        return ctx.ring.ptr;
    }

    fn ringV(d: *Mtp, ctx: *Ctx) u64 {
        return ctx.ring.ptr + d.capacity * d.kvWidth() * 2;
    }

    /// out (rows, n) = x (rows, k) through `l`, as the target's decode tiles or prompt GEMM, with the drafter's sums.
    fn proj(d: *Mtp, l: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize) !void {
        const e = d.e;
        if (rows > 16) return e.affine.prompt(e.stream, x, l, out, rows);
        try e.run(e.fns.sums, .{ rows, 1, 1 }, .{ 32, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(l.k)), @as(u64, 0) });
        try e.affine.decode(e.stream, x, d.xs, l, out, rows);
    }

    /// out = norm(h + r) with w; h takes the sum (r 0: no residual).
    fn norm(d: *Mtp, h: u64, r: u64, w: u64, out: u64, rows: usize) !void {
        const e = d.e;
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ h, r, w, out, e.g.eps, @as(c_int, @intCast(e.g.hidden)), @as(u64, 0) });
    }

    /// The layer over `rows` rows at slots pos0..: d.ids their tokens, d.hin their states; x gets their normed outputs.
    fn layer(d: *Mtp, ctx: *Ctx, rows: usize, pos0: u64) !void {
        const e = d.e;
        const g = e.g;
        const w = &d.w;
        const aw = g.attnWidth();
        const kw = d.kvWidth();
        const p0: c_longlong = @intCast(pos0);
        try e.run(e.fns.embed, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.ids, e.embed_w, e.embed_s, e.embed_b, d.emb, @as(c_int, @intCast(g.hidden)), @as(u64, 0) });
        try d.norm(d.emb, 0, w.norm_e, d.en, rows);
        try d.norm(d.hin, 0, w.norm_h, d.hn, rows);
        try d.proj(w.fc_e, d.en, d.h, rows);
        try d.proj(w.fc_h, d.hn, d.r, rows);
        try d.norm(d.h, d.r, w.input_norm, d.x, rows);
        try d.proj(w.q, d.x, d.qraw, rows);
        try d.proj(w.k, d.x, d.kraw, rows);
        try d.proj(w.v, d.x, d.vb, rows);
        try e.run(d.fns.norm_rope, .{ rows * g.q_heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.qraw, w.q_norm, d.q, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), p0, g.eps, geometry.theta });
        try e.run(d.fns.norm_rope, .{ rows * g.kv_heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.kraw, w.k_norm, d.kb, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(kw)), @as(c_int, head_dim), p0, g.eps, geometry.theta });
        try e.run(d.fns.attn, .{ rows, g.q_heads, 1 }, .{ 32, 1, 1 }, .{ d.q, d.qraw, @as(c_int, @intCast(2 * aw)), d.kb, d.vb, d.ringK(ctx), d.ringV(ctx), @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(d.capacity)), @as(c_longlong, @intCast(ctx.begin)), p0, d.att });
        try d.proj(w.o, d.att, d.r, rows);
        try d.norm(d.h, d.r, w.post_norm, d.x, rows);
        try e.routedRows(w.mlp.moe.router, w.mlp.moe.experts, d.x, d.r, rows, rows > most_window);
        try d.norm(d.h, d.r, w.norm, d.x, rows);
        try e.run(d.ring_fns.ring_put, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.kb, d.vb, d.ringK(ctx), d.ringV(ctx), @as(c_int, @intCast(kw)), p0, @as(c_int, @intCast(d.capacity)) });
    }

    /// Row `row` of x through the target's head: its argmax into `tok` (the host sees it after a synchronize).
    fn draw(d: *Mtp, row: usize) !void {
        const e = d.e;
        try d.proj(e.head, d.x + row * e.g.hidden * 2, d.logits, 1);
        try e.run(e.fns.argmax, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, .{ d.logits, @as(c_int, @intCast(e.g.vocab)), d.tok });
    }

    fn token(d: *Mtp) !u32 {
        const e = d.e;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(d.staged.bytes.ptr + tok_at, d.tok, 4, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
        return std.mem.bytesToValue(u32, d.staged.bytes[tok_at..][0..4]);
    }

    fn putIds(d: *Mtp, ids: []const u32) !void {
        const e = d.e;
        @memcpy(std.mem.bytesAsSlice(u32, d.staged.bytes[ids_at..][0 .. ids.len * 4]), ids);
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.ids, d.staged.bytes.ptr + ids_at, ids.len * 4, e.stream.handle), "cuMemcpyHtoDAsync");
    }

    /// Target rows [start, start + n) from the tap ring with the token after each; `rows`, when given, picks them.
    fn absorb(d: *Mtp, ctx: *Ctx, tap_ring: u64, start_: u64, n_: usize, rows_: ?[]const u32, follow: []const u32) !void {
        const e = d.e;
        const g = e.g;
        var start = start_;
        var n = n_;
        var fi: usize = 0;
        if (follow.len < n) return error.BadDraftRows;
        if (rows_) |rs| if (rs.len != n or n > most_window) return error.BadDraftRows;
        if (start < ctx.end) ctx.end = start;
        if (start > ctx.end) ctx.begin = start; // a gap (a prompt restored from a cache): the context starts here
        const held = engine.tap_rows - max_rows; // the target's ring holds this many rows behind its newest
        if (n > held) {
            fi = n - held;
            start += n - held;
            n = held;
            ctx.begin = start;
        }
        ctx.end = start;
        while (n > 0) {
            const rows = @min(n, max_rows);
            if (rows_) |rs| {
                var tr: TapRows = .{ .n = @intCast(rows), .rows = @splat(0) };
                for (rs, 0..) |r, i| tr.rows[i] = @intCast(r);
                try e.run(d.ring_fns.gather_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), tr, @as(c_int, @intCast(g.hidden)), d.taps_in });
            } else try e.run(d.ring_fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(g.hidden)), d.taps_in });
            try d.norm(d.taps_in, 0, e.final_norm, d.hin, rows); // the target's final norm: the rows the head read
            try d.putIds(follow[fi..][0..rows]);
            try d.layer(ctx, rows, start);
            start += rows;
            n -= rows;
            fi += rows;
            ctx.end = start;
            // the newest row: its state kept for the chain, its draft drawn
            try ctx.last.copyFrom(0, d.x + (rows - 1) * g.hidden * 2, g.hidden * 2, e.stream.handle);
            try d.draw(rows - 1);
        }
        ctx.first = try d.token();
    }

    /// `depth` drafts after `pending` at pos0 (the pending token's slot): the first came with the newest absorbed row.
    fn hold(d: *Mtp, ctx: *Ctx, pending: u32, pos0: u64, depth: usize) !void {
        const e = d.e;
        const g = e.g;
        if (ctx.end != pos0) return error.DraftContextBehind;
        ctx.held[0] = ctx.first;
        ctx.held_n = 1;
        _ = pending; // the newest absorbed row's follow token, whose draft ctx.first is
        var k: usize = 1;
        while (k < depth and k < ctx.held.len) : (k += 1) {
            // row k: (the previous row's normed output, the draft before) at slot pos0 + k - 1
            const src = if (k == 1) ctx.last.ptr else d.x;
            try d.scratch.copyFrom(d.hin - d.scratch.ptr, src, g.hidden * 2, e.stream.handle);
            try d.putIds(&.{ctx.held[k - 1]});
            try d.layer(ctx, 1, pos0 + k - 1);
            try d.draw(0);
            ctx.held[k] = try d.token();
            ctx.held_n = k + 1;
        }
    }

    // the Drafter contract

    pub fn drafter(d: *Mtp) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(p: *anyopaque) *Mtp {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        return cast(p).taps();
    }

    fn factsFn(p: *anyopaque) dr.Facts {
        const d = cast(p);
        return .{ .depth = d.depth, .step_ms = d.step_ms, .prior = &prior, .plain_guard = true, .batched = false, .stream_bytes = d.ringBytes() + d.e.g.hidden * 2, .trees = false, .window_costs = d.costs[0..d.n_costs], .lane_costs = d.costs[0..d.n_costs] };
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(p);
        if (d.streams.contains(s)) return error.StreamOpen;
        const ctx = d.pool.pop() orelse try d.create();
        errdefer d.pool.append(d.gpa, ctx) catch d.drop(ctx);
        d.resetCtx(ctx);
        try d.streams.put(d.gpa, s, ctx);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("mtp absorb", d.io, t0);
        for (items) |it| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
            // the target's features: its ring, `offset` the first row's position (cuda_native.zig's featuresFn)
            if (it.rows) |rs| {
                try d.absorb(ctx, it.features.buffer, it.features.offset, rs.len, rs, it.follow);
            } else try d.absorb(ctx, it.features.buffer, it.features.offset, it.features.rows, null, it.follow);
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("mtp hold", d.io, t0);
        for (items) |it| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            try d.hold(ctx, it.pending, it.position - 1, it.depth);
        }
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(p);
        for (streams, out) |s, o| {
            const ctx = d.streams.get(s) orelse return error.UnknownStream;
            if (o.len > ctx.held_n) return error.TooFewDrafts;
            @memcpy(o, ctx.held[0..o.len]);
        }
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const d = cast(p);
        if (d.streams.fetchRemove(s)) |kv| d.pool.append(d.gpa, kv.value) catch d.drop(kv.value);
    }
};

/// Acceptance by depth until a stream has its own (the frozen engine's chains kept about three of four).
const prior = [_]f64{ 0.7, 0.55, 0.45, 0.4, 0.35, 0.3, 0.3 };
