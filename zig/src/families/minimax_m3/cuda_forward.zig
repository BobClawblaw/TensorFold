//! MiniMax-M3.1-Flash's TP forward on CUDA: the Metal family's layer loop with every rank's partials gathered.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const kern = @import("cuda_kernels.zig");

pub const State = state.State;
pub const Buffers = state.Buffers;

/// The kernels a forward launches from: the lane matmul (MXFP8 and bf16 linears), the expert formats, our glue.
pub const Modules = struct {
    mods: [5]cuda.Module, // experts, experts_prefill, experts_pack, experts3, experts3_prefill
    lane: cuda.qmmf.Lane,
    nvfp4: cuda.experts.Nvfp4Experts,
    a3: cuda.experts3.Affine3Experts,
    a4: cuda.experts.Affine4Experts,
    router: cuda.grouped.Router,
    k: kern.Kernels,

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Modules {
        const images = [_][]const u8{ cuda.kernels.experts, cuda.kernels.experts_prefill, cuda.kernels.experts_pack, cuda.kernels.experts3, cuda.kernels.experts3_prefill };
        var m: Modules = undefined;
        var n: usize = 0;
        errdefer for (m.mods[0..n]) |*x| x.unload();
        for (&m.mods, images) |*x, image| {
            x.* = try cuda.Module.load(d, image);
            n += 1;
        }
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        m.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer m.lane.unload();
        m.nvfp4 = try cuda.experts.Nvfp4Experts.load(d, sms);
        errdefer m.nvfp4.unload();
        m.a4 = try cuda.experts.Affine4Experts.resolve(m.mods[0], m.mods[1], m.mods[2], sms);
        m.a3 = try cuda.experts3.Affine3Experts.resolve(m.mods[3], m.mods[4], sms);
        m.router = try cuda.grouped.Router.resolve(m.mods[0]);
        m.k = try kern.Kernels.load(d);
        return m;
    }

    pub fn deinit(m: *Modules) void {
        m.k.unload();
        m.nvfp4.unload();
        m.lane.unload();
        for (&m.mods) |*x| x.unload();
        m.* = undefined;
    }

    /// What the loader packs with.
    pub fn tools(m: *const Modules, d: *const cuda.Driver, s: cuda.Stream) weights.Tools {
        return .{ .d = d, .stream = s, .a3 = &m.a3, .a4 = &m.a4 };
    }

    fn grouped(m: *const Modules) cuda.experts.Grouped {
        return .{ .nvfp4 = &m.nvfp4, .affine3 = &m.a3, .affine4 = &m.a4 };
    }
};

/// How the ranks' partials meet: one rank alone, or every rank's slot gathered over NCCL (RoCE for small ones).
pub const Exchange = union(enum) {
    local,
    nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, group: ?*const cuda.tp_group.Group = null },
};

/// Called after each layer with the residual rows (a reference check's dumps).
pub const Watch = struct { ctx: *anyopaque, call: *const fn (*anyopaque, usize, u64, usize) anyerror!void };

/// Section timers (TF_MINIMAX_PROF): GPU ms accumulated by section between events on the forward's stream.
pub const Prof = struct {
    pub const Section = enum { proj, prep, index, attn, oproj, gather, glue, router, experts, shared, head };
    ev: [2]cuda.Event,
    ms: [@typeInfo(Section).@"enum".field_names.len]f64 = @splat(0),
    open: bool = false,

    pub fn init(d: *const cuda.Driver) !Prof {
        return .{ .ev = .{ try cuda.Event.init(d, true), try cuda.Event.init(d, true) } };
    }

    pub fn deinit(p: *Prof) void {
        for (&p.ev) |*e| e.deinit();
    }

    /// Times `s` from the last mark (synchronizing: for measurement only).
    fn mark(p: *Prof, stream: cuda.Stream, s: Section) !void {
        if (!p.open) {
            try p.ev[0].record(stream);
            p.open = true;
            return;
        }
        try p.ev[1].record(stream);
        try p.ev[1].synchronize();
        p.ms[@intFromEnum(s)] += try cuda.Event.elapsedMs(p.ev[0], p.ev[1]);
        try p.ev[0].record(stream);
    }

    pub fn print(p: *const Prof) void {
        var total: f64 = 0;
        for (p.ms) |x| total += x;
        for (p.ms, 0..) |x, i| std.debug.print("  {s:<8} {d:9.2} ms  {d:5.1}%\n", .{ @typeInfo(Section).@"enum".field_names[i], x, 100 * x / @max(total, 1e-9) });
        std.debug.print("  total    {d:9.2} ms\n", .{total});
    }
};

fn decodeOnTensorUnits() bool {
    const v = std.c.getenv("TF_MINIMAX_DECODE_ATTN") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "rows");
}

/// The engine's image rows and the sequence the forward runs on (they splice into that sequence's rows only).
pub const ImageRows = struct { rows: u64, spans: []const @import("cuda_vision.zig").Span };

/// One stream's rows in a shared forward: rows [row0, row0 + rows) on its sequence.
pub const Seg = struct { st: *State, row0: usize, rows: usize };

pub const Forward = struct {
    d: *const cuda.Driver,
    m: *const Modules,
    w: *const weights.Weights,
    stream: cuda.Stream,
    b: *Buffers,
    exchange: Exchange = .local,
    watch: ?Watch = null,
    taps: ?u64 = null, // a drafter's feature rows [rows][taps][hidden] bf16
    tap_at: []const i32 = &.{}, // each layer's tap slot, or -1
    prof: ?*Prof = null,
    images: ?ImageRows = null, // a prompt pass's image rows over their positions (its sequence only)

    pub fn at(f: *const Forward, s: Prof.Section) !void {
        if (f.prof) |p| try p.mark(f.stream, s);
    }

    fn c(f: *const Forward) *const cfg.Config {
        return &f.w.config;
    }

    /// Every rank's partial of `rows` rows (`slots` a row) into every rank's gathered buffer, rank order.
    pub fn gather(f: *const Forward, rows: usize, slots: usize) !void {
        const bytes = rows * slots * f.c().hidden * 4;
        switch (f.exchange) {
            .local => {},
            .nccl => |n| {
                const mine = f.b.gathered + f.w.share.rank * bytes;
                if (n.group) |g| return g.allGather(f.stream, mine, f.b.gathered, bytes);
                try n.lib.check(n.lib.api.ncclAllGather(mine, f.b.gathered, rows * slots * f.c().hidden, .f32, n.comm, f.stream.handle), "ncclAllGather");
            },
        }
    }

    /// This rank's slot of the gathered buffer for a site of `slots` a row.
    pub fn slot(f: *const Forward, rows: usize, slots: usize) u64 {
        return f.b.gathered + f.w.share.rank * rows * slots * f.c().hidden * 4;
    }

    pub fn linear(f: *const Forward, w: weights.Linear, x: u64, ldx: usize, rows: usize, y: u64, out: cuda.qmmf.Out) !void {
        try f.m.lane.matmulOut(f.stream, x, ldx, rows, w, y, out, f.b.part);
    }

    /// `ids` at each segment's positions: the last row's logits (a prompt chunk) or every row's, the caches written.
    pub fn run(f: *const Forward, ids: []const i32, segs: []const Seg) !void {
        const R = ids.len;
        const b = f.b;
        const co = f.c();
        if (R == 0 or R > b.rows or segs.len == 0) return error.BadWindow;
        var total: usize = 0;
        for (segs) |g| {
            if (g.rows == 0 or g.st.pos + g.rows > g.st.capacity or g.st.capacity > b.context) return error.ContextFull;
            total += g.rows;
        }
        if (total != R) return error.BadWindow;
        try cuda.copy_source.enqueue(f.d, b.ids, std.mem.sliceAsBytes(ids), f.stream);
        try kern.embed(&f.m.k, f.stream, b.ids, f.w.embed, b.resid, R, co.hidden);
        if (f.images) |im| if (segs.len == 1) try @import("cuda_vision.zig").splice(&f.m.k, f.stream, im.rows, im.spans, segs[0].st.pos, segs[0].st.pos + R, b.resid);
        try kern.norm(&f.m.k, f.stream, b.resid, f.w.layers[0].in_norm, b.x, R, co.hidden, co.eps);
        for (f.w.layers, 0..) |l, j| {
            try f.layer(l, j, R, segs, if (j + 1 < f.w.layers.len) f.w.layers[j + 1].in_norm else 0);
            if (f.taps) |t| if (j < f.tap_at.len and f.tap_at[j] >= 0) try kern.tap(&f.m.k, f.stream, b.resid, t, R, co.hidden, 5, @intCast(f.tap_at[j]));
            if (f.watch) |wt| try wt.call(wt.ctx, j, b.resid, R);
        }
        for (segs) |g| g.st.pending = g.rows;
        const first: usize = if (b.prefill) R - 1 else 0;
        const n = R - first;
        try kern.norm(&f.m.k, f.stream, b.resid + first * co.hidden * 4, f.w.norm, b.fnormed, n, co.hidden, co.eps);
        try f.linear(f.w.head, b.fnormed, co.hidden, n, b.logits, .bf16);
        try f.at(.head);
    }

    /// Layer `l` on every segment's rows (b.x their normed input); the last add norms the residual by `next` (0: none).
    fn layer(f: *const Forward, l: weights.Layer, j: usize, R: usize, segs: []const Seg, next: u64) !void {
        const b = f.b;
        const co = f.c();
        const h: usize = co.hidden;
        const world = f.w.share.world;
        try f.at(.glue);
        try f.linear(l.attn.qkv, b.x, h, R, b.qkv, .bf16);
        try f.at(.proj);
        for (segs) |g| try f.attention(l.attn, j, g);
        try f.linear(l.attn.o, b.o, f.w.share.queries(co).len() * co.head_dim, R, f.slot(R, 1), .f32);
        try f.at(.oproj);
        try f.gather(R, 1);
        try f.at(.gather);
        try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, world, 1, 1.0, l.post_norm, b.x, R, h, co.eps);
        try f.at(.glue);
        switch (l.ffn) {
            .dense => |m| {
                try f.mlp(m, f.w.share.dense.len(), R, f.slot(R, 1), .f32);
                try f.gather(R, 1);
                try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, world, 1, 1.0, next, b.x, R, h, co.eps);
            },
            .moe => |m| {
                try f.moe(m, R);
                try f.at(.shared);
                try f.gather(R, 2);
                try f.at(.gather);
                try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, world, 2, co.routed_scale, next, b.x, R, h, co.eps);
            },
        }
    }

    /// gate | up of b.x, SwiGLU-OAI, down into `out` (fp32 partial rows).
    pub fn mlp(f: *const Forward, m: weights.Mlp, inter: usize, R: usize, out: u64, kind: cuda.qmmf.Out) !void {
        const b = f.b;
        const co = f.c();
        try f.linear(m.gu, b.x, co.hidden, R, b.gu, .bf16);
        try kern.swigluOai(&f.m.k, f.stream, b.gu, b.act, R, inter, 2 * inter, co.swiglu_alpha, co.swiglu_limit);
        try f.linear(m.down, b.act, inter, R, out, kind);
    }

    /// The routed and shared experts on b.x: this rank's partial rows (routed weighted, shared) into its slot.
    fn moe(f: *const Forward, m: weights.Moe, R: usize) !void {
        const b = f.b;
        const co = f.c();
        const h: usize = co.hidden;
        const topk: usize = co.topk;
        const pairs = R * topk;
        try kern.router(&f.m.k, f.stream, b.x, m.router, b.logits32, R, h, co.experts);
        try kern.route(&f.m.k, f.stream, b.logits32, m.bias, b.pick, b.wts, R, co.experts, topk);
        const tile: usize = if (b.prefill) 64 else 16;
        try f.m.router.route(f.stream, b.pick, pairs, co.experts, tile, b.plan());
        try f.at(.router);
        const g = f.m.grouped();
        const in_up: cuda.experts.Rows = .{ .x = b.x, .stride = h, .slots = topk };
        const in_down: cuda.experts.Rows = .{ .x = b.eact, .stride = m.routed.width };
        if (b.prefill) {
            try g.prompt(f.stream, .up, in_up, m.routed, b.plan(), b.eact, b.max_items, -1);
            try g.prompt(f.stream, .down_bf16, in_down, m.routed, b.plan(), b.ey, b.max_items, -1);
        } else {
            try g.decode(f.stream, .up, in_up, m.routed, b.plan(), b.eact, b.max_items, -1);
            try g.decode(f.stream, .down_f32, in_down, m.routed, b.plan(), b.ey, b.max_items, -1);
        }
        try f.at(.experts);
        try f.mlp(m.shared, f.w.share.shared.len(), R, b.sy, .f32);
        try kern.moePartial(&f.m.k, f.stream, b.ey, b.prefill, b.wts, b.sy, f.slot(R, 2), R, topk, h);
    }

    /// One segment's rows: q, k, v and the index pair prepared, the block selection, attention into b.o.
    fn attention(f: *const Forward, a: weights.Attention, j: usize, g: Seg) !void {
        const b = f.b;
        const co = f.c();
        const s = f.w.share;
        const hq = s.queries(co).len();
        const hk = s.kv.len();
        const hi = hk;
        const D: usize = co.head_dim;
        const qkv_rows = (hq + 2 * hk) * D + (hi + 1) * co.index_dim;
        const st = g.st;
        const R = g.rows;
        const pos = st.pos;
        const base = g.row0;
        const q = b.q + base * hq * D * 2;
        const iq = b.iq + base * hi * co.index_dim * 2;
        const o = b.o + base * hq * D * 2;
        try f.at(.proj);
        try kern.prep(&f.m.k, f.stream, .{
            .x = b.qkv + base * qkv_rows * 2,
            .stride = qkv_rows,
            .qw = a.q_norm,
            .kw = a.k_norm,
            .iqw = a.iq_norm,
            .ikw = a.ik_norm,
            .q = q,
            .iq = iq,
            .kc = st.keys(j),
            .vc = st.values(co, j),
            .ik = st.indexKeys(co, j),
            .rows = R,
            .pos = pos,
            .hq = hq,
            .hk = hk,
            .hi = hi,
            .rotary = co.rotary,
            .eps = co.eps,
            .theta = co.theta,
        });
        const nb = co.blocks(pos + R);
        const topk: usize = co.topk_blocks;
        const all = nb <= topk;
        const sel_at = base * hi;
        const scores = b.scores + sel_at * co.blocks(b.context) * 4;
        const sel = b.sel + sel_at * topk * 4;
        // decode rows on the tensor units too (each row's bits window-independent: drafted == plain); =rows: quarters
        const tc = b.prefill or decodeOnTensorUnits();
        const words = if (tc) b.selWords(co) else 0;
        const bits = b.bits + sel_at * words * 4;
        try f.at(.prep);
        if (!all) {
            try kern.indexScores(&f.m.k, f.stream, iq, st.indexKeys(co, j), scores, R, pos, hi, nb);
            try kern.indexSelect(&f.m.k, f.stream, scores, sel, bits, R, pos, hi, nb, topk, words);
        }
        try f.at(.index);
        defer f.at(.attn) catch {};
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        if (tc) {
            return kern.attnPrompt(&f.m.k, f.stream, q, st.keys(j), st.values(co, j), bits, o, R, pos, hq, hk, if (all) 0 else words, scale);
        }
        const quarters = topk * 4;
        const part = base * hq * quarters;
        try kern.attnDecode(&f.m.k, f.stream, .{
            .q = q,
            .kc = st.keys(j),
            .vc = st.values(co, j),
            .sel = sel,
            .pm = b.pm + part * 4,
            .pl = b.pl + part * 4,
            .po = b.po + part * D * 4,
            .o = o,
            .rows = R,
            .pos = pos,
            .hq = hq,
            .hk = hk,
            .topk = topk,
            .all = all,
            .scale = scale,
        });
    }

    /// The first `keep` rows of the last forward committed on each segment's sequence.
    pub fn commit(segs: []const Seg, keep: []const usize) !void {
        for (segs, keep) |g, k| {
            if (k == 0 or k > g.st.pending) return error.BadCommit;
            g.st.pos += k;
            g.st.pending = 0;
        }
    }
};
