//! Flash Next (qwen4_exp) on CUDA, one rank of two: the forward the Python engine runs for our recipe (int8 dense
//! linears, affine 4-bit routed experts, the int8 shared expert, a 4-bit MTP draft head), as code. Each launch takes
//! the grid, constexprs and arguments the Python engine gives it (kern.zig), so every row equals Python's bit for bit.
//!
//! The engine's calls (``prefill``, ``verify``, ``keep``, ``draft``) are the lane backend's; both ranks make the same
//! calls in the same order. Tokens are drawn inside the calls from both ranks' vocabulary shards (an all-gather of
//! each rank's best candidate), so both ranks reach the same tokens with no host decision that could differ.

const std = @import("std");
const cuda = @import("cuda");
const api = @import("api.zig");
const kern = @import("kern.zig");
const ngram_mod = @import("ngram.zig");
const grow_mod = @import("grow.zig");
const draw_mod = @import("draw.zig");
const lanes = @import("lanes");
pub const devstore = @import("devstore.zig");
pub const api_ = api;

const K = kern.K;
const D = kern.D;
const WIDE = kern.WIDE;

pub const LAYERS = 48;
pub const LIN = 36; // DeltaNet layers (every layer but each fourth)
pub const ATT = 12;
pub const PREFILL_ROWS: i64 = 2048;
pub const ATT_ROWS: i64 = 256;
pub const SLOTS: i64 = 11; // ten routed experts and the shared one
pub const CONV_DIM: i64 = 5120;
pub const PROJ_W: i64 = 8240;
pub const HEAD_N: i64 = 124160; // a rank's share of the vocabulary
pub const DRAFT_N: i64 = 39796;

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

// -- weights --------------------------------------------------------------------------------------------------------
const HCW = struct { scale: u64, down: kern.Face, up: kern.Face };
const MoEW = struct { router: u64, up: u64, down: u64, sgu: kern.Face, sdown: kern.Face };
const AttnW = struct { proj: kern.Face, q_scale: u64, k_scale: u64, iq_scale: u64, ik_scale: u64, o: kern.Face };
const GdnW = struct { proj: kern.Face, conv: u64, a_log: u64, dt_bias: u64, norm: u64, out: kern.Face };
const PleW = struct { key: kern.Face, value: kern.Face, norm_key: u64, norm_query: u64, norm_conv: u64, conv: u64 };
const LayerW = struct {
    attn_hc: HCW,
    mlp_hc: HCW,
    moe: MoEW,
    linear: bool,
    li: usize, // its DeltaNet or attention index
    gdn: GdnW = undefined,
    attn: AttnW = undefined,
    ple: ?PleW = null,
};
const MtpW = struct { norm_e: u64, norm_h: u64, fc_e: kern.Face, fc_h: kern.Face, layer: LayerW, mixer: HCW };

fn face(store: *const api.Store, prefix: []const u8) !kern.Face {
    var f: kern.Face = .{ .k = 0 };
    var buf: [256]u8 = undefined;
    for (0..2) |i| {
        const name = try std.fmt.bufPrint(&buf, "{s}.parts.{d}.weight", .{ prefix, i });
        const w: api.Tensor = if (i == 0) try store.get(name) else (store.map.get(name) orelse break); // a second part is optional
        const sc = store.map.get(try std.fmt.bufPrint(&buf, "{s}.parts.{d}.scale", .{ prefix, i }));
        const n = w.shape[0];
        const kk = w.shape[1];
        f.k = kk;
        f.parts[i] = .{ .w = w.ptr, .s = if (sc) |s| s.ptr else 0, .n = n, .gs = if (sc) |s| @divExact(kk, s.shape[1]) else 0 };
        f.count = i + 1;
    }
    return f;
}

fn ptr(store: *const api.Store, comptime fmt: []const u8, args: anytype) !u64 {
    return (try store.getf(fmt, args)).ptr;
}

fn hcw(store: *const api.Store, prefix: []const u8) !HCW {
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    return .{ .scale = (try store.get(try std.fmt.bufPrint(&b1, "{s}.scale", .{prefix}))).ptr,
        .down = try face(store, try std.fmt.bufPrint(&b1, "{s}.down", .{prefix})),
        .up = try face(store, try std.fmt.bufPrint(&b2, "{s}.up", .{prefix})) };
}

fn layerW(store: *const api.Store, p: []const u8, linear: bool, li: usize) !LayerW {
    var b: [160]u8 = undefined;
    var l: LayerW = .{
        .attn_hc = try hcw(store, try std.fmt.bufPrint(&b, "{s}.attn_hc", .{p})),
        .mlp_hc = try hcw(store, try std.fmt.bufPrint(&b, "{s}.mlp_hc", .{p})),
        .moe = .{ .router = try ptr(store, "{s}.moe.router", .{p}), .up = try ptr(store, "{s}.moe.experts.routed_experts.up", .{p}),
            .down = try ptr(store, "{s}.moe.experts.routed_experts.down", .{p}),
            .sgu = try face(store, try std.fmt.bufPrint(&b, "{s}.moe.experts.shared_gu", .{p})),
            .sdown = try face(store, try std.fmt.bufPrint(&b, "{s}.moe.experts.shared_down", .{p})) },
        .linear = linear,
        .li = li,
    };
    if (linear) {
        l.gdn = .{ .proj = try face(store, try std.fmt.bufPrint(&b, "{s}.gdn.proj", .{p})), .conv = try ptr(store, "{s}.gdn.conv", .{p}),
            .a_log = try ptr(store, "{s}.gdn.a_log", .{p}), .dt_bias = try ptr(store, "{s}.gdn.dt_bias", .{p}), .norm = try ptr(store, "{s}.gdn.norm", .{p}),
            .out = try face(store, try std.fmt.bufPrint(&b, "{s}.gdn.out", .{p})) };
    } else {
        l.attn = .{ .proj = try face(store, try std.fmt.bufPrint(&b, "{s}.attn.proj", .{p})), .q_scale = try ptr(store, "{s}.attn.q_scale", .{p}),
            .k_scale = try ptr(store, "{s}.attn.k_scale", .{p}), .iq_scale = try ptr(store, "{s}.attn.iq_scale", .{p}),
            .ik_scale = try ptr(store, "{s}.attn.ik_scale", .{p}), .o = try face(store, try std.fmt.bufPrint(&b, "{s}.attn.o", .{p})) };
    }
    if (store.map.get(try std.fmt.bufPrint(&b, "{s}.ple.conv", .{p}))) |_| {
        l.ple = .{ .key = try face(store, try std.fmt.bufPrint(&b, "{s}.ple.key", .{p})), .value = try face(store, try std.fmt.bufPrint(&b, "{s}.ple.value", .{p})),
            .norm_key = try ptr(store, "{s}.ple.norm_key", .{p}), .norm_query = try ptr(store, "{s}.ple.norm_query", .{p}),
            .norm_conv = try ptr(store, "{s}.ple.norm_conv", .{p}), .conv = try ptr(store, "{s}.ple.conv", .{p}) };
    }
    return l;
}

// -- device memory: one allocation, carved -------------------------------------------------------------------------
const Carve = struct {
    base: u64 = 0,
    at: u64 = 0,
    fn take(self: *Carve, bytes: i64) u64 {
        const p = self.base + self.at;
        self.at += std.mem.alignForward(u64, @intCast(@max(bytes, 1)), 512);
        return p;
    }
};

/// Scratch for windows of up to ``rows`` rows (state.py's Buffers): the main window's, the MTP head's, a prompt's.
pub const Buffers = struct {
    rows: i64,
    prefill: bool,
    moe_prefill: bool, // the experts' prompt arithmetic (bf16 slots) also for windows: the main window's buffers
    nb: i64, // key blocks of the context (the indexer's scores row)
    mem: cuda.DeviceBuffer = undefined,
    ids: u64 = 0, h: u64 = 0, pss: u64 = 0, normed: u64 = 0, xs_normed: u64 = 0, act: u64 = 0, xs_act: u64 = 0,
    inj_a: u64 = 0, inj_m: u64 = 0, up: u64 = 0, mixed: u64 = 0, xs_mixed: u64 = 0,
    pa: u64 = 0, q: u64 = 0, iq: u64 = 0,
    a_po: u64 = 0, a_pm: u64 = 0, a_pl: u64 = 0, a_out: u64 = 0, a_ids: u64 = 0, a_nk: u64 = 0, a_sparse: u64 = 0, a_scores: u64 = 0,
    gated: u64 = 0, xs_gated: u64 = 0,
    m_logits: u64 = 0, m_pick: u64 = 0, m_wts: u64 = 0, m_act: u64 = 0, m_y: u64 = 0, plan: kern.Plan = undefined,
    proj: u64 = 0, windows: u64 = 0, sid: u64 = 0, pos_blk: u64 = 0,
    gout: u64 = 0, gxs: u64 = 0, attn_o: u64 = 0, streams: u64 = 0, logits: u64 = 0,
    part_branch: u64 = 0, part_moe: u64 = 0, g_branch: u64 = 0, g_moe: u64 = 0,
    cand: u64 = 0, gath: u64 = 0,
    ple_v: u64 = 0, ple_emb: u64 = 0, xs_ple: u64 = 0, ple_keys: u64 = 0, ple_vals: u64 = 0, ple_gated: u64 = 0, ple_pss: u64 = 0, ple_nrow: u64 = 0,
    mtp_e: u64 = 0, mtp_eo: u64 = 0, mtp_hn: u64 = 0, mtp_xh: u64 = 0, mtp_hs: u64 = 0, mtp_in: u64 = 0,
    kpart: u64 = 0, hcpart: u64 = 0, dnf: u64 = 0, sgu: u64 = 0,
    seg_nk: u64 = 0, seg_sparse: u64 = 0, // a shared round's per-stream key counts, 64 bytes a stream (16-aligned)
    gdn_tab: u64 = 0, // a shared round's DeltaNet table: each layer's streams (ChainSeg), or each layer's replays at a keep
    attn_ints: u64 = 0, attn_ptrs: u64 = 0, // a shared round's attention tables (attnTable)
    attn: ?kern.MultiTab = null, // set while a forward's streams attend through the tables
    tk: u64 = 0, gtk: u64 = 0, invt: u64 = 0, // sampled rows: each row's candidates, both ranks' gathered, 1 / temperature
    // several prompts in one pass (prefillMany): each row's conv taps and sequence, each DeltaNet layer's conv windows
    mwindows: u64 = 0, msid: u64 = 0, conv_tab: u64 = 0,
    multi_prompts: bool = false,
    chains_staged: bool = false, // gdn_tab holds this forward's chains (a shared round of the main forward)

    fn lay(b: *Buffers, c: *Carve) void {
        const R = b.rows;
        const ar = if (b.prefill) @min(R, ATT_ROWS) else R;
        b.ids = c.take(R * 4);
        b.h = c.take(R * WIDE * 2);
        b.pss = c.take(R * 10 * 4 * 4);
        b.normed = c.take(R * WIDE * 2);
        b.xs_normed = c.take(R * (WIDE / 32) * 4);
        b.act = c.take(R * kern.LOW * 2);
        b.xs_act = c.take(R * 10 * 4);
        b.inj_a = c.take(R * 4 * 2);
        b.inj_m = c.take(R * 4 * 2);
        b.up = c.take(R * WIDE * 2);
        b.mixed = c.take(R * D * 2);
        b.xs_mixed = c.take(R * (D / 32) * 4);
        b.pa = c.take(R * 7296 * 2);
        b.q = c.take(R * 12 * 256 * 2);
        b.iq = c.take(R * 4 * 128 * 2);
        b.a_po = c.take(ar * kern.NCH * 12 * 256 * 4);
        b.a_pm = c.take(ar * kern.NCH * 12 * 4);
        b.a_pl = c.take(ar * kern.NCH * 12 * 4);
        b.a_out = c.take(ar * 12 * 256 * 2);
        b.a_ids = c.take(ar * kern.IDW * 4);
        b.a_nk = c.take(ar * 4);
        b.a_sparse = c.take(ar * 4);
        b.a_scores = c.take(ar * b.nb * 4);
        b.gated = c.take(R * 3072 * 2);
        b.xs_gated = c.take(R * 96 * 4);
        const pairs = R * SLOTS;
        b.m_logits = c.take(R * 513 * 4);
        b.m_pick = c.take(R * SLOTS * 4);
        b.m_wts = c.take(R * SLOTS * 4);
        const wide = pairs > 1024;
        b.plan = .{ .members = c.take(pairs * 4), .items = c.take(kern.maxItems(pairs, 512, 16) * 3 * 4), .counts = c.take(2 * 4),
            .rank = c.take(if (wide) pairs * 4 else 4), .hist = c.take(if (wide) cdiv(pairs, 1024) * 512 * 4 else 4),
            .tile = @as(i64, if (b.moe_prefill) 64 else 16) };
        b.m_act = c.take(pairs * kern.LOW * 2);
        b.m_y = c.take(pairs * D * @as(i64, if (b.moe_prefill) 2 else 4));
        b.proj = c.take(@as(i64, if (b.prefill) 1 else LIN) * R * PROJ_W * 2);
        if (b.prefill) {
            b.windows = c.take(R * 4 * 4);
            b.sid = c.take(R * 4);
            b.pos_blk = c.take(4);
            b.mwindows = c.take(R * 4 * 4);
            b.msid = c.take(R * 4);
            b.conv_tab = c.take(LIN * max_parts * 8);
        }
        b.gout = c.take(R * 3072 * 2);
        b.gxs = c.take(R * 96 * 4);
        b.attn_o = c.take(R * 12 * 256 * 2);
        b.streams = c.take(R * WIDE * 2);
        b.logits = c.take((if (b.prefill) 16 else R) * HEAD_N * 2);
        b.part_branch = c.take(R * D * 4);
        b.part_moe = c.take(R * D * 4);
        b.g_branch = c.take(2 * R * D * 4);
        b.g_moe = c.take(2 * R * D * 4);
        b.cand = c.take(@max(R, 16) * 4 * 4);
        b.gath = c.take(2 * @max(R, 16) * 4 * 4);
        b.ple_v = c.take(R * 16 * 160 * 2);
        b.ple_emb = c.take(R * D * 2);
        b.xs_ple = c.take(R * (D / 32) * 4);
        b.ple_keys = c.take(R * WIDE * 2);
        b.ple_vals = c.take(R * D * 2);
        b.ple_gated = c.take(R * WIDE * 2);
        b.ple_pss = c.take(R * 4 * 4);
        b.ple_nrow = c.take(R * WIDE * 2);
        b.mtp_e = c.take(R * D * 2);
        b.mtp_eo = c.take(R * D * 2);
        b.mtp_hn = c.take(R * WIDE * 2);
        b.mtp_xh = c.take(R * (WIDE / 32) * 4);
        b.mtp_hs = c.take(R * S4 * D * 2);
        b.mtp_in = c.take(R * WIDE * 2);
        const small = @min(R * S4, 128);
        b.kpart = c.take(4 * small * 10240 * 4);
        b.hcpart = c.take(8 * @min(R, 128) * 324 * 4);
        b.dnf = c.take(R * 324 * 4);
        b.sgu = c.take(R * 640 * 2);
        const srows = if (b.prefill) max_parts else R; // a prompt draws its last row only (prefillMany: each prompt's)
        b.tk = c.take(srows * draw_mod.W * 4);
        b.gtk = c.take(2 * srows * draw_mod.W * 4);
        b.invt = c.take(srows * 4);
        b.seg_nk = c.take(max_parts * 64);
        b.seg_sparse = c.take(max_parts * 64);
        b.gdn_tab = c.take(LIN * max_parts * @sizeOf(ChainSeg));
        b.attn_ints = c.take((2 * R + 2 * max_parts) * 4);
        b.attn_ptrs = c.take((ATT + 1) * 6 * max_parts * 8);
    }

    fn init(d: *const cuda.Driver, rows: i64, is_prefill: bool, moe_prefill: bool, nb: i64) !Buffers {
        var b: Buffers = .{ .rows = rows, .prefill = is_prefill, .moe_prefill = moe_prefill, .nb = nb };
        var c: Carve = .{};
        b.lay(&c);
        b.mem = try cuda.DeviceBuffer.alloc(d, c.at);
        try d.check(d.api.cuMemsetD8_v2(b.mem.ptr, 0, c.at), "zero buffers");
        c = .{ .base = b.mem.ptr };
        b.lay(&c);
        if (is_prefill) { // conv taps [state (3) | the chunk's rows]: row r reads taps r .. r + 3
            const w = try std.heap.page_allocator.alloc(i32, @intCast(rows * 4));
            defer std.heap.page_allocator.free(w);
            for (0..@intCast(rows)) |r| for (0..4) |t| {
                w[r * 4 + t] = @intCast(r + t);
            };
            try d.check(d.api.cuMemcpyHtoD_v2(b.windows, w.ptr, w.len * 4), "windows");
        }
        return b;
    }
};

const S4: i64 = 4;

/// A prompt's state after ``at`` tokens (forward.py's cut_snapshot): each DeltaNet layer's recurrent state and conv
/// window, the n-gram tail and history, the MTP head's length (every row but the point's last) and that last row's
/// streams, which the head absorbs with the next prompt's token when a turn resumes here.
pub const Snap = struct {
    buf: ?cuda.DeviceBuffer = null,
    at: i64 = 0, // 0: none kept
    mtp_len: i64 = 0,
    hist: ngram_mod.History = .{},

    const REC: u64 = @intCast(LIN * Seq.REC_LAYER);
    const CONV: u64 = @intCast(LIN * 3 * CONV_DIM * 2);
    const PLE: u64 = @intCast(9 * WIDE * 2);
    const TAIL: u64 = @intCast(WIDE * 2);

    fn rec(sn: *const Snap, li: usize) u64 {
        return sn.buf.?.ptr + li * @as(u64, @intCast(Seq.REC_LAYER));
    }
    fn conv(sn: *const Snap, li: usize) u64 {
        return sn.buf.?.ptr + REC + li * @as(u64, @intCast(3 * CONV_DIM * 2));
    }
    fn ple(sn: *const Snap) u64 {
        return sn.buf.?.ptr + REC + CONV;
    }
    fn tail(sn: *const Snap) u64 {
        return sn.buf.?.ptr + REC + CONV + PLE;
    }
};

/// One sequence's committed state (state.py's State) on the device, plus its host bookkeeping.
pub const Seq = struct {
    capacity: i64,
    mem: cuda.DeviceBuffer = undefined,
    pos_dev: u64 = 0, mtp_pos: u64 = 0,
    conv: u64 = 0, rec: u64 = 0, conv_ptrs: u64 = 0,
    sc_k: [LIN]u64 = undefined, sc_v: [LIN]u64 = undefined, sc_g: [LIN]u64 = undefined, sc_b: [LIN]u64 = undefined,
    kc_k: [ATT + 1]u64 = undefined, kc_v: [ATT + 1]u64 = undefined, kc_ks: [ATT + 1]u64 = undefined, kc_vs: [ATT + 1]u64 = undefined,
    ikc: [ATT + 1]u64 = undefined, pooled: [ATT + 1]u64 = undefined,
    ple_tail: u64 = 0,
    last_streams: u64 = 0,
    grow: grow_mod.Range = undefined, // the caches that grow with the position (kc_*, ikc, pooled)
    img: ?Image = null, // an image prompt's rotary table and (until its prefill ends) its features
    sampling: ?lanes.Sampling = null, // the stream's draw (null: greedy); its rows step eagerly
    mask: ?Mask = null, // the next forward's grammar rows (set before each call, cleared by it)
    cut_at: ?i64 = null, // the next prefill keeps its state after this many prompt tokens (snap)
    snap: Snap = .{}, // the kept prompt state a next turn resumes from (multi.py's kept points)
    // host
    pos: i64 = 0,
    mtp_len: i64 = 0,
    mtp_drafted: i64 = 0,
    cur: [LIN]u1 = @splat(0),
    hist: ngram_mod.History = .{},
    fresh: bool = true, // the next draft absorbs the prompt's last streams
    last_rows: i64 = 0, // the last verify's rows
    last_row0: i64 = 0, // where they sat in the window's buffers (a shared round packs several streams)
    last_tokens: [16]i64 = undefined,

    const REC_LAYER: i64 = 24 * 128 * 128 * 4;

    fn lay(s: *Seq, c: *Carve, max_rows: i64) void {
        const cap = s.capacity;
        s.pos_dev = c.take(4);
        s.mtp_pos = c.take(4);
        s.conv = c.take(LIN * 3 * CONV_DIM * 2);
        s.rec = c.take(2 * LIN * REC_LAYER);
        s.conv_ptrs = c.take(LIN * 8);
        for (0..LIN) |i| {
            s.sc_k[i] = c.take(max_rows * 8 * 128 * 4);
            s.sc_v[i] = c.take(max_rows * 24 * 128 * 2);
            s.sc_g[i] = c.take(max_rows * 24 * 4);
            s.sc_b[i] = c.take(max_rows * 24 * 4);
        }
        _ = cap; // the position-sized caches live in ``grow`` (layGrow)
        s.ple_tail = c.take(9 * WIDE * 2);
        s.last_streams = c.take(WIDE * 2);
    }

    /// Each attention layer's caches as regions of one reserved range (mapped as positions reach them).
    fn layGrow(s: *Seq, e: *Engine) !void {
        s.grow = .{ .d = e.ctx.d, .gpa = e.gpa, .gran = e.gran, .budget = &e.budget };
        errdefer s.grow.deinit();
        const cap: u64 = @intCast(s.capacity);
        var offs: [ATT + 1][6]u64 = undefined;
        for (&offs) |*o| {
            o[0] = try s.grow.add(cap * 256, 256, 1);
            o[1] = try s.grow.add(cap * 256, 256, 1);
            o[2] = try s.grow.add(cap * 16, 16, 1);
            o[3] = try s.grow.add(cap * 16, 16, 1);
            o[4] = try s.grow.add(cap * 256, 256, 1);
            o[5] = try s.grow.add(@as(u64, @intCast(cdiv(s.capacity, 4))) * 256, 256, 4);
        }
        try s.grow.reserve();
        s.pointGrow();
    }

    fn pointGrow(s: *Seq) void {
        const r = s.grow.regions.items;
        for (0..ATT + 1) |i| {
            s.kc_k[i] = s.grow.base + r[6 * i].off;
            s.kc_v[i] = s.grow.base + r[6 * i + 1].off;
            s.kc_ks[i] = s.grow.base + r[6 * i + 2].off;
            s.kc_vs[i] = s.grow.base + r[6 * i + 3].off;
            s.ikc[i] = s.grow.base + r[6 * i + 4].off;
            s.pooled[i] = s.grow.base + r[6 * i + 5].off;
        }
    }

    fn recAt(s: *const Seq, half: u1, li: usize) u64 {
        return s.rec + (@as(u64, half) * LIN + li) * REC_LAYER;
    }
};

pub const Options = struct { context: usize, max_rows: u32, depth: u32 };

/// An image prompt on a sequence: the rotary positions of its rows ([length, 3] int32, then the decode delta) and,
/// while the prompt is prefilled, the features that replace its image placeholder rows (``rows``: their indexes).
pub const Image = struct {
    table: cuda.DeviceBuffer,
    length: i64,
    rows: []u32,
    feats: ?cuda.DeviceBuffer,

    fn rope(im: *const Image) kern.Rope {
        return .{ .table = im.table.ptr, .delta = im.table.ptr + deltaAt(im.length), .length = im.length };
    }

    fn deltaAt(length: i64) u64 {
        return std.mem.alignForward(u64, @intCast(length * 12), 16);
    }
};

/// ``s`` gets an image prompt: its rows' rotary positions (``positions`` [3, length] as the Python engine computes
/// them), the decode ``delta``, and the features (``feats``: device bf16 [rows.len, 2560], copied) for its
/// placeholder ``rows``. Before prefill; an image sequence runs its steps eagerly (its table is its own).
pub fn attach(e: *Engine, s: *Seq, rows: []const u32, feats: u64, positions: []const i32, delta: i64) !void {
    detach(e, s);
    const length: i64 = @intCast(positions.len / 3);
    var table = try cuda.DeviceBuffer.alloc(e.ctx.d, @intCast(Image.deltaAt(length) + 16));
    errdefer table.free();
    const host = try e.gpa.alloc(i32, @intCast(length * 3 + 4));
    defer e.gpa.free(host);
    @memset(host, 0);
    for (0..@intCast(length)) |i| for (0..3) |a| {
        host[i * 3 + a] = positions[a * @as(usize, @intCast(length)) + i];
    };
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyHtoD_v2(table.ptr, host.ptr, @intCast(length * 12)), "image positions");
    const d32: i32 = @intCast(delta);
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyHtoD_v2(table.ptr + Image.deltaAt(length), &d32, 4), "image delta");
    var fb = try cuda.DeviceBuffer.alloc(e.ctx.d, rows.len * 2560 * 2);
    errdefer fb.free();
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoD_v2(fb.ptr, feats, rows.len * 2560 * 2), "image features");
    s.img = .{ .table = table, .length = length, .rows = try e.gpa.dupe(u32, rows), .feats = fb };
}

pub fn detach(e: *Engine, s: *Seq) void {
    if (s.img) |*im| {
        im.table.free();
        if (im.feats) |*f| f.free();
        e.gpa.free(im.rows);
    }
    s.img = null;
}


/// One sequence's rows of a forward: rows ``row0 .. row0 + rows`` of the buffers belong to ``s`` (a shared round
/// packs several streams' windows; a lone window or a prompt chunk is one segment from row 0).
pub const Seg = struct { s: *Seq, row0: i64 = 0, rows: i64, cut: i64 = 0 }; // cut: a prompt piece's kept point (prefillMany)

fn rowAt(base: u64, row0: i64, row_bytes: i64) u64 {
    return base + @as(u64, @intCast(row0 * row_bytes));
}

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: *api.Ctx,
    k: K,
    layers: [LAYERS]LayerW,
    mixer: HCW,
    mtp: MtpW,
    embed: u64,
    head: kern.Face,
    inv_freq: u64,
    dh: kern.DraftHead,
    draft_ids: cuda.DeviceBuffer,
    vocab_offset: i64,
    ng: ngram_mod.NGram,
    opts: Options,
    capacity: i64,
    buf: Buffers,
    mbuf: Buffers,
    pbuf: Buffers,
    host: cuda.HostBuffer,
    // per-call staging on the host
    idbuf: []i32,
    rowbuf: []i64,
    valbuf: []u16,
    confidence: f64 = 0,
    draft_p: [16]f64 = undefined, // the last draftUpTo's head probability for each draft it returned
    gran: u64 = 0, // device memory's mapping granularity
    cand_host: []i32 = &.{}, // gathered candidates, read back
    full: cuda.DeviceBuffer = undefined, // a full row of both ranks' logits (a rule past the candidates)
    full_host: []u16 = &.{},
    gbits: cuda.DeviceBuffer = undefined, // a window's grammar rows: their allowed bits over this rank's columns
    grows: cuda.DeviceBuffer = undefined,
    gbits_host: []u32 = &.{},
    budget: grow_mod.Budget = .{ .cap = std.math.maxInt(u64) }, // what every sequence's caches may map together
    /// Verify windows and MTP head steps as CUDA graphs (each width, DeltaNet parity and attention geometry).
    use_graphs: bool = true,
    /// Shared rounds attend every stream in one launch a kernel (the kernel set holds attn_multi's kernels).
    multi_attn: bool = false,
    /// Shared rounds run every stream's DeltaNet chain (and every keep's replays) in one launch.
    multi_gdn: bool = true,
    multi_layers: [2]u64 = .{ 0, 0 }, // shared attention layers in one launch a kernel, and those the set had no variant for
    stage_ms: f64 = 0, // host time staging windows (ids, n-gram rows)
    gpu_ms: f64 = 0, // GPU time of verify windows (events around the step), when ``timing``
    timing: bool = false,
    split: [3]f64 = .{ 0, 0, 0 }, // staging: ids upload, n-gram ids, table rows
    graphs: std.AutoHashMap(GKey, cuda.graph.Exec) = undefined,
};

const GKey = struct { kind: u8, rows: i64, par: u1, sig: [3]i64, seq: usize };

/// The attention launches' geometry at ``keys`` attended keys: score programs, chunk count, selector width.
fn geometry(keys: i64, nb: i64) [3]i64 {
    const blocks = @min(nb, @max(1, cdiv(keys, 4)));
    return .{ cdiv(blocks, 64), @min(5, cdiv(@min(keys, 2051), 512)), @intCast(std.math.ceilPowerOfTwo(u64, @intCast(blocks)) catch unreachable) };
}

/// The step's GPU work (capturable): a verify window's forward, head and candidates; or the MTP head over its rows.
fn body(e: *Engine, kind: u8, s: *Seq, R: i64) !void {
    if (kind == 0) {
        const b = &e.buf;
        const pending = try mainForward(e, b, &.{.{ .s = s, .rows = R }}, R);
        _ = try finish(e, b, e.mixer, R, pending);
        try kern.matmul(&e.k, b.mixed, D, e.head, b.logits, HEAD_N, false, R, b.kpart);
        try candidates(e, b, b.logits, HEAD_N, 0, e.vocab_offset, R);
    } else try mtpCompute(e, &e.mbuf, s, R);
}

/// Captures (without running) every verify width at both DeltaNet parities and every MTP head width, at the
/// sequence's current geometry: the rounds that follow launch graphs only (until the context grows past it).
pub fn warm(e: *Engine, s: *Seq, widths: u32) !void {
    if (!e.use_graphs) return;
    const cur = s.cur;
    defer s.cur = cur;
    for (1..@as(usize, widths) + 1) |r| {
        for ([_]u1{ 0, 1 }) |par| {
            s.cur = @splat(par);
            try capture(e, 0, s, @intCast(r));
        }
        try capture(e, 1, s, @intCast(r));
    }
}

fn keyOf(e: *Engine, kind: u8, s: *Seq, R: i64) GKey {
    const keys = (if (kind == 0) s.pos else s.mtp_len) + R;
    return .{ .kind = kind, .rows = R, .par = if (kind == 0) s.cur[0] else 0, .sig = geometry(keys, e.buf.nb), .seq = @intFromPtr(s) };
}

/// ``body`` captured into this key's graph (a new geometry replaces the width's older graph: the context only grows).
fn capture(e: *Engine, kind: u8, s: *Seq, R: i64) !void {
    const key = keyOf(e, kind, s, R);
    if (e.graphs.contains(key)) return;
    var it = e.graphs.iterator();
    var stale: ?GKey = null;
    while (it.next()) |en| {
        const o = en.key_ptr.*;
        if (o.kind == key.kind and o.rows == key.rows and o.par == key.par and o.seq == key.seq) stale = o;
    }
    if (stale) |o| {
        var ex = e.graphs.fetchRemove(o).?.value;
        ex.deinit();
    }
    try cuda.graph.beginCapture(e.k.stream, .thread_local);
    body(e, kind, s, R) catch |err| {
        if (cuda.graph.endCapture(e.k.stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return err;
    };
    var g = try cuda.graph.endCapture(e.k.stream);
    defer g.deinit();
    const ex = try g.instantiate();
    try ex.upload(e.k.stream);
    try e.graphs.put(key, ex);
}

/// ``body`` from its graph (captured the first time this width, parity and geometry run).
fn step(e: *Engine, kind: u8, s: *Seq, R: i64) !void {
    if (!e.use_graphs or e.k.sync_each or s.img != null or (kind == 0 and s.sampling != null)) return body(e, kind, s, R); // image and sampled sequences: eager
    try capture(e, kind, s, R);
    return e.graphs.get(keyOf(e, kind, s, R)).?.launchOn(e.k.stream);
}


pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *api.Ctx, kernels: *api.Kernels, store: *const api.Store, opts: Options) !*Engine {
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.gpa = gpa;
    e.io = io;
    e.ctx = ctx;
    e.opts = opts;
    e.capacity = @intCast(opts.context + opts.depth + 1);
    e.gran = try grow_mod.Range.granularity(ctx.d);
    e.cand_host = try gpa.alloc(i32, @intCast(2 * @max(rows_for(opts), 1) * draw_mod.W));
    e.full = try cuda.DeviceBuffer.alloc(ctx.d, 2 * HEAD_N * 2);
    e.full_host = try gpa.alloc(u16, 2 * HEAD_N);
    e.gbits = try cuda.DeviceBuffer.alloc(ctx.d, @intCast(max_round_rows * MASK_WORDS * 4));
    e.grows = try cuda.DeviceBuffer.alloc(ctx.d, max_round_rows * 4);
    e.gbits_host = try gpa.alloc(u32, @intCast(max_round_rows * MASK_WORDS));
    e.budget = .{ .cap = std.math.maxInt(u64) };
    e.k = try K.init(gpa, ctx.d, ctx.ctx, ctx.stream, &kernels.triton, &kernels.ext);
    var ai: usize = 0;
    var li: usize = 0;
    var b: [64]u8 = undefined;
    for (0..LAYERS) |i| {
        const linear = i % 4 != 3;
        e.layers[i] = try layerW(store, try std.fmt.bufPrint(&b, "layers.{d}", .{i}), linear, if (linear) li else ai);
        if (linear) li += 1 else ai += 1;
    }
    e.mixer = try hcw(store, "mixer");
    e.mtp = .{ .norm_e = try ptr(store, "mtp.norm_e", .{}), .norm_h = try ptr(store, "mtp.norm_h", .{}), .fc_e = try face(store, "mtp.fc_e"),
        .fc_h = try face(store, "mtp.fc_h"), .layer = try layerW(store, "mtp.layer", false, ATT), .mixer = try hcw(store, "mtp.mixer") };
    e.embed = try ptr(store, "embed.0", .{});
    e.head = try face(store, "head");
    e.inv_freq = try ptr(store, "inv_freq", .{});
    const dw = try store.get("draft_head.weight");
    const ds = try store.get("draft_head.scales");
    const ids = try store.get("draft_ids");
    e.dh = .{ .w = dw.ptr, .scales = ds.ptr, .biases = try ptr(store, "draft_head.biases", .{}), .n = ids.shape[0], .npad = ds.shape[1] };
    // the draft vocabulary's global ids as int32 for the candidate kernel (the loader's host copy, else the device's)
    const n_ids: usize = @intCast(ids.shape[0]);
    e.draft_ids = try cuda.DeviceBuffer.alloc(ctx.d, n_ids * 4);
    if (store.host.get("draft_ids")) |h| {
        if (h.len != n_ids * 4) return error.DraftIdsMismatch;
        try e.draft_ids.upload(0, h);
    } else {
        const host64 = try gpa.alloc(i64, n_ids);
        defer gpa.free(host64);
        try ctx.d.check(ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(host64.ptr), ids.ptr, n_ids * 8), "draft ids");
        const host32 = try gpa.alloc(i32, n_ids);
        defer gpa.free(host32);
        for (host64, host32) |x, *y| y.* = @intCast(x);
        try e.draft_ids.upload(0, std.mem.sliceAsBytes(host32));
    }
    e.vocab_offset = @as(i64, ctx.rank) * e.head.n();
    e.ng = try ngram_mod.NGram.open(gpa, io, store.host.get("ngram.json") orelse return error.MissingNgramGeometry, store.host.get("ngram_root") orelse ".");
    const nb = cdiv(e.capacity, 4);
    const rows: i64 = @intCast(@max(opts.max_rows, 1));
    e.buf = try Buffers.init(ctx.d, rows, false, true, nb);
    e.mbuf = try Buffers.init(ctx.d, rows, false, false, nb);
    e.pbuf = try Buffers.init(ctx.d, PREFILL_ROWS, true, true, nb);
    e.host = try cuda.HostBuffer.alloc(ctx.d, @intCast(2 * @max(rows, 16) * 4 * 4));
    e.idbuf = try gpa.alloc(i32, @intCast(PREFILL_ROWS));
    e.rowbuf = try gpa.alloc(i64, @intCast(PREFILL_ROWS * 16));
    e.valbuf = try gpa.alloc(u16, @intCast(PREFILL_ROWS * 16 * 160));
    e.use_graphs = true;
    e.multi_attn = kern.hasMulti(&e.k);
    e.multi_gdn = true;
    e.confidence = 0;
    e.stage_ms = 0;
    e.split = .{ 0, 0, 0 };
    e.graphs = .init(gpa);
    return e;
}

/// Pages the n-gram tables in (after the communicator is up: NCCL registers its buffers first).
pub fn prefetchTables(e: *Engine) !void {
    try e.ng.map(true);
}

pub fn deinit(e: *Engine) void {
    e.buf.mem.free();
    e.mbuf.mem.free();
    e.pbuf.mem.free();
    e.draft_ids.free();
    e.host.free();
    e.gpa.free(e.idbuf);
    e.gpa.free(e.rowbuf);
    e.gpa.free(e.cand_host);
    e.gpa.free(e.full_host);
    e.full.free();
    e.gbits.free();
    e.grows.free();
    e.gpa.free(e.gbits_host);
    e.gpa.free(e.valbuf);
    e.gpa.destroy(e);
}

pub fn maxLen(e: *const Engine) usize {
    return e.opts.context;
}

pub fn seqBytes(e: *const Engine) usize {
    var s: Seq = .{ .capacity = e.capacity };
    var c: Carve = .{};
    s.lay(&c, windowRows(e));
    // and the caches' first growth step (the rest is mapped as the sequence grows, within the engine's budget)
    const per_pos: u64 = (ATT + 1) * (256 + 256 + 16 + 16 + 256 + 64);
    return c.at + per_pos * @as(u64, @intCast(grow_mod.step_positions));
}

/// A sequence's widest window: the pending token and its drafts (a shared round's rows are several such windows).
fn windowRows(e: *const Engine) i64 {
    return @min(@as(i64, e.opts.depth) + 1, 16);
}

/// The bytes every sequence's growing caches may map together (the server's room for streams).
pub fn setGrowthBudget(e: *Engine, cap: u64) void {
    e.budget.cap = cap;
}

/// ``s``'s caches backed through ``positions`` (its main and MTP positions), before a forward writes them.
fn room(e: *Engine, s: *Seq, extra: i64) !void {
    try s.grow.ensure(@max(s.pos, s.mtp_len) + extra, e.k.stream.handle);
}

pub fn newSeq(e: *Engine) !*Seq {
    const s = try e.gpa.create(Seq);
    errdefer e.gpa.destroy(s);
    s.* = .{ .capacity = e.capacity };
    var c: Carve = .{};
    s.lay(&c, windowRows(e));
    s.mem = try cuda.DeviceBuffer.alloc(e.ctx.d, c.at);
    try e.ctx.d.check(e.ctx.d.api.cuMemsetD8_v2(s.mem.ptr, 0, c.at), "zero state");
    c = .{ .base = s.mem.ptr };
    s.lay(&c, windowRows(e));
    var ptrs: [LIN]u64 = undefined;
    for (0..LIN) |i| ptrs[i] = s.conv + i * 3 * CONV_DIM * 2;
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyHtoD_v2(s.conv_ptrs, &ptrs, LIN * 8), "conv pointers");
    s.hist = e.ng.start();
    try s.layGrow(e);
    return s;
}

/// ``s`` as newSeq leaves a sequence, in its own memory (zeroed on the stream, no allocation): a server reuses one
/// sequence's ~11 GiB at a 1M window instead of allocating and zeroing it per request, and its graphs stay valid.
pub fn resetSeq(e: *Engine, s: *Seq) !void {
    detach(e, s);
    const mem = s.mem;
    var g = s.grow;
    g.shrink(); // the caches' memory back to the budget (mapped again, zeroed, as the next request grows)
    const snap_buf = s.snap.buf;
    s.* = .{ .capacity = e.capacity };
    s.snap.buf = snap_buf;
    s.mem = mem;
    s.grow = g;
    s.pointGrow();
    var c: Carve = .{ .base = mem.ptr };
    s.lay(&c, windowRows(e));
    try e.ctx.d.check(e.ctx.d.api.cuMemsetD8Async(mem.ptr, 0, c.at, e.k.stream.handle), "zero state");
    var ptrs: [LIN]u64 = undefined;
    for (0..LIN) |i| ptrs[i] = s.conv + i * 3 * CONV_DIM * 2;
    try e.k.upload(s.conv_ptrs, std.mem.sliceAsBytes(&ptrs));
    s.hist = e.ng.start();
}

pub fn freeSeq(e: *Engine, s: *Seq) void {
    var gone: std.ArrayList(GKey) = .empty;
    defer gone.deinit(e.gpa);
    var it = e.graphs.iterator();
    while (it.next()) |en| if (en.key_ptr.seq == @intFromPtr(s)) gone.append(e.gpa, en.key_ptr.*) catch {};
    for (gone.items) |key| {
        var ex = e.graphs.fetchRemove(key).?.value;
        ex.deinit();
    }
    detach(e, s);
    s.grow.deinit();
    s.mem.free();
    if (s.snap.buf) |*sb| sb.free();
    e.gpa.destroy(s);
}

// -- the forward ----------------------------------------------------------------------------------------------------
const Pending = struct { g: u64, inj: u64 };

fn allGather(e: *Engine, send: u64, recv: u64, count: usize) !void {
    try e.ctx.nccl.check(e.ctx.nccl.api.ncclAllGather(send, recv, count, .f32, e.ctx.comm, e.k.stream.handle), "ncclAllGather");
}

/// forward._readout_b16: norm (unless done), down with SiLU and inject gates, up and the mix -> mixed [R, D].
fn readout(e: *Engine, b: *Buffers, hc: HCW, h: u64, R: i64, inject: ?u64, normed: bool) !void {
    const k = &e.k;
    if (!normed) try kern.hcNormed(k, h, b.pss, hc.scale, b.normed, b.xs_normed, R);
    const ndn = hc.down.n();
    if (try kern.partials(k, b.normed, WIDE, hc.down, b.hcpart, R)) {
        try kern.hcReduceAct(k, b.hcpart, b.act, b.xs_act, inject, R, ndn);
    } else {
        try kern.matmul(k, b.normed, WIDE, hc.down, b.dnf, ndn, true, R, b.kpart);
        try kern.hcAct(k, b.dnf, b.act, b.xs_act, inject, R, ndn);
    }
    if (b.prefill and R > 128 and hc.up.count == 1 and hc.up.parts[0].gs != 0) {
        try kern.upmix(k, b.act, hc.up, b.normed, b.mixed, b.xs_mixed, R);
        return;
    }
    try kern.matmul(k, b.act, kern.LOW, hc.up, b.up, WIDE, false, R, b.kpart);
    try kern.hcMix(k, b.up, b.normed, b.mixed, b.xs_mixed, R);
}

/// forward.hc_block: write the pending branch back into h, then the read-out (and this block's inject gates).
fn hcBlock(e: *Engine, b: *Buffers, hc: HCW, R: i64, pending: ?Pending, inj_out: u64) !void {
    const mode: i64 = if (pending != null) 3 else 0;
    const br = if (pending) |p| p.g else b.h;
    const inj = if (pending) |p| p.inj else b.h;
    if (b.prefill and R > 16) {
        try kern.writeNorm(&e.k, b.h, b.pss, hc.scale, b.normed, b.xs_normed, br, inj, R, mode);
        return readout(e, b, hc, b.h, R, inj_out, true);
    }
    try kern.hcWriteback(&e.k, b.h, b.pss, br, inj, R, mode);
    try readout(e, b, hc, b.h, R, inj_out, false);
}

/// A block's output projection across ranks: this rank's fp32 partials, gathered in rank order.
fn outProj(e: *Engine, b: *Buffers, x: u64, x_stride: i64, f: kern.Face, R: i64) !u64 {
    try kern.matmul(&e.k, x, x_stride, f, b.part_branch, D, true, R, b.kpart);
    try allGather(e, b.part_branch, b.g_branch, @intCast(R * D));
    return b.g_branch;
}

fn gdnBlock(e: *Engine, b: *Buffers, segs: []const Seg, l: LayerW, R: i64) !u64 {
    const k = &e.k;
    const g = l.gdn;
    const li = l.li;
    if (b.prefill) { // a prompt chunk, or several prompts' pieces (prefillMany): each sequence's rows on its own state
        try kern.matmul(k, b.mixed, D, g.proj, b.proj, PROJ_W, false, R, b.kpart);
        // q, k (fp32), v (bf16), g, beta of the piece: carved from the MTP/ple scratch of the prompt buffers
        const qb = b.ple_keys; // [R, 8, 128] fp32 = R * 4096 bytes (ple_keys holds R * 20480)
        const kb = b.ple_keys + @as(u64, @intCast(R * 4096));
        const vb = b.ple_gated; // [R, 24, 128] bf16
        const gb = b.ple_vals; // [R, 24] fp32
        const bb = b.ple_vals + @as(u64, @intCast(R * 96));
        const yb = b.mtp_hn; // [R, 24, 128] bf16 (the MTP head runs after the main forward)
        if (b.multi_prompts) {
            try kern.gdnFront(k, b.proj, b.conv_tab + li * max_parts * 8, b.msid, b.mwindows, g.conv, g.a_log, g.dt_bias, qb, kb, vb, gb, bb, R);
        } else try kern.gdnFront(k, b.proj, segs[0].s.conv_ptrs + li * 8, b.sid, b.windows, g.conv, g.a_log, g.dt_bias, qb, kb, vb, gb, bb, R);
        for (segs) |sg| {
            const s = sg.s;
            const cur = s.cur[li];
            const o: u64 = @intCast(sg.row0);
            const n = sg.rows;
            const m = sg.cut; // a kept point inside the piece: the rows before it, its state, then the rest from it
            if (m > 0 and m < n) {
                const sn = &s.snap;
                const u: u64 = o + @as(u64, @intCast(m));
                try kern.gdnPrefill(k, qb + o * 4096, kb + o * 4096, vb + o * 6144, gb + o * 96, bb + o * 96, s.recAt(cur, li), sn.rec(li), yb + o * 6144, m);
                try kern.gdnPrefill(k, qb + u * 4096, kb + u * 4096, vb + u * 6144, gb + u * 96, bb + u * 96, sn.rec(li), s.recAt(1 - cur, li), yb + u * 6144, n - m);
                try k.copy(sn.conv(li), s.conv + li * 3 * CONV_DIM * 2, 3 * CONV_DIM * 2);
                try kern.shiftWindows(k, sn.conv(li), rowAt(b.proj, sg.row0, PROJ_W * 2), m, 3 * CONV_DIM, b.rows * PROJ_W, PROJ_W, 1, CONV_DIM, 3);
            } else try kern.gdnPrefill(k, qb + o * 4096, kb + o * 4096, vb + o * 6144, gb + o * 96, bb + o * 96, s.recAt(cur, li), s.recAt(1 - cur, li), yb + o * 6144, n);
        }
        try kern.gdnBack(k, yb, b.proj, g.norm, b.gout, b.gxs, R);
        for (segs) |sg| {
            const s = sg.s;
            const cur = s.cur[li];
            s.cur[li] = 1 - cur;
            try kern.shiftWindows(k, s.conv + li * 3 * CONV_DIM * 2, rowAt(b.proj, sg.row0, PROJ_W * 2), sg.rows, 3 * CONV_DIM, b.rows * PROJ_W, PROJ_W, 1, CONV_DIM, 3);
            if (sg.cut == sg.rows) { // the point ends the piece: the state as committed
                try k.copy(s.snap.rec(li), s.recAt(1 - cur, li), @intCast(Seq.REC_LAYER));
                try k.copy(s.snap.conv(li), s.conv + li * 3 * CONV_DIM * 2, 3 * CONV_DIM * 2);
            }
        }
        return outProj(e, b, b.gout, 3072, g.out, R);
    }
    const proj = b.proj + @as(u64, @intCast(@as(i64, @intCast(li)) * b.rows * PROJ_W * 2));
    try kern.matmul(k, b.mixed, D, g.proj, proj, PROJ_W, false, R, b.kpart);
    if (segs.len > 1 and b.chains_staged) { // every stream's chain in one launch (the round's table, chainTable)
        try kern.gdnChainMulti(k, b.gdn_tab + li * max_parts * @sizeOf(ChainSeg), @intCast(segs.len), g.conv, g.a_log, g.dt_bias, g.norm, b.gout, b.gxs);
        return outProj(e, b, b.gout, 3072, g.out, R);
    }
    for (segs) |sg| { // each stream's chain on its own state, over its rows
        const s = sg.s;
        const cur = s.cur[li];
        try kern.gdnChain(k, rowAt(proj, sg.row0, PROJ_W * 2), s.conv + li * 3 * CONV_DIM * 2, g.conv, s.recAt(cur, li), g.a_log, g.dt_bias, g.norm, sg.rows,
            rowAt(b.gout, sg.row0, 3072 * 2), rowAt(b.gxs, sg.row0, 96 * 4), s.recAt(1 - cur, li), s.sc_k[li], s.sc_v[li], s.sc_g[li], s.sc_b[li]);
    }
    return outProj(e, b, b.gout, 3072, g.out, R);
}

fn attnBlock(e: *Engine, b: *Buffers, segs: []const Seg, l: LayerW, R: i64, mtp: bool) !u64 {
    const k = &e.k;
    const a = l.attn;
    const ai = if (mtp) ATT else l.li;
    try kern.matmul(k, b.mixed, D, a.proj, b.pa, 7296, false, R, b.kpart);
    if (b.attn) |tab0| if (segs.len > 1) { // every stream in one launch a kernel (attn_multi.layer)
        var tab = tab0;
        tab.cp = b.attn_ptrs + ai * 6 * max_parts * 8;
        if (try kern.attentionMulti(k, tab, b.pa, a.q_scale, a.k_scale, a.iq_scale, a.ik_scale, e.inv_freq, b.q, b.iq, b.a_po, b.a_pm, b.a_pl, b.a_ids, b.a_nk, b.a_out)) {
            e.multi_layers[0] += 1;
            try kern.attnGate(k, b.a_out, b.pa, b.gated, b.xs_gated, R);
            return outProj(e, b, b.gated, 3072, a.o, R);
        }
        e.multi_layers[1] += 1;
    };
    for (segs) |sg| { // each stream's rows into its own caches, at its own positions
        const s = sg.s;
        const pos = if (mtp) s.mtp_pos else s.pos_dev;
        try kern.attnPrep(k, rowAt(b.pa, sg.row0, 7296 * 2), pos, a.q_scale, a.k_scale, a.iq_scale, e.inv_freq, rowAt(b.q, sg.row0, 12 * 256 * 2),
            s.kc_k[ai], s.kc_v[ai], s.kc_ks[ai], s.kc_vs[ai], rowAt(b.iq, sg.row0, 4 * 128 * 2), s.ikc[ai], sg.rows, if (s.img) |*im| im.rope() else null);
        try kern.pool(k, s.ikc[ai], s.pooled[ai], pos, a.ik_scale, e.inv_freq, sg.rows, if (s.img) |*im| im.rope() else null);
    }
    if (b.prefill) for (segs) |sg| { // a prompt chunk, or several prompts' pieces: each in blocks of ATT_ROWS rows
        const s = sg.s;
        const host_pos = if (mtp) s.mtp_len else s.pos;
        var r0: i64 = 0;
        while (r0 < sg.rows) : (r0 += ATT_ROWS) {
            const n = @min(ATT_ROWS, sg.rows - r0);
            try k.memset32(b.pos_blk, @bitCast(@as(i32, @intCast(host_pos + r0))), 1);
            const ends = host_pos + r0 + n;
            const ro: u64 = @intCast(sg.row0 + r0);
            try kern.qsaRows(k, b.iq + ro * 4 * 128 * 2, s.pooled[ai], b.pos_blk, b.a_scores, b.a_ids, b.a_nk, b.a_sparse, b.nb, n, ends);
            try kern.attention(k, b.q + ro * 12 * 256 * 2, s.kc_k[ai], s.kc_v[ai], s.kc_ks[ai], s.kc_vs[ai], b.pos_blk, b.a_po, b.a_pm, b.a_pl,
                b.a_ids, b.a_nk, b.a_sparse, b.attn_o + ro * 12 * 256 * 2, n, ends);
        }
    };
    if (b.prefill) {
        try kern.attnGate(k, b.attn_o, b.pa, b.gated, b.xs_gated, R);
    } else {
        for (segs, 0..) |sg, j| {
            const s = sg.s;
            const pos = if (mtp) s.mtp_pos else s.pos_dev;
            const keys = (if (mtp) s.mtp_len else s.pos) + sg.rows;
            const r0 = sg.row0;
            // per-row counts at row0 * 4 bytes would break the captured variants' 16-byte alignment: a slot per stream
            const nk = if (segs.len == 1) b.a_nk else b.seg_nk + j * 64;
            const sp = if (segs.len == 1) b.a_sparse else b.seg_sparse + j * 64;
            try kern.qsaRows(k, rowAt(b.iq, r0, 4 * 128 * 2), s.pooled[ai], pos, rowAt(b.a_scores, r0, b.nb * 4), rowAt(b.a_ids, r0, kern.IDW * 4), nk,
                sp, b.nb, sg.rows, keys);
            try kern.attention(k, rowAt(b.q, r0, 12 * 256 * 2), s.kc_k[ai], s.kc_v[ai], s.kc_ks[ai], s.kc_vs[ai], pos, rowAt(b.a_po, r0, kern.NCH * 12 * 256 * 4),
                rowAt(b.a_pm, r0, kern.NCH * 12 * 4), rowAt(b.a_pl, r0, kern.NCH * 12 * 4), rowAt(b.a_ids, r0, kern.IDW * 4), nk, sp,
                rowAt(b.a_out, r0, 12 * 256 * 2), sg.rows, keys);
        }
        try kern.attnGate(k, b.a_out, b.pa, b.gated, b.xs_gated, R);
    }
    return outProj(e, b, b.gated, 3072, a.o, R);
}

fn pleBlock(e: *Engine, b: *Buffers, segs: []const Seg, p: PleW, R: i64) !void {
    const k = &e.k;
    try kern.pleEmbedBf16(k, b.ple_v, b.ple_emb, b.xs_ple, R);
    try kern.matmul(k, b.ple_emb, D, p.key, b.ple_keys, WIDE, false, R, b.kpart);
    try kern.matmul(k, b.ple_emb, D, p.value, b.ple_vals, D, false, R, b.kpart);
    try kern.pleGate(k, b.ple_keys, b.ple_vals, b.h, p.norm_key, p.norm_query, b.ple_gated, b.ple_pss, R);
    for (segs) |sg| // the causal conv over each stream's own tail
        try kern.pleConv(k, rowAt(b.ple_gated, sg.row0, WIDE * 2), rowAt(b.ple_pss, sg.row0, 4 * 4), p.norm_conv, sg.s.ple_tail, p.conv, rowAt(b.h, sg.row0, WIDE * 2),
            rowAt(b.ple_nrow, sg.row0, WIDE * 2), sg.rows);
}

fn moeBlock(e: *Engine, b: *Buffers, m: MoEW, R: i64) !u64 {
    const k = &e.k;
    try kern.router(k, b.mixed, m.router, b.m_logits, R, D);
    try kern.topkRows(k, b.m_logits, b.m_pick, b.m_wts, R);
    const pairs = R * SLOTS;
    try kern.expertsPlan(k, b.m_pick, pairs, b.plan);
    const items = kern.maxItems(pairs, 512, b.plan.tile);
    const pf = b.moe_prefill;
    try kern.expertsCall(k, pf, 64, 2, b.mixed, D, SLOTS, m.up, 40, 10, b.plan, b.m_act, kern.LOW, if (pf) items else items * 10);
    try kern.matmul(k, b.mixed, D, m.sgu, b.sgu, 640, false, R, b.kpart);
    const shared_act = b.m_act + 10 * kern.LOW * 2;
    try kern.swiglu(k, b.sgu, shared_act, 640, SLOTS * kern.LOW, R);
    try kern.expertsCall(k, pf, 64, if (pf) 3 else 0, b.m_act, kern.LOW, 0, m.down, 5, 80, b.plan, b.m_y, D, if (pf) items else items * 80);
    const ysz: u64 = if (pf) 2 else 4;
    try kern.matmul(k, shared_act, SLOTS * kern.LOW, m.sdown, b.m_y + @as(u64, @intCast(10 * D)) * ysz, SLOTS * D, !pf, R, b.kpart);
    try kern.moePartial(k, b.m_y, !pf, b.m_wts, b.part_moe, R);
    try allGather(e, b.part_moe, b.g_moe, @intCast(R * D));
    return b.g_moe;
}

fn layerForward(e: *Engine, b: *Buffers, segs: []const Seg, l: LayerW, R: i64, pending_in: ?Pending, mtp: bool) !Pending {
    var pending = pending_in;
    if (l.ple) |p| {
        if (pending) |pd| {
            try kern.hcWriteback(&e.k, b.h, b.pss, pd.g, pd.inj, R, 3);
            pending = null;
        }
        try pleBlock(e, b, segs, p, R);
    }
    try hcBlock(e, b, l.attn_hc, R, pending, b.inj_a);
    const g = if (l.linear) try gdnBlock(e, b, segs, l, R) else try attnBlock(e, b, segs, l, R, mtp);
    try hcBlock(e, b, l.mlp_hc, R, .{ .g = g, .inj = b.inj_a }, b.inj_m);
    return .{ .g = try moeBlock(e, b, l.moe, R), .inj = b.inj_m };
}

/// forward.finish without the head: the last write-back into the streams and the mixer's read-out (a prompt: its
/// last row only, into row 0).
fn finish(e: *Engine, b: *Buffers, mixer: HCW, R: i64, pending: Pending) !i64 {
    const k = &e.k;
    try k.copy(b.streams, b.h, @intCast(R * WIDE * 2));
    try kern.hcWriteback(k, b.streams, b.pss, pending.g, pending.inj, R, 3);
    if (b.prefill) {
        const last: u64 = @intCast(R - 1);
        try k.copy(b.pss, b.pss + last * 160, 160);
        try readout(e, b, mixer, b.streams + last * WIDE * 2, 1, null, false);
        return 1;
    }
    try readout(e, b, mixer, b.streams, R, null, false);
    return R;
}

/// Each row's greedy token across both ranks' shards: our candidate kernel and an all-gather (rows <= 16).
fn candidates(e: *Engine, b: *Buffers, logits: u64, cols: i64, id_map: u64, offset: i64, rows: i64) !void {
    try kern.rowsTop(&e.k, logits, cols, id_map, offset, b.cand, rows);
    try e.ctx.nccl.check(e.ctx.nccl.api.ncclAllGather(b.cand, b.gath, @intCast(4 * rows), .i32, e.ctx.comm, e.k.stream.handle), "candidates");
}

const Pick = struct { tok: i64, p: f64 };

fn readPicks(e: *Engine, b: *Buffers, rows: i64, out: []Pick) !void {
    const d = e.ctx.d;
    const n: usize = @intCast(2 * rows * 4);
    const w: []i32 = @alignCast(std.mem.bytesAsSlice(i32, e.host.bytes[0 .. n * 4]));
    try d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(w.ptr), b.gath, n * 4, e.k.stream.handle), "candidates");
    try e.k.stream.synchronize();
    const R: usize = @intCast(rows);
    for (0..R) |r| {
        const a = w[4 * r ..][0..4];
        const c = w[4 * (R + r) ..][0..4];
        const v0: f32 = @bitCast(a[1]);
        const v1: f32 = @bitCast(c[1]);
        const second = v1 > v0 or (v1 == v0 and c[0] < a[0]);
        const l0: f64 = @as(f32, @bitCast(a[2]));
        const l1: f64 = @as(f32, @bitCast(c[2]));
        const top = @max(l0, l1);
        const total = top + @log(@exp(l0 - top) + @exp(l1 - top));
        const v: f64 = if (second) v1 else v0;
        out[r] = .{ .tok = if (second) c[0] else a[0], .p = @exp(v - total) };
    }
}

/// stage: the window's token ids and each row's n-gram table rows into the buffers (host gathers, then uploads).
fn stage(e: *Engine, b: *Buffers, s: *const Seq, tokens: []const i64) !void {
    const t0 = std.Io.Timestamp.now(e.io, .awake);
    defer e.stage_ms += @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(e.io, .awake)).toNanoseconds())) / 1e6;
    const n = tokens.len;
    for (tokens, 0..) |t, i| e.idbuf[i] = @intCast(t);
    try e.k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..n]));
    const t1 = std.Io.Timestamp.now(e.io, .awake);
    const heads = e.ng.heads;
    try e.ng.ids(&s.hist, tokens, e.rowbuf[0 .. n * heads]);
    const t2 = std.Io.Timestamp.now(e.io, .awake);
    try e.ng.gather(e.rowbuf[0 .. n * heads], e.valbuf[0 .. n * heads * e.ng.width]);
    const t3 = std.Io.Timestamp.now(e.io, .awake);
    e.split[0] += @as(f64, @floatFromInt(t0.durationTo(t1).toNanoseconds())) / 1e6;
    e.split[1] += @as(f64, @floatFromInt(t1.durationTo(t2).toNanoseconds())) / 1e6;
    e.split[2] += @as(f64, @floatFromInt(t2.durationTo(t3).toNanoseconds())) / 1e6;
    // pageable sources: the async copy returns once they are staged, so the buffers can be reused at once
    try e.k.upload(b.ple_v, std.mem.sliceAsBytes(e.valbuf[0 .. n * heads * e.ng.width]));
}

/// An image prompt's features over its placeholder rows in this piece (rows s.pos .. s.pos + rows of the prompt).
fn splice(e: *Engine, b: *Buffers, sg: Seg) !void {
    const im = &(sg.s.img orelse return);
    const fb = im.feats orelse return;
    var pairs: std.ArrayList(i32) = .empty;
    defer pairs.deinit(e.gpa);
    for (im.rows, 0..) |p, i| {
        const at: i64 = p;
        if (at >= sg.s.pos and at < sg.s.pos + sg.rows) try pairs.appendSlice(e.gpa, &.{ @intCast(sg.row0 + at - sg.s.pos), @intCast(i) });
    }
    if (pairs.items.len == 0) return;
    var dp = try cuda.DeviceBuffer.alloc(e.ctx.d, pairs.items.len * 4);
    defer dp.free();
    try e.k.upload(dp.ptr, std.mem.sliceAsBytes(pairs.items));
    const f = try e.k.ext(.vision, "fn_vis_splice");
    var a: cuda.Args = .{};
    a.add(b.h); a.add(fb.ptr); a.add(dp.ptr); a.add(@as(i32, @intCast(D))); a.add(@as(i32, 4));
    try e.k.go(f, .{ .x = @intCast(pairs.items.len / 2) }, 256, 0, &a);
    try e.k.stream.synchronize(); // the pairs' buffer is freed on return
}

/// stage for a shared round: every part's ids, each part's n-gram rows from its own history, then one upload each.
fn stageParts(e: *Engine, b: *Buffers, parts: []const Part, segs: []const Seg) !void {
    const heads = e.ng.heads;
    var n: usize = 0;
    for (parts, segs) |p, sg| {
        var toks: [16]i64 = undefined;
        for (p.ids, 0..) |t, i| {
            toks[i] = t;
            e.idbuf[n + i] = @intCast(t);
        }
        const r0: usize = @intCast(sg.row0);
        try e.ng.ids(&p.s.hist, toks[0..p.ids.len], e.rowbuf[r0 * heads .. (r0 + p.ids.len) * heads]);
        n += p.ids.len;
    }
    try e.k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..n]));
    try e.ng.gather(e.rowbuf[0 .. n * heads], e.valbuf[0 .. n * heads * e.ng.width]);
    try e.k.upload(b.ple_v, std.mem.sliceAsBytes(e.valbuf[0 .. n * heads * e.ng.width]));
}

/// gdn_multi.cu's table entries (its ChainSeg and ReplaySeg, 80 and 64 bytes).
pub const ChainSeg = extern struct { p: u64, cs: u64, state_in: u64, state_out: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, rows: i64, row0: i64 };
pub const ReplaySeg = extern struct { state_in: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, state_out: u64, rows: i64, pad: i64 = 0 };

/// A shared round's chains into ``b.gdn_tab``: each DeltaNet layer's streams (their projection rows, conv windows,
/// states at their parities, replay scratch), one upload for the whole forward.
fn chainTable(e: *Engine, b: *Buffers, segs: []const Seg) !void {
    var tab: [LIN * max_parts]ChainSeg = undefined;
    for (0..LIN) |li| {
        const proj = b.proj + @as(u64, @intCast(@as(i64, @intCast(li)) * b.rows * PROJ_W * 2));
        for (segs, 0..) |sg, j| {
            const s = sg.s;
            const cur = s.cur[li];
            tab[li * max_parts + j] = .{ .p = rowAt(proj, sg.row0, PROJ_W * 2), .cs = s.conv + li * 3 * CONV_DIM * 2, .state_in = s.recAt(cur, li),
                .state_out = s.recAt(1 - cur, li), .k_save = s.sc_k[li], .v_save = s.sc_v[li], .g_save = s.sc_g[li], .b_save = s.sc_b[li],
                .rows = sg.rows, .row0 = sg.row0 };
        }
    }
    try e.k.upload(b.gdn_tab, std.mem.sliceAsBytes(&tab));
}

/// A shared forward's attention tables (attn_multi.Step): every row's position and stream, each stream's first
/// position and rows, each attention layer's cache pointers [6][streams] (the MTP head's: its own caches, slot ATT).
/// Null when a stream has an image (its own rotary table) or a stream's keys reach the sparse selection: those
/// forwards attend stream by stream, as do kernel sets packed before the shared kernels.
fn attnTable(e: *Engine, b: *Buffers, segs: []const Seg, R: i64, mtp: bool) !?kern.MultiTab {
    if (!e.multi_attn or segs.len < 2) return null;
    var most: i64 = 0;
    var keys: i64 = 0;
    for (segs) |sg| {
        if (sg.s.img != null) return null;
        const end = (if (mtp) sg.s.mtp_len else sg.s.pos) + sg.rows;
        if (end > kern.KEYS_MAX) return null;
        most = @max(most, sg.rows);
        keys = @max(keys, end);
    }
    const n = segs.len;
    const rows: usize = @intCast(R);
    var ints: [2 * max_round_rows + 2 * max_parts]i32 = undefined;
    for (segs, 0..) |sg, j| {
        const p0 = if (mtp) sg.s.mtp_len else sg.s.pos;
        for (0..@intCast(sg.rows)) |r| {
            ints[@as(usize, @intCast(sg.row0)) + r] = @intCast(p0 + @as(i64, @intCast(r)));
            ints[rows + @as(usize, @intCast(sg.row0)) + r] = @intCast(j);
        }
        ints[2 * rows + j] = @intCast(p0);
        ints[2 * rows + n + j] = @intCast(sg.rows);
    }
    try e.k.upload(b.attn_ints, std.mem.sliceAsBytes(ints[0 .. 2 * rows + 2 * n]));
    var ptrs: [(ATT + 1) * 6 * max_parts]u64 = undefined;
    const layers: []const usize = if (mtp) &.{ATT} else &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    for (layers) |ai| {
        const base = ai * 6 * max_parts;
        for (segs, 0..) |sg, j| {
            const s = sg.s;
            for ([_]u64{ s.kc_k[ai], s.kc_v[ai], s.kc_ks[ai], s.kc_vs[ai], s.ikc[ai], s.pooled[ai] }, 0..) |x, f| ptrs[base + f * n + j] = x;
        }
    }
    if (mtp) {
        try e.k.upload(b.attn_ptrs + ATT * 6 * max_parts * 8, std.mem.sliceAsBytes(ptrs[ATT * 6 * max_parts ..][0 .. 6 * n]));
    } else try e.k.upload(b.attn_ptrs, std.mem.sliceAsBytes(ptrs[0 .. ATT * 6 * max_parts]));
    return .{ .posr = b.attn_ints, .sid = b.attn_ints + rows * 4, .first = b.attn_ints + 2 * rows * 4, .counts = b.attn_ints + (2 * rows + n) * 4,
        .cp = b.attn_ptrs, .streams = @intCast(n), .rows = R, .most = most, .keys = keys };
}

fn mainForward(e: *Engine, b: *Buffers, segs: []const Seg, R: i64) !Pending {
    try kern.embed(&e.k, b.ids, e.embed, b.h, R, 4);
    for (segs) |sg| try splice(e, b, sg);
    var pending: ?Pending = null;
    for (e.layers) |l| pending = try layerForward(e, b, segs, l, R, pending, false);
    return pending.?;
}

/// The MTP head over ``n`` staged rows (ids in b.ids, input streams in b.mtp_in): its layer, mixer and draft head.
fn peek(e: *Engine, what: []const u8, at: u64, f32_: bool) void {
    if (!e.k.sync_each) return;
    e.k.stream.synchronize() catch return;
    var raw: [8]u32 = undefined;
    _ = e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(&raw), at, 32);
    var v: [8]f32 = undefined;
    for (0..8) |i| v[i] = if (f32_) @bitCast(raw[i]) else @bitCast(raw[i / 2] << @intCast(16 * (1 - i % 2)) & 0xffff0000);
    std.debug.print("peek {s}: {any}\n", .{ what, v });
}

fn mtpCompute(e: *Engine, b: *Buffers, s: *Seq, n: i64) !void {
    return mtpComputeSegs(e, b, &.{.{ .s = s, .rows = n }}, n);
}

/// The draft head's logits rows (a stream's row k of a batched level at ``k * head_stride`` elements, 16-aligned).
fn headStride(e: *const Engine) i64 {
    return std.mem.alignForward(i64, e.dh.n, 8);
}

/// mtpCompute over several streams' rows (a batched draft level): the MTP layer over every row, each stream's
/// attention on its own MTP cache, and the draft head on each stream's last row into logits row k.
fn mtpComputeSegs(e: *Engine, b: *Buffers, segs: []const Seg, n: i64) !void {
    const k = &e.k;
    const m = e.mtp;
    defer if (k.sync_each and b.prefill) {
        peek(e, "mtp_e", b.mtp_e, false);
        peek(e, "mtp_eo", b.mtp_eo, false);
        peek(e, "mtp_in", b.mtp_in, false);
        peek(e, "mtp_hs", b.mtp_hs, false);
        peek(e, "streams", b.streams, false);
        peek(e, "mixed", b.mixed, false);
        peek(e, "m_y", b.m_y, !b.moe_prefill);
        peek(e, "g_moe", b.g_moe, true);
        peek(e, "logits", b.logits, false);
    };
    try kern.embed(k, b.ids, e.embed, b.mtp_e, n, 1);
    try kern.rmsnorm(k, b.mtp_e, m.norm_e, b.mixed, b.xs_mixed, D, n, D);
    try kern.matmul(k, b.mixed, D, m.fc_e, b.mtp_eo, D, false, n, b.kpart);
    try kern.rmsnorm(k, b.mtp_in, m.norm_h, b.mtp_hn, b.mtp_xh, WIDE, n, WIDE);
    try kern.matmul(k, b.mtp_hn, D, m.fc_h, b.mtp_hs, D, false, n * 4, b.kpart);
    try kern.addStreams(k, b.mtp_eo, b.mtp_hs, b.h, n);
    if (!b.prefill) b.attn = try attnTable(e, b, segs, n, true);
    defer b.attn = null;
    const pending = try layerForward(e, b, segs, m.layer, n, null, true);
    _ = try finish(e, b, m.mixer, n, pending);
    if (b.prefill and segs.len > 1) return; // several prompts absorbed: no draft from a prompt pass
    if (b.prefill) {
        try kern.qmmPrefill(k, b.mixed, e.dh, b.logits);
    } else for (segs, 0..) |sg, j| {
        const last: u64 = @intCast(sg.row0 + sg.rows - 1);
        try kern.qmmDecode(k, b.mixed + last * D * 2, b.xs_mixed + last * (D / 32) * 4, e.dh, b.logits + j * @as(u64, @intCast(headStride(e))) * 2, 0);
    }
    if (k.sync_each and !b.prefill) {
        peek(e, "w.mtp_hs", b.mtp_hs, false);
        peek(e, "w.streams", b.streams, false);
        peek(e, "w.mixed", b.mixed, false);
        peek(e, "w.m_y", b.m_y, true);
        peek(e, "w.g_moe", b.g_moe, true);
        peek(e, "w.logits", b.logits, false);
    }
    if (segs.len == 1) try candidates(e, b, b.logits, e.dh.n, e.draft_ids.ptr, 0, 1) else {
        const rows: i64 = @intCast(segs.len);
        try kern.rowsTopStrided(k, b.logits, e.dh.n, headStride(e), e.draft_ids.ptr, 0, b.cand, rows);
        try e.ctx.nccl.check(e.ctx.nccl.api.ncclAllGather(b.cand, b.gath, @intCast(4 * rows), .i32, e.ctx.comm, k.stream.handle), "candidates");
    }
}

fn setPos(e: *Engine, s: *Seq, pos: i64) !void {
    s.pos = pos;
    try e.k.memset32(s.pos_dev, @bitCast(@as(i32, @intCast(pos))), 1);
}

fn setMtpLen(e: *Engine, s: *Seq, n: i64) !void {
    s.mtp_len = n;
    try e.k.memset32(s.mtp_pos, @bitCast(@as(i32, @intCast(n))), 1);
}

/// The whole prompt in chunks of 2048 rows, the MTP head absorbing each; returns the greedy first token.
/// ``start`` > 0: resume from the kept point there (the sequence's snap); the prompt's rows from it on.
pub fn prefill(e: *Engine, s: *Seq, prompt_u: []const u32, start: usize) !u32 {
    const prompt = try e.gpa.alloc(i64, prompt_u.len);
    defer e.gpa.free(prompt);
    for (prompt_u, prompt) |t, *x| x.* = t;
    if (prompt.len == 0) return error.EmptyPrompt;
    if (s.img) |im| if (im.length != @as(i64, @intCast(prompt.len))) return error.ImageLengthMismatch;
    if (@as(i64, @intCast(prompt.len)) + @as(i64, e.opts.depth) + 1 > e.capacity) return error.PromptTooLong;
    const b = &e.pbuf;
    var at: usize = start;
    if (start > 0) {
        if (start >= prompt.len or s.snap.at != @as(i64, @intCast(start))) return error.NoKeptState;
        try restore(e, b, s, prompt[start]);
    }
    s.snap.at = 0; // the kept point is this prompt's (or none)
    const cut = s.cut_at;
    s.cut_at = null;
    if (cut != null and s.snap.buf == null) s.snap.buf = try cuda.DeviceBuffer.alloc(e.ctx.d, Snap.REC + Snap.CONV + Snap.PLE + Snap.TAIL);
    var first: Pick = undefined;
    while (at < prompt.len) {
        const end = @min(at + @as(usize, @intCast(PREFILL_ROWS)), prompt.len);
        const R: i64 = @intCast(end - at);
        const final = end == prompt.len;
        const row: i64 = if (cut) |kp| (if (kp > @as(i64, @intCast(at)) and kp <= @as(i64, @intCast(end))) kp - @as(i64, @intCast(at)) else 0) else 0;

        try room(e, s, R + 1);
        try stage(e, b, s, prompt[at..end]);
        const pending = try mainForward(e, b, &.{.{ .s = s, .rows = R, .cut = row }}, R);
        if (final) { // the head on the last row, its candidates before the MTP head reuses the logits buffer
            _ = try finish(e, b, e.mixer, R, pending);
            try kern.matmul(&e.k, b.mixed, D, e.head, b.logits, HEAD_N, false, 1, b.kpart);
            _ = try applyMasks(e, b.logits, &.{.{ .s = s, .row0 = 0, .rows = 1 }});
            try candidates(e, b, b.logits, HEAD_N, 0, e.vocab_offset, 1);
            var picks: [1]Pick = undefined;
            try readPicks(e, b, 1, &picks);
            first = picks[0];
            if (s.sampling) |rule| { // the first token draws at the prompt's length
                var tok: [1]u32 = .{@intCast(first.tok)};
                try drawRows(e, b, b.logits, 1, &.{.{ .row = 0, .s = rule, .position = prompt.len }}, &tok);
                first.tok = tok[0];
            }
            try e.k.copy(s.last_streams, b.streams + @as(u64, @intCast(R - 1)) * WIDE * 2, WIDE * 2);
        } else {
            try e.k.copy(b.streams, b.h, @intCast(R * WIDE * 2));
            try kern.hcWriteback(&e.k, b.streams, b.pss, pending.g, pending.inj, R, 3);
        }
        if (row > 0) { // the kept point: its last row's streams, MTP length, n-gram windows, before the head and the commit
            const sn = &s.snap;
            try e.k.copy(sn.tail(), b.streams + @as(u64, @intCast(row - 1)) * WIDE * 2, WIDE * 2);
            sn.mtp_len = s.mtp_len + row - 1;
            sn.hist = s.hist;
            sn.hist.advance(prompt[at..][0..@intCast(row)]);
            try e.k.copy(sn.ple(), s.ple_tail, Snap.PLE);
            try kern.shiftWindows(&e.k, sn.ple(), b.ple_nrow, row, 9 * WIDE, b.rows * WIDE, WIDE, 1, WIDE, 9);
            sn.at = cut.?;
        }
        // the MTP head takes the rows with their next tokens (a final piece: every row but the last)
        const nn: i64 = if (final) R - 1 else R;
        if (nn > 0) {
            const nxt = prompt[at + 1 .. at + 1 + @as(usize, @intCast(nn))];
            for (nxt, 0..) |t, i| e.idbuf[i] = @intCast(t);
            try e.k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..nxt.len]));
            try e.k.copy(b.mtp_in, b.streams, @intCast(nn * WIDE * 2));
            try mtpCompute(e, b, s, nn);
            try setMtpLen(e, s, s.mtp_len + nn);
        }
        // commit: the n-gram tail keeps the piece's rows, positions move on
        try kern.shiftWindows(&e.k, s.ple_tail, b.ple_nrow, R, 9 * WIDE, b.rows * WIDE, WIDE, 1, WIDE, 9);
        s.hist.advance(prompt[at..end]);
        try setPos(e, s, s.pos + R);
        at = end;
    }
    s.fresh = true;
    s.mtp_drafted = 0;
    try e.k.stream.synchronize();
    if (s.img) |*im| if (im.feats) |*f| { // the features end with the prompt; the rotary table stays for decode
        f.free();
        im.feats = null;
    };
    return @intCast(first.tok);
}

/// A prompt in ``prefillMany``: its sequence, its tokens and where it resumes (a kept point, as ``prefill``).
pub const Prompt = struct { s: *Seq, ids: []const u32, start: usize = 0 };

/// Whether ``prompts`` fit one prompt pass together: text prompts whose rows past their kept points fill one chunk.
pub fn prefillsTogether(e: *const Engine, prompts: []const Prompt) bool {
    if (prompts.len < 2 or prompts.len > max_parts) return false;
    var rows: usize = 0;
    for (prompts) |p| {
        if (p.s.img != null or p.ids.len == 0 or p.start >= p.ids.len) return false;
        rows += p.ids.len - p.start;
    }
    return rows <= @as(usize, @intCast(e.pbuf.rows));
}

/// ``prefill`` for several short prompts in one pass (multi.py's prompt passes): every prompt's rows through each
/// layer's shared weights once (a prompt alone reads most of the experts), each sequence's DeltaNet, conv, attention
/// and n-gram rows on its own state; ``firsts`` gets each prompt's first token. The prompts must fit together
/// (``prefillsTogether``); a sequence's kept point (``cut_at``) and kept state (``start``) as ``prefill``.
pub fn prefillMany(e: *Engine, prompts: []const Prompt, firsts: []u32) !void {
    if (!prefillsTogether(e, prompts)) return error.NotTogether;
    const b = &e.pbuf;
    const k = &e.k;
    var segs: [max_parts]Seg = undefined;
    var cuts: [max_parts]?i64 = undefined;
    var R: i64 = 0;
    for (prompts, 0..) |p, j| {
        const s = p.s;
        if (@as(i64, @intCast(p.ids.len)) + @as(i64, e.opts.depth) + 1 > e.capacity) return error.PromptTooLong;
        if (p.start > 0 and s.snap.at != @as(i64, @intCast(p.start))) return error.NoKeptState;
        const rows: i64 = @intCast(p.ids.len - p.start);
        cuts[j] = s.cut_at;
        const cut_row: i64 = if (s.cut_at) |c| (if (c > p.start and c <= p.ids.len) c - @as(i64, @intCast(p.start)) else 0) else 0;
        segs[j] = .{ .s = s, .row0 = R, .rows = rows, .cut = cut_row };
        R += rows;
    }
    const n = prompts.len;
    for (prompts, 0..) |p, j| {
        const s = p.s;
        if (p.start > 0) try restore(e, b, s, p.ids[p.start]);
        s.snap.at = 0; // the kept point is this prompt's (or none)
        s.cut_at = null;
        if (cuts[j] != null and s.snap.buf == null) s.snap.buf = try cuda.DeviceBuffer.alloc(e.ctx.d, Snap.REC + Snap.CONV + Snap.PLE + Snap.TAIL);
        try room(e, s, segs[j].rows + 1);
    }
    // the rows' ids and n-gram rows (each from its own history), each row's conv taps and sequence, the layers' windows
    const heads = e.ng.heads;
    var toks: [PREFILL_ROWS]i64 = undefined;
    const tk = toks[0..@intCast(R)];
    var wins: [4 * PREFILL_ROWS]i32 = undefined;
    var sids: [PREFILL_ROWS]i32 = undefined;
    for (prompts, segs[0..n], 0..) |p, sg, j| {
        const r0: usize = @intCast(sg.row0);
        for (p.ids[p.start..], 0..) |t_, i| {
            tk[r0 + i] = t_;
            e.idbuf[r0 + i] = @intCast(t_);
            sids[r0 + i] = @intCast(j);
            for (0..4) |tap| {
                const src = i + tap;
                wins[(r0 + i) * 4 + tap] = @intCast(if (src < 3) src else r0 + src);
            }
        }
        try e.ng.ids(&p.s.hist, tk[r0..][0..@intCast(sg.rows)], e.rowbuf[r0 * heads .. (r0 + @as(usize, @intCast(sg.rows))) * heads]);
    }
    const rows_u: usize = @intCast(R);
    try k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..rows_u]));
    try e.ng.gather(e.rowbuf[0 .. rows_u * heads], e.valbuf[0 .. rows_u * heads * e.ng.width]);
    try k.upload(b.ple_v, std.mem.sliceAsBytes(e.valbuf[0 .. rows_u * heads * e.ng.width]));
    try k.upload(b.mwindows, std.mem.sliceAsBytes(wins[0 .. 4 * rows_u]));
    try k.upload(b.msid, std.mem.sliceAsBytes(sids[0..rows_u]));
    var conv: [LIN * max_parts]u64 = undefined;
    for (0..LIN) |li| for (segs[0..n], 0..) |sg, j| {
        conv[li * max_parts + j] = sg.s.conv + li * 3 * CONV_DIM * 2;
    };
    try k.upload(b.conv_tab, std.mem.sliceAsBytes(&conv));
    b.multi_prompts = true;
    defer b.multi_prompts = false;
    const pending = try mainForward(e, b, segs[0..n], R);
    // every row's streams; each prompt's last row (streams and sums) gathered to rows 0 .. n - 1 for the head
    try k.copy(b.streams, b.h, @intCast(R * WIDE * 2));
    try kern.hcWriteback(k, b.streams, b.pss, pending.g, pending.inj, R, 3);
    for (segs[0..n], 0..) |sg, j| {
        const last: u64 = @intCast(sg.row0 + sg.rows - 1);
        try k.copy(b.pss + j * 160, b.pss + last * 160, 160);
        try k.copy(b.mtp_hn + j * WIDE * 2, b.streams + last * WIDE * 2, WIDE * 2);
        try k.copy(sg.s.last_streams, b.streams + last * WIDE * 2, WIDE * 2);
    }
    const nr: i64 = @intCast(n);
    try readout(e, b, e.mixer, b.mtp_hn, nr, null, false);
    try kern.matmul(k, b.mixed, D, e.head, b.logits, HEAD_N, false, nr, b.kpart);
    var heads_segs: [max_parts]Seg = undefined;
    for (segs[0..n], 0..) |sg, j| heads_segs[j] = .{ .s = sg.s, .row0 = @intCast(j), .rows = 1 };
    _ = try applyMasks(e, b.logits, heads_segs[0..n]);
    try candidates(e, b, b.logits, HEAD_N, 0, e.vocab_offset, nr);
    var picks: [max_parts]Pick = undefined;
    try readPicks(e, b, nr, picks[0..n]);
    for (picks[0..n], 0..) |pk, j| firsts[j] = @intCast(pk.tok);
    var want: [max_parts]Want = undefined;
    var nw: usize = 0;
    for (prompts, 0..) |p, j| if (p.s.sampling) |rule| { // the first token draws at the prompt's length
        want[nw] = .{ .row = j, .s = rule, .position = p.ids.len };
        nw += 1;
    };
    try drawRows(e, b, b.logits, nr, want[0..nw], firsts[0..n]);
    // kept points: the state before each prompt's point (its last row's streams, MTP length, n-gram windows)
    for (prompts, segs[0..n]) |p, sg| if (sg.cut > 0) {
        const s = sg.s;
        const sn = &s.snap;
        try k.copy(sn.tail(), b.streams + @as(u64, @intCast(sg.row0 + sg.cut - 1)) * WIDE * 2, WIDE * 2);
        sn.mtp_len = s.mtp_len + sg.cut - 1;
        sn.hist = s.hist;
        sn.hist.advance(tk[@intCast(sg.row0)..][0..@intCast(sg.cut)]);
        try k.copy(sn.ple(), s.ple_tail, Snap.PLE);
        try kern.shiftWindows(k, sn.ple(), rowAt(b.ple_nrow, sg.row0, WIDE * 2), sg.cut, 9 * WIDE, b.rows * WIDE, WIDE, 1, WIDE, 9);
        sn.at = @intCast(p.start + @as(usize, @intCast(sg.cut)));
    };
    // the MTP head takes each prompt's rows with their next tokens (every row but its last)
    var msegs: [max_parts]Seg = undefined;
    var nm: usize = 0;
    var M: i64 = 0;
    for (prompts, segs[0..n]) |p, sg| {
        const nn = sg.rows - 1;
        if (nn <= 0) continue;
        for (p.ids[p.start + 1 ..][0..@intCast(nn)], 0..) |t_, i| e.idbuf[@as(usize, @intCast(M)) + i] = @intCast(t_);
        try k.copy(rowAt(b.mtp_in, M, WIDE * 2), rowAt(b.streams, sg.row0, WIDE * 2), @intCast(nn * WIDE * 2));
        msegs[nm] = .{ .s = sg.s, .row0 = M, .rows = nn };
        nm += 1;
        M += nn;
    }
    if (nm > 0) {
        try k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..@intCast(M)]));
        try mtpComputeSegs(e, b, msegs[0..nm], M);
        for (msegs[0..nm]) |sg| try setMtpLen(e, sg.s, sg.s.mtp_len + sg.rows);
    }
    // commit: each prompt's n-gram tail keeps its rows, positions move on
    for (segs[0..n]) |sg| {
        const s = sg.s;
        try kern.shiftWindows(k, s.ple_tail, rowAt(b.ple_nrow, sg.row0, WIDE * 2), sg.rows, 9 * WIDE, b.rows * WIDE, WIDE, 1, WIDE, 9);
        s.hist.advance(tk[@intCast(sg.row0)..][0..@intCast(sg.rows)]);
        try setPos(e, s, s.pos + sg.rows);
        s.fresh = true;
        s.mtp_drafted = 0;
    }
    try k.stream.synchronize();
}

/// One window over ``ids`` (the pending token, then drafts): out[r] is the greedy token after row r.
/// A stream's grammar rows for its next forward: each row of its window the grammar constrains (in window order)
/// and that row's allowed bits over this rank's columns, MASK_WORDS words a row.
pub const Mask = struct { rows: []const u32, bits: []const u32 };
pub const MASK_WORDS: usize = @intCast(@divExact(HEAD_N, 32));

/// Every segment's grammar rows to -inf in ``logits`` (both ranks, before the picks); whether any was masked.
fn applyMasks(e: *Engine, logits: u64, segs: []const Seg) !bool {
    var n: usize = 0;
    var rows: [max_round_rows]i32 = undefined;
    for (segs) |sg| if (sg.s.mask) |m| {
        for (m.rows, 0..) |r, i| {
            if (n == rows.len) return error.WindowTooWide;
            rows[n] = @intCast(sg.row0 + @as(i64, r));
            @memcpy(e.gbits_host[n * MASK_WORDS ..][0..MASK_WORDS], m.bits[i * MASK_WORDS ..][0..MASK_WORDS]);
            n += 1;
        }
        sg.s.mask = null;
    };
    if (n == 0) return false;
    const k = &e.k;
    try k.upload(e.gbits.ptr, std.mem.sliceAsBytes(e.gbits_host[0 .. n * MASK_WORDS]));
    try k.upload(e.grows.ptr, std.mem.sliceAsBytes(rows[0..n]));
    const f = try k.ext(.sample, "fn_grammar_rows");
    var a: cuda.Args = .{};
    a.add(logits); a.add(@as(i32, @intCast(HEAD_N))); a.add(@as(i32, @intCast(HEAD_N)));
    a.add(e.gbits.ptr); a.add(@as(i32, @intCast(MASK_WORDS))); a.add(e.grows.ptr);
    try k.go(f, .{ .x = @intCast(n) }, 256, 0, &a);
    return true;
}

fn rows_for(opts: Options) i64 {
    return @max(@as(i64, opts.max_rows), 16);
}

/// A sampled row of a window: its row in the logits, its stream's rule and the position its draw is keyed at.
const Want = struct { row: usize, s: lanes.Sampling, position: u64 };

/// The sampled rows' tokens into ``out`` (by row): each rank's 64 candidates of every row and its log-sum-exps
/// (fn_cand_topk at each row's temperature), gathered, drawn on the host (draw.zig); a rule past the candidates draws
/// over the row's gathered logits. Both ranks run it with the same rows and rules, so their gathers pair up.
fn drawRows(e: *Engine, b: *Buffers, logits: u64, rows: i64, want: []const Want, out: []u32) !void {
    if (want.len == 0) return;
    const k = &e.k;
    var inv: [max_round_rows]f32 = @splat(1);
    for (want) |w| inv[w.row] = @floatCast(1.0 / @max(w.s.temperature, 1e-6));
    try k.upload(b.invt, std.mem.sliceAsBytes(inv[0..@intCast(rows)]));
    {
        const f = try k.ext(.sample, "fn_cand_topk");
        var a: cuda.Args = .{};
        a.add(logits); a.add(@as(i32, @intCast(HEAD_N))); a.add(@as(i32, @intCast(HEAD_N))); a.add(@as(i32, @intCast(e.vocab_offset)));
        a.add(b.invt); a.add(b.tk);
        try k.go(f, .{ .x = @intCast(rows) }, 256, 0, &a);
    }
    const n: usize = @intCast(rows * draw_mod.W);
    try e.ctx.nccl.check(e.ctx.nccl.api.ncclAllGather(b.tk, b.gtk, n, .i32, e.ctx.comm, k.stream.handle), "candidates");
    try k.stream.synchronize();
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(e.cand_host.ptr), b.gtk, 2 * n * 4), "candidates");
    for (want) |w| {
        const blocks = [2][]const i32{ e.cand_host[w.row * draw_mod.W ..][0..draw_mod.W], e.cand_host[n + w.row * draw_mod.W ..][0..draw_mod.W] };
        switch (try draw_mod.draw(e.gpa, blocks, w.s, w.position)) {
            .token => |t| out[w.row] = t,
            .full => out[w.row] = try drawFull(e, logits + @as(u64, @intCast(w.row)) * HEAD_N * 2, w.s, w.position),
        }
    }
}

/// A row's draw over the whole vocabulary: both ranks' logits gathered (rank r's columns are ids r * HEAD_N + c).
fn drawFull(e: *Engine, row: u64, s: lanes.Sampling, position: u64) !u32 {
    try e.ctx.nccl.check(e.ctx.nccl.api.ncclAllGather(row, e.full.ptr, @intCast(HEAD_N), .bf16, e.ctx.comm, e.k.stream.handle), "full row");
    try e.k.stream.synchronize();
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(e.full_host.ptr), e.full.ptr, e.full_host.len * 2), "full row");
    const values = try e.gpa.alloc(f64, e.full_host.len);
    defer e.gpa.free(values);
    for (values, e.full_host) |*v, h| v.* = @as(f32, @bitCast(@as(u32, h) << 16));
    return draw_mod.drawFull(e.gpa, values, s, position);
}

/// The live state the kept point's (snap): its DeltaNet states and windows, n-gram tail and history, positions; the
/// MTP head then absorbs the point's last row with ``next`` (the resumed prompt's token there).
fn restore(e: *Engine, b: *Buffers, s: *Seq, next: i64) !void {
    const sn = &s.snap;
    const k = &e.k;
    detach(e, s);
    for (0..LIN) |li| {
        try k.copy(s.recAt(0, li), sn.rec(li), @intCast(Seq.REC_LAYER));
        s.cur[li] = 0;
    }
    try k.copy(s.conv, sn.conv(0), Snap.CONV);
    try k.copy(s.ple_tail, sn.ple(), Snap.PLE);
    s.hist = sn.hist;
    s.mtp_drafted = 0;
    try setPos(e, s, sn.at);
    try setMtpLen(e, s, sn.mtp_len);
    try room(e, s, 2);
    e.idbuf[0] = @intCast(next);
    try k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..1]));
    try k.copy(b.mtp_in, sn.tail(), WIDE * 2);
    try mtpCompute(e, b, s, 1);
    try setMtpLen(e, s, s.mtp_len + 1);
}

pub fn verify(e: *Engine, s: *Seq, ids: []const u32, out: []u32) !void {
    const R: i64 = @intCast(ids.len);
    if (R > e.buf.rows) return error.WindowTooWide;
    if (s.pos + R > e.capacity) return error.PromptTooLong;
    const b = &e.buf;
    var toks: [16]i64 = undefined;
    for (ids, 0..) |t, i| toks[i] = t;
    try room(e, s, R + 1);
    try stage(e, b, s, toks[0..ids.len]);
    var ev: [2]cuda.Event = undefined;
    if (e.timing) {
        ev[0] = try cuda.Event.init(e.ctx.d, true);
        ev[1] = try cuda.Event.init(e.ctx.d, true);
        try ev[0].record(e.k.stream);
    }
    try step(e, 0, s, R);
    if (e.timing) try ev[1].record(e.k.stream);
    defer if (e.timing) {
        e.gpu_ms += cuda.Event.elapsedMs(ev[0], ev[1]) catch 0;
        ev[0].deinit();
        ev[1].deinit();
    };
    if (try applyMasks(e, b.logits, &.{.{ .s = s, .row0 = 0, .rows = R }})) try candidates(e, b, b.logits, HEAD_N, 0, e.vocab_offset, R);
    var picks: [16]Pick = undefined;
    try readPicks(e, b, R, picks[0..ids.len]);
    for (out[0..ids.len], picks[0..ids.len]) |*o, p| o.* = @intCast(p.tok);
    if (s.sampling) |rule| { // row r draws the token at position pos + 1 + r
        var want: [16]Want = undefined;
        for (0..ids.len) |r| want[r] = .{ .row = r, .s = rule, .position = @intCast(s.pos + 1 + @as(i64, @intCast(r))) };
        try drawRows(e, b, b.logits, R, want[0..ids.len], out);
    }
    s.last_rows = R;
    s.last_row0 = 0;
    @memcpy(s.last_tokens[0..ids.len], toks[0..ids.len]);
}

/// A stream's window in a shared round: its sequence and its rows' tokens (the pending one, then its drafts).
pub const Part = struct { s: *Seq, ids: []const u32 };

/// One forward over several streams' windows (eager): shared layers over every row, each stream's attention,
/// DeltaNet, conv and n-gram rows on its own state; ``out`` gets every row's greedy token, the parts in order.
pub fn verifyShared(e: *Engine, parts: []const Part, out: []u32) !void {
    if (parts.len == 0 or parts.len > max_parts) return error.TooManyParts;
    var segs: [max_parts]Seg = undefined;
    var R: i64 = 0;
    for (parts, 0..) |p, i| {
        const n: i64 = @intCast(p.ids.len);
        if (n == 0 or n > 16) return error.WindowTooWide;
        if (p.s.pos + n > e.capacity) return error.PromptTooLong;
        segs[i] = .{ .s = p.s, .row0 = R, .rows = n };
        R += n;
    }
    const b = &e.buf;
    if (R > b.rows) return error.WindowTooWide;
    for (segs[0..parts.len]) |sg| try room(e, sg.s, sg.rows + 1);
    try stageParts(e, b, parts, segs[0..parts.len]);
    if (e.multi_gdn) try chainTable(e, b, segs[0..parts.len]);
    b.chains_staged = e.multi_gdn;
    defer b.chains_staged = false;
    b.attn = try attnTable(e, b, segs[0..parts.len], R, false);
    defer b.attn = null;
    const pending = try mainForward(e, b, segs[0..parts.len], R);
    _ = try finish(e, b, e.mixer, R, pending);
    try kern.matmul(&e.k, b.mixed, D, e.head, b.logits, HEAD_N, false, R, b.kpart);
    _ = try applyMasks(e, b.logits, segs[0..parts.len]);
    try candidates(e, b, b.logits, HEAD_N, 0, e.vocab_offset, R);
    var picks: [max_parts * 16]Pick = undefined;
    try readPicks(e, b, R, picks[0..@intCast(R)]);
    for (out[0..@intCast(R)], picks[0..@intCast(R)]) |*o, p| o.* = @intCast(p.tok);
    var want: [max_parts * 16]Want = undefined;
    var nw: usize = 0;
    for (parts, segs[0..parts.len]) |p, sg| if (p.s.sampling) |rule| {
        for (0..@intCast(sg.rows)) |r| {
            want[nw] = .{ .row = @intCast(sg.row0 + @as(i64, @intCast(r))), .s = rule, .position = @intCast(p.s.pos + 1 + @as(i64, @intCast(r))) };
            nw += 1;
        }
    };
    try drawRows(e, b, b.logits, R, want[0..nw], out);
    for (parts, segs[0..parts.len]) |p, sg| {
        p.s.last_rows = sg.rows;
        p.s.last_row0 = sg.row0;
        for (p.ids, 0..) |t, i| p.s.last_tokens[i] = t;
    }
}

/// Streams a shared round packs at most.
pub const max_parts = 16;
/// A shared round's rows at most: every stream's widest window.
pub const max_round_rows = max_parts * 16;

/// forward.commit: keep the first ``kept`` of the last window's ``rows`` (DeltaNet replays, conv and n-gram windows).
pub fn keep(e: *Engine, s: *Seq, rows: u32, kept: u32) !void {
    const k = &e.k;
    const R: i64 = rows;
    const kp: i64 = kept;
    if (kp < 1 or kp > R) return error.BadKeep;
    for (0..LIN) |li| {
        const cur = s.cur[li];
        if (kp < R) try kern.gdnReplay(k, s.recAt(cur, li), s.sc_k[li], s.sc_v[li], s.sc_g[li], s.sc_b[li], kp, s.recAt(1 - cur, li));
        s.cur[li] = 1 - cur;
    }
    const b = &e.buf;
    // the stream's own rows of the window (row0: where a shared round put them)
    try kern.shiftWindows(k, s.conv, rowAt(b.proj, s.last_row0, PROJ_W * 2), kp, 3 * CONV_DIM, b.rows * PROJ_W, PROJ_W, LIN, CONV_DIM, 3);
    try kern.shiftWindows(k, s.ple_tail, rowAt(b.ple_nrow, s.last_row0, WIDE * 2), kp, 9 * WIDE, b.rows * WIDE, WIDE, 1, WIDE, 9);
    s.hist.advance(s.last_tokens[0..@intCast(kp)]);
    try setPos(e, s, s.pos + kp);
}

/// A stream's keep in ``keepMany``: its last window's rows and how many of them it keeps.
pub const Keep = struct { s: *Seq, rows: u32, kept: u32 };

/// ``keep`` for several streams: every stream's DeltaNet replays (all layers) in one launch, then each stream's
/// windows and positions; the states and windows each stream's own keep gives.
pub fn keepMany(e: *Engine, items: []const Keep) !void {
    if (items.len == 1 or !e.multi_gdn) {
        for (items) |it| try keep(e, it.s, it.rows, it.kept);
        return;
    }
    if (items.len == 0) return;
    if (items.len > max_parts) return error.TooManyParts;
    const k = &e.k;
    var tab: [LIN * max_parts]ReplaySeg = undefined;
    var any = false;
    for (items, 0..) |it, j| {
        if (it.kept < 1 or it.kept > it.rows) return error.BadKeep;
        const s = it.s;
        const replay: i64 = if (it.kept < it.rows) it.kept else 0;
        any = any or replay > 0;
        for (0..LIN) |li| {
            const cur = s.cur[li];
            tab[li * items.len + j] = .{ .state_in = s.recAt(cur, li), .k_save = s.sc_k[li], .v_save = s.sc_v[li], .g_save = s.sc_g[li], .b_save = s.sc_b[li],
                .state_out = s.recAt(1 - cur, li), .rows = replay };
        }
    }
    const b = &e.buf;
    if (any) {
        try k.upload(b.gdn_tab, std.mem.sliceAsBytes(tab[0 .. LIN * items.len]));
        try kern.gdnReplayMulti(k, b.gdn_tab, @intCast(items.len));
    }
    for (items) |it| {
        const s = it.s;
        for (0..LIN) |li| s.cur[li] = 1 - s.cur[li];
        const kp: i64 = it.kept;
        try kern.shiftWindows(k, s.conv, rowAt(b.proj, s.last_row0, PROJ_W * 2), kp, 3 * CONV_DIM, b.rows * PROJ_W, PROJ_W, LIN, CONV_DIM, 3);
        try kern.shiftWindows(k, s.ple_tail, rowAt(b.ple_nrow, s.last_row0, WIDE * 2), kp, 9 * WIDE, b.rows * WIDE, WIDE, 1, WIDE, 9);
        s.hist.advance(s.last_tokens[0..@intCast(kp)]);
        try setPos(e, s, s.pos + kp);
    }
}

/// decode.draft: the MTP head absorbs the kept rows (``follow``: the token after each), then chains ``depth`` drafts.
pub fn draft(e: *Engine, s: *Seq, follow: []const u32, depth: u32, out: []u32) !void {
    _ = try draftUpTo(e, s, follow, depth, 0, out);
}

/// One stream's request in a batched draft: the token after each kept row, how deep to draft, where the drafts and
/// the head's chance for each go; ``got`` is how many it drafted.
pub const DraftReq = struct { s: *Seq, follow: []const u32, depth: u32, out: []u32, probs: []f64, got: usize = 0 };

/// draftUpTo for several streams at once: each level one MTP forward over every stream still chaining (the absorb
/// over each stream's kept rows first), one read-back a level; the same drafts each would get alone.
pub fn draftBatch(e: *Engine, reqs: []DraftReq, confidence: f64) !void {
    if (reqs.len == 0) return;
    if (reqs.len > max_parts) return error.TooManyParts;
    const k = &e.k;
    const b = &e.mbuf;
    var segs: [max_parts]Seg = undefined;
    var R: i64 = 0;
    for (reqs, 0..) |*r, i| {
        const s = r.s;
        const n: i64 = @intCast(r.follow.len);
        if (s.mtp_drafted > 0) {
            try setMtpLen(e, s, s.mtp_len - s.mtp_drafted);
            s.mtp_drafted = 0;
        }
        try room(e, s, n + r.depth + 1);
        for (r.follow, 0..) |t, j| e.idbuf[@as(usize, @intCast(R)) + j] = @intCast(t);
        if (s.fresh) {
            try k.copy(rowAt(b.mtp_in, R, WIDE * 2), s.last_streams, WIDE * 2);
            s.fresh = false;
        } else try k.copy(rowAt(b.mtp_in, R, WIDE * 2), rowAt(e.buf.streams, s.last_row0, WIDE * 2), @intCast(n * WIDE * 2));
        segs[i] = .{ .s = s, .row0 = R, .rows = n };
        r.got = 0;
        R += n;
    }
    if (R > b.rows) return error.WindowTooWide;
    try k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..@intCast(R)]));
    try mtpComputeSegs(e, b, segs[0..reqs.len], R);
    for (reqs, segs[0..reqs.len]) |r, sg| try setMtpLen(e, r.s, r.s.mtp_len + sg.rows);
    // levels: who chains on, and the row of b.streams its next input comes from
    var live: [max_parts]usize = undefined;
    var from: [max_parts]i64 = undefined;
    var nlive: usize = reqs.len;
    for (0..reqs.len) |i| {
        live[i] = i;
        from[i] = segs[i].row0 + segs[i].rows - 1;
    }
    var j: u32 = 0;
    var picks: [max_parts]Pick = undefined;
    while (nlive > 0) : (j += 1) {
        try readPicks(e, b, @intCast(nlive), picks[0..nlive]);
        var next: usize = 0;
        var row: i64 = 0;
        for (live[0..nlive], 0..) |i, m| {
            const r = &reqs[i];
            if (j >= r.depth) continue;
            const p = picks[m];
            const low = confidence > 0 and p.p < confidence;
            if (low and j > 0) continue;
            r.out[r.got] = @intCast(p.tok);
            r.probs[r.got] = p.p;
            r.got += 1;
            if (low or j + 1 >= r.depth) continue;
            // chains on: its pick as the next row's id, its last row's streams as the next row's input
            e.idbuf[@intCast(row)] = @intCast(p.tok);
            try k.copy(rowAt(b.mtp_in, row, WIDE * 2), rowAt(b.streams, from[m], WIDE * 2), WIDE * 2);
            segs[next] = .{ .s = r.s, .row0 = row, .rows = 1 };
            live[next] = i;
            from[next] = row;
            next += 1;
            row += 1;
        }
        nlive = next;
        if (nlive == 0) break;
        try k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..nlive]));
        try mtpComputeSegs(e, b, segs[0..nlive], @intCast(nlive));
        for (segs[0..nlive]) |sg| {
            try setMtpLen(e, sg.s, sg.s.mtp_len + 1);
            sg.s.mtp_drafted += 1;
        }
    }
}

/// ``draft`` that stops before a later draft the head gives under ``confidence`` (0: never; the first draft stays).
pub fn draftUpTo(e: *Engine, s: *Seq, follow: []const u32, depth: u32, confidence: f64, out: []u32) !usize {
    const k = &e.k;
    const b = &e.mbuf;
    const n: i64 = @intCast(follow.len);
    if (s.mtp_drafted > 0) {
        try setMtpLen(e, s, s.mtp_len - s.mtp_drafted);
        s.mtp_drafted = 0;
    }
    try room(e, s, n + depth + 1);
    for (follow, 0..) |t, i| e.idbuf[i] = @intCast(t);
    try k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..follow.len]));
    if (s.fresh) {
        try k.copy(b.mtp_in, s.last_streams, WIDE * 2);
        s.fresh = false;
    } else try k.copy(b.mtp_in, rowAt(e.buf.streams, s.last_row0, WIDE * 2), @intCast(n * WIDE * 2));
    try step(e, 1, s, n);
    try setMtpLen(e, s, s.mtp_len + n);
    var got: usize = 0;
    var j: u32 = 0;
    while (j < depth) : (j += 1) {
        var pick: [1]Pick = undefined;
        try readPicks(e, b, 1, &pick);
        const low = confidence > 0 and pick[0].p < confidence;
        if (low and j > 0) break;
        out[got] = @intCast(pick[0].tok);
        e.draft_p[got] = pick[0].p;
        got += 1;
        if (low) break;
        if (j + 1 < depth) {
            const prev = b.streams + @as(u64, @intCast(if (j == 0) n - 1 else 0)) * WIDE * 2;
            e.idbuf[0] = @intCast(pick[0].tok);
            try k.upload(b.ids, std.mem.sliceAsBytes(e.idbuf[0..1]));
            try k.copy(b.mtp_in, prev, WIDE * 2);
            try step(e, 1, s, 1);
            try setMtpLen(e, s, s.mtp_len + 1);
            s.mtp_drafted += 1;
        }
    }
    return got;
}
