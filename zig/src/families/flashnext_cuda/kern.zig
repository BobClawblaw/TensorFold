//! Flash Next on CUDA: every launch the forward makes, with the Python engine's grids, constexprs and arguments.
//! Triton kernels come from the captured set (a variant picked as Triton picks it: constexprs, 16-divisibility, the
//! value 1 folded); extension kernels from the build's fatbins (SASS-equal to the Python extensions), found by name.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

pub const Arg = aot.Arg;
pub const Const = aot.Const;
pub const ci = aot.ci;
pub const cf = aot.cf;
pub const cnone = struct {
    fn f(name: []const u8) Const {
        return .{ .name = name };
    }
}.f;

pub fn P(name: []const u8, addr: u64) Arg {
    return aot.ptr(name, "", addr);
}
pub fn I(name: []const u8, v: anytype) Arg {
    return aot.int(name, @intCast(v));
}
pub fn F(name: []const u8, v: f32) Arg {
    return aot.float(name, v);
}

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

fn nextPow2(x: u64) u64 {
    return std.math.ceilPowerOfTwo(u64, @max(x, 1)) catch unreachable;
}

/// aot.zig's matching with pointer types left to the caller (the constexprs tell this engine's variants apart).
fn matches(k: anytype, args: []const Arg, consts: []const Const) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1) return false;
                if (p.div16 != (!p.nospec and @mod(x, 16) == 0)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (p.type[0] != '*' or p.div16 != (x.addr % 16 == 0)) return false;
                if (x.ty.len > 0 and !std.mem.eql(u8, p.type, x.ty)) return false; // where variants differ by it
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64") or p.div16 != (x % 16 == 0)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}

pub const Ext = enum(u8) { experts, experts_prefill, gdn_prefill, qmm, qmm_prefill, gdn, gdn_io, sample, vision };

/// The launch context: the stream, the Triton set, the extension modules' functions by name.
pub const K = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    set: *const aot.Set,
    names: [16][]const [:0]const u8 = @splat(&.{}),
    funcs: [16][]cuda.Function = @splat(&.{}),
    sms: i64 = 48,
    cache: std.AutoHashMap(u64, cuda.Function) = undefined,
    gpa: std.mem.Allocator,
    /// Development: synchronize after every launch and name the one that faulted.
    sync_each: bool = false,

    fn after(self: *K, what: []const u8) !void {
        if (!self.sync_each) return;
        self.stream.synchronize() catch |e| {
            std.log.err("flash next: launch {s} faulted", .{what});
            return e;
        };
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, stream: cuda.Stream, set: *const aot.Set, modules: []const cuda.Module) !K {
        var k: K = .{ .d = d, .stream = stream, .set = set, .gpa = gpa };
        k.cache = .init(gpa);
        k.sms = try ctx.attribute(.multiprocessor_count);
        for (modules, 0..) |m, i| {
            var n: c_uint = 0;
            try d.check(d.api.cuModuleGetFunctionCount(&n, m.handle), "cuModuleGetFunctionCount");
            const hs = try gpa.alloc(cuda.abi.Function, n);
            defer gpa.free(hs);
            try d.check(d.api.cuModuleEnumerateFunctions(hs.ptr, n, m.handle), "cuModuleEnumerateFunctions");
            const names = try gpa.alloc([:0]const u8, n);
            const fs = try gpa.alloc(cuda.Function, n);
            for (hs, 0..) |h, j| {
                var s: ?[*:0]const u8 = null;
                try d.check(d.api.cuFuncGetName(&s, h), "cuFuncGetName");
                names[j] = try gpa.dupeSentinel(u8, std.mem.span(s.?), 0);
                fs[j] = .{ .d = d, .handle = h };
            }
            k.names[i] = names;
            k.funcs[i] = fs;
        }
        return k;
    }

    /// An extension kernel whose mangled name contains ``frag`` (one match only).
    pub fn ext(self: *K, m: Ext, frag: []const u8) !cuda.Function {
        const key = std.hash.Wyhash.hash(@intFromEnum(m), frag);
        if (self.cache.get(key)) |f| return f;
        var found: ?cuda.Function = null;
        for (self.names[@intFromEnum(m)], 0..) |n, j| {
            if (std.mem.indexOf(u8, n, frag) != null) {
                if (found != null) {
                    std.log.err("flash next: two kernels in {s} match {s}", .{ @tagName(m), frag });
                    return error.AmbiguousKernel;
                }
                found = self.funcs[@intFromEnum(m)][j];
            }
        }
        const f = found orelse {
            std.log.err("flash next: no kernel in {s} matches {s}", .{ @tagName(m), frag });
            return error.MissingKernel;
        };
        try self.cache.put(key, f);
        return f;
    }

    pub fn go(self: *K, f: cuda.Function, grid: cuda.Dim3, block: u32, shared: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = block }, .shared = shared }, self.stream, args);
        if (self.sync_each) {
            var nm: ?[*:0]const u8 = null;
            _ = self.d.api.cuFuncGetName(&nm, f.handle);
            try self.after(if (nm) |x| std.mem.span(x) else "extension kernel");
        }
    }

    /// One Triton launch: the variant whose constexprs and specialization fit, its runtime arguments in order.
    pub fn tri(self: *K, name: []const u8, grid: [3]i64, args: []const Arg, consts: []const Const) !void {
        for (self.set.variants) |*v| {
            if (!std.mem.eql(u8, v.spec.@"fn", name)) continue;
            if (!matches(v.spec, args, consts)) continue;
            var packed_args: cuda.Args = .{};
            for (v.spec.params) |p| {
                const a = for (args) |x| {
                    if (std.mem.eql(u8, x.name, p.name)) break x;
                } else unreachable;
                switch (a.value) {
                    .ptr => |x| packed_args.add(x.addr),
                    .i32 => |x| packed_args.add(x),
                    .f32 => |x| packed_args.add(x),
                    .u64 => |x| packed_args.add(x),
                }
            }
            try v.kernel.launchOn(.{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, self.stream, &packed_args, .{}, &.{});
            return self.after(name);
        }
        std.log.err("flash next: no captured {s} variant for this launch (grid {any})", .{ name, grid });
        for (consts) |c| std.log.err("  const {s} = {?d} {?d}", .{ c.name, c.int, c.f32 });
        for (args) |a| switch (a.value) {
            .ptr => |p| std.log.err("  {s}: ptr {x}", .{ a.name, p.addr }),
            .i32 => |x| std.log.err("  {s}: i32 {d}", .{ a.name, x }),
            .f32 => |x| std.log.err("  {s}: f32 {d}", .{ a.name, x }),
            .u64 => |x| std.log.err("  {s}: u64 {d}", .{ a.name, x }),
        };
        return error.MissingTritonVariant;
    }

    pub fn memset32(self: *K, dst: u64, v: u32, n: usize) !void {
        try self.d.check(self.d.api.cuMemsetD32Async(dst, v, n, self.stream.handle), "memset32");
    }
    pub fn copy(self: *K, dst: u64, src: u64, n: usize) !void {
        try self.d.check(self.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, self.stream.handle), "copy");
    }
    pub fn upload(self: *K, dst: u64, bytes: []const u8) !void {
        try self.d.check(self.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, self.stream.handle), "upload");
    }
};

// -- the model's fixed dimensions (Qwen3.8 Flash Next, two ranks) ---------------------------------------------------
pub const D: i64 = 2560; // hidden
pub const S: i64 = 4; // hyper-connection streams
pub const WIDE: i64 = D * S;
pub const LOW: i64 = 320;
pub const EPS: f32 = 1e-6;

// -- glue.py --------------------------------------------------------------------------------------------------------
pub fn embed(k: *K, ids: u64, table: u64, out: u64, R: i64, copies: i64) !void {
    try k.tri("_embed", .{ R, D / 256, 1 }, &.{ P("IDS", ids), P("T", table), P("OUT", out) }, &.{ ci("D", D), ci("S", copies), ci("BLOCK", 256) });
}

/// mode 0: no branch (every optional pointer is h); mode 3: gathered fp32 partials ``br`` with ``inj``.
pub fn hcWriteback(k: *K, h: u64, pss: u64, br: u64, inj: u64, R: i64, mode: i64) !void {
    const world: i64 = if (mode == 3) 2 else 1;
    try k.tri("_hc_writeback", .{ R, D / 256, 1 }, &.{ P("H", h), P("HOUT", h), P("PSS", pss), P("BR", if (mode == 0) h else br),
        P("INJ", if (mode == 0) h else inj), P("Y", h), P("WTS", h), I("RS", R * D) },
        &.{ ci("D", D), ci("S", S), ci("MODE", mode), ci("TOPK", 1), ci("SLOTS", 1), ci("BLOCK", 256), ci("WORLD", world) });
}

pub fn hcNormed(k: *K, h: u64, pss: u64, scale: u64, normed: u64, xs: u64, R: i64) !void {
    try k.tri("_hc_normed", .{ R, WIDE / 512, 1 }, &.{ P("H", h), P("PSS", pss), P("SCALE", scale), P("NORMED", normed), P("XS", xs), F("eps", EPS) },
        &.{ ci("D", D), ci("S", S), ci("NC", D / 256), ci("BLOCK", 512) });
}

/// The prompt's fused write-back and norm (hc_check's fuser).
pub fn writeNorm(k: *K, h: u64, pss: u64, scale: u64, normed: u64, xs: u64, br: u64, inj: u64, R: i64, mode: i64) !void {
    const world: i64 = if (mode == 3) 2 else 1;
    try k.tri("_write_norm", .{ R, 1, 1 }, &.{ P("H", h), P("PSS", pss), P("SCALE", scale), P("NORMED", normed), P("XS", xs),
        P("BR", if (mode == 0) h else br), P("INJ", if (mode == 0) h else inj), P("Y", h), P("WTS", h), I("RS", R * D), F("eps", EPS) },
        &.{ ci("D", D), ci("S", S), ci("MODE", mode), ci("SLOTS", 1), ci("WORLD", world), ci("PAD", 4096) });
}

pub fn hcReduceAct(k: *K, part: u64, act: u64, xs: u64, inj: ?u64, R: i64, ndn: i64) !void {
    try k.tri("_hc_reduce_act", .{ R, 1, 1 }, &.{ P("PART", part), P("ACT", act), P("XS", xs), P("INJ", inj orelse act), I("M", R) },
        &.{ ci("SK", 8), ci("S", S), ci("LOW", LOW), ci("LOWP", 512), ci("HAS_INJ", @intFromBool(inj != null)), ci("NDN", ndn) });
}

pub fn hcAct(k: *K, dn: u64, act: u64, xs: u64, inj: ?u64, R: i64, ndn: i64) !void {
    try k.tri("_hc_act", .{ R, 1, 1 }, &.{ P("DN", dn), P("ACT", act), P("XS", xs), P("INJ", inj orelse act) },
        &.{ ci("S", S), ci("LOW", LOW), ci("LOWP", 512), ci("HAS_INJ", @intFromBool(inj != null)), ci("NDN", ndn) });
}

pub fn hcMix(k: *K, up: u64, normed: u64, mixed: u64, xs: u64, R: i64) !void {
    try k.tri("_hc_mix", .{ R, D / 256, 1 }, &.{ P("UP", up), P("NORMED", normed), P("MIXED", mixed), P("XS", xs) }, &.{ ci("D", D), ci("S", S), ci("BLOCK", 256) });
}

pub fn rmsnorm(k: *K, x: u64, w: u64, out: u64, xs: u64, x_stride: i64, R: i64, d: i64) !void {
    try k.tri("_rmsnorm", .{ R, 1, 1 }, &.{ P("X", x), P("W", w), P("OUT", out), P("XS", xs), F("eps", EPS), I("x_stride", x_stride) },
        &.{ ci("D", d), ci("G", d), ci("BLOCK", @intCast(nextPow2(@intCast(d)))) });
}

pub fn addStreams(k: *K, e: u64, hs: u64, out: u64, R: i64) !void {
    try k.tri("_add_streams", .{ R, D / 256, 1 }, &.{ P("E", e), P("HS", hs), P("OUT", out) }, &.{ ci("D", D), ci("S", S), ci("BLOCK", 256) });
}

/// ``y_f32``: the slots are fp32 (a window's MTP head), else bf16; the two variants differ by Y's type alone.
pub fn moePartial(k: *K, y: u64, y_f32: bool, wts: u64, out: u64, R: i64) !void {
    try k.tri("_moe_partial", .{ R, D / 256, 1 }, &.{ aot.ptr("Y", if (y_f32) "*fp32" else "*bf16", y), P("WTS", wts), P("OUT", out) }, &.{ ci("D", D), ci("TOPK", 10), ci("SLOTS", 11), ci("BLOCK", 256) });
}

pub fn pleEmbedBf16(k: *K, v: u64, out: u64, xs: u64, R: i64) !void {
    try k.tri("_ple_embed_bf16", .{ R, 16, 1 }, &.{ P("V", v), P("OUT", out), P("XS", xs) }, &.{ ci("HEADS", 16), ci("DH", 160), cf("SCALE", 1.0) });
}

pub fn pleGate(k: *K, keys: u64, vals: u64, h: u64, nk: u64, nq: u64, gated: u64, pss: u64, R: i64) !void {
    try k.tri("_ple_gate", .{ R, 1, 1 }, &.{ P("KEYS", keys), P("VALS", vals), P("H", h), P("NK", nk), P("NQ", nq), P("GATED", gated), P("PSS", pss), F("eps", EPS) },
        &.{ ci("D", D), ci("S", S), ci("BLOCK", 512) });
}

pub fn pleConv(k: *K, gated: u64, pss: u64, nc: u64, tail: u64, cw: u64, h: u64, nrow: u64, R: i64) !void {
    try k.tri("_ple_conv", .{ R, WIDE / 512, 1 }, &.{ P("GATED", gated), P("PSS", pss), P("NC", nc), P("TAIL", tail), P("CW", cw), P("H", h), P("HOUT", h),
        P("NROW", nrow), F("eps", EPS), I("R", R) }, &.{ ci("D", D), ci("S", S), ci("TAPS", 4), ci("DIL", 3), ci("BLOCK", 512) });
}

/// A sequence's image positions (rotary MODE 2): ``table`` [length, 3] (t, h, w) int32 for the prompt's rows, and the
/// device ``delta`` the rows past the prompt add to their position.
pub const Rope = struct { table: u64, delta: u64, length: i64 };

pub fn attnPrep(k: *K, p: u64, pos: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, kc: u64, vc: u64, ks: u64, vs: u64, iq: u64, ikc: u64, R: i64, rope: ?Rope) !void {
    const table = if (rope) |r| r.table else pos;
    const delta = if (rope) |r| r.delta else pos;
    const length = if (rope) |r| r.length else 0;
    try k.tri("_attn_prep", .{ R, 12 + 1 + 4 + 1, 1 }, &.{ P("P", p), P("POS0", pos), P("QW", qw), P("KW", kw), P("IW", iw), P("INV", inv), P("Q", q),
        P("KC", kc), P("VC", vc), P("KS", ks), P("VS", vs), P("IQ", iq), P("IKC", ikc), P("ROPE", table), P("DELTA", delta), I("length", length), F("eps", EPS) },
        &.{ ci("PW", 7296), ci("NQ", 12), ci("NKV", 1), ci("HD", 256), ci("NI", 4), ci("IHD", 128), ci("HALF", 32), ci("BITS", 8), ci("MODE", if (rope != null) 2 else 0), ci("S1", 11), ci("S2", 10) });
}

pub fn attnGate(k: *K, o: u64, p: u64, out: u64, xs: u64, R: i64) !void {
    try k.tri("_attn_gate", .{ R, 12, 1 }, &.{ P("O", o), P("P", p), P("OUT", out), P("XS", xs) }, &.{ ci("PW", 7296), ci("NQ", 12), ci("HD", 256) });
}

// -- attention.py ----------------------------------------------------------------------------------------------------
pub const RATIO: i64 = 4;
pub const TOP: i64 = 512;
pub const IDW: i64 = 2052;
pub const CHUNK: i64 = 512;
pub const NCH: i64 = 5;
pub const KEYS_MAX: i64 = 2051; // (budget / ratio + 1) * ratio - 1

pub fn pool(k: *K, ikc: u64, pooled: u64, pos: u64, w: u64, inv: u64, R: i64, rope: ?Rope) !void {
    if (rope) |r| return k.tri("_pool", .{ @divFloor(R, RATIO) + 2, 1, 1 }, &.{ P("IKC", ikc), P("POOLED", pooled), P("POS0", pos), P("W", w), P("INV", inv),
        P("ROPE", r.table), P("DELTA", r.delta), F("eps", EPS), I("R", R), I("length", r.length) },
        &.{ ci("DI", 128), ci("HALF", 32), ci("RATIO", RATIO), ci("MODE", 2), ci("S1", 11), ci("S2", 10) });
    try k.tri("_pool", .{ @divFloor(R, RATIO) + 2, 1, 1 }, &.{ P("IKC", ikc), P("POOLED", pooled), P("POS0", pos), P("W", w), P("INV", inv), F("eps", EPS), I("R", R), I("length", 0) },
        &.{ ci("DI", 128), ci("HALF", 32), ci("RATIO", RATIO), cnone("ROPE"), cnone("DELTA"), ci("MODE", 0), ci("S1", 11), ci("S2", 10) });
}

/// qsa_rows: score and select the rows' blocks (``context`` keys bound the launches).
pub fn qsaRows(k: *K, iq: u64, pooled: u64, pos: u64, sc: u64, ids: u64, nkr: u64, spr: u64, nb: i64, rows: i64, context: i64) !void {
    const blocks = @min(nb, @max(1, cdiv(context, RATIO)));
    try k.tri("_scores", .{ rows, cdiv(blocks, 64), 1 }, &.{ P("IQ", iq), P("POOLED", pooled), P("POS0", pos), P("SC", sc), I("NB", nb) },
        &.{ ci("HI", 4), ci("DI", 128), ci("RATIO", RATIO), ci("TOP", TOP), ci("BB", 64) });
    const width: i64 = @intCast(nextPow2(@intCast(blocks)));
    const sargs = [_]Arg{ P("SC", sc), P("POS0", pos), P("IDS", ids), P("NKR", nkr), P("SPR", spr), I("NB", nb) };
    if (width <= 32768) {
        try k.tri("_select", .{ rows, 1, 1 }, &sargs, &.{ ci("RATIO", RATIO), ci("TOP", TOP), ci("IDW", IDW), ci("BLOCK", width) });
    } else {
        const tb: i64 = if (rows >= 64) 4096 else 8192;
        try k.tri("_select_tiles", .{ rows, 1, 1 }, &sargs, &.{ ci("RATIO", RATIO), ci("TOP", TOP), ci("IDW", IDW), ci("TB", tb) });
    }
}

/// attention: 512-key chunks over at most 2051 keys, then the merge into ``out``.
pub fn attention(k: *K, q: u64, kc: u64, vc: u64, ks: u64, vs: u64, pos: u64, po: u64, pm: u64, pl: u64, ids: u64, nkr: u64, spr: u64, out: u64, rows: i64, context: i64) !void {
    const keys = @min(context, KEYS_MAX);
    const chunks = @min(NCH, cdiv(keys, CHUNK));
    try k.tri("_chunks", .{ rows, 1, chunks }, &.{ P("Q", q), P("KC", kc), P("VC", vc), P("KSC", ks), P("VSC", vs), P("POS0", pos), P("PO", po), P("PM", pm), P("PL", pl),
        P("IDS", ids), P("NKR", nkr), P("SPR", spr) }, &.{ ci("H", 12), ci("HK", 1), ci("D", 256), ci("G", 12), ci("CH", CHUNK), ci("NCH", NCH),
        cf("SCALE", 0.0625), ci("IDW", IDW), ci("QSA", 1), ci("BITS", 8) });
    try k.tri("_merge", .{ rows, 1, 1 }, &.{ P("PO", po), P("PM", pm), P("PL", pl), P("POS0", pos), P("OUT", out), P("NKR", nkr), P("SPR", spr) },
        &.{ ci("H", 12), ci("HK", 1), ci("D", 256), ci("G", 12), ci("CH", CHUNK), ci("NCH", NCH), ci("QSA", 1), ci("BITS", 8) });
}

// -- moe.py, affine_moe.py --------------------------------------------------------------------------------------------
pub fn router(k: *K, x: u64, w: u64, out: u64, m: i64, x_stride: i64) !void {
    const bm: i64 = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
    const be: i64, const bk: i64 = if (bm == 16) .{ 32, 256 } else .{ 64, 64 };
    try k.tri("_router", .{ cdiv(m, bm), cdiv(513, be), 1 }, &.{ P("X", x), P("W", w), P("OUT", out), I("M", m), I("x_stride", x_stride) },
        &.{ ci("D", D), ci("NE", 513), ci("BM", bm), ci("BLOCK_E", be), ci("BK", bk) });
}

pub fn topkRows(k: *K, l: u64, pick: u64, wts: u64, R: i64) !void {
    try k.tri("_topk_rows", .{ R, 1, 1 }, &.{ P("L", l), P("PICK", pick), P("WTS", wts) },
        &.{ ci("NE", 512), ci("NL", 513), ci("TOPK", 10), ci("SLOTS", 11), ci("BLOCK", 1024), ci("SLOTP", 16) });
}

pub fn swiglu(k: *K, g: u64, out: u64, g_stride: i64, out_stride: i64, R: i64) !void {
    try k.tri("_swiglu", .{ R, 1, 1 }, &.{ P("G", g), P("OUT", out), I("g_stride", g_stride), I("out_stride", out_stride) }, &.{ ci("NI", 320), ci("BLOCK", 512) });
}

/// experts.max_items: an item per used expert, plus one per ``tile`` pairs past its first.
pub fn maxItems(pairs: i64, experts: i64, tile: i64) i64 {
    return @min(pairs, experts) + @divFloor(pairs, tile);
}

pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64, tile: i64 };

/// experts_v7.plan: picks [R, 11] -> the plan (the shared slot's id E falls past the table).
pub fn expertsPlan(k: *K, picks: u64, pairs: i64, plan: Plan) !void {
    var a: cuda.Args = .{};
    const E: i32 = 512;
    if (pairs <= 1024) {
        a.add(picks); a.add(@as(i32, @intCast(pairs))); a.add(E); a.add(@as(i32, @intCast(plan.tile))); a.add(plan.members); a.add(plan.items); a.add(plan.counts);
        try k.go(try k.ext(.experts, "11plan_kernelE"), .{ .x = 1 }, 1024, 0, &a);
    } else {
        const nblk = cdiv(pairs, 1024);
        a.add(picks); a.add(@as(i32, @intCast(pairs))); a.add(E); a.add(plan.rank); a.add(plan.hist);
        try k.go(try k.ext(.experts, "9plan_rankE"), .{ .x = @intCast(nblk) }, 1024, 0, &a);
        var b: cuda.Args = .{};
        b.add(@as(i32, @intCast(nblk))); b.add(E); b.add(@as(i32, @intCast(plan.tile))); b.add(plan.hist); b.add(plan.items); b.add(plan.counts);
        try k.go(try k.ext(.experts, "12plan_offsetsE"), .{ .x = 1 }, 1024, 0, &b);
        var s: cuda.Args = .{};
        s.add(picks); s.add(@as(i32, @intCast(pairs))); s.add(E); s.add(plan.rank); s.add(plan.hist); s.add(plan.members);
        try k.go(try k.ext(.experts, "12plan_scatterE"), .{ .x = @intCast(cdiv(pairs, 256)) }, 256, 0, &s);
    }
}

/// experts_v7.prefill / run: gate-up (epi 2, 11 slots of x) or down (epi 3 prompt bf16 / 0 fp32) over the plan.
pub fn expertsCall(k: *K, prefill: bool, gs: i64, epi: i64, x: u64, x_stride: i64, slots: i64, w: u64, kb: i64, nb: i64, plan: Plan, out: u64, width: i64, items: i64) !void {
    const m: i64 = if (epi == 2) 2 else 1;
    var a: cuda.Args = .{};
    a.add(x); a.add(@as(i32, @intCast(x_stride))); a.add(@as(i32, @intCast(slots))); a.add(w); a.add(@as(i32, @intCast(kb)));
    a.add(@as(i32, @intCast(nb))); a.add(plan.items); a.add(plan.counts); a.add(plan.members); a.add(out); a.add(@as(i32, @intCast(width)));
    a.add(@as(f32, 0.0));
    var buf: [128]u8 = undefined;
    if (prefill) {
        const f = try k.ext(.experts_prefill, try std.fmt.bufPrint(&buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi2ELi2ELi4E", .{ gs, m, epi }));
        const block: i64 = 32 * @divExact(4 * gs, 128) + 4 * 2;
        const su: i64 = 64 * 8 + 4 * @divExact(64, gs) * m * block;
        const sb = su * 16;
        const stages: i64 = if (sb * 4 <= 49152) 4 else if (sb * 3 <= 49152) 3 else 2;
        const smem: u32 = @intCast(stages * sb);
        try f.allowDynamicShared(smem);
        const grid = items * cdiv(nb, 4);
        if (grid >= 1) try k.go(f, .{ .x = @intCast(grid) }, 256, smem, &a);
    } else {
        const f = try k.ext(.experts, try std.fmt.bufPrint(&buf, "13expert_kernelILi{d}ELi{d}ELi{d}ELi2ELi4E", .{ gs, m, epi }));
        const per_sm: i64 = @max(1, try f.occupancy(128, 0));
        const grid = @min(cdiv(items, 4), per_sm * k.sms);
        if (grid >= 1) try k.go(f, .{ .x = @intCast(grid) }, 128, 0, &a);
    }
}

// -- q8.py: int8 faces ----------------------------------------------------------------------------------------------
pub const Part = struct { w: u64, s: u64, n: i64, gs: i64 }; // gs 0: a bf16 part (no scales)
pub const Face = struct {
    parts: [2]Part = undefined,
    count: usize = 0,
    k: i64,
    pub fn n(f: Face) i64 {
        var t: i64 = 0;
        for (f.parts[0..f.count]) |p| t += p.n;
        return t;
    }
};

pub const Cfg = struct { bm: i64, bn: i64, bk: i64, sk: i64 };

pub fn config(n: i64, kk: i64, gs: i64, m: i64) Cfg {
    const bm: i64 = if (m > 128) 128 else 16;
    if (n >= 65536 and @mod(gs, 128) == 0 and @mod(kk, 128) == 0) return .{ .bm = bm, .bn = 128, .bk = 128, .sk = 1 };
    if (n <= 512) return .{ .bm = bm, .bn = 32, .bk = 64, .sk = if (@mod(@divFloor(kk, 64), 8) == 0) 8 else 1 };
    if (kk <= 512) return .{ .bm = bm, .bn = 128, .bk = 64, .sk = 1 };
    if (n < 4096) return .{ .bm = bm, .bn = 128, .bk = 64, .sk = if (@mod(@divFloor(kk, 64), 4) == 0) 4 else 1 };
    return .{ .bm = bm, .bn = 32, .bk = 64, .sk = if (@mod(@divFloor(kk, 64), 4) == 0) 4 else 1 };
}

/// q8.matmul: x [m, K] -> out [m, N] (bf16, or fp32 sums); split K partials summed by _reduce through ``part``.
pub fn matmul(k: *K, x: u64, x_stride: i64, face: Face, out: u64, out_stride: i64, fp32: bool, m: i64, part: u64) !void {
    var col0: i64 = 0;
    for (face.parts[0..face.count]) |p| {
        const gs_cfg: i64 = if (p.gs != 0) p.gs else 128;
        var c = config(p.n, face.k, gs_cfg, m);
        var inline_: i64 = 1;
        if (m > 128) {
            inline_ = c.sk;
            c.sk = 1;
            c.bm = 64;
            c.bn = 128;
        }
        const tgt = if (c.sk > 1) part else out;
        try k.tri("_q8mm", .{ cdiv(m, c.bm), cdiv(p.n, c.bn), c.sk }, &.{ P("X", x), P("W", p.w), P("S", if (p.gs != 0) p.s else p.w), P("OUT", out), P("PART", tgt),
            I("M", m), I("x_stride", x_stride), I("out_stride", out_stride), I("col0", col0) },
            &.{ ci("N", p.n), ci("K", face.k), ci("GS", @max(p.gs, 1)), ci("SK", c.sk), ci("BM", c.bm), ci("BLOCK_N", c.bn), ci("BLOCK_K", c.bk),
            ci("SCALED", @intFromBool(p.gs != 0)), ci("F32", @intFromBool(fp32)), ci("INLINE", inline_) });
        if (c.sk > 1) {
            try k.tri("_reduce", .{ cdiv(m * p.n, 1024), 1, 1 }, &.{ P("PART", part), P("OUT", out), I("M", m), I("out_stride", out_stride), I("col0", col0) },
                &.{ ci("N", p.n), ci("SK", c.sk), ci("BLOCK", 1024), ci("F32", @intFromBool(fp32)) });
        }
        col0 += p.n;
    }
}

/// q8.partials: the face's K slices unreduced [SK, m, N] fp32 into ``part``; false when the parts split differently.
pub fn partials(k: *K, x: u64, x_stride: i64, face: Face, part: u64, m: i64) !bool {
    if (m > 128) return false;
    var cfgs: [2]Cfg = undefined;
    for (face.parts[0..face.count], 0..) |p, i| cfgs[i] = config(p.n, face.k, if (p.gs != 0) p.gs else 128, m);
    const sk = cfgs[0].sk;
    if (sk == 1) return false;
    for (cfgs[0..face.count]) |c| if (c.sk != sk) return false;
    const n_all = face.n();
    var col0: i64 = 0;
    for (face.parts[0..face.count], 0..) |p, i| {
        const c = cfgs[i];
        try k.tri("_q8slices", .{ cdiv(m, c.bm), cdiv(p.n, c.bn), sk }, &.{ P("X", x), P("W", p.w), P("S", if (p.gs != 0) p.s else p.w), P("PART", part),
            I("M", m), I("x_stride", x_stride), I("col0", col0) }, &.{ ci("N", p.n), ci("NALL", n_all), ci("K", face.k), ci("GS", @max(p.gs, 1)),
            ci("SK", sk), ci("BM", c.bm), ci("BLOCK_N", c.bn), ci("BLOCK_K", c.bk), ci("SCALED", @intFromBool(p.gs != 0)) });
        col0 += p.n;
    }
    return true;
}

/// q8.hc_upmix for prompt rows (m > 128): the up projection and the mix in one pass.
pub fn upmix(k: *K, act: u64, face: Face, normed: u64, mixed: u64, xs: u64, m: i64) !void {
    const p = face.parts[0];
    try k.tri("_q8upmix", .{ cdiv(m, 32), @divExact(D, 64), 1 }, &.{ P("A", act), P("W", p.w), P("S", p.s), P("NORMED", normed), P("MIXED", mixed), P("XS", xs),
        I("M", m), I("a_stride", face.k) }, &.{ ci("D", D), ci("K", face.k), ci("GS", p.gs), ci("STREAMS", S), ci("BM", 32), ci("BLOCK_N", 64), ci("BLOCK_K", 64) });
}

// -- gdn.py, gdn_io.py, kernels/gdn.py ------------------------------------------------------------------------------
pub const NK: i64 = 8;
pub const NV: i64 = 24;

/// qwen4_exp_gdn.chain: a decode window's DeltaNet step per row (the conv state, the recurrent state in and out).
pub fn gdnChain(k: *K, p: u64, conv: u64, conv_w: u64, state_in: u64, a_log: u64, dt_bias: u64, norm: u64, rows: i64, out: u64, xs: u64, state_out: u64, sk: u64, sv: u64, sg: u64, sb: u64) !void {
    var buf: [96]u8 = undefined;
    const f = try k.ext(.gdn, try std.fmt.bufPrint(&buf, "12chain_kernelILi{d}ELi{d}ELb{d}E", .{ NK, NV, @intFromBool(rows > 1) }));
    var a: cuda.Args = .{};
    for ([_]u64{ p, conv, conv_w, state_in, a_log, dt_bias, norm }) |x| a.add(x);
    a.add(EPS); a.add(@as(i32, @intCast(rows)));
    for ([_]u64{ out, xs, state_out, sk, sv, sg, sb }) |x| a.add(x);
    try k.go(f, .{ .x = @intCast(NV) }, 1024, 0, &a);
}

pub fn gdnReplay(k: *K, state_in: u64, sk: u64, sv: u64, sg: u64, sb: u64, rows: i64, state_out: u64) !void {
    var buf: [96]u8 = undefined;
    const f = try k.ext(.gdn, try std.fmt.bufPrint(&buf, "13replay_kernelILi{d}ELi{d}E", .{ NK, NV }));
    var a: cuda.Args = .{};
    for ([_]u64{ state_in, sk, sv, sg, sb }) |x| a.add(x);
    a.add(@as(i32, @intCast(rows))); a.add(state_out);
    try k.go(f, .{ .x = @intCast(NV) }, 1024, 0, &a);
}

/// gdn_io.front: a prompt piece's conv taps and gates -> q, k (fp32), v (bf16), g, beta.
pub fn gdnFront(k: *K, p: u64, conv_ptrs: u64, sid: u64, windows: u64, conv_w: u64, a_log: u64, dt_bias: u64, q: u64, kk: u64, v: u64, g: u64, beta: u64, n: i64) !void {
    var buf: [96]u8 = undefined;
    const f = try k.ext(.gdn_io, try std.fmt.bufPrint(&buf, "12front_kernelILi{d}ELi{d}E", .{ NK, NV }));
    var a: cuda.Args = .{};
    for ([_]u64{ p, conv_ptrs, sid, windows, conv_w, a_log, dt_bias, q, kk, v, g, beta }) |x| a.add(x);
    try k.go(f, .{ .x = @intCast(n), .y = @intCast(NK + NV) }, 128, 0, &a);
}

/// kernels/gdn.py chain (gdn_v2.prefill): the piece's rows from state_in into state_out, outputs y [n, NV, 128].
pub fn gdnPrefill(k: *K, q: u64, kk: u64, v: u64, g: u64, beta: u64, state_in: u64, state_out: u64, y: u64, n: i64) !void {
    if (n <= 0) return;
    const rows: i64 = if (NV >= k.sms) 128 else 64;
    var buf: [96]u8 = undefined;
    const f = try k.ext(.gdn_prefill, try std.fmt.bufPrint(&buf, "12chain_kernelIfLi{d}ELi32E", .{rows}));
    var a: cuda.Args = .{};
    for ([_]u64{ q, kk, v, g, beta, state_in, state_out, y }) |x| a.add(x);
    a.add(@as(i32, @intCast(n))); a.add(@as(i32, @intCast(NK))); a.add(@as(i32, @intCast(NV)));
    try k.go(f, .{ .x = @intCast(NV), .y = @intCast(@divExact(128, rows)) }, @intCast(2 * rows), 0, &a);
}

pub fn gdnBack(k: *K, y: u64, p: u64, norm: u64, out: u64, xs: u64, n: i64) !void {
    var buf: [96]u8 = undefined;
    const f = try k.ext(.gdn_io, try std.fmt.bufPrint(&buf, "11back_kernelILi{d}ELi{d}E", .{ NK, NV }));
    var a: cuda.Args = .{};
    a.add(y); a.add(p); a.add(norm); a.add(EPS); a.add(out); a.add(xs);
    try k.go(f, .{ .x = @intCast(n), .y = @intCast(NV) }, 128, 0, &a);
}

/// forward.shift_windows: old [L, T, C] keeps the last T rows of (old, the kept ``keep`` rows of new).
pub fn shiftWindows(k: *K, old: u64, new: u64, keep: i64, old_l: i64, new_l: i64, new_row: i64, layers: i64, channels: i64, taps: i64) !void {
    try k.tri("_shift_windows", .{ layers, cdiv(channels, 256), 1 }, &.{ P("OLD", old), P("NEW", new), I("keep", keep), I("OLD_L", old_l), I("NEW_L", new_l), I("NEW_ROW", new_row) },
        &.{ ci("C", channels), ci("T", taps), ci("TP", @intCast(nextPow2(@intCast(taps)))), ci("BLOCK", 256) });
}

// -- qmm.py: the MTP draft head (4-bit rows of the draft vocabulary) ------------------------------------------------
pub const DraftHead = struct { w: u64, scales: u64, biases: u64, n: i64, npad: i64 };

pub fn qmmDecode(k: *K, x: u64, xs: u64, h: DraftHead, out: u64, part: u64) !void {
    const gs: i64 = 32;
    const bm: i64 = 16;
    const K_: i64 = D;
    const M: i64 = 1;
    var buf: [128]u8 = undefined;
    const f = try k.ext(.qmm, try std.fmt.bufPrint(&buf, "10qmm_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb0ELb0ELb0E", .{ gs, bm }));
    const stage = bm * gs * 2 + 64 * @divExact(gs, 2) + 2 * 64 * 2 + bm * 4;
    const partials_ = @divExact(bm, 16) * 2 * 4 * 128 * 4;
    const smem: u32 = @intCast(@max(4 * stage, partials_));
    try f.allowDynamicShared(smem);
    const rows_t = cdiv(M, bm);
    const group = @max(1, @min(rows_t, @divFloor(@as(i64, 12 << 20), bm * K_ * 2)));
    var a: cuda.Args = .{};
    a.add(x); a.add(xs); a.add(h.w); a.add(h.scales); a.add(h.biases); a.add(out); a.add(@as(u64, 0));
    a.add(@as(i32, @intCast(M))); a.add(@as(i32, @intCast(h.n))); a.add(@as(i32, @intCast(K_))); a.add(@as(i32, 1)); a.add(@as(i32, @intCast(h.npad)));
    a.add(@as(i32, @intCast(K_))); a.add(@as(i32, @intCast(group)));
    _ = part;
    try k.go(f, .{ .x = @intCast(rows_t * cdiv(h.n, 64)), .y = 1, .z = 1 }, 128, smem, &a);
}

pub fn qmmPrefill(k: *K, x: u64, h: DraftHead, out: u64) !void {
    const gs: i64 = 32;
    const K_: i64 = D;
    const M: i64 = 1;
    const tl = [5]i64{ 128, 128, 2, 4, 3 }; // tile 0
    const bm = tl[0];
    const bn = tl[1];
    var buf: [128]u8 = undefined;
    const f = try k.ext(.qmm_prefill, try std.fmt.bufPrint(&buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELb0E", .{ gs, bm, bn, tl[2], tl[3], tl[4] }));
    const smem: u32 = @intCast(tl[4] * (bm * gs * 2 + bn * @divExact(gs, 2) + 2 * bn * 2));
    try f.allowDynamicShared(smem);
    const rows_t = cdiv(M, bm);
    const group = @max(1, @min(rows_t, @divFloor(@as(i64, 12 << 20), bm * K_ * 2)));
    var a: cuda.Args = .{};
    a.add(x); a.add(h.w); a.add(h.scales); a.add(h.biases); a.add(out);
    a.add(@as(i32, @intCast(M))); a.add(@as(i32, @intCast(h.n))); a.add(@as(i32, @intCast(K_))); a.add(@as(i32, @intCast(h.npad)));
    a.add(@as(i32, @intCast(K_))); a.add(@as(i32, @intCast(group)));
    try k.go(f, .{ .x = @intCast(rows_t * cdiv(h.n, bn)) }, @intCast(tl[2] * tl[3] * 32), smem, &a);
}

/// fn_rows_top (our sample.cu): each bf16 row's first maximum as a global id, its value and log-sum-exp.
pub fn rowsTop(k: *K, logits: u64, cols: i64, id_map: u64, offset: i64, out: u64, rows: i64) !void {
    return rowsTopStrided(k, logits, cols, cols, id_map, offset, out, rows);
}

/// rowsTop over rows ``stride`` elements apart (rows padded to an alignment).
pub fn rowsTopStrided(k: *K, logits: u64, cols: i64, stride: i64, id_map: u64, offset: i64, out: u64, rows: i64) !void {
    const f = try k.ext(.sample, "fn_rows_top");
    var a: cuda.Args = .{};
    a.add(logits); a.add(@as(i32, @intCast(cols))); a.add(@as(i32, @intCast(stride))); a.add(id_map); a.add(@as(i32, @intCast(offset))); a.add(out);
    try k.go(f, .{ .x = @intCast(rows) }, 1024, 0, &a);
}
