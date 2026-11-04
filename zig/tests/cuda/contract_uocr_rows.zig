//! Unlimited-OCR-2's CUDA row kernels under the contract (norms, activations, rotary, adds), each with a control.
const std = @import("std");
const cuda = @import("cuda");
const uocr = @import("uocr_cuda");
const rig_ = @import("contract_rig.zig");
const u = @import("contract_uocr.zig");
const ref = @import("contract_uocr_rows_ref.zig");

const c = ref.c;
const g = ref.g;
const cfg = uocr.config;
const Kind = ref.Kind;
const R = ref.R;

/// A registration: its kernel, which call it serves (the add serves the tower and the decoder), its control symbol.
const Case = struct { kind: Kind, name: []const u8, eps: f32 = 0, decoder: bool = false, ctl: ?[:0]const u8 = null };

const cases = [_]Case{
    .{ .kind = .add, .name = "cuda uocr rows add tower (residual in place)" },
    .{ .kind = .add, .name = "control: uocr rows add rounds toward zero", .ctl = "tf_ctl_uocr_rows_add_rz" },
    .{ .kind = .add, .name = "cuda uocr rows add decoder (residual in place)", .decoder = true },
    .{ .kind = .layer_norm, .name = "cuda uocr rows layer_norm SAM (eps 1e-6)", .eps = cfg.sam_eps },
    .{ .kind = .layer_norm, .name = "cuda uocr rows layer_norm CLIP (eps 1e-5)", .eps = cfg.clip_eps },
    .{ .kind = .layer_norm, .name = "control: uocr rows layer_norm rounds (x - u) inv to bf16", .eps = cfg.clip_eps, .ctl = "tf_ctl_uocr_rows_layer_norm_xhat" },
    .{ .kind = .layer_norm2d, .name = "cuda uocr rows layer_norm2d neck (eps 1e-6)", .eps = cfg.sam_eps },
    .{ .kind = .layer_norm2d, .name = "control: uocr rows layer_norm2d leaves bf16((x - u) / den) in fp32", .eps = cfg.sam_eps, .ctl = "tf_ctl_uocr_rows_layer_norm2d_quot" },
    .{ .kind = .gelu, .name = "cuda uocr rows gelu (erf, in place)" },
    .{ .kind = .gelu, .name = "control: uocr rows gelu rounds 1 + erf to bf16", .ctl = "tf_ctl_uocr_rows_gelu_e1" },
    .{ .kind = .quick_gelu, .name = "cuda uocr rows quick_gelu (in place)" },
    .{ .kind = .quick_gelu, .name = "control: uocr rows quick_gelu leaves bf16(sigmoid) in fp32", .ctl = "tf_ctl_uocr_rows_quick_gelu_sig" },
    .{ .kind = .rms_norm, .name = "cuda uocr rows rms_norm", .eps = cfg.eps },
    .{ .kind = .rms_norm, .name = "control: uocr rows rms_norm leaves bf16(x r) in fp32", .eps = cfg.eps, .ctl = "tf_ctl_uocr_rows_rms_norm_xr" },
    .{ .kind = .rope, .name = "cuda uocr rows rope page2d (in place)" },
    .{ .kind = .rope, .name = "control: uocr rows rope leaves its bf16 products in fp32", .ctl = "tf_ctl_uocr_rows_rope_prod" },
    .{ .kind = .router_norm, .name = "cuda uocr rows router_norm (Gemma norm, clamp, page2d)" },
    .{ .kind = .router_norm, .name = "control: uocr rows router_norm leaves the bf16 normed row in fp32", .ctl = "tf_ctl_uocr_rows_router_norm_row" },
    .{ .kind = .silu_mul, .name = "cuda uocr rows silu_mul" },
    .{ .kind = .silu_mul, .name = "control: uocr rows silu_mul leaves bf16(silu) in fp32", .ctl = "tf_ctl_uocr_rows_silu_mul_silu" },
};

fn inPlace(k: Kind) bool {
    return k == .add or k == .gelu or k == .quick_gelu or k == .rope;
}

fn sums(k: Kind) bool {
    return k == .layer_norm or k == .layer_norm2d or k == .rms_norm or k == .router_norm;
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A row's sums: each of 256 threads' strided chain, the warp's 5 shuffles, then the 8 warps' partials in 5 more.
fn rowDepth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 5 + 5;
}

/// router_norm's: one square a thread, the warp's 5 shuffles, then the 4 warps' partials in 5 more.
fn routerDepth(_: c.Shape) usize {
    return 1 + 5 + 5;
}

const norm_why = "bf16 outputs after the sums' 1 / sqrtf: a bf16 sum's error is a few output ulps, not provably 100x";

fn spec(v: Case) c.Spec {
    // the engine's shapes, an odd prompt chunk or the CLIP rows first (their tails fall off a 256-thread block)
    const served: []const c.Shape = switch (v.kind) {
        .add => if (v.decoder) &.{ .{ .rows = 37, .n = cfg.hidden, .k = 1 }, .{ .rows = 2048, .n = cfg.hidden, .k = 1 }, .{ .rows = 1, .n = cfg.hidden, .k = 1 } } else &.{ .{ .rows = 257, .n = cfg.clip_width, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_width, .k = 1 }, .{ .rows = 1600, .n = cfg.sam_width, .k = 1 }, .{ .rows = 101, .n = cfg.clip_width, .k = 1 } },
        .layer_norm => if (v.eps == cfg.clip_eps) &.{ .{ .rows = 257, .n = cfg.clip_width, .k = 1 }, .{ .rows = 101, .n = cfg.clip_width, .k = 1 } } else &.{ .{ .rows = 1600, .n = cfg.sam_width, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_width, .k = 1 } },
        .layer_norm2d => &.{ .{ .rows = 1600, .n = cfg.neck_width, .k = 1 }, .{ .rows = 4096, .n = cfg.neck_width, .k = 1 } },
        .gelu => &.{ .{ .rows = 1600, .n = cfg.sam_ffn, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_ffn, .k = 1 } },
        .quick_gelu => &.{ .{ .rows = 257, .n = cfg.clip_ffn, .k = 1 }, .{ .rows = 101, .n = cfg.clip_ffn, .k = 1 } },
        .rms_norm, .rope => &.{ .{ .rows = 37, .n = cfg.hidden, .k = 1 }, .{ .rows = 2048, .n = cfg.hidden, .k = 1 }, .{ .rows = 1, .n = cfg.hidden, .k = 1 } },
        .router_norm => &.{ .{ .rows = 37, .n = cfg.router_dim, .k = 1 }, .{ .rows = 2048, .n = cfg.router_dim, .k = 1 }, .{ .rows = 1, .n = cfg.router_dim, .k = 1 } },
        .silu_mul => &.{ .{ .rows = 37, .n = cfg.dense_ffn, .k = 1 }, .{ .rows = 2048, .n = cfg.dense_ffn, .k = 1 }, .{ .rows = 1, .n = cfg.shared_ffn, .k = 1 }, .{ .rows = 2048, .n = cfg.shared_ffn, .k = 1 }, .{ .rows = 300, .n = cfg.expert_ffn, .k = 1 } },
    };
    var s: c.Spec = .{ .name = v.name, .accum = .none, .depth = noSums, .out = .bf16, .in_place = inPlace(v.kind), .served = served };
    s.tiles[1] = switch (v.kind) {
        .rope => .{ .size = 128, .step = 128, .min = 128 },
        .router_norm => .{},
        else => .{ .size = 256, .step = 1, .min = 1 },
    };
    if (sums(v.kind)) {
        s.accum = .f32;
        s.depth = if (v.kind == .router_norm) routerDepth else rowDepth;
        s.sums = .{ .bf16 = 0, .why = norm_why };
    }
    if (v.kind == .router_norm) {
        s.out = .f32;
        s.nan = .{ .stated = "a non-finite input reads as 0 and a non-finite result is written as 0 (dash_attention.py's nan_to_num)" };
    }
    return s;
}

fn uni(r: std.Random, lo: f32, hi: f32) u16 {
    return c.toBf16(lo + r.float(f32) * (hi - lo));
}

const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

pub const Rows = struct {
    rig: *rig_.Rig,
    k: uocr.kernels.Kernels,
    v: Case,
    shape: c.Shape = undefined,
    /// The first bf16 operand (a, x or the gate), the second (b, the weight or up), the layer norm's bias.
    x: []u16 = &.{},
    y: []u16 = &.{},
    z: []u16 = &.{},
    /// The rows' token positions, then their pages.
    pos: []u32 = &.{},
    inv: [cfg.rope_token_pairs]f32 = cfg.invTok(),
    /// The device buffers in the kernel's argument order (the in-place output where it reads its operand).
    refs: [6]u64 = @splat(0),
    io: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(t: *Rows) void {
        t.free();
    }
    fn free(t: *Rows) void {
        const a = t.rig.gpu.gpa;
        inline for (.{ "x", "y", "z", "pos" }) |name| {
            a.free(@field(t, name));
            @field(t, name) = &.{};
        }
    }
    pub fn kernel(t: *Rows) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = if (t.v.kind == .router_norm) &vt_router else &vt, .control = if (t.v.ctl != null) .shapes else null };
    }
    const vt = blk: {
        var h = g.hooks(Rows);
        h.overlap = overlap;
        h.check = null;
        break :blk h;
    };
    const vt_router = g.hooks(Rows);
    /// The in-place kernels write over the input they read (the engine binds one buffer as both).
    fn overlap(ctx: *anyopaque) bool {
        const t: *Rows = @ptrCast(@alignCast(ctx));
        return inPlace(t.v.kind);
    }

    pub fn depth(t: *const Rows) usize {
        return spec(t.v).depth(t.shape);
    }

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = t.v.kind;
        if (shape.rows == 0 or shape.n == 0 or (k == .rope and shape.n % 128 != 0) or (k == .router_norm and shape.n != 128)) return false;
        if (fill == .stress and shape.n % 2 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const rows = shape.rows;
        const n = shape.n;
        const total = rows * n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const ln = k == .layer_norm or k == .layer_norm2d;
        const pairs = k == .add or k == .silu_mul;
        const rotary = k == .rope or k == .router_norm;
        t.x = try a.alloc(u16, total);
        t.y = try a.alloc(u16, if (pairs) total else if (sums(k)) n else 0);
        t.z = try a.alloc(u16, if (ln) n else 0);
        t.pos = try a.alloc(u32, if (rotary) 2 * rows else 0);
        const span: [2]f32 = switch (k) {
            .layer_norm, .layer_norm2d => .{ -2.25, 3.75 },
            .gelu => .{ -6, 6 },
            .quick_gelu => .{ -8, 8 },
            .silu_mul => .{ -20, 12 },
            else => .{ -4, 4 },
        };
        for (t.x) |*v| v.* = uni(rng, span[0], span[1]);
        for (t.y) |*v| v.* = if (sums(k)) uni(rng, -1.5, 1.5) else uni(rng, -2, 2);
        for (t.z) |*v| v.* = uni(rng, -0.5, 0.5);
        for (t.pos, 0..) |*v, i| v.* = rng.uintLessThan(u32, if (i < rows) cfg.max_context else 64);
        // a router weight past the clamp: (1 + w) times the normed value leaves +-1e4
        if (k == .router_norm and n > 5) t.y[5] = c.toBf16(2000);
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 and overflow fp16 in any order; mirrored halves (mean 0)
            for (t.x) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(24 + rng.uintLessThan(u32, 17))) * (if (rng.boolean()) @as(f32, 1) else -1));
            if (ln) for (0..rows) |r| for (0..n / 2) |j| {
                t.x[r * n + n / 2 + j] = t.x[r * n + j] ^ 0x8000;
            };
            for (t.y) |*v| v.* = c.toBf16(if (k == .router_norm) 0 else 1);
            @memset(t.z, 0);
        }
        if (fill == .nonfinite) {
            // a row each, one row holding both infinities, in the first operand; then the second's and the bias's
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 3, 5 } };
            const vals = [_]f32{ planted[0], planted[1], planted[2], planted[1], planted[2] };
            for (at, vals) |p, v| if (p[0] < rows and p[1] < n) {
                t.x[p[0] * n + p[1]] = c.toBf16(v);
            };
            if (rows > 4 and n > 7 and k != .router_norm) {
                if (t.y.len == total) t.y[4 * n + 6] = c.toBf16(planted[0]) else if (t.y.len == n) t.y[6] = c.toBf16(planted[0]);
                if (t.z.len > 0) t.z[7] = c.toBf16(planted[1]);
            }
        }
        // the buffers in the order the kernel takes them, as cuda_vision.zig and cuda_decoder.zig bind them
        t.io = try t.rig.output(total * @as(usize, if (k == .router_norm) 4 else 2));
        const b = std.mem.sliceAsBytes;
        const ins: []const []const u8 = switch (k) {
            .add => &.{ &.{}, b(t.y), &.{} },
            .gelu, .quick_gelu => &.{&.{}},
            .rope => &.{ &.{}, b(t.pos[0..rows]), b(t.pos[rows..]), b(&t.inv) },
            .router_norm => &.{ b(t.x), b(t.y), b(t.pos[0..rows]), b(t.pos[rows..]), b(&t.inv), &.{} },
            .rms_norm, .silu_mul => &.{ b(t.x), b(t.y), &.{} },
            .layer_norm, .layer_norm2d => &.{ b(t.x), b(t.y), b(t.z), &.{} },
        };
        // an empty entry is the output: the last buffer, or the operand an in-place kernel overwrites
        for (ins, 0..) |data, i| t.refs[i] = if (data.len == 0) t.io else try t.rig.input(data);
        return true;
    }

    /// One launch as the family's cuda_vision.zig and cuda_decoder.zig make it, waited for.
    pub fn run(t: *Rows) !void {
        const rows = t.shape.rows;
        const n = t.shape.n;
        const total = rows * n;
        const s = t.rig.s;
        const k = &t.k;
        const r = t.refs;
        if (inPlace(t.v.kind)) {
            try s.synchronize();
            const io: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.io, .len = total * 2 };
            try io.upload(0, std.mem.sliceAsBytes(t.x));
        }
        const flat: [3]usize = .{ uocr.kernels.blocks(total), 1, 1 };
        const len: i64 = @intCast(total);
        switch (t.v.kind) {
            .add => try k.run("add", s, flat, 256, .{ r[0], r[1], r[2], len }),
            .gelu => try k.run("gelu", s, flat, 256, .{ r[0], len }),
            .quick_gelu => try k.run("quick_gelu", s, flat, 256, .{ r[0], len }),
            .silu_mul => try k.run("silu_mul", s, flat, 256, .{ r[0], r[1], r[2], len }),
            .layer_norm => try k.run("layer_norm", s, .{ rows, 1, 1 }, 256, .{ r[0], r[1], r[2], r[3], n, t.v.eps }),
            .layer_norm2d => try k.run("layer_norm2d", s, .{ rows, 1, 1 }, 256, .{ r[0], r[1], r[2], r[3], n, t.v.eps }),
            .rms_norm => try k.run("rms_norm", s, .{ rows, 1, 1 }, 256, .{ r[0], r[1], r[2], n, t.v.eps }),
            .rope => try k.run("rope", s, .{ rows, n / 128, 1 }, 128, .{ r[0], r[1], r[2], r[3], n }),
            .router_norm => try k.run("router_norm", s, .{ rows, 1, 1 }, 128, .{ r[0], r[1], r[2], r[3], r[4], r[5] }),
        }
        try s.synchronize();
    }

    pub fn count(t: *Rows) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Every output's value and spread (freed by the caller).
    fn eval(t: *Rows, sim: ref.Sim) !struct { want: []f64, rg: []R } {
        const a = t.rig.gpu.gpa;
        const n = t.shape.n;
        const want = try a.alloc(f64, t.count() + 3 * n);
        errdefer a.free(want);
        const rg = try a.alloc(R, t.count() + n);
        for (0..t.shape.rows) |r| ref.row(t, r, want[r * n ..][0..n], rg[r * n ..][0..n], want[t.count()..], rg[t.count()..], sim);
        return .{ .want = want, .rg = rg };
    }
    fn done(t: *Rows, e: anytype) void {
        t.rig.gpu.gpa.free(e.want);
        t.rig.gpu.gpa.free(e.rg);
    }

    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        const e = try t.eval(null);
        defer t.done(e);
        @memcpy(want, e.want[0..want.len]);
        @memset(mag, 0);
    }

    pub fn intervals(t: *Rows, low: []f64, high: []f64) !void {
        const e = try t.eval(null);
        defer t.done(e);
        for (low, high, e.rg[0..low.len]) |*l, *h, o| {
            l.* = o.lo;
            h.* = o.hi;
        }
    }

    /// The farther end of each output's spread from its float64 value.
    pub fn bounds(t: *Rows, want: []const f64, _: []const f64, allow: []f64) !void {
        const e = try t.eval(null);
        defer t.done(e);
        for (allow, want, e.rg[0..want.len]) |*l, w, o| l.* = if (std.math.isFinite(w)) @max(@abs(w - o.lo), @abs(o.hi - w)) else 0;
    }

    pub fn simulate(t: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        const e = try t.eval(.{ .acc = acc, .order = order });
        defer t.done(e);
        @memcpy(out, e.want[0..out.len]);
    }

    /// router_norm's stated rule: every output finite, within its spread about the sanitized float64 value.
    pub fn check(t: *Rows, _: g.Gate) !?[]const u8 {
        const e = try t.eval(null);
        defer t.done(e);
        const out = try t.rig.outputBytes();
        for (0..t.count()) |i| {
            const got = c.read(.f32, out, i);
            const o = e.rg[i];
            if (!std.math.isFinite(got) or got < o.lo or got > o.hi)
                return try std.fmt.bufPrint(&t.msg, "output {d} is {e:.6} where the rule gives {e:.6} in [{e:.6}, {e:.6}]", .{ i, got, e.want[i], o.lo, o.hi });
        }
        return null;
    }
};

pub const Group = struct {
    gpa: std.mem.Allocator = undefined,
    tests: std.ArrayList(*Rows) = .empty,

    /// The group's tests, appended to `list` (each control after the kernel it breaks).
    pub fn init(group: *Group, ctx: u.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        group.gpa = ctx.gpu.gpa;
        for (cases) |v| {
            var k = ctx.k;
            if (v.ctl) |sym| {
                const f = try ctx.ctl.function(sym);
                k = switch (v.kind) {
                    inline else => |tag| ctx.k.with(@tagName(tag), f),
                };
            }
            const x = try group.gpa.create(Rows);
            x.* = .{ .rig = ctx.rig, .k = k, .v = v };
            group.tests.append(group.gpa, x) catch |err| {
                group.gpa.destroy(x);
                return err;
            };
            try list.append(a, x.kernel());
        }
    }

    pub fn deinit(group: *Group) void {
        for (group.tests.items) |x| {
            x.deinit();
            group.gpa.destroy(x);
        }
        group.tests.deinit(group.gpa);
    }
};
