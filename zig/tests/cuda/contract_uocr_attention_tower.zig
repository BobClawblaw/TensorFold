//! uocr_vision.cu's tf_uocr_attention (SAM windows and global blocks with rel-pos bias, CLIP) under the contract.
const std = @import("std");
const core = @import("core");
const uocr = @import("uocr_cuda");
const rig_ = @import("contract_rig.zig");
const at = @import("contract_uocr_attention.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const ar = core.contract.attention_range;
const cfg = uocr.config;

// --- tf_uocr_attention: a warp a (row, head), softmax(q k^T / 8 + bf16(rel_h + rel_w)) v over `len` rows ---

pub const Tower = enum { sam_window, sam_global, clip };
/// The engine's kernel, and two planted controls: rel_w read one column over (SAM windows), P x V without its last key.
pub const Case = enum { plain, rel_w_shift, last_key };

pub const entries = [_]struct { Tower, Case }{ .{ .sam_window, .plain }, .{ .sam_global, .plain }, .{ .clip, .plain }, .{ .sam_window, .rel_w_shift }, .{ .clip, .last_key } };

/// A control's kernel in the controls image; null for the engine's own.
pub fn control(k: Case) ?[:0]const u8 {
    return switch (k) {
        .plain => null,
        .rel_w_shift => "tf_ctl_uocr_attention_rel_w_shift",
        .last_key => "tf_ctl_uocr_attention_last_key",
    };
}

fn name(t: Tower, k: Case) []const u8 {
    return switch (k) {
        .plain => switch (t) {
            .sam_window => "cuda uocr attention tf_uocr_attention sam 14x14 windows, 12 heads of 64, relative-position bias",
            .sam_global => "cuda uocr attention tf_uocr_attention sam global, 12 heads of 64, relative-position bias",
            .clip => "cuda uocr attention tf_uocr_attention clip, 16 heads of 64",
        },
        .rel_w_shift => "control: uocr attention tf_uocr_attention sam windows, rel_w read one column over",
        .last_key => "control: uocr attention tf_uocr_attention clip, P x V without the last key",
    };
}

fn widthOf(t: Tower) usize {
    return if (t == .clip) cfg.clip_width else cfg.sam_width;
}

fn spec(t: Tower, k: Case) c.Spec {
    return .{
        .name = name(t, k),
        .accum = .f32,
        .depth = depth,
        .ops = at.div_roundings,
        .out = .bf16,
        // CLIP's keys at the kernel's 64-key tile edges (any group length runs; SAM's are squares of its side)
        .tiles = if (t == .clip) .{ .{}, .{}, .{ .size = 64, .min = 1 } } else .{ .{}, .{}, .{} },
        // rows: query rows checked from row 0 (groups of k rows, each launched whole); n: the width; k: keys a group
        .served = switch (t) {
            // a 1024 view's 25 windows (64 padded to 70), a 640 tile's 9 (40 padded to 42)
            .sam_window => &.{ .{ .rows = 25 * 196, .n = cfg.sam_width, .k = 196 }, .{ .rows = 9 * 196, .n = cfg.sam_width, .k = 196 } },
            // a 1024 view's 64 x 64 and a 640 tile's 40 x 40: each one's first query rows
            .sam_global => &.{ .{ .rows = 64, .n = cfg.sam_width, .k = 4096 }, .{ .rows = 200, .n = cfg.sam_width, .k = 1600 } },
            // a 1024 view's 257 rows and a 640 tile's 101
            .clip => &.{ .{ .rows = 257, .n = cfg.clip_width, .k = 257 }, .{ .rows = 101, .n = cfg.clip_width, .k = 101 } },
        },
        .sums = .{ .exact = true },
    };
}

/// A lane's P x V chain over every key; the bounds hook replaces the generic bound.
fn depth(s: c.Shape) usize {
    return s.k + 1;
}

/// tf_uocr_attention over n keys: 64 products and adds (--fmad=false) and the bias add, expf, bf16 P, lane chains.
fn form(n: usize) ar.Form {
    return .{
        .score = 64 + 1,
        .score_round = 1,
        .fast_exp = false,
        .exp_ulps = at.exp_ulps,
        .chunk_exp = false,
        .p_rel = 0x1p-8,
        // P x V: a lane's products and adds over every key; l: a lane's keys, then the warp's 5-level tree
        .num = @floatFromInt(n + 1),
        .den = @floatFromInt(n / 32 + 7),
        .div = at.div_roundings,
    };
}

const scale: f32 = 1.0 / @sqrt(@as(f32, cfg.sam_head_dim)); // cuda_vision.zig's, 0.125 for both towers

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: *rig_.Rig,
    k: uocr.kernels.Kernels,
    tower: Tower,
    case: Case,
    rows: usize = 0,
    len: usize = 0,
    groups: usize = 0,
    side: usize = 0,
    heads: usize = 0,
    width: usize = 0,
    fill: g.Fill = .random,
    qkv: []u16 = &.{},
    rh: []u16 = &.{},
    rw: []u16 = &.{},
    point: []f64 = &.{},
    half: []f64 = &.{},
    dev: [3]u64 = undefined,
    out: u64 = 0,

    pub fn free(p: *Probe) void {
        for ([_][]u16{ p.qkv, p.rh, p.rw }) |x| p.a.free(x);
        for ([_][]f64{ p.point, p.half }) |x| p.a.free(x);
        p.qkv = &.{};
        p.rh = &.{};
        p.rw = &.{};
        p.point = &.{};
        p.half = &.{};
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.tower, p.case), .ctx = p, .vt = &vt, .control = if (p.case == .plain) null else .shapes };
    }

    /// The rig's hooks, but the output is the checked rows alone: the launch writes every row of its groups.
    const vt = blk: {
        var v = g.hooks(Probe);
        v.output = output;
        break :blk v;
    };

    fn output(ctx: *anyopaque) anyerror![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        return (try p.rig.outputBytes())[0 .. p.count() * 2];
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const width = widthOf(p.tower);
        if (shape.rows == 0 or shape.n != width or shape.k == 0 or shape.k > 4096) return false;
        const side: usize = if (p.tower == .clip) 0 else std.math.sqrt(shape.k);
        if (p.tower != .clip and side * side != shape.k) return false;
        p.rig.reset();
        p.free();
        p.rows = shape.rows;
        p.len = shape.k;
        p.groups = (shape.rows + shape.k - 1) / shape.k;
        p.side = side;
        p.width = width;
        p.heads = width / 64;
        p.fill = fill;
        const total = p.groups * p.len;
        const stride = 3 * width;
        p.qkv = try p.a.alloc(u16, total * stride);
        const rel = @max(1, p.heads * p.groups * p.len * side);
        p.rh = try p.a.alloc(u16, rel);
        p.rw = try p.a.alloc(u16, rel);
        p.point = try p.a.alloc(f64, p.count());
        p.half = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        at.fillRandom(rng, p.qkv, 3);
        for (0..total) |r| at.fillRandom(rng, p.qkv[r * stride + 2 * width ..][0..width], 2); // values in [-2, 2)
        at.fillRandom(rng, p.rh, 2);
        at.fillRandom(rng, p.rw, 2);
        switch (fill) {
            .random => {},
            .nonfinite => p.plant(),
            .stress => {
                // q and k zero, no bias: every weight exactly 1; the values in periods, a partial last period zero
                @memset(p.qkv, 0);
                @memset(p.rh, 0);
                @memset(p.rw, 0);
                const base = try at.stressBase(p.a, rng, width);
                defer p.a.free(base);
                for (0..total) |r| {
                    const j = r % p.len;
                    for (0..width) |col| p.qkv[r * stride + 2 * width + col] = c.toBf16(if (j < p.len / at.period * at.period) at.stressValue(base, width, j, col) else 0);
                }
            },
        }
        for ([_][]u16{ p.qkv, p.rh, p.rw }, &p.dev) |x, *d| d.* = try p.rig.input(std.mem.sliceAsBytes(x));
        p.out = try p.rig.output(total * width * 2);
        return true;
    }

    /// Gate 1's plants in group 0: NaN/Inf in q, k, v and SAM's bias, plus two rows whose every score is -Inf.
    fn plant(p: *Probe) void {
        const stride = 3 * p.width;
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        const put = struct {
            fn q(t: *Probe, row: usize, h: usize, e: usize, v: f32) void {
                t.qkv[row * 3 * t.width + h * 64 + e] = c.toBf16(v);
            }
        }.q;
        put(p, 0, 1, 3, nan);
        put(p, 1, 0, 5, inf);
        put(p, 2, 2, 5, inf);
        put(p, 2, 2, 9, -inf);
        p.qkv[4 * stride + p.width + 3 * 64] = c.toBf16(nan); // key 4's k, head 3
        put(p, 4, 9, 0, -inf);
        for (0..p.len) |j| {
            const k = &p.qkv[j * stride + p.width + 9 * 64];
            k.* = c.toBf16(@abs(c.fromBf16(k.*)) + 0.5);
        }
        if (p.len > 7) {
            p.qkv[6 * stride + 2 * p.width + 4 * 64 + 7] = c.toBf16(inf);
            p.qkv[6 * stride + 2 * p.width + 4 * 64 + 8] = c.toBf16(nan);
            p.qkv[7 * stride + 2 * p.width + 4 * 64 + 9] = c.toBf16(-inf);
        }
        if (p.side > 0) {
            const rel = struct {
                fn of(t: *Probe, h: usize, qi: usize) usize {
                    return (h * t.groups * t.len + qi) * t.side;
                }
            }.of;
            p.rh[rel(p, 5, 4)] = c.toBf16(nan);
            p.rw[rel(p, 6, 0) + 1] = c.toBf16(-inf);
            p.rw[rel(p, 7, 3) + 2] = c.toBf16(inf);
            @memset(p.rh[rel(p, 8, 1)..][0..p.side], c.toBf16(-inf));
        }
    }

    /// As cuda_vision.zig's attention(): 8 rows a 256-thread block, every row of every group, a block row a head.
    pub fn run(p: *Probe) !void {
        try p.k.run("attention", p.rig.s, .{ p.groups * ((p.len + 7) / 8), p.heads, 1 }, 256, .{ p.dev[0], p.dev[1], p.dev[2], p.out, p.groups, p.len, p.width, p.side, scale });
        try p.rig.s.synchronize();
    }

    pub fn count(p: *const Probe) usize {
        return p.rows * p.width;
    }

    /// Each row's heads over its group's keys, bias bf16(rel_h + rel_w) as a 65th dot term: float64 points and ranges.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const L = p.len;
        const dk = 65;
        const buf = try p.a.alloc(f64, 2 * L * dk + 2 * L + 3 * dk);
        defer p.a.free(buf);
        const ks = buf[0 .. L * dk];
        const vs = buf[L * dk .. 2 * L * dk];
        const w = buf[2 * L * dk ..][0..L];
        const tau = buf[2 * L * dk + L ..][0..L];
        const qv = buf[2 * L * dk + 2 * L ..][0..dk];
        const pt = buf[2 * L * dk + 2 * L + dk ..][0..dk];
        const hf = buf[2 * L * dk + 2 * L + 2 * dk ..][0..dk];
        const stride = 3 * p.width;
        const sc: f64 = scale;
        const f = form(L);
        for (0..p.groups) |grp| for (0..p.heads) |h| {
            for (0..L) |j| {
                const row = grp * L + j;
                for (0..64) |e| {
                    ks[j * dk + e] = c.fromBf16(p.qkv[row * stride + p.width + h * 64 + e]);
                    vs[j * dk + e] = c.fromBf16(p.qkv[row * stride + 2 * p.width + h * 64 + e]);
                }
                ks[j * dk + 64] = 0;
                vs[j * dk + 64] = 0;
            }
            for (grp * L..@min(p.rows, (grp + 1) * L)) |r| {
                const qi = r - grp * L;
                for (0..64) |e| qv[e] = c.fromBf16(p.qkv[r * stride + h * 64 + e]);
                qv[64] = 1;
                if (p.side > 0) {
                    const rel = ((h * p.groups + grp) * L + qi) * p.side;
                    for (0..L) |j| {
                        const b = at.bf(@as(f64, c.fromBf16(p.rh[rel + j / p.side])) + c.fromBf16(p.rw[rel + j % p.side]));
                        ks[j * dk + 64] = b / sc;
                    }
                }
                ar.head(qv, ks, vs, L, dk, sc, f, w, tau, pt, hf);
                const o = r * p.width + h * 64;
                @memcpy(want[o..][0..64], pt[0..64]);
                @memcpy(p.half[o..][0..64], hf[0..64]);
            }
        };
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    /// The range's allowance (attention_range.allowBf16); gate 4 allows the division and the rounding alone.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.half) |*l, w, h| l.* = if (p.fill == .stress) c.boundExact(spec(p.tower, p.case), w) else ar.allowBf16(w, h, 0, 0);
    }

    /// Each bf16 output within its range's rounded ends; gate 4's are open.
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        for (low, high, p.point, p.half) |*l, *hi, pt, h| {
            const e = if (p.fill == .stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(pt, h, 0, 0);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    /// Gate 4: every weight 1, so each output is its group's values summed in `acc` over the keys, then / len.
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.len);
        defer p.a.free(terms);
        const stride = 3 * p.width;
        for (0..p.rows) |r| for (0..p.width) |col| {
            const grp = r / p.len;
            for (terms, 0..) |*x, j| x.* = c.fromBf16(p.qkv[(grp * p.len + j) * stride + 2 * p.width + col]);
            out[r * p.width + col] = at.bf(core.contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(p.len)));
        };
    }
};
