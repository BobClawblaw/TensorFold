//! uocr_decoder.cu's tf_uocr_attend under the contract (DASH: 12 routed 32-key blocks, then the 128-key window).
const std = @import("std");
const core = @import("core");
const uocr = @import("uocr_cuda");
const rig_ = @import("contract_rig.zig");
const at = @import("contract_uocr_attention.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const ar = core.contract.attention_range;
const cfg = uocr.config;

// --- tf_uocr_attend: a 256-thread block a (row, head), its selected blocks before the window, then the window ---

const block = cfg.block;
const local = cfg.local;
const reads = cfg.read_blocks;
const max_context = cfg.max_context;
const D = cfg.head_dim;
const H = cfg.hidden;

/// The kernel and 3 controls: a key past the row; the 12th slot ignored; a block's keys in the window counted twice.
pub const Case = enum { plain, past_row, slot_dropped, overlap_twice };
pub const cases = std.enums.values(Case);

/// A control's kernel in the controls image; null for the engine's own.
pub fn control(k: Case) ?[:0]const u8 {
    return switch (k) {
        .plain => null,
        .past_row => "tf_ctl_uocr_attention_past_row",
        .slot_dropped => "tf_ctl_uocr_attention_slot_dropped",
        .overlap_twice => "tf_ctl_uocr_attention_overlap_twice",
    };
}

fn name(k: Case) []const u8 {
    return switch (k) {
        .plain => "cuda uocr attention tf_uocr_attend DASH, 12 routed blocks of 32 and a 128-key window, 10 heads of 128",
        .past_row => "control: uocr attention tf_uocr_attend, a key past the row",
        .slot_dropped => "control: uocr attention tf_uocr_attend, the 12th selected block ignored",
        .overlap_twice => "control: uocr attention tf_uocr_attend, a selected block's keys inside the window counted again",
    };
}

/// Rows: query rows at consecutive positions; n: the read slots a row fills (the rest -1); k: the first row's position.
fn spec(k: Case) c.Spec {
    return .{
        .name = name(k),
        .accum = .f32,
        .depth = depth,
        .ops = at.div_roundings,
        .out = .bf16,
        // the window's edges: a first row at 0, 127, 128, 129
        .tiles = .{ .{}, .{}, .{ .size = local, .min = 0 } },
        .served = &.{
            // a short chunk whose 6 remote blocks leave 6 slots on blocks at or past the window (each ignored)
            .{ .rows = 8, .n = reads, .k = 300 },
            // decode at the trained context's last position, and mid-context
            .{ .rows = 1, .n = reads, .k = max_context - 1 },
            .{ .rows = 1, .n = reads, .k = 5000 },
            // a prompt's first chunk of 2,048 rows (fewer than 128 keys, then blocks overlapping the window, then 12)
            .{ .rows = 2048, .n = reads, .k = 0 },
            // a query near position 0; a chunk across the window's edge with 3 slots; rows reading no blocks
            .{ .rows = 5, .n = reads, .k = 0 },
            .{ .rows = 4, .n = 3, .k = 126 },
            .{ .rows = 64, .n = 0, .k = 1000 },
            // the blocks next to the window, where the last remote block overlaps it
            .{ .rows = 33, .n = reads, .k = 160 },
        },
        .sums = .{ .exact = true },
    };
}

fn depth(_: c.Shape) usize {
    return (reads * block + local) / 8 + 10;
}

/// tf_uocr_attend over n keys: a lane's 4 products and adds, the 5-level tree, * scale, expf, fp32 P; 8 warp chains.
fn form(n: usize) ar.Form {
    return .{
        .score = 4 + 5 + 2,
        .score_round = 1,
        .fast_exp = false,
        .exp_ulps = at.exp_ulps,
        .chunk_exp = false,
        .p_rel = 0,
        // a warp's products and adds over every 8th key, then the 8 warps' partials in order
        .num = @floatFromInt((n + 7) / 8 + 1 + 8),
        .den = @floatFromInt((n + 7) / 8 + 8),
        .div = at.div_roundings,
    };
}

const scale: f32 = 1.0 / @sqrt(@as(f32, cfg.head_dim)); // cuda_decoder.zig's

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: *rig_.Rig,
    k: uocr.kernels.Kernels,
    case: Case,
    rows: usize = 0,
    n: usize = 0,
    p0: usize = 0,
    ctx: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    sel: []i32 = &.{},
    bias: []f32 = &.{},
    qpos: []u32 = &.{},
    point: []f64 = &.{},
    half: []f64 = &.{},
    dev: [6]u64 = undefined,
    out: u64 = 0,

    pub fn free(p: *Probe) void {
        for ([_][]u16{ p.q, p.kc, p.vc }) |x| p.a.free(x);
        for ([_][]f64{ p.point, p.half }) |x| p.a.free(x);
        p.a.free(p.sel);
        p.a.free(p.bias);
        p.a.free(p.qpos);
        p.q = &.{};
        p.kc = &.{};
        p.vc = &.{};
        p.point = &.{};
        p.half = &.{};
        p.sel = &.{};
        p.bias = &.{};
        p.qpos = &.{};
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.case), .ctx = p, .vt = &vt, .control = if (p.case == .plain) null else .shapes };
    }
    const vt = g.hooks(Probe);

    fn low(pos: usize) usize {
        return (pos + 1) -| local;
    }

    /// Row r's keys in the kernel's order: each slot's block up to the window (its bias), then the window (no bias).
    fn keys(p: *const Probe, r: usize, list: *std.ArrayList(usize), bs: *std.ArrayList(f64)) !void {
        list.clearRetainingCapacity();
        bs.clearRetainingCapacity();
        const pos = p.p0 + r;
        const lo = low(pos);
        for (0..reads) |s| {
            const cb = p.sel[r * reads + s];
            if (cb < 0) continue;
            const first: usize = @as(usize, @intCast(cb)) * block;
            const last = @max(first, @min(first + block, lo));
            for (first..last) |j| {
                try list.append(p.a, j);
                try bs.append(p.a, p.bias[r * reads + s]);
            }
        }
        for (lo..pos + 1) |j| {
            try list.append(p.a, j);
            try bs.append(p.a, 0);
        }
    }

    /// Each row's slots as tf_uocr_route fills them (distinct remote blocks, then ignored, rest -1); stress: bias 0.
    fn route(p: *Probe, rng: std.Random) void {
        @memset(p.sel, -1);
        for (p.bias) |*b| b.* = rng.float(f32) * 5 - 4;
        for (0..p.rows) |r| {
            if (p.fill == .stress) {
                for (0..reads) |s| {
                    p.sel[r * reads + s] = @intCast(4 * (s / 2) + 2 * (s % 2));
                    p.bias[r * reads + s] = 0;
                }
                continue;
            }
            const pos = p.p0 + r;
            const last_remote: i64 = @as(i64, @intCast(pos / block)) - 4;
            const remote: usize = @intCast(@max(0, last_remote + 1));
            var picks: [reads]i32 = undefined;
            var got: usize = 0;
            if (p.n > 0 and remote > 0) {
                picks[0] = @intCast(last_remote);
                got = 1;
            }
            while (got < p.n and got < remote) {
                const cand: i32 = @intCast(rng.uintLessThan(usize, remote));
                if (std.mem.indexOfScalar(i32, picks[0..got], cand) == null) {
                    picks[got] = cand;
                    got += 1;
                }
            }
            var next: i32 = @intCast(@max(0, last_remote + 1));
            while (got < p.n) : (next += 1) {
                picks[got] = next;
                got += 1;
            }
            var slots: [reads]u8 = undefined;
            for (&slots, 0..) |*s, i| s.* = @intCast(i);
            rng.shuffle(u8, &slots);
            for (picks[0..got], slots[0..got]) |cb, s| p.sel[r * reads + s] = cb;
        }
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n > reads or shape.k + shape.rows > max_context) return false;
        if (fill == .stress and (shape.n != reads or shape.k < 1024)) return false;
        p.rig.reset();
        p.free();
        p.rows = shape.rows;
        p.n = shape.n;
        p.p0 = shape.k;
        p.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.sel = try p.a.alloc(i32, p.rows * reads);
        p.bias = try p.a.alloc(f32, p.rows * reads);
        p.qpos = try p.a.alloc(u32, p.rows);
        for (p.qpos, 0..) |*x, r| x.* = @intCast(p.p0 + r);
        p.route(rng);
        // the cache holds a key past the chunk's end and reaches every selected block's end; unread keys hold NaN
        p.ctx = p.p0 + p.rows + 1;
        for (p.sel) |cb| if (cb >= 0) {
            p.ctx = @max(p.ctx, (@as(usize, @intCast(cb)) + 1) * block);
        };
        p.q = try p.a.alloc(u16, p.rows * H);
        p.kc = try p.a.alloc(u16, p.ctx * H);
        p.vc = try p.a.alloc(u16, p.ctx * H);
        p.point = try p.a.alloc(f64, p.count());
        p.half = try p.a.alloc(f64, p.count());
        @memset(p.kc, at.nan16);
        @memset(p.vc, at.nan16);
        at.fillRandom(rng, p.q, 3);
        const base: []f32 = if (fill == .stress) try at.stressBase(p.a, rng, H) else &.{};
        defer p.a.free(base);
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(p.a);
        var bs: std.ArrayList(f64) = .empty;
        defer bs.deinit(p.a);
        const read = try p.a.alloc(bool, p.ctx);
        defer p.a.free(read);
        @memset(read, false);
        for (0..p.rows) |r| {
            try p.keys(r, &list, &bs);
            for (list.items) |j| read[j] = true;
        }
        for (read, 0..) |seen, j| if (seen) {
            if (fill == .stress) {
                @memset(p.kc[j * H ..][0..H], 0);
                for (0..H) |col| p.vc[j * H + col] = c.toBf16(at.stressValue(base, H, j, col));
            } else {
                at.fillRandom(rng, p.kc[j * H ..][0..H], 3);
                at.fillRandom(rng, p.vc[j * H ..][0..H], 2);
            }
        };
        if (fill == .stress) @memset(p.q, 0);
        if (fill == .nonfinite) p.plant();
        p.dev[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.dev[1] = try p.rig.input(std.mem.sliceAsBytes(p.kc));
        p.dev[2] = try p.rig.input(std.mem.sliceAsBytes(p.vc));
        p.dev[3] = try p.rig.input(std.mem.sliceAsBytes(p.sel));
        p.dev[4] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
        p.dev[5] = try p.rig.input(std.mem.sliceAsBytes(p.qpos));
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// Gate 1's plants (rows 0 to 7): NaN/Inf in q, keys, values and bias, an all -Inf row; unread keys hold NaN.
    fn plant(p: *Probe) void {
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        if (p.rows < 7) return;
        p.q[0 * H + 1 * D + 3] = c.toBf16(nan);
        p.q[1 * H + 0 * D + 5] = c.toBf16(inf);
        p.q[2 * H + 2 * D + 5] = c.toBf16(inf);
        p.q[2 * H + 2 * D + 9] = c.toBf16(-inf);
        p.kc[(p.p0 + 3) * H + 3 * D] = c.toBf16(nan);
        p.q[3 * H + 6 * D] = c.toBf16(-inf);
        for (0..p.ctx) |j| {
            const k = &p.kc[j * H + 6 * D];
            if (k.* != at.nan16) k.* = c.toBf16(@abs(c.fromBf16(k.*)) + 0.5);
        }
        p.vc[(p.p0 + 4) * H + 4 * D + 7] = c.toBf16(inf);
        p.vc[(p.p0 + 4) * H + 4 * D + 8] = c.toBf16(nan);
        const last_remote = @as(i64, @intCast(p.p0 / block)) - 4;
        if (last_remote >= 0) p.vc[@as(usize, @intCast(last_remote)) * block * H + 5 * D + 9] = c.toBf16(-inf);
        for ([_]usize{ 5, 6 }, [_]f32{ -inf, nan }) |r, v| for (0..reads) |s| {
            if (p.sel[r * reads + s] >= 0 and p.sel[r * reads + s] <= last_remote) {
                p.bias[r * reads + s] = v;
                break;
            }
        };
    }

    /// As cuda_decoder.zig: a 256-thread block a (row, head).
    pub fn run(p: *Probe) !void {
        try p.k.run("attend", p.rig.s, .{ p.rows, cfg.heads, 1 }, 256, .{ p.dev[0], p.dev[1], p.dev[2], p.dev[3], p.dev[4], p.dev[5], p.out, H, scale, local });
        try p.rig.s.synchronize();
    }

    pub fn count(p: *const Probe) usize {
        return p.rows * H;
    }

    /// Each row's heads over its keys, a block's bias folded into the dot as a 129th term (bias / scale against 1).
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const most = reads * block + local;
        const dk = D + 1;
        const buf = try p.a.alloc(f64, 2 * most * dk + 2 * most + 3 * dk);
        defer p.a.free(buf);
        const ks = buf[0 .. most * dk];
        const vs = buf[most * dk .. 2 * most * dk];
        const w = buf[2 * most * dk ..][0..most];
        const tau = buf[2 * most * dk + most ..][0..most];
        const qv = buf[2 * most * dk + 2 * most ..][0..dk];
        const pt = buf[2 * most * dk + 2 * most + dk ..][0..dk];
        const hf = buf[2 * most * dk + 2 * most + 2 * dk ..][0..dk];
        const sc: f64 = scale;
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(p.a);
        var bs: std.ArrayList(f64) = .empty;
        defer bs.deinit(p.a);
        for (0..p.rows) |r| {
            try p.keys(r, &list, &bs);
            const n = list.items.len;
            const f = form(n);
            for (0..cfg.heads) |h| {
                for (list.items, bs.items, 0..) |j, b, i| {
                    for (0..D) |e| {
                        ks[i * dk + e] = c.fromBf16(p.kc[j * H + h * D + e]);
                        vs[i * dk + e] = c.fromBf16(p.vc[j * H + h * D + e]);
                    }
                    ks[i * dk + D] = b / sc;
                    vs[i * dk + D] = 0;
                }
                for (0..D) |e| qv[e] = c.fromBf16(p.q[r * H + h * D + e]);
                qv[D] = 1;
                ar.head(qv, ks[0 .. n * dk], vs[0 .. n * dk], n, dk, sc, f, w, tau, pt, hf);
                const o = r * H + h * D;
                @memcpy(want[o..][0..D], pt[0..D]);
                @memcpy(p.half[o..][0..D], hf[0..D]);
            }
        }
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.half) |*l, w, h| l.* = if (p.fill == .stress) c.boundExact(spec(p.case), w) else ar.allowBf16(w, h, 0, 0);
    }

    pub fn intervals(p: *Probe, lo: []f64, hi: []f64) !void {
        for (lo, hi, p.point, p.half) |*l, *u, pt, h| {
            const e = if (p.fill == .stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(pt, h, 0, 0);
            l.* = e[0];
            u.* = e[1];
        }
    }

    /// Gate 4: every weight 1, so each output is its keys' values summed in `acc` in the kernel's key order, / n.
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(p.a);
        var bs: std.ArrayList(f64) = .empty;
        defer bs.deinit(p.a);
        var terms: [reads * block + local]f64 = undefined;
        for (0..p.rows) |r| {
            try p.keys(r, &list, &bs);
            const n = list.items.len;
            for (0..H) |col| {
                for (list.items, terms[0..n]) |j, *x| x.* = c.fromBf16(p.vc[j * H + col]);
                out[r * H + col] = at.bf(core.contract.rows.sumIn(acc, order, terms[0..n]) / @as(f64, @floatFromInt(n)));
            }
        }
    }
};
