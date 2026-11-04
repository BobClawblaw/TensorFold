//! Unlimited-OCR-2's CUDA moves contract, part two: SAM's rel_pos, the CLIP | SAM concat and CLIP's embeddings.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_uocr_moves_kit.zig");

const c = kit.c;
const g = kit.g;
const Kernels = kit.Kernels;
const blocks = kit.blocks;
const planted = kit.planted;
const bits = kit.bits;
const val = kit.val;
const bf = kit.bf;
const release = kit.release;
const upload = kit.upload;
const sam_width = kit.sam_width;
const sam_heads = kit.sam_heads;
const sam_head_dim = kit.sam_head_dim;
const sam_window = kit.sam_window;
const clip_width = kit.clip_width;

// --- tf_uocr_rel_pos: rel_h[h, g, q, kh] = bf16(q . Rh[qh - kh + side - 1]), rel_w likewise over kw; fp32 sums ---

fn sixtyFour(_: c.Shape) usize {
    return sam_head_dim;
}

pub const RelPos = struct {
    rig: *rig_.Rig,
    k: Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    qkv: []u16 = &.{},
    th: []u16 = &.{},
    tw: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    outs: [2]u64 = @splat(0),

    /// Rows: windows (1 global, 25 or 9); n: side (14, or 64/40 global); k: dots' head width. Checks rel_h, rel_w.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = sixtyFour,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 16 }, .{} },
            .served = &.{ .{ .rows = 1, .n = 40, .k = sam_head_dim }, .{ .rows = 1, .n = 64, .k = sam_head_dim }, .{ .rows = 9, .n = sam_window, .k = sam_head_dim }, .{ .rows = 25, .n = sam_window, .k = sam_head_dim } },
        };
    }
    pub fn deinit(t: *RelPos) void {
        release(t.rig.gpu.gpa, t, .{ "qkv", "th", "tw" });
    }
    pub fn kernel(t: *RelPos) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(RelPos);

    fn len(t: *const RelPos) usize {
        return t.shape.n * t.shape.n;
    }

    pub fn prepare(t: *RelPos, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != sam_head_dim) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        release(a, t, .{ "qkv", "th", "tw" });
        t.shape = shape;
        const side = shape.n;
        const qrows = shape.rows * t.len();
        const trows = 2 * side - 1;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.qkv = try a.alloc(u16, qrows * 3 * sam_width);
        t.th = try a.alloc(u16, trows * sam_head_dim);
        t.tw = try a.alloc(u16, trows * sam_head_dim);
        if (fill == .stress) {
            // integer products, exact fp32 sums; second half = first rotated by 7, q negated: every dot is 0
            const half = sam_head_dim / 2;
            @memset(t.qkv, 0);
            for (0..qrows) |r| for (0..sam_heads) |h| {
                const q = t.qkv[r * 3 * sam_width + h * sam_head_dim ..][0..sam_head_dim];
                for (0..half) |d| q[d] = bits(@floatFromInt(128 + rng.uintLessThan(u32, 129)));
                for (0..half) |d| q[half + d] = bits(-c.fromBf16(q[(d + 7) % half]));
            };
            for ([_][]u16{ t.th, t.tw }) |tab| for (0..trows) |r| {
                const row = tab[r * sam_head_dim ..][0..sam_head_dim];
                for (0..half) |d| row[d] = bits(@floatFromInt(128 + rng.uintLessThan(u32, 129)));
                for (0..half) |d| row[half + d] = row[(d + 7) % half];
            };
        } else {
            for (t.qkv) |*v| v.* = kit.value(rng, 1);
            for (t.th) |*v| v.* = kit.value(rng, 1);
            for (t.tw) |*v| v.* = kit.value(rng, 1);
        }
        if (fill == .nonfinite) {
            // q rows 0, 1 and 2 (heads 0, 1, 2), the last holding both infinities, and one row of Rh
            const row = 3 * sam_width;
            if (qrows > 2) {
                t.qkv[0 * row + 0 * sam_head_dim + 5] = bits(planted[0]);
                t.qkv[1 * row + 1 * sam_head_dim + 7] = bits(planted[1]);
                t.qkv[2 * row + 2 * sam_head_dim + 3] = bits(planted[1]);
                t.qkv[2 * row + 2 * sam_head_dim + 4] = bits(planted[2]);
            }
            t.th[(trows - 1) * sam_head_dim + 9] = bits(planted[2]);
        }
        t.ins[0] = try upload(t.rig, t.qkv);
        t.ins[1] = try upload(t.rig, t.th);
        t.ins[2] = try upload(t.rig, t.tw);
        t.outs[0] = try t.rig.output(t.each() * 2);
        t.outs[1] = try t.rig.output(t.each() * 2);
        return true;
    }

    pub fn run(t: *RelPos) !void {
        const side = t.shape.n;
        const groups = t.shape.rows;
        // as the encoder launches it: a block a query row and head, 2 side threads rounded up to whole warps
        const threads: u32 = @intCast(std.mem.alignForward(usize, 2 * side, 32));
        const args = .{ t.ins[0], t.ins[1], t.ins[2], t.outs[0], t.outs[1], groups, side, sam_width };
        try t.k.run("rel_pos", t.rig.s, .{ groups * t.len(), sam_heads, 1 }, threads, args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *RelPos) usize {
        return 2 * t.each();
    }
    /// Outputs in each of rel_h and rel_w.
    fn each(t: *const RelPos) usize {
        return sam_heads * t.shape.rows * t.len() * t.shape.n;
    }

    /// Output i's 64 products (exact in fp32: bf16 times bf16).
    fn terms(t: *const RelPos, i: usize, out: *[sam_head_dim]f64) void {
        const side = t.shape.n;
        const groups = t.shape.rows;
        const w = i >= t.each();
        var o = i % t.each();
        const kk = o % side;
        o /= side;
        const qi = o % t.len();
        o /= t.len();
        const grp = o % groups;
        const h = o / groups;
        const pos = if (w) qi % side else qi / side;
        const tab = if (w) t.tw else t.th;
        const q = t.qkv[(grp * t.len() + qi) * 3 * sam_width + h * sam_head_dim ..][0..sam_head_dim];
        const r = tab[(pos + side - 1 - kk) * sam_head_dim ..][0..sam_head_dim];
        for (out, q, r) |*x, a, b| x.* = val(a) * val(b);
    }

    pub fn reference(t: *RelPos, want: []f64, mag: []f64) !void {
        var buf: [sam_head_dim]f64 = undefined;
        for (want, mag, 0..) |*o, *m, i| {
            t.terms(i, &buf);
            var s: f64 = 0;
            var a: f64 = 0;
            for (buf) |x| {
                s += x;
                a += @abs(x);
            }
            o.* = s;
            m.* = if (std.math.isFinite(a)) a else 0;
        }
    }

    pub fn simulate(t: *RelPos, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [sam_head_dim]f64 = undefined;
        for (out, 0..) |*o, i| {
            t.terms(i, &buf);
            o.* = bf(core.contract.rows.sumIn(acc, order, &buf));
        }
    }
};

// --- tf_uocr_concat: y [rows, left + right] = [a row | b row] -------------------------------------------------------

pub const Concat = struct {
    rig: *rig_.Rig,
    k: Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    l: []u16 = &.{},
    r: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: the view's tokens (16^2 or 10^2); n: CLIP's width (left); k: SAM's net3 width (right).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = kit.noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256 }, .{ .size = 256 } },
            .served = &.{ .{ .rows = 256, .n = clip_width, .k = 1024 }, .{ .rows = 100, .n = clip_width, .k = 1024 } },
        };
    }
    pub fn deinit(t: *Concat) void {
        release(t.rig.gpu.gpa, t, .{ "l", "r" });
    }
    pub fn kernel(t: *Concat) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Concat);

    pub fn prepare(t: *Concat, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        release(a, t, .{ "l", "r" });
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.l = try a.alloc(u16, shape.rows * shape.n);
        t.r = try a.alloc(u16, shape.rows * shape.k);
        for (t.l) |*v| v.* = kit.finiteBits(rng);
        for (t.r) |*v| v.* = kit.finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 3) {
            t.l[0 * shape.n + 1] = bits(planted[0]);
            t.r[1 * shape.k + 2 % shape.k] = bits(planted[1]);
            t.l[2 * shape.n + 3 % shape.n] = bits(planted[2]);
            t.l[3 * shape.n + shape.n - 1] = bits(planted[1]);
            t.r[3 * shape.k] = bits(planted[2]);
        }
        t.ins[0] = try upload(t.rig, t.l);
        t.ins[1] = try upload(t.rig, t.r);
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Concat) !void {
        const s = t.shape;
        try t.k.run("concat", t.rig.s, .{ s.rows, blocks(s.n + s.k), 1 }, 256, .{ t.ins[0], t.ins[1], t.out, s.rows, s.n, s.k });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Concat) usize {
        return t.shape.rows * (t.shape.n + t.shape.k);
    }

    pub fn reference(t: *Concat, want: []f64, mag: []f64) !void {
        const w = t.shape.n + t.shape.k;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / w;
            const col = i % w;
            o.* = if (col < t.shape.n) val(t.l[row * t.shape.n + col]) else val(t.r[row * t.shape.k + col - t.shape.n]);
            m.* = 0;
        }
    }
};

// --- tf_uocr_clip_embed: y[0] = bf16(cls + pos[0]), y[r] = bf16(sam[r - 1] + pos[r]) -----------------------------

pub const ClipEmbed = struct {
    rig: *rig_.Rig,
    k: Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    cls: []u16 = &.{},
    sam: []u16 = &.{},
    pos: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: CLIP's tokens (the class row and 16^2 or 10^2 of SAM's); n: CLIP's width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = kit.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 8 }, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 257, .n = clip_width, .k = 1 }, .{ .rows = 101, .n = clip_width, .k = 1 } },
        };
    }
    pub fn deinit(t: *ClipEmbed) void {
        release(t.rig.gpu.gpa, t, .{ "cls", "sam", "pos" });
    }
    pub fn kernel(t: *ClipEmbed) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(ClipEmbed);

    pub fn prepare(t: *ClipEmbed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        release(a, t, .{ "cls", "sam", "pos" });
        t.shape = shape;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.cls = try a.alloc(u16, n);
        t.sam = try a.alloc(u16, (shape.rows - 1) * n);
        t.pos = try a.alloc(u16, shape.rows * n);
        for (t.cls) |*v| v.* = kit.value(rng, 4);
        for (t.sam) |*v| v.* = kit.value(rng, 4);
        for (t.pos) |*v| v.* = kit.value(rng, 1);
        if (fill == .nonfinite and shape.rows > 3 and n > 9) {
            t.cls[2] = bits(planted[0]); // row 0
            t.sam[0 * n + 5] = bits(planted[1]); // row 1
            t.pos[2 * n + 7] = bits(planted[2]); // row 2
            t.sam[2 * n + 9] = bits(planted[1]); // row 3, both infinities
            t.pos[3 * n + 9] = bits(planted[2]);
        }
        t.ins[0] = try upload(t.rig, t.cls);
        t.ins[1] = try upload(t.rig, t.sam);
        t.ins[2] = try upload(t.rig, t.pos);
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *ClipEmbed) !void {
        const s = t.shape;
        try t.k.run("clip_embed", t.rig.s, .{ s.rows, blocks(s.n), 1 }, 256, .{ t.ins[0], t.ins[1], t.ins[2], t.out, s.n });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *ClipEmbed) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *ClipEmbed, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / n;
            const v = if (row == 0) val(t.cls[i]) else val(t.sam[i - n]);
            o.* = bf(v + val(t.pos[i]));
            m.* = 0;
        }
    }
};
