//! Unlimited-OCR-2's CUDA moves contract, part one: shared helpers and the im2col and window cases.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const uocr = @import("uocr_cuda");
const rig_ = @import("contract_rig.zig");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
pub const Kernels = uocr.kernels.Kernels;
pub const blocks = uocr.kernels.blocks;

const cfg = uocr.config;
pub const hidden = cfg.hidden;
pub const topk = cfg.top_k;
pub const vocab = cfg.vocab;
pub const image_token = cfg.image_token;
pub const sam_width = cfg.sam_width;
pub const clip_width = cfg.clip_width;
pub const sam_window = cfg.sam_window;
pub const sam_heads = cfg.sam_heads;
pub const sam_head_dim = cfg.sam_head_dim;
/// A prompt chunk's rows at most, and its expert slots (top-6 a row).
pub const chunk = uocr.serve.prompt_rows;
pub const slots = chunk * topk;
pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 round it.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Any finite bf16 (every exponent, subnormals and both zeros): a copy must keep its bits.
pub fn finiteBits(rng: std.Random) u16 {
    while (true) {
        const v = rng.int(u16);
        if (std.math.isFinite(c.fromBf16(v))) return v;
    }
}

/// A bf16 value in (-scale, scale): two of them sum exactly in float64, so bf() of the sum is fp32's add then bf16.
pub fn value(rng: std.Random, scale: f32) u16 {
    return c.toBf16((rng.float(f32) * 2 - 1) * scale);
}

pub fn bits(v: f32) u16 {
    return c.toBf16(v);
}

pub fn val(h: u16) f64 {
    return c.fromBf16(h);
}

/// Distinct ids from [0, k) in random order (a shuffled prefix): an expert's token rows or slots.
pub fn distinct(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize) ![]u32 {
    const all = try a.alloc(u32, k);
    defer a.free(all);
    for (all, 0..) |*v, i| v.* = @intCast(i);
    for (0..n) |i| std.mem.swap(u32, &all[i], &all[i + rng.uintLessThan(usize, k - i)]);
    return a.dupe(u32, all[0..n]);
}

/// Frees each named slice field of `t` with `a` and empties it.
pub fn release(a: std.mem.Allocator, t: anytype, comptime fields: anytype) void {
    inline for (fields) |f| {
        a.free(@field(t, f));
        @field(t, f) = &.{};
    }
}

/// `bytes` from device address `from` into `to` on the rig's stream: an output's contents before a partial write.
pub fn preset(rig: *rig_.Rig, to: u64, from: u64, bytes: usize) !void {
    try (cuda.DeviceBuffer{ .d = rig.gpu.d, .ptr = to, .len = bytes }).copyFrom(0, from, bytes, rig.s.handle);
}

pub fn upload(rig: *rig_.Rig, x: anytype) !u64 {
    return rig.input(std.mem.sliceAsBytes(x));
}

// --- tf_uocr_im2col: convolution patches, row (oy, ox) of [out^2, C k k] in (c, ky, kx) order, zero outside ------

pub const Geo = struct {
    name: []const u8,
    k: u32,
    stride: u32,
    pad: u32,
    /// The input as [C, h, w] (pixels) or [h, w, C] (feature maps).
    chw: bool,
    served: []const c.Shape,
    tiles: [3]c.Tile,
};

/// Rows: output side; n: input channels; k: input side. The engine's 4 launches a view, at the 1024 view and 640 tile.
pub const geos = [_]Geo{
    .{
        .name = "cuda uocr moves im2col patch 16x16 stride 16 (pixels [3, h, w])",
        .k = 16,
        .stride = 16,
        .pad = 0,
        .chw = true,
        .served = &.{ .{ .rows = 64, .n = 3, .k = 1024 }, .{ .rows = 40, .n = 3, .k = 640 } },
        // input sides short of the 64-patch grid: the last patches' outside pixels are zero
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 1008, .min = 1000 } },
    },
    .{
        .name = "cuda uocr moves im2col neck 3x3 stride 1 pad 1 (256 channels)",
        .k = 3,
        .stride = 1,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 40, .n = 256, .k = 40 }, .{ .rows = 64, .n = 256, .k = 64 } },
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 32 } },
    },
    .{
        .name = "cuda uocr moves im2col net2 3x3 stride 2 pad 1 (256 channels)",
        .k = 3,
        .stride = 2,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 20, .n = 256, .k = 40 }, .{ .rows = 32, .n = 256, .k = 64 } },
        // odd input sides reach the bottom and right padding
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 32 } },
    },
    .{
        .name = "cuda uocr moves im2col net3 3x3 stride 2 pad 1 (512 channels)",
        .k = 3,
        .stride = 2,
        .pad = 1,
        .chw = false,
        .served = &.{ .{ .rows = 10, .n = 512, .k = 20 }, .{ .rows = 16, .n = 512, .k = 32 } },
        .tiles = .{ .{ .size = 8 }, .{ .size = 32 }, .{ .size = 16 } },
    },
};

pub const Im2col = struct {
    rig: rig_.Rig,
    k: Kernels,
    geo: Geo,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    pub fn spec(t: *const Im2col) c.Spec {
        return .{ .name = t.name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = t.geo.tiles, .served = t.geo.served };
    }
    pub fn deinit(t: *Im2col) void {
        release(t.rig.gpu.gpa, t, .{"x"});
        t.rig.deinit();
    }
    pub fn kernel(t: *Im2col) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Im2col);

    fn index(t: *const Im2col, y: usize, x: usize, ch: usize) usize {
        const side = t.shape.k;
        return if (t.geo.chw) (ch * side + y) * side + x else (y * side + x) * t.shape.n + ch;
    }

    pub fn prepare(t: *Im2col, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        release(t.rig.gpu.gpa, t, .{"x"});
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.rig.gpu.gpa.alloc(u16, shape.k * shape.k * shape.n);
        for (t.x) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite) {
            // a pixel each, one pixel row holding both infinities; each read by the first outputs' patches
            const at = [_][3]usize{ .{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 2, 2, 2 }, .{ 3, 3, 0 }, .{ 3, 4, 1 } };
            const v = [_]f32{ planted[0], planted[1], planted[2], planted[1], planted[2] };
            for (at, v) |p, x| if (p[0] < shape.k and p[1] < shape.k) {
                t.x[t.index(p[0], p[1], p[2] % shape.n)] = bits(x);
            };
        }
        t.in = try upload(&t.rig, t.x);
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    fn width(t: *const Im2col) usize {
        return t.shape.n * t.geo.k * t.geo.k;
    }

    pub fn run(t: *Im2col) !void {
        const side = t.shape.k;
        const out = t.shape.rows;
        const geo = t.geo;
        const args = .{ t.in, t.out, side, side, t.shape.n, @as(usize, geo.k), @as(usize, geo.stride), @as(usize, geo.pad), out, out, @as(usize, @intFromBool(geo.chw)) };
        try t.k.run("im2col", t.rig.s, .{ out * out, blocks(t.width()), 1 }, 256, args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Im2col) usize {
        return t.shape.rows * t.shape.rows * t.width();
    }

    pub fn reference(t: *Im2col, want: []f64, mag: []f64) !void {
        const w = t.width();
        const k = t.geo.k;
        const out = t.shape.rows;
        const side: isize = @intCast(t.shape.k);
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / w;
            const col = i % w;
            const ch = col / (k * k);
            const ky = (col / k) % k;
            const kx = col % k;
            const iy = @as(isize, @intCast((row / out) * t.geo.stride + ky)) - @as(isize, @intCast(t.geo.pad));
            const ix = @as(isize, @intCast((row % out) * t.geo.stride + kx)) - @as(isize, @intCast(t.geo.pad));
            const inside = iy >= 0 and ix >= 0 and iy < side and ix < side;
            o.* = if (inside) val(t.x[t.index(@intCast(iy), @intCast(ix), ch)]) else 0;
            m.* = 0;
        }
    }
};

// --- tf_uocr_window and tf_uocr_unwindow_add: SAM's 14x14 windows over a zero-padded grid, and back with the add ---

pub const Window = struct {
    rig: rig_.Rig,
    k: Kernels,
    back: bool,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    win: []u16 = &.{},
    /// The window rows (forward: the grid) and, for the add, the residual before the launch.
    in: u64 = 0,
    old: u64 = 0,
    out: u64 = 0,

    /// Rows: the patch grid's side (64 or 40); n: SAM's channels. Sides off the window multiple pad with zeros.
    pub fn spec(t: *const Window) c.Spec {
        return .{
            .name = t.name,
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .in_place = t.back,
            .tiles = .{ .{ .size = sam_window }, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 64, .n = sam_width, .k = 1 }, .{ .rows = 40, .n = sam_width, .k = 1 } },
        };
    }
    pub fn deinit(t: *Window) void {
        release(t.rig.gpu.gpa, t, .{ "x", "win" });
        t.rig.deinit();
    }
    pub fn kernel(t: *Window) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Window);

    fn padded(t: *const Window) usize {
        return (t.shape.rows + sam_window - 1) / sam_window * sam_window;
    }
    /// The windows' row holding grid position (y, x).
    fn winRow(t: *const Window, y: usize, x: usize) usize {
        const per = t.padded() / sam_window;
        return ((y / sam_window) * per + x / sam_window) * sam_window * sam_window + (y % sam_window) * sam_window + x % sam_window;
    }

    pub fn prepare(t: *Window, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        release(a, t, .{ "x", "win" });
        t.shape = shape;
        const n = shape.n;
        const side = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try a.alloc(u16, side * side * n);
        t.win = try a.alloc(u16, if (t.back) t.padded() * t.padded() * n else 0);
        if (t.back) {
            for (t.x) |*v| v.* = value(rng, 4);
            for (t.win) |*v| v.* = value(rng, 4);
        } else for (t.x) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite) {
            // grid positions (0, 0), (0, 1), (1, 0), (1, 1): NaN, +Inf, -Inf, both infinities (the add's two terms)
            const pos = [_][2]usize{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
            for (pos, 0..) |p, j| if (p[0] < side and p[1] < side) {
                const col = (j * 7) % n;
                const tok = (p[0] * side + p[1]) * n + col;
                if (j < 3) {
                    // the residual for the first two, the windows' row for the third
                    if (t.back and j == 2) t.win[t.winRow(p[0], p[1]) * n + col] = bits(planted[2]) else t.x[tok] = bits(planted[j]);
                } else {
                    t.x[tok] = bits(planted[1]);
                    if (t.back) t.win[t.winRow(p[0], p[1]) * n + col] = bits(planted[2]) else t.x[tok + @min(1, n - 1 - col)] = bits(planted[2]);
                }
            };
            // a NaN in the padding past the grid's bottom edge: the crop must leave it out
            if (t.back and t.padded() > side) t.win[t.winRow(side, 0) * n] = bits(planted[0]);
        }
        if (t.back) {
            t.in = try upload(&t.rig, t.win);
            t.old = try upload(&t.rig, t.x);
            t.out = try t.rig.output(t.x.len * 2);
        } else {
            t.in = try upload(&t.rig, t.x);
            t.out = try t.rig.output(t.padded() * t.padded() * n * 2);
        }
        return true;
    }

    pub fn run(t: *Window) !void {
        const n = t.shape.n;
        const side = t.shape.rows;
        const p = t.padded();
        if (t.back) {
            // in place: the residual goes back before each independent run
            try preset(&t.rig, t.out, t.old, t.x.len * 2);
            try t.k.run("unwindow_add", t.rig.s, .{ side * side, blocks(n), 1 }, 256, .{ t.out, t.in, side, p, sam_window, n });
        } else {
            try t.k.run("window", t.rig.s, .{ p * p, blocks(n), 1 }, 256, .{ t.in, t.out, side, p, sam_window, n });
        }
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Window) usize {
        return if (t.back) t.x.len else t.padded() * t.padded() * t.shape.n;
    }

    pub fn reference(t: *Window, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const side = t.shape.rows;
        const per = t.padded() / sam_window;
        const w2 = sam_window * sam_window;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / n;
            const col = i % n;
            m.* = 0;
            if (t.back) {
                o.* = bf(val(t.x[i]) + val(t.win[t.winRow(row / side, row % side) * n + col]));
                continue;
            }
            const win = row / w2;
            const in = row % w2;
            const gy = (win / per) * sam_window + in / sam_window;
            const gx = (win % per) * sam_window + in % sam_window;
            o.* = if (gy < side and gx < side) val(t.x[(gy * side + gx) * n + col]) else 0;
        }
    }
};
