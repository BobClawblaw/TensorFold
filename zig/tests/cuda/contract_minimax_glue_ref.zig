//! MiniMax-M3.1-Flash's glue under the contract: float64 at the kernels' rounding points, and the range fp32 allows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const sumIn = core.contract.rows.sumIn;

/// fp32's unit roundoff (half an ulp at 1).
pub const unit = 0x1p-24;
/// The libm ulps assumed for nvcc's single-precision functions: expf, cosf, sinf and rsqrtf 2.
pub const libm_ulps = 2;

/// A value as float64 gives it at the kernel's rounding points (m), and the range its fp32 steps allow it (lo, hi).
pub const Iv = struct {
    m: f64,
    lo: f64,
    hi: f64,

    pub fn of(v: f64) Iv {
        return .{ .m = v, .lo = v, .hi = v };
    }

    /// The range [lo, hi] widened by `rel` of itself either side (one fp32 rounding: 2^-24, a libm function: n ulps).
    fn spread(m: f64, lo: f64, hi: f64, rel: f64) Iv {
        return .{ .m = m, .lo = lo - @abs(lo) * rel - 0x1p-149, .hi = hi + @abs(hi) * rel + 0x1p-149 };
    }

    pub fn neg(a: Iv) Iv {
        return .{ .m = -a.m, .lo = -a.hi, .hi = -a.lo };
    }

    /// a + b in fp32: one rounding.
    pub fn add(a: Iv, b: Iv) Iv {
        return spread(a.m + b.m, a.lo + b.lo, a.hi + b.hi, unit);
    }

    pub fn sub(a: Iv, b: Iv) Iv {
        return a.add(b.neg());
    }

    /// a b in fp32: one rounding.
    pub fn mul(a: Iv, b: Iv) Iv {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return spread(a.m * b.m, @min(@min(p[0], p[1]), @min(p[2], p[3])), @max(@max(p[0], p[1]), @max(p[2], p[3])), unit);
    }

    /// a / b in fp32 (IEEE division, one rounding), b's range positive.
    pub fn div(a: Iv, b: Iv) Iv {
        const r = Iv{ .m = 1 / b.m, .lo = 1 / b.hi, .hi = 1 / b.lo };
        const q = a.mul(r);
        return spread(a.m / b.m, q.lo, q.hi, 0);
    }

    /// To bf16, nearest even: rounding is monotone, so the range's ends round to the range's ends.
    pub fn bf(a: Iv) Iv {
        return .{ .m = bfr.bf16(a.m), .lo = bfr.bf16(a.lo), .hi = bfr.bf16(a.hi) };
    }

    /// rsqrtf of a positive range (decreasing), within libm_ulps.
    pub fn rsqrt(a: Iv) Iv {
        return spread(1 / @sqrt(a.m), 1 / @sqrt(a.hi), 1 / @sqrt(a.lo), libm_ulps * 0x1p-23);
    }

    /// expf (increasing), within libm_ulps.
    pub fn exp(a: Iv) Iv {
        return spread(@exp(a.m), @exp(a.lo), @exp(a.hi), libm_ulps * 0x1p-23);
    }

    /// The kernels' sigmoid, 1 / (1 + expf(-x)): expf, an add and a division.
    pub fn sigmoid(a: Iv) Iv {
        return of(1).div(of(1).add(a.neg().exp()));
    }

    /// min(a, v) by a comparison (a > v ? v : a): a NaN stays NaN, as torch.clamp keeps it.
    pub fn atMost(a: Iv, v: f64) Iv {
        return .{ .m = if (a.m > v) v else a.m, .lo = @min(a.lo, v), .hi = @min(a.hi, v) };
    }

    pub fn atLeast(a: Iv, v: f64) Iv {
        return .{ .m = if (a.m < v) v else a.m, .lo = @max(a.lo, v), .hi = @max(a.hi, v) };
    }

    /// cosf (sin: false) or sinf (true) of an angle range: 1-Lipschitz, plus libm_ulps of 1, within [-1, 1].
    pub fn trig(a: Iv, sin: bool) Iv {
        const m = if (sin) @sin(a.m) else @cos(a.m);
        const d = @max(a.m - a.lo, a.hi - a.m) + libm_ulps * 0x1p-23;
        return .{ .m = m, .lo = @max(-1.0, m - d), .hi = @min(1.0, m + d) };
    }
};

/// Gate 4's simulated sums: every add rounded to `acc`, in `order`.
pub const Sim = struct { acc: c.Accum, order: g.Order };

/// How a solve sums: float64 with fp32's spread, or a simulated accumulator; `exact`: gate 4's inputs, fp32 sums exact.
pub const Sums = struct {
    sim: ?Sim = null,
    exact: bool = false,

    /// An fp32 sum of `t` whose longest rounding chain is `depth`: gamma(depth) of the terms' magnitude either side.
    pub fn sum(s: Sums, t: []const Iv, depth: usize, scratch: []f64) Iv {
        if (s.sim) |x| {
            for (scratch[0..t.len], t) |*v, y| v.* = y.m;
            return Iv.of(sumIn(x.acc, x.order, scratch[0..t.len]));
        }
        var r = Iv.of(0);
        var mag: f64 = 0;
        for (t) |y| {
            r.m += y.m;
            r.lo += y.lo;
            r.hi += y.hi;
            mag += @max(@abs(y.lo), @abs(y.hi));
        }
        const rad = if (s.exact) 0 else c.gamma(@floatFromInt(depth), unit) * mag + mag * 0x1p-50;
        return .{ .m = r.m, .lo = r.lo - rad, .hi = r.hi + rad };
    }
};

/// A kernel's outputs in float64 against its own ranges: K uploads, launches and solves; this holds the hooks.
pub fn Gated(comptime K: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        k: K,
        exact: bool = false,
        cache: []Iv = &.{},
        fresh: bool = false,

        pub fn deinit(self: *Self) void {
            const a = self.rig.gpu.gpa;
            a.free(self.cache);
            self.k.free(a);
            self.rig.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = self.k.spec(), .ctx = self, .vt = &vt, .control = self.k.control() };
        }

        const vt = blk: {
            var v = g.hooks(Self);
            if (!@hasDecl(K, "sums")) v.simulate = null;
            break :blk v;
        };

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            self.rig.reset();
            self.fresh = false;
            self.exact = fill == .stress;
            var prng = std.Random.DefaultPrng.init(seed);
            return self.k.prepare(self.rig.gpu.gpa, &self.rig, shape, fill, prng.random());
        }

        pub fn run(self: *Self) !void {
            try self.k.run(&self.rig);
            try self.rig.s.synchronize();
        }

        pub fn count(self: *Self) usize {
            return self.k.count();
        }

        fn solved(self: *Self) ![]const Iv {
            if (self.fresh) return self.cache;
            const a = self.rig.gpu.gpa;
            a.free(self.cache);
            self.cache = &.{};
            self.cache = try a.alloc(Iv, self.count());
            try self.k.solve(a, .{ .exact = self.exact }, self.cache);
            self.fresh = true;
            return self.cache;
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            for (want, mag, try self.solved()) |*w, *m, v| {
                w.* = v.m;
                m.* = 0;
            }
        }

        /// The farthest end of the output's range from float64's value: a kernel inside its range is inside its bound.
        pub fn bounds(self: *Self, want: []const f64, _: []const f64, allow: []f64) !void {
            for (allow, want, try self.solved()) |*l, w, v| l.* = if (std.math.isFinite(w)) @max(0, @max(w - v.lo, v.hi - w)) + 0x1p-140 else std.math.floatMax(f64);
        }

        pub fn intervals(self: *Self, low: []f64, high: []f64) !void {
            for (low, high, try self.solved()) |*l, *h, v| {
                l.* = v.lo;
                h.* = v.hi;
            }
        }

        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            const a = self.rig.gpu.gpa;
            const tmp = try a.alloc(Iv, out.len);
            defer a.free(tmp);
            try self.k.solve(a, .{ .sim = .{ .acc = acc, .order = order }, .exact = true }, tmp);
            for (out, tmp) |*o, v| o.* = v.m;
        }
    };
}

/// A kernel whose outputs are ids or exact words, checked by its own stated rule: K uploads, launches and checks.
pub fn Checked(comptime K: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        k: K,
        msg: [200]u8 = undefined,

        pub fn deinit(self: *Self) void {
            self.k.free(self.rig.gpu.gpa);
            self.rig.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = self.k.spec(), .ctx = self, .vt = &vt, .control = self.k.control() };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            self.rig.reset();
            var prng = std.Random.DefaultPrng.init(seed);
            return self.k.prepare(self.rig.gpu.gpa, &self.rig, shape, fill, prng.random());
        }

        pub fn run(self: *Self) !void {
            try self.k.run(&self.rig);
            try self.rig.s.synchronize();
        }

        pub fn count(self: *Self) usize {
            return self.k.count();
        }

        pub fn reference(_: *Self, want: []f64, mag: []f64) !void {
            @memset(want, 0);
            @memset(mag, 0);
        }

        pub fn check(self: *Self, _: g.Gate) !?[]const u8 {
            const a = self.rig.gpu.gpa;
            const got = try a.dupe(u8, try self.rig.outputBytes());
            defer a.free(got);
            return self.k.check(a, got, &self.msg);
        }
    };
}

/// No sums: the kernel compares or moves values.
pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// `n` bf16 values uniform in [lo, hi).
pub fn bfs(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
    return v;
}

/// `n` fp32 values uniform in [lo, hi).
pub fn f32s(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]f32 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = lo + rng.float(f32) * (hi - lo);
    return v;
}

/// A whole number in [lo, hi], as fp32.
pub fn int(rng: std.Random, lo: i32, hi: i32) f32 {
    return @floatFromInt(lo + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(hi - lo + 1)))));
}

pub fn input(rig: *rig_.Rig, x: anytype) !u64 {
    return rig.input(std.mem.sliceAsBytes(x));
}

/// Bytes put back where the device reads and writes them in place, before a launch.
pub fn restore(rig: *rig_.Rig, ptr: u64, bytes: []const u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = rig.gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
}

pub fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub fn readF32(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
}

pub fn readI32(bytes: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, bytes[4 * i ..][0..4], .little);
}

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);
