//! The rows group's float64 values at the CUDA kernels' bf16 points, and each output's spread through their fp32 steps.
const std = @import("std");
const core = @import("core");
const uocr = @import("uocr_cuda");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
const cfg = uocr.config;

/// fp32's unit roundoff: one correctly rounded fp32 step moves a value by at most this share of it.
const U: f64 = 0x1p-24;
/// x / y under --prec-div: IEEE division, correctly rounded.
const div_u: f64 = 1;
/// sqrtf under --prec-sqrt: correctly rounded.
const sqrt_u: f64 = 1;
/// 1 / sqrtf(v): sqrtf's rounding, then the division's, with one unit of slack for their product.
const rsqrt_u: f64 = 3;
/// v / (1 + expf(-z)): expf's 2 ulps (4 U), the add's 1 and the division's 1, rounded up.
const sig_u: f64 = 8;
/// sinf and cosf within 2 ulps of a value at most 1: an absolute spread.
const trig_abs: f64 = 0x1p-22;
/// 1 + erf_(x * f32(1 / sqrt 2)): the faithful erf_, the argument's rounding and the add, each under 2^-24.
const erf_abs: f64 = 0x1p-22;

/// Each kernel by its name in the family's Kernels (tf_uocr_<name>).
pub const Kind = enum { add, layer_norm, layer_norm2d, gelu, quick_gelu, rms_norm, rope, router_norm, silu_mul };

/// A value as fp32 then bf16 round it (to nearest even, twice).
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

pub fn val(h: u16) f64 {
    return c.fromBf16(h);
}

/// A value's spread through fp32 steps and bf16 rounding points.
pub const R = struct {
    lo: f64,
    hi: f64,
    fn pt(v: f64) R {
        return .{ .lo = v, .hi = v };
    }
    fn wid(a: R, e: f64) R {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    /// Widened by k fp32 units of its larger end (k = 1: one correctly rounded step).
    fn rel(a: R, k: f64) R {
        return a.wid(k * U * @max(@abs(a.lo), @abs(a.hi)));
    }
    fn bf16(a: R) R {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    fn add(a: R, b: R) R {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    fn neg(a: R) R {
        return .{ .lo = -a.hi, .hi = -a.lo };
    }
    fn mul(a: R, b: R) R {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    fn times(a: R, v: f64) R {
        return a.mul(pt(v));
    }
    fn sq(a: R) R {
        if (a.lo >= 0) return .{ .lo = a.lo * a.lo, .hi = a.hi * a.hi };
        if (a.hi <= 0) return .{ .lo = a.hi * a.hi, .hi = a.lo * a.lo };
        return .{ .lo = 0, .hi = @max(a.lo * a.lo, a.hi * a.hi) };
    }
    fn rsqrt(a: R) R {
        return .{ .lo = 1 / @sqrt(a.hi), .hi = 1 / @sqrt(a.lo) };
    }
    fn clamp(a: R, m: f64) R {
        return .{ .lo = std.math.clamp(a.lo, -m, m), .hi = std.math.clamp(a.hi, -m, m) };
    }
};

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// torch.clamp: NaN stays NaN.
fn clampNan(v: f64, m: f64) f64 {
    return if (std.math.isNan(v)) v else std.math.clamp(v, -m, m);
}

/// erf in float64 to about 1e-15: its Maclaurin series below 3, erfc's continued fraction (60 terms) above, 1 past 6.
fn erf(z: f64) f64 {
    if (std.math.isNan(z)) return z;
    if (z < 0) return -erf(-z);
    if (z > 6) return 1;
    if (z < 3) {
        var term = z;
        var s: f64 = z;
        var n: f64 = 1;
        while (n < 200) : (n += 1) {
            term *= -z * z / n;
            const t = term / (2 * n + 1);
            s += t;
            if (@abs(t) < 1e-18 * @abs(s)) break;
        }
        return s * 2 / @sqrt(std.math.pi);
    }
    var f = z;
    var k: f64 = 60;
    while (k >= 1) : (k -= 1) f = z + (k / 2) / f;
    return 1 - @exp(-z * z) / (@sqrt(std.math.pi) * f);
}

/// A simulated accumulator for gate 4, or none: float64's own sums.
pub const Sim = ?struct { acc: c.Accum, order: g.Order };

fn sumOf(sim: Sim, terms: []const f64) f64 {
    if (sim) |s| return core.contract.rows.sumIn(s.acc, s.order, terms);
    var s: f64 = 0;
    for (terms) |v| s += v;
    return s;
}

/// The page2d angle of row r's column d: an fp32 product (torch's float32 freqs), or the page's parity.
fn angle(t: anytype, r: usize, d: usize) f64 {
    const j = d % 64;
    if (j == 63) return @floatFromInt(t.pos[t.shape.rows + r] % 2);
    return @as(f32, @floatCast(@as(f64, @floatFromInt(t.pos[r])) * t.inv[j]));
}

/// Row r's float64 values at the kernel's rounding points (gate 4: its sums in `sim`) and their fp32 spreads.
pub fn row(t: anytype, r: usize, want: []f64, rg: []R, buf: []f64, rbuf: []R, sim: Sim) void {
    const n = t.shape.n;
    const nf: f64 = @floatFromInt(n);
    const gam = c.gamma(@floatFromInt(t.depth()), U);
    const eps: f64 = t.v.eps;
    const x = t.x[r * n ..][0..n];
    const terms = buf[0..n];
    switch (t.v.kind) {
        .add, .gelu, .quick_gelu, .silu_mul => for (want, rg, x, 0..) |*w, *o, xh, j| {
            const v = val(xh);
            switch (t.v.kind) {
                .add => {
                    const s = v + val(t.y[r * n + j]);
                    w.* = bf(s);
                    o.* = R.pt(s).rel(1).bf16();
                },
                .gelu => {
                    const e1 = 1 + erf(v / std.math.sqrt2);
                    w.* = bf(0.5 * v * e1);
                    o.* = R.pt(e1).wid(erf_abs).times(0.5 * v).rel(1).bf16();
                },
                .quick_gelu => {
                    const z = @as(f64, @as(f32, 1.702)) * v;
                    w.* = bf(v * bf(sigmoid(bf(z))));
                    const zr = R.pt(z).rel(1).bf16();
                    o.* = R.rel(.{ .lo = sigmoid(zr.lo), .hi = sigmoid(zr.hi) }, sig_u).bf16().times(v).bf16();
                },
                else => {
                    const u = val(t.y[r * n + j]);
                    w.* = bf(bf(v * sigmoid(v)) * u);
                    o.* = R.pt(v * sigmoid(v)).rel(sig_u).bf16().times(u).bf16();
                },
            }
        },
        .rms_norm => {
            for (terms, x) |*q, v| q.* = val(v) * val(v);
            const se = sumOf(null, terms);
            const inv = 1 / @sqrt(sumOf(sim, terms) / nf + eps);
            const ir = R.pt(se).wid(gam * se).times(1 / nf).rel(div_u).add(R.pt(eps)).rel(1).rsqrt().rel(rsqrt_u);
            for (want, rg, x, 0..) |*w, *o, xv, j| {
                w.* = bf(val(t.y[j]) * bf(val(xv) * inv));
                o.* = ir.times(val(xv)).rel(1).bf16().times(val(t.y[j])).bf16();
            }
        },
        .layer_norm, .layer_norm2d => {
            for (terms, x) |*q, v| q.* = val(v);
            var abs: f64 = 0;
            for (x) |v| abs += @abs(val(v));
            const mr = R.pt(sumOf(null, terms)).wid(gam * abs).times(1 / nf).rel(div_u);
            const two = t.v.kind == .layer_norm2d;
            const m = sumOf(sim, terms) / nf;
            const u = if (two) bf(m) else m;
            const ur = if (two) mr.bf16() else mr;
            const ds = buf[n .. 2 * n];
            var vr = R.pt(0);
            for (x, terms, ds, rbuf) |v, *q, *d, *dr| {
                d.* = if (two) bf(val(v) - u) else val(v) - u;
                q.* = if (two) bf(d.* * d.*) else d.* * d.*;
                dr.* = R.pt(val(v)).add(ur.neg()).rel(1);
                if (two) dr.* = dr.bf16();
                vr = vr.add(if (two) dr.sq().bf16() else dr.sq().rel(1));
            }
            const vs = sumOf(sim, terms) / nf;
            vr = vr.wid(gam * vr.hi).times(1 / nf).rel(div_u);
            // LayerNorm2d rounds var, var + eps, sqrt and the quotient; LayerNorm takes 1 / sqrtf of the fp32 variance
            const den = bf(@sqrt(bf(bf(vs) + eps)));
            const dn = vr.bf16().add(R.pt(eps)).rel(1).bf16();
            const dnr = R.rel(.{ .lo = @sqrt(dn.lo), .hi = @sqrt(dn.hi) }, sqrt_u).bf16();
            const ir = vr.add(R.pt(eps)).rel(1).rsqrt().rel(rsqrt_u);
            for (want, rg, ds, rbuf, 0..) |*w, *o, d, dr, j| {
                const wt = val(t.y[j]);
                const b = val(t.z[j]);
                if (two) {
                    w.* = bf(bf(wt * bf(d / den)) + b);
                    o.* = dr.mul(.{ .lo = 1 / dnr.hi, .hi = 1 / dnr.lo }).rel(div_u).bf16().times(wt).bf16().add(R.pt(b)).rel(1).bf16();
                } else {
                    w.* = bf(d / @sqrt(vs + eps) * wt + b);
                    o.* = dr.mul(ir).rel(1).times(wt).rel(1).add(R.pt(b)).rel(1).bf16();
                }
            }
        },
        .rope => for (want, rg, x, 0..) |*w, *o, xv, d| {
            const dd = d % 128;
            const rh = if (dd < 64) -val(x[d + 64]) else val(x[d - 64]);
            const ang = angle(t, r, dd);
            const cr = R.pt(@cos(ang)).wid(trig_abs).bf16();
            const sr = R.pt(@sin(ang)).wid(trig_abs).bf16();
            w.* = bf(bf(val(xv) * bf(@cos(ang))) + bf(rh * bf(@sin(ang))));
            o.* = cr.times(val(xv)).bf16().add(sr.times(rh).bf16()).rel(1).bf16();
        },
        .router_norm => {
            const lim = 1e4;
            const reps: f64 = @as(f32, 1e-6);
            const vs = buf[n .. 2 * n];
            for (vs, terms, x) |*v, *q, xv| {
                v.* = if (std.math.isFinite(val(xv))) val(xv) else 0;
                q.* = v.* * v.*;
            }
            const se = sumOf(null, terms);
            const inv = 1 / @sqrt(sumOf(sim, terms) / 128 + reps);
            // s / 128 is exact; then + eps, 1 / sqrtf, the product, 1 + w and its product, each clamped as torch does
            const ir = R.pt(se).wid(gam * se).times(1.0 / 128.0).add(R.pt(reps)).rel(1).rsqrt().rel(rsqrt_u);
            const nw = buf[2 * n .. 3 * n];
            for (vs, nw, rbuf, 0..) |v, *w, *nr, j| {
                const w1 = 1 + val(t.y[j]);
                w.* = bf(clampNan(clampNan(v * inv, lim) * w1, lim));
                nr.* = ir.times(v).rel(1).clamp(lim).mul(R.pt(w1).rel(1)).rel(1).clamp(lim).bf16();
            }
            for (want, rg, 0..) |*w, *o, d| {
                const pair = if (d < 64) d + 64 else d - 64;
                const ang = angle(t, r, d);
                const cr = R.pt(@cos(ang)).wid(trig_abs).bf16();
                const sr = R.pt(@sin(ang)).wid(trig_abs).bf16();
                const rh = if (d < 64) -nw[pair] else nw[pair];
                const rhr = if (d < 64) rbuf[pair].neg() else rbuf[pair];
                const out = bf(bf(nw[d] * bf(@cos(ang))) + bf(rh * bf(@sin(ang))));
                w.* = if (std.math.isFinite(out)) out else 0;
                o.* = rbuf[d].mul(cr).bf16().add(rhr.mul(sr).bf16()).rel(1).bf16();
            }
        },
    }
}
