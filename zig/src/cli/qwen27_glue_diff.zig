//! tf-qwen27-cuda glue-diff: each native glue kernel (TF_QWEN27_GLUE's) against the captured Triton kernel it
//! replaces, on the same random rows at the 27B's and DFlash2's shapes: the elements that differ and by how much.

const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen27");
const tri = qwen.triton;
const glue = qwen.glue;
const c = qwen.config;

const aot = cuda.aot;
const p = aot.ptr;

const Bench = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    rng: std.Random.DefaultPrng,
    held: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn put(b: *Bench, bytes: []const u8) !u64 {
        const x = try cuda.DeviceBuffer.fromHost(b.d, bytes);
        try b.held.append(b.gpa, x);
        return x.ptr;
    }

    fn zeros(b: *Bench, n: usize) !u64 {
        const z = try b.gpa.alloc(u8, @max(n, 16));
        defer b.gpa.free(z);
        @memset(z, 0);
        return b.put(z);
    }

    fn bf16(b: *Bench, n: usize, lo: f32, hi: f32) !u64 {
        const h = try b.gpa.alloc(u16, n);
        defer b.gpa.free(h);
        for (h) |*v| v.* = @truncate(@as(u32, @bitCast(lo + b.rng.random().float(f32) * (hi - lo))) >> 16);
        return b.put(std.mem.sliceAsBytes(h));
    }

    fn f32s(b: *Bench, n: usize, lo: f32, hi: f32) !u64 {
        const h = try b.gpa.alloc(f32, n);
        defer b.gpa.free(h);
        for (h) |*v| v.* = lo + b.rng.random().float(f32) * (hi - lo);
        return b.put(std.mem.sliceAsBytes(h));
    }

    /// TF_GLUE_DUMP=DIR: the buffer's bytes to DIR/name (for working out a reduction order on the host).
    fn dump(b: *Bench, io: std.Io, name: []const u8, at: u64, bytes: usize) !void {
        const dir = std.mem.span(std.c.getenv("TF_GLUE_DUMP") orelse return);
        try b.s.synchronize();
        const h = try b.fetch(u8, at, bytes);
        defer b.gpa.free(h);
        const path = try std.fs.path.join(b.gpa, &.{ dir, name });
        defer b.gpa.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = h });
    }

    fn deinit(b: *Bench) void {
        for (b.held.items) |*x| x.free();
        b.held.deinit(b.gpa);
    }

    fn fetch(b: *Bench, comptime T: type, at: u64, n: usize) ![]T {
        const out = try b.gpa.alloc(T, n);
        const view: cuda.DeviceBuffer = .{ .d = b.d, .ptr = at, .len = n * @sizeOf(T), .borrowed = true };
        try view.download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    /// Element `n` bf16 (u16) or fp32 (u32) values at `x` (Triton) and `y` (native): how many differ, the largest
    /// distance in ulps, and the largest absolute difference.
    fn diff(b: *Bench, comptime T: type, label: []const u8, x: u64, y: u64, n: usize) !bool {
        try b.s.synchronize();
        const a = try b.fetch(T, x, n);
        defer b.gpa.free(a);
        const e = try b.fetch(T, y, n);
        defer b.gpa.free(e);
        var differ: usize = 0;
        var over: usize = 0; // more than one ulp of the larger value apart
        var most: f64 = 0;
        var abs: f64 = 0;
        var nan: usize = 0;
        for (a, e) |u, v| {
            if (u == v) continue;
            differ += 1;
            const fu = value(T, u);
            const fv = value(T, v);
            if (std.math.isNan(fu) or std.math.isNan(fv)) {
                nan += 1;
                continue;
            }
            const ulps = @abs(fu - fv) / ulp(T, @max(@abs(fu), @abs(fv)));
            if (ulps > 1.0001) over += 1;
            most = @max(most, ulps);
            abs = @max(abs, @abs(fu - fv));
        }
        std.debug.print("  {s:<10} {d:>9} of {d:>9} differ ({d:.3}%), {d} by more than 1 ulp, most {d:.1} ulp, |d| {e:.2}{s}\n", .{ label, differ, n, 100.0 * @as(f64, @floatFromInt(differ)) / @as(f64, @floatFromInt(n)), over, most, abs, if (nan > 0) " NaN" else "" });
        return differ == 0;
    }
};

fn value(comptime T: type, v: T) f64 {
    return if (T == u16) @as(f32, @bitCast(@as(u32, v) << 16)) else @as(f32, @bitCast(v));
}

/// One ulp of a bf16 (8 significant bits) or fp32 (24) value of magnitude `m`.
fn ulp(comptime T: type, m: f64) f64 {
    const bits: i32 = if (T == u16) 8 else 24;
    if (m == 0) return std.math.floatMin(f32);
    return std.math.pow(f64, 2, @as(f64, @floatFromInt(std.math.ilogb(m) - bits + 1)));
}

/// Both sides' outputs: Triton's and the native kernel's buffers for each output.
fn pair(b: *Bench, n: usize) ![2]u64 {
    return .{ try b.zeros(n), try b.zeros(n) };
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, kernels_dir: []const u8, rows: usize) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var k = try qwen.kernels.Kernels.load(gpa, io, &ctx, kernels_dir, c.the_27b.linear_v_heads);
    defer k.deinit();
    var g = try glue.Glue.loadSet(&driver, std.EnumSet(glue.Kernel).full);
    defer g.unload();
    var b: Bench = .{ .gpa = gpa, .d = &driver, .s = stream, .rng = .init(27) };
    defer b.deinit();
    const ts = [2]tri.Tri{ .{ .set = &k.triton, .s = stream }, .{ .set = &k.triton, .s = stream, .g = &g } };
    const R = rows;
    const geo = c.the_27b;
    var same: usize = 0;
    var total: usize = 0;

    {
        const d = geo.hidden;
        std.debug.print("add_rmsnorm ({d} rows of {d}, with r):\n", .{ R, d });
        const x = try b.bf16(R * d, -2, 2);
        const r = try b.bf16(R * d, -2, 2);
        const w = try b.bf16(d, 0.5, 1.5);
        const h = try pair(&b, R * d * 2);
        const y = try pair(&b, R * d * 2);
        const xs = try pair(&b, R * d / 64 * 4);
        for (ts, 0..) |t, i| try t.addRmsnorm(x, r, w, h[i], y[i], xs[i], R, d, c.eps);
        try b.dump(io, "add_rmsnorm.out", y[0], R * d * 2);
        try b.dump(io, "add_rmsnorm.xs0", xs[0], R * d / 64 * 4);
        try b.dump(io, "add_rmsnorm.xs1", xs[1], R * d / 64 * 4);
        same += @intFromBool(try b.diff(u16, "h", h[0], h[1], R * d));
        same += @intFromBool(try b.diff(u16, "y", y[0], y[1], R * d));
        same += @intFromBool(try b.diff(u32, "xs", xs[0], xs[1], R * d / 64));
        total += 3;
        std.debug.print("add_rmsnorm ({d} rows of {d}, no r):\n", .{ R, d });
        for (ts, 0..) |t, i| try t.addRmsnorm(x, null, w, 0, y[i], xs[i], R, d, c.eps);
        same += @intFromBool(try b.diff(u16, "y", y[0], y[1], R * d));
        same += @intFromBool(try b.diff(u32, "xs", xs[0], xs[1], R * d / 64));
        total += 2;
    }
    {
        const vh = geo.linear_v_heads;
        const dv = c.linear_dim;
        std.debug.print("gated_norm ({d} rows, {d} heads of {d}):\n", .{ R, vh, dv });
        const y = try b.bf16(R * vh * dv, -2, 2);
        const z = try b.bf16(R * vh * dv, -4, 4);
        const w = try b.bf16(dv, 0.5, 1.5);
        const out = try pair(&b, R * vh * dv * 2);
        const xs = try pair(&b, R * vh * dv / 64 * 4);
        for (ts, 0..) |t, i| try t.gatedNorm(y, z, w, out[i], xs[i], R, vh, dv, c.eps);
        try b.dump(io, "gated_norm.out", out[0], R * vh * dv * 2);
        try b.dump(io, "gated_norm.xs0", xs[0], R * vh * dv / 64 * 4);
        try b.dump(io, "gated_norm.xs1", xs[1], R * vh * dv / 64 * 4);
        same += @intFromBool(try b.diff(u16, "out", out[0], out[1], R * vh * dv));
        same += @intFromBool(try b.diff(u32, "xs", xs[0], xs[1], R * vh * dv / 64));
        total += 2;
    }
    {
        const n = geo.intermediate;
        std.debug.print("swiglu ({d} rows of {d}):\n", .{ R, n });
        const gate = try b.bf16(R * n, -6, 6);
        const up = try b.bf16(R * n, -4, 4);
        const out = try pair(&b, R * n * 2);
        const xs = try pair(&b, R * n / 64 * 4);
        for (ts, 0..) |t, i| try t.swiglu(gate, up, out[i], xs[i], R, n);
        try b.dump(io, "swiglu.out", out[0], R * n * 2);
        try b.dump(io, "swiglu.xs0", xs[0], R * n / 64 * 4);
        try b.dump(io, "swiglu.xs1", xs[1], R * n / 64 * 4);
        same += @intFromBool(try b.diff(u16, "out", out[0], out[1], R * n));
        same += @intFromBool(try b.diff(u32, "xs", xs[0], xs[1], R * n / 64));
        total += 2;
    }
    {
        const H = geo.query_heads;
        const D = c.head_dim;
        std.debug.print("gate_mul ({d} rows, {d} heads of {d}):\n", .{ R, H, D });
        const o = try b.bf16(R * H * D, -2, 2);
        const qg = try b.bf16(R * H * 2 * D, -6, 6);
        const out = try pair(&b, R * H * D * 2);
        const xs = try pair(&b, R * H * D / 64 * 4);
        for (ts, 0..) |t, i| try t.gateMul(o, qg, out[i], xs[i], R, H, D);
        try b.dump(io, "gate_mul.out", out[0], R * H * D * 2);
        try b.dump(io, "gate_mul.xs0", xs[0], R * H * D / 64 * 4);
        try b.dump(io, "gate_mul.xs1", xs[1], R * H * D / 64 * 4);
        same += @intFromBool(try b.diff(u16, "out", out[0], out[1], R * H * D));
        same += @intFromBool(try b.diff(u32, "xs", xs[0], xs[1], R * H * D / 64));
        total += 2;
    }
    {
        const a: tri.Attn = .{ .heads = geo.query_heads, .kv_heads = geo.kv_heads, .dim = c.head_dim, .half = c.rotary_dim / 2 };
        std.debug.print("attn_prep ({d} rows, {d}+{d} heads of {d}, rotary {d}):\n", .{ R, a.heads, a.kv_heads, a.dim, 2 * a.half });
        const qg = try b.bf16(R * a.heads * 2 * a.dim, -2, 2);
        const kv = try b.bf16(R * a.kv_heads * a.dim, -2, 2);
        const qn = try b.bf16(a.dim, 0.5, 1.5);
        const kn = try b.bf16(a.dim, 0.5, 1.5);
        const pos_h = try gpa.alloc(i32, R);
        defer gpa.free(pos_h);
        for (pos_h, 0..) |*v, i| v.* = @intCast(1000 + 4093 * i);
        const pos = try b.put(std.mem.sliceAsBytes(pos_h));
        var inv_h: [64]f32 = undefined;
        for (inv_h[0..a.half], 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, c.theta, -@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(2 * a.half))));
        const inv = try b.put(std.mem.sliceAsBytes(inv_h[0..a.half]));
        const qo = try pair(&b, R * a.heads * a.dim * 2);
        const ko = try pair(&b, R * a.kv_heads * a.dim * 2);
        for (ts, 0..) |t, i| try t.attnPrep(qg, kv, qn, kn, pos, inv, qo[i], ko[i], R, a, c.eps);
        same += @intFromBool(try b.diff(u16, "q", qo[0], qo[1], R * a.heads * a.dim));
        same += @intFromBool(try b.diff(u16, "k", ko[0], ko[1], R * a.kv_heads * a.dim));
        total += 2;
    }
    {
        const m: tri.Gdn = .{ .c = geo.convDim(), .kh = geo.linear_k_heads, .vh = geo.linear_v_heads, .dk = c.linear_dim, .keep = c.conv_taps - 1 };
        std.debug.print("gdn_pre ({d} rows, {d} channels, {d}+{d} heads of {d}, {d} kept):\n", .{ R, m.c, m.kh, m.vh, m.dk, m.keep });
        const qkv = try b.bf16(R * m.c, -2, 2);
        const cs = try b.bf16(m.keep * m.c, -2, 2);
        const cw = try b.bf16(m.c * (m.keep + 1), -0.6, 0.6);
        const win_h = try gpa.alloc(i32, R * (m.keep + 1));
        defer gpa.free(win_h);
        for (0..R) |r| for (0..m.keep + 1) |j| {
            win_h[r * (m.keep + 1) + j] = @intCast(r + j);
        };
        const win = try b.put(std.mem.sliceAsBytes(win_h));
        const av = try b.bf16(R * m.vh, -4, 4);
        const bv = try b.bf16(R * m.vh, -4, 4);
        const alog = try b.f32s(m.vh, -1, 2.5);
        const dtb = try b.f32s(m.vh, -1, 1);
        const q = try pair(&b, R * m.kh * m.dk * 2);
        const kk = try pair(&b, R * m.kh * m.dk * 2);
        const v = try pair(&b, R * m.vh * m.dk * 2);
        const gg = try pair(&b, R * m.vh * 4);
        const beta = try pair(&b, R * m.vh * 4);
        for (ts, 0..) |t, i| try t.gdnPre(qkv, cs, cw, win, null, av, bv, alog, dtb, q[i], kk[i], v[i], gg[i], beta[i], R, m);
        same += @intFromBool(try b.diff(u16, "q", q[0], q[1], R * m.kh * m.dk));
        same += @intFromBool(try b.diff(u16, "k", kk[0], kk[1], R * m.kh * m.dk));
        same += @intFromBool(try b.diff(u16, "v", v[0], v[1], R * m.vh * m.dk));
        same += @intFromBool(try b.diff(u32, "g", gg[0], gg[1], R * m.vh));
        same += @intFromBool(try b.diff(u32, "beta", beta[0], beta[1], R * m.vh));
        total += 5;
    }
    // DFlash2's: hidden 5120, 32 query and 8 key heads of 128, conv groups of 16, 2,047 context rows kept
    const H = 5120;
    const D = 128;
    const dh = 32;
    const dkv = 8;
    const gs = 16;
    const L = 16;
    const S = (R + L - 1) / L;
    const DR = S * L;
    {
        std.debug.print("dconv ({d} rows of {d}, groups of {d}, {d}-row blocks):\n", .{ DR, H, gs, L });
        const x = try b.bf16(DR * H, -2, 2);
        const dyn = try b.bf16(DR * 2 * 2 * (H / gs), -0.3, 0.3);
        const base = try b.bf16(2 * 2 * H, -1, 1);
        const res = try b.bf16(DR * H, -2, 2);
        const out = try pair(&b, DR * H * 2);
        for (0..2) |branch| {
            const r: ?u64 = if (branch == 1) res else null;
            try k.triton.run(stream, "_dconv_kernel", .{ @intCast(DR), (H + 1023) / 1024, 1 }, &.{
                p("X", "*bf16", x), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", r orelse x), p("OUT", "*bf16", out[0]), aot.int("SEG", L),
            }, &.{ aot.ci("D", H), aot.ci("G", H / gs), aot.ci("GS", gs), aot.ci("BRANCH", @intCast(branch)), aot.ci("HAS_RES", @intFromBool(r != null)), aot.ci("BLOCK", 1024) });
            try g.act.dconv(stream, x, dyn, base, r, out[1], DR, H, gs, branch, L);
            same += @intFromBool(try b.diff(u16, if (branch == 1) "out, res" else "out", out[0], out[1], DR * H));
            total += 1;
        }
    }
    {
        std.debug.print("prep (DFlash2: {d} rows, {d}+{d}+{d} heads of {d}, full rotation):\n", .{ DR, dh, dkv, dkv, D });
        const stride = (dh + 2 * dkv) * D;
        const qkv = try b.bf16(DR * stride, -2, 2);
        const qn = try b.bf16(D, 0.5, 1.5);
        const kn = try b.bf16(D, 0.5, 1.5);
        const pos_h = try gpa.alloc(i32, DR);
        defer gpa.free(pos_h);
        for (pos_h, 0..) |*v, i| v.* = @intCast(5000 + 7 * i);
        const pos = try b.put(std.mem.sliceAsBytes(pos_h));
        const inv = try b.zeros(D / 2 * 4);
        const cos = try b.zeros(DR * D / 2 * 4);
        const sin = try b.zeros(DR * D / 2 * 4);
        {
            var a: cuda.Args = .{};
            a.add(inv);
            a.add(@as(f32, 10000000));
            a.add(@as(c_int, D / 2));
            a.add(@as(c_int, D));
            try cuda.launch.launch(k.inv_freq, .{ .grid = .{ .x = 1 }, .block = .{ .x = D / 2 } }, stream, &a);
            var r: cuda.Args = .{};
            for ([_]u64{ pos, inv, cos, sin }) |v| r.add(v);
            r.add(@as(c_int, D / 2));
            try cuda.launch.launch(k.rotary, .{ .grid = .{ .x = @intCast(DR) }, .block = .{ .x = D / 2 } }, stream, &r);
        }
        const q = try pair(&b, dh * DR * D * 2);
        const kk = try pair(&b, dkv * DR * D * 2);
        const v = try pair(&b, dkv * DR * D * 2);
        try k.triton.run(stream, "_prep_kernel", .{ @intCast(DR), dh + 2 * dkv, 1 }, &.{
            p("QKV", "*bf16", qkv), p("QN", "*bf16", qn),    p("KN", "*bf16", kn),   p("COS", "*fp32", cos),     p("SIN", "*fp32", sin),
            p("QO", "*bf16", q[0]), p("KO", "*bf16", kk[0]), p("VO", "*bf16", v[0]), aot.int("L", @intCast(DR)), aot.int("stride", stride),
            aot.float("eps", 1e-6),
        }, &.{ aot.ci("H", dh), aot.ci("HKV", dkv), aot.ci("HALF", D / 2) });
        try g.qk.rows(stream, qkv, .{ .h = dh, .hk = dkv, .dim = D }, .{ .norms = qn, .norm_k = kn, .norms_bf16 = true, .inv = inv, .rot = D, .eps = 1e-6 }, pos, null, .{ .q = q[1], .k = kk[1], .v = v[1], .q_ld = D, .q_hs = DR * D, .kv_ld = D, .kv_hs = DR * D }, DR);
        same += @intFromBool(try b.diff(u16, "q", q[0], q[1], dh * DR * D));
        same += @intFromBool(try b.diff(u16, "k", kk[0], kk[1], dkv * DR * D));
        same += @intFromBool(try b.diff(u16, "v", v[0], v[1], dkv * DR * D));
        total += 3;
        // the same angles as Triton's: rotary()'s [rows, D / 2] tables widened to qk_rows' [pos, D], rows as positions
        try stream.synchronize();
        const ch = try b.fetch(f32, cos, DR * D / 2);
        defer gpa.free(ch);
        const sh = try b.fetch(f32, sin, DR * D / 2);
        defer gpa.free(sh);
        const cw = try gpa.alloc(f32, DR * D);
        defer gpa.free(cw);
        const sw = try gpa.alloc(f32, DR * D);
        defer gpa.free(sw);
        for (0..DR) |r| for (0..D) |j| {
            cw[r * D + j] = ch[r * D / 2 + j % (D / 2)];
            sw[r * D + j] = sh[r * D / 2 + j % (D / 2)];
        };
        const cos2 = try b.put(std.mem.sliceAsBytes(cw));
        const sin2 = try b.put(std.mem.sliceAsBytes(sw));
        const rows_h = try gpa.alloc(i32, DR);
        defer gpa.free(rows_h);
        for (rows_h, 0..) |*x, i| x.* = @intCast(i);
        const iota = try b.put(std.mem.sliceAsBytes(rows_h));
        std.debug.print("prep, Triton's angles as qk_rows tables:\n", .{});
        try g.qk.rows(stream, qkv, .{ .h = dh, .hk = dkv, .dim = D }, .{ .norms = qn, .norm_k = kn, .norms_bf16 = true, .cos = cos2, .sin = sin2, .rot = D, .eps = 1e-6 }, iota, null, .{ .q = q[1], .k = kk[1], .v = v[1], .q_ld = D, .q_hs = DR * D, .kv_ld = D, .kv_hs = DR * D }, DR);
        same += @intFromBool(try b.diff(u16, "q", q[0], q[1], dh * DR * D));
        same += @intFromBool(try b.diff(u16, "k", kk[0], kk[1], dkv * DR * D));
        total += 2;
    }
    {
        const window = 2047;
        const lens_h = [_]i32{ 300, 2047, 1000, 64, 2047, 5, 700, 1500 };
        std.debug.print("block_attention (DFlash2: {d} streams of {d} rows, {d}+{d} heads of {d}, context {d}..):\n", .{ S, L, dh, dkv, D, lens_h[0] });
        const q = try b.bf16(dh * DR * D, -2, 2);
        const kb = try b.bf16(dkv * DR * D, -2, 2);
        const vb = try b.bf16(dkv * DR * D, -2, 2);
        const table_h = try gpa.alloc(u8, S * 16 + S * 4);
        defer gpa.free(table_h);
        for (0..S) |j| {
            const len: usize = @intCast(lens_h[j % lens_h.len]);
            const kc = try b.bf16(dkv * len * D, -2, 2);
            const vc = try b.bf16(dkv * len * D, -2, 2);
            std.mem.writeInt(u64, table_h[j * 16 ..][0..8], kc, .little);
            std.mem.writeInt(u64, table_h[j * 16 + 8 ..][0..8], vc, .little);
            std.mem.writeInt(i32, table_h[S * 16 + j * 4 ..][0..4], @intCast(len), .little);
        }
        const tb = try b.put(table_h);
        const out = try pair(&b, DR * dh * D * 2);
        const scale: f32 = @floatCast(std.math.pow(f64, D, -0.5));
        try k.triton.run(stream, "_block_attention", .{ @intCast(S), dkv, 1 }, &.{
            p("Q", "*bf16", q),        p("KB", "*bf16", kb),      p("VB", "*bf16", vb),       p("TABLE", "*i64", tb), p("LENS", "*i32", tb + S * 16), p("O", "*bf16", out[0]),
            aot.float("scale", scale), aot.int("window", window), aot.int("R", @intCast(DR)),
        }, &.{ aot.ci("G", dh / dkv), aot.ci("HKV", dkv), aot.ci("L", L), aot.ci("LP", 16), aot.ci("D", D), aot.ci("BN", 64), aot.ci("CAUSAL", 0) });
        try g.block.run(stream, q, kb, vb, tb, tb + S * 16, out[1], .{ .h = dh, .hk = dkv, .rows = DR, .block = L, .window = window, .scale = scale });
        same += @intFromBool(try b.diff(u16, "out", out[0], out[1], DR * dh * D));
        total += 1;
    }
    std.debug.print("{d} of {d} outputs bit-identical\n", .{ same, total });
    return if (same == total) 0 else 1;
}
