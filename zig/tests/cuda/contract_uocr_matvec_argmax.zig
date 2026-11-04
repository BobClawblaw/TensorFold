//! tf_uocr_argmax under the contract: the greedy pick under the page guard, checked against a host scan of each row.
const std = @import("std");
const rig_ = @import("contract_rig.zig");
const mv = @import("contract_uocr_matvec.zig");

const c = mv.c;
const g = mv.g;
const eos = mv.cfg.eos;

const rule = "the lowest id of the largest bf16 logit, EOS (id 1) skipped when bar_eos; a NaN never wins; " ++
    "with no logit above -Inf, id 0";

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Launches alternate the bar (odd rows barred); rows 1 and 2 of every 4 lift EOS above every other logit.
fn barred(r: usize) bool {
    return r % 2 == 1;
}
fn spiked(r: usize) bool {
    return r % 4 == 1 or r % 4 == 2;
}

pub const Argmax = struct {
    pub const Variant = enum { plain, nan_as_inf, high_tie, eos_open };
    rig: *rig_.Rig,
    k: mv.Kernels,
    v: Variant,
    shape: c.Shape = undefined,
    logits: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,
    msg: [160]u8 = undefined,

    /// The control's symbol in the controls image.
    pub fn symbol(v: Variant) ?[:0]const u8 {
        return switch (v) {
            .plain => null,
            .nan_as_inf => "tf_ctl_uocr_matvec_argmax_nan",
            .high_tie => "tf_ctl_uocr_matvec_argmax_tie",
            .eos_open => "tf_ctl_uocr_matvec_argmax_eos",
        };
    }

    /// Rows: picks, each its own launch (two with EOS barred, two not); n: the vocabulary; k unused.
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = switch (v) {
                .plain => "cuda uocr matvec tf_uocr_argmax",
                .nan_as_inf => "control: uocr matvec tf_uocr_argmax reads a NaN as +Inf",
                .high_tie => "control: uocr matvec tf_uocr_argmax ties to the higher id",
                .eos_open => "control: uocr matvec tf_uocr_argmax never bars EOS",
            },
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = rule },
            .tiles = .{ .{}, .{ .size = 1024 }, .{} },
            .served = &.{.{ .rows = 4, .n = mv.cfg.vocab, .k = 1 }},
        };
    }

    pub fn deinit(t: *Argmax) void {
        t.free();
    }
    fn free(t: *Argmax) void {
        t.rig.gpu.gpa.free(t.logits);
        t.logits = &.{};
    }
    pub fn kernel(t: *Argmax) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt, .control = switch (t.v) {
            .plain => null,
            .nan_as_inf => .nonfinite,
            .high_tie, .eos_open => .shapes,
        } };
    }
    const vt = g.hooks(Argmax);

    /// Quarter steps in [-16, 16) with -0 among the zeros, so each maximum is held by many ids.
    pub fn prepare(t: *Argmax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        t.logits = try t.rig.gpu.gpa.alloc(u16, shape.rows * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.logits) |*v| {
            const level: i32 = @as(i32, @intCast(rng.uintLessThan(u32, 128))) - 64;
            v.* = c.toBf16(if (level == 0 and rng.boolean()) -0.0 else @as(f32, @floatFromInt(level)) * 0.25);
        }
        if (n > eos) for (0..shape.rows) |r| if (spiked(r)) {
            t.logits[r * n + eos] = c.toBf16(64);
        };
        if (fill == .nonfinite) t.plant();
        t.in = try mv.input(t.rig, t.logits);
        t.out = try t.rig.output(shape.rows * 4);
        return true;
    }

    /// Row 0: NaN at its maximum, 0 and n - 1; 1: +Inf at EOS (barred), n/2, n/3; 2: half -Inf; 3: -Inf; 4: NaN.
    fn plant(t: *Argmax) void {
        const n = t.shape.n;
        const nan = c.toBf16(mv.planted[0]);
        const inf = c.toBf16(mv.planted[1]);
        const ninf = c.toBf16(mv.planted[2]);
        const best = t.scan(0) orelse 0;
        for ([_]usize{ best, 0, n - 1 }) |i| t.logits[i] = nan;
        if (t.shape.rows > 1) for ([_]usize{ eos, n / 2, n / 3 }) |i| if (i < n) {
            t.logits[n + i] = inf;
        };
        if (t.shape.rows > 2) @memset(t.logits[2 * n ..][0 .. n / 2], ninf);
        if (t.shape.rows > 3) @memset(t.logits[3 * n ..][0..n], ninf);
        if (t.shape.rows > 4) @memset(t.logits[4 * n ..][0..n], nan);
    }

    /// The host's scan of row `r`: from -Inf, each logit above the best so far becomes it (EOS skipped when barred).
    fn scan(t: *const Argmax, r: usize) ?usize {
        const n = t.shape.n;
        var best = -std.math.inf(f32);
        var at: ?usize = null;
        for (t.logits[r * n ..][0..n], 0..) |b, i| {
            if (barred(r) and i == eos) continue;
            const v = c.fromBf16(b);
            if (v > best) {
                best = v;
                at = i;
            }
        }
        return at;
    }

    /// Decoder.argmax's launch a pick: one block of 1,024 threads over the vocabulary.
    pub fn run(t: *Argmax) !void {
        const n = t.shape.n;
        for (0..t.shape.rows) |r| {
            const bar: c_int = @intFromBool(barred(r));
            try t.k.run("argmax", t.rig.s, .{ 1, 1, 1 }, 1024, .{ t.in + @as(u64, r * n * 2), n, bar, t.out + @as(u64, r * 4) });
        }
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Argmax) usize {
        return t.shape.rows * 4;
    }

    pub fn reference(_: *Argmax, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Every pick against the host's scan of its row.
    pub fn check(t: *Argmax, _: g.Gate) !?[]const u8 {
        const got = try t.rig.outputBytes();
        for (0..t.shape.rows) |r| {
            const id = std.mem.readInt(u32, got[r * 4 ..][0..4], .little);
            const due: u32 = @intCast(t.scan(r) orelse 0);
            if (id != due) return try std.fmt.bufPrint(&t.msg, "pick {d} (EOS barred: {}): id {d} where {d} is due", .{ r, barred(r), id, due });
        }
        return null;
    }
};
