//! Llama's checkpoint on Metal: the release's bf16 matrices in place, or an MLX affine checkpoint's triples in place.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const rows = @import("../../core/row_projection.zig");
const cfg = @import("config.zig");

pub const Tensor = ck.Tensor;

/// A bf16 [n, k] matrix as the release stores it, at `off` in `buf`.
pub const Dense = struct { buf: mtl.Buffer, off: usize, n: usize, k: usize };

/// A matrix's weights: the release's bf16 rows, or one MLX affine width in groups of 64 (mlx_lm's layout).
pub const Linear = union(enum) { bf16: Dense, affine: rows.Weights };

/// q, k and v write side by side into one row, gate and up into another, each from its own matrix.
pub const Attention = struct { q: Linear, k: Linear, v: Linear, o: Linear };
pub const Mlp = struct { gate: Linear, up: Linear, down: Linear };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attn: Attention, mlp: Mlp };

pub const Weights = struct {
    embed: Linear,
    head: Linear,
    norm: Tensor,
    layers: []Layer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// `t` is `dtype` with exactly `dims`, its bytes those dims' extent.
fn require(t: Tensor, dtype: @FieldType(Tensor, "dtype"), dims: []const usize) error{UnexpectedLlamaTensor}!void {
    if (t.dtype != dtype or t.rank != dims.len or t.rank > t.shape.len) return error.UnexpectedLlamaTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedLlamaTensor;
    var bytes: usize = if (dtype == .bf16) 2 else 4;
    for (dims) |n| bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedLlamaTensor;
    if (t.bytes != bytes) return error.UnexpectedLlamaTensor;
}

fn norm(c: *const ck.Checkpoint, hidden: usize, prefix: []const u8, suffix: []const u8) !Tensor {
    var buf: [256]u8 = undefined;
    const n = try name(&buf, prefix, suffix);
    const t = try c.get(n);
    require(t, .bf16, &.{hidden}) catch |err| {
        std.log.err("llama: {s} is not bf16[{d}]", .{ n, hidden });
        return err;
    };
    return t;
}

/// The release's bf16 [n, k] matrix in place: k whole 256s for the GEMM's steps, 8-byte aligned for the row kernel.
fn dense(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Dense {
    var buf: [256]u8 = undefined;
    const full = try name(&buf, prefix, suffix);
    const w = try c.get(full);
    require(w, .bf16, &.{ n, k }) catch |err| {
        std.log.err("llama: {s} is not bf16[{d}, {d}]", .{ full, n, k });
        return err;
    };
    if (k % 256 != 0 or w.offset % 8 != 0) return error.UnexpectedLlamaTensor;
    return .{ .buf = w.buffer, .off = w.offset, .n = n, .k = k };
}

/// `prefix ++ base`'s affine [n, k] as mlx_lm writes it: u32 words [n, k * bits / 32], bf16 scales, biases [n, k / 64].
fn affine(c: *const ck.Checkpoint, bits: u8, prefix: []const u8, base: []const u8, n: usize, k: usize) !rows.Weights {
    var wb: [256]u8 = undefined;
    var sb: [256]u8 = undefined;
    var bb: [256]u8 = undefined;
    const wn = try std.fmt.bufPrint(&wb, "{s}{s}.weight", .{ prefix, base });
    const w = try c.get(wn);
    const s = try c.get(try std.fmt.bufPrint(&sb, "{s}{s}.scales", .{ prefix, base }));
    const z = try c.get(try std.fmt.bufPrint(&bb, "{s}{s}.biases", .{ prefix, base }));
    require(w, .u32, &.{ n, k * bits / 32 }) catch |err| {
        std.log.err("llama: {s} is not u32[{d}, {d}] ({d}-bit words)", .{ wn, n, k * bits / 32, bits });
        return err;
    };
    try require(s, .bf16, &.{ n, k / cfg.group });
    try require(z, .bf16, &.{ n, k / cfg.group });
    if (n % 64 != 0) return error.UnexpectedLlamaTensor;
    const out = rows.Weights{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = z.buffer, .b_off = z.offset, .n = n, .k = k, .group = cfg.group, .bits = bits, .sum = .f32 };
    try out.validate();
    return out;
}

/// `prefix ++ base`: the release's bf16 `.weight`, or the MLX checkpoint's affine triple.
fn linear(c: *const ck.Checkpoint, bits: u8, prefix: []const u8, base: []const u8, n: usize, k: usize) !Linear {
    if (bits != cfg.bf16) return .{ .affine = try affine(c, bits, prefix, base, n, k) };
    var buf: [128]u8 = undefined;
    return .{ .bf16 = try dense(c, prefix, try name(&buf, base, ".weight"), n, k) };
}

pub fn load(gpa: std.mem.Allocator, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    const h = f.hidden;
    var w = Weights{
        .embed = try linear(c, f.bits, "model.", "embed_tokens", f.vocab, h),
        .head = try linear(c, f.bits, "", "lm_head", f.vocab, h),
        .norm = try norm(c, h, "model.", "norm.weight"),
        .layers = try gpa.alloc(Layer, f.layers),
    };
    errdefer w.deinit(gpa);
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        var ab: [96]u8 = undefined;
        const a = try name(&ab, p, "self_attn.");
        var mb: [96]u8 = undefined;
        const m = try name(&mb, p, "mlp.");
        l.* = .{
            .input_norm = try norm(c, h, p, "input_layernorm.weight"),
            .post_norm = try norm(c, h, p, "post_attention_layernorm.weight"),
            .attn = .{
                .q = try linear(c, f.bits, a, "q_proj", f.queryWidth(), h),
                .k = try linear(c, f.bits, a, "k_proj", f.kvWidth(), h),
                .v = try linear(c, f.bits, a, "v_proj", f.kvWidth(), h),
                .o = try linear(c, f.bits, a, "o_proj", h, f.queryWidth()),
            },
            .mlp = .{
                .gate = try linear(c, f.bits, m, "gate_proj", f.intermediate, h),
                .up = try linear(c, f.bits, m, "up_proj", f.intermediate, h),
                .down = try linear(c, f.bits, m, "down_proj", h, f.intermediate),
            },
        };
    }
    return w;
}
