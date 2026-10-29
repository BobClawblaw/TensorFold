//! An EXL3 pack's DSpark head (the release's under dspark/): MXFP8 projections widened to bf16 at load, exactly.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const ds = @import("dspark.zig");

const pre = "language_model.model.dspark.";

/// e4m3fn byte -> bf16 bits (exact: 3 mantissa bits fit bf16's 7); 0x7f and 0xff are NaN and refused.
fn e4m3(u: u8) u16 {
    const e: u32 = (u >> 3) & 0xF;
    const m: u32 = u & 7;
    const v: f32 = if (e == 0) @as(f32, @floatFromInt(m)) / 8.0 * 0x1p-6 else (1.0 + @as(f32, @floatFromInt(m)) / 8.0) * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(e)) - 7.0);
    const bits: u32 = @bitCast(if (u & 0x80 != 0) -v else v);
    return @truncate(bits >> 16);
}

const lut = blk: {
    @setEvalBranchQuota(100000);
    var out: [256]u16 = undefined;
    for (&out, 0..) |*x, i| x.* = e4m3(@intCast(i));
    break :blk out;
};

/// rows [n, k] of e4m3 with e8m0 scales a 32-block into `dst` as bf16; a value outside bf16's normal range refused.
fn widen(c: *const ck.Checkpoint, base: []const u8, n: usize, k: usize, dst: []u16) !void {
    var b1: [192]u8 = undefined;
    var b2: [192]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&b1, "{s}{s}.weight", .{ pre, base }));
    const s = try c.get(try std.fmt.bufPrint(&b2, "{s}{s}.weight_scale_inv", .{ pre, base }));
    if (w.dtype != .f8_e4m3 or w.rank != 2 or w.shape[0] != n or w.shape[1] != k or s.rank != 2 or s.shape[0] != n or s.shape[1] != k / 32 or k % 32 != 0)
        return error.UnexpectedDSparkTensor;
    const codes = w.host(u8);
    const scales = s.host(u8);
    for (0..n) |r| for (0..k / 32) |g| {
        const shift: i32 = @as(i32, scales[r * (k / 32) + g]) - 127;
        for (0..32) |j| {
            const at = r * k + g * 32 + j;
            const code = codes[at];
            if (code & 0x7f == 0x7f) return error.UnexpectedDSparkTensor;
            const v = lut[code];
            if (v & 0x7fff == 0) {
                dst[at] = v;
                continue;
            }
            const exp: i32 = @as(i32, (v >> 7) & 0xff) + shift;
            if (exp <= 0 or exp >= 255) return error.UnexpectedDSparkTensor;
            dst[at] = (v & 0x807f) | (@as(u16, @intCast(exp)) << 7);
        }
    };
}

fn bf16(c: *const ck.Checkpoint, base: []const u8, dims: []const usize) !ck.Tensor {
    var buf: [192]u8 = undefined;
    const x = try c.get(try std.fmt.bufPrint(&buf, "{s}{s}", .{ pre, base }));
    if (x.dtype != .bf16 or x.rank != dims.len or !std.mem.eql(usize, x.shape[0..x.rank], dims)) return error.UnexpectedDSparkTensor;
    return x;
}

/// Matrices stacked by rows, widened into one new buffer.
fn stack(device: mtl.Device, c: *const ck.Checkpoint, owned: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator, parts: []const []const u8, ns: []const usize, k: usize) !ds.Lin {
    var rows: usize = 0;
    for (ns) |n| rows += n;
    const buf = try device.buffer(rows * k * 2, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    var at: usize = 0;
    for (parts, ns) |p, n| {
        try widen(c, p, n, k, buf.slice(u16, rows * k)[at * k .. (at + n) * k]);
        at += n;
    }
    try owned.append(gpa, buf);
    return .{ .bf16 = .{ .buffer = buf, .offset = 0, .n = rows, .k = k } };
}

/// Bytes the widened matrices take (beside the shard itself).
pub fn widenedBytes(cfg: *const ds.Config) u64 {
    const h: u64 = cfg.hidden;
    const qw: u64 = cfg.heads * 128;
    const kvw: u64 = cfg.kv_heads * 128;
    const layer = (qw + 2 * kvw) * h + h * qw + 2 * @as(u64, cfg.ffn) * h + h * cfg.ffn + 2 * kvw * h;
    return layer * cfg.layers * 2;
}

/// The head from the pack's dspark shard (in `c`): bf16 fc, norms and Markov tables as stored, the rest widened.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, cfg: *const ds.Config, vocab: usize, owned: *std.ArrayList(mtl.Buffer)) !ds.Weights {
    const h: usize = cfg.hidden;
    const qw: usize = cfg.heads * 128;
    const kvw: usize = cfg.kv_heads * 128;
    const fc = try bf16(c, "fc.weight", &.{ h, h * cfg.taps.len });
    var w: ds.Weights = .{
        .fc = .{ .bf16 = .{ .buffer = fc.buffer, .offset = fc.offset, .n = h, .k = h * cfg.taps.len } },
        .hidden_norm = try bf16(c, "hidden_norm.weight", &.{h}),
        .norm = try bf16(c, "final_norm.weight", &.{h}),
        .ctx_kv = undefined,
        .w1 = try bf16(c, "markov_head.markov_w1.weight", &.{ vocab, cfg.rank }),
        .w2 = .{ .bf16 = try bf16(c, "markov_head.markov_w2.weight", &.{ vocab, cfg.rank }) },
        .layers = undefined,
    };
    var ctx_parts: [16][]const u8 = undefined;
    var ctx_names: [16][96]u8 = undefined;
    var ctx_ns: [16]usize = undefined;
    for (w.layers[0..cfg.layers], 0..) |*l, i| {
        var nb: [8][96]u8 = undefined;
        const p = struct {
            fn of(buf: *[96]u8, layer: usize, z: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "layers.{d}.decoder_layer.{s}", .{ layer, z });
            }
        }.of;
        const q = try p(&nb[0], i, "self_attn.q_proj");
        const k = try p(&nb[1], i, "self_attn.k_proj");
        const v = try p(&nb[2], i, "self_attn.v_proj");
        var b3: [96]u8 = undefined;
        var b4: [96]u8 = undefined;
        var b5: [96]u8 = undefined;
        var b6: [96]u8 = undefined;
        l.* = .{
            .in_norm = try bf16(c, try p(&b3, i, "input_layernorm.weight"), &.{h}),
            .post_norm = try bf16(c, try p(&b4, i, "post_attention_layernorm.weight"), &.{h}),
            .q_norm = try bf16(c, try p(&b5, i, "self_attn.q_norm.weight"), &.{128}),
            .k_norm = try bf16(c, try p(&b6, i, "self_attn.k_norm.weight"), &.{128}),
            .qkv = try stack(device, c, owned, gpa, &.{ q, k, v }, &.{ qw, kvw, kvw }, h),
            .o = try stack(device, c, owned, gpa, &.{try p(&nb[3], i, "self_attn.o_proj")}, &.{h}, qw),
            .gu = try stack(device, c, owned, gpa, &.{ try p(&nb[4], i, "mlp.gate_proj"), try p(&nb[5], i, "mlp.up_proj") }, &.{ cfg.ffn, cfg.ffn }, h),
            .down = try stack(device, c, owned, gpa, &.{try p(&nb[6], i, "mlp.down_proj")}, &.{h}, cfg.ffn),
        };
        // every layer's k | v again: context rows project through all of them at once
        for ([_][]const u8{ "self_attn.k_proj", "self_attn.v_proj" }, 0..) |z, j| {
            ctx_parts[2 * i + j] = try p(&ctx_names[2 * i + j], i, z);
            ctx_ns[2 * i + j] = kvw;
        }
    }
    w.ctx_kv = try stack(device, c, owned, gpa, ctx_parts[0 .. 2 * cfg.layers], ctx_ns[0 .. 2 * cfg.layers], h);
    return w;
}

test "e4m3 widening is exact on every code" {
    for (0..256) |i| {
        if (i & 0x7f == 0x7f) continue;
        const bits: u32 = @as(u32, lut[i]) << 16;
        const want: f32 = blk: {
            const e: u32 = @intCast((i >> 3) & 0xF);
            const m: f32 = @floatFromInt(i & 7);
            const mag: f32 = if (e == 0) m / 8.0 * 0x1p-6 else (1.0 + m / 8.0) * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(e)) - 7.0);
            break :blk if (i & 0x80 != 0) -mag else mag;
        };
        try std.testing.expectEqual(want, @as(f32, @bitCast(bits)));
    }
}
