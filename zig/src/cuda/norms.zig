//! norms.cu's RMS norm rows for every CUDA family: a block a row, its bits independent of the row count.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Element kinds, numbered as the kernels (and hip/ops/common.hpp) number them.
pub const Kind = enum(u8) { f32 = 0, f16 = 1, bf16 = 2 };

const threads = 1024;
/// norms.cu's 16-byte forms: VNT threads, VEC_MOST loads of 8 a thread.
const vec_threads = 256;
const vec_most = 4;

/// The 16-byte forms take the row: width a multiple of 8 within VEC_MOST * 8 * VNT, every address 16-byte aligned.
fn vectorOk(width: usize, ptrs: []const u64) bool {
    if (width % 8 != 0 or width > vec_most * 8 * vec_threads) return false;
    for (ptrs) |p| if (p % 16 != 0) return false;
    return true;
}

pub const Norms = struct {
    mod: Module,
    rms_fn: Function,
    add_rms_fn: Function,
    rms_bf16_fn: Function,
    add_rms_bf16_fn: Function,

    pub fn load(d: *const Driver) !Norms {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.norms);
        errdefer mod.unload();
        return .{ .mod = mod, .rms_fn = try mod.function("tf_rms_rows"), .add_rms_fn = try mod.function("tf_add_rms_rows"), .rms_bf16_fn = try mod.function("tf_rms_rows_bf16"), .add_rms_bf16_fn = try mod.function("tf_add_rms_rows_bf16") };
    }

    pub fn unload(n: *Norms) void {
        n.mod.unload();
    }

    /// `rows` rows of `width`: y = x * rsqrt(mean(x^2) + eps) * w (w null: no weight), fp32 weights.
    pub fn rms(n: *const Norms, s: Stream, x: u64, x_kind: Kind, w: ?u64, y: u64, y_kind: Kind, rows: usize, width: usize, eps: f32) !void {
        if (rows == 0) return;
        if (x_kind == .bf16 and y_kind == .bf16 and vectorOk(width, &.{ x, y, w orelse 0 })) {
            var a: launch_.Args = .{};
            a.add(x);
            a.add(w orelse 0);
            a.add(y);
            a.add(@as(i32, @intCast(width)));
            a.add(eps);
            return launch_.launch(n.rms_bf16_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
        }
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @backingInt(x_kind)));
        a.add(w orelse 0);
        a.add(y);
        a.add(@as(i32, @backingInt(y_kind)));
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        try launch_.launch(n.rms_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// The residual step: res (fp32, in place) += rms(x) * w1 (w1 null: res += x), then y = rms(res) * w2.
    pub fn addRms(n: *const Norms, s: Stream, x: u64, x_kind: Kind, res: u64, w1: ?u64, w2: u64, y: u64, y_kind: Kind, rows: usize, width: usize, eps: f32) !void {
        if (rows == 0) return;
        if (x_kind == .bf16 and y_kind == .bf16 and vectorOk(width, &.{ x, y, res, w1 orelse 0, w2 })) {
            var a: launch_.Args = .{};
            a.add(x);
            a.add(res);
            a.add(w1 orelse 0);
            a.add(w2);
            a.add(y);
            a.add(@as(i32, @intCast(width)));
            a.add(eps);
            return launch_.launch(n.add_rms_bf16_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
        }
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @backingInt(x_kind)));
        a.add(res);
        a.add(w1 orelse 0);
        a.add(w2);
        a.add(y);
        a.add(@as(i32, @backingInt(y_kind)));
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        try launch_.launch(n.add_rms_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};

test "kinds keep the kernels' numbering" {
    try std.testing.expectEqual(@as(u8, 2), @backingInt(Kind.bf16));
}
