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

const threads = 256;

pub const Norms = struct {
    mod: Module,
    rms_fn: Function,
    add_rms_fn: Function,

    pub fn load(d: *const Driver) !Norms {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.norms);
        errdefer mod.unload();
        return .{ .mod = mod, .rms_fn = try mod.function("tf_rms_rows"), .add_rms_fn = try mod.function("tf_add_rms_rows") };
    }

    pub fn unload(n: *Norms) void {
        n.mod.unload();
    }

    /// `rows` rows of `width`: y = x * rsqrt(mean(x^2) + eps) * w (w null: no weight), fp32 weights.
    pub fn rms(n: *const Norms, s: Stream, x: u64, x_kind: Kind, w: ?u64, y: u64, y_kind: Kind, rows: usize, width: usize, eps: f32) !void {
        if (rows == 0) return;
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
