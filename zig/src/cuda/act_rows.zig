//! act_rows.cu's activation rows for every CUDA family: SwiGLU, sigmoid output gates and 64-value group sums.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

const threads = 256;

fn aligned(ptrs: []const u64, strides: []const usize) bool {
    for (ptrs) |p| if (p % 16 != 0) return false;
    for (strides) |v| if (v % 8 != 0) return false;
    return true;
}

fn grid(rows: usize, n: usize) launch_.Dim3 {
    return .{ .x = @intCast(rows), .y = @intCast((n / 8 + threads - 1) / threads), .z = 1 };
}

pub const ActRows = struct {
    mod: Module,
    swiglu_fn: Function,
    gate_fn: Function,
    sums_fn: Function,

    pub fn load(d: *const Driver) !ActRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.act_rows);
        errdefer mod.unload();
        return .{ .mod = mod, .swiglu_fn = try mod.function("tf_swiglu_rows"), .gate_fn = try mod.function("tf_gate_rows"), .sums_fn = try mod.function("tf_group_sums_rows") };
    }

    pub fn unload(g: *ActRows) void {
        g.mod.unload();
    }

    /// out [rows, n] = bf16(silu(gate) * up), rows at strides g_ld, u_ld; xs (or null): out's 64-value sums.
    pub fn swiglu(g: *const ActRows, s: Stream, gate: u64, g_ld: usize, up: u64, u_ld: usize, out: u64, xs: ?u64, rows: usize, n: usize) !void {
        if (n % 64 != 0 or !aligned(&.{ gate, up, out }, &.{ g_ld, u_ld })) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(gate);
        a.add(@as(i64, @intCast(g_ld)));
        a.add(up);
        a.add(@as(i64, @intCast(u_ld)));
        a.add(out);
        a.add(xs orelse 0);
        a.add(@as(i32, @intCast(n)));
        try launch_.launch(g.swiglu_fn, .{ .grid = grid(rows, n), .block = .{ .x = threads } }, s, &a);
    }

    /// out = bf16(o * sigmoid(gate)), head h's gate at r * g_ld + h * g_hs ([q | gate]: q + d, 2 d); xs: sums.
    pub fn outputGate(g: *const ActRows, s: Stream, o: u64, gate_: u64, g_ld: usize, g_hs: usize, out: u64, xs: ?u64, rows: usize, heads: usize, d: usize) !void {
        if (d % 64 != 0 or !aligned(&.{ o, gate_, out }, &.{ g_ld, g_hs })) return error.RowShape;
        if (rows == 0 or heads == 0) return;
        var a: launch_.Args = .{};
        a.add(o);
        a.add(gate_);
        a.add(@as(i64, @intCast(g_ld)));
        a.add(@as(i64, @intCast(g_hs)));
        a.add(out);
        a.add(xs orelse 0);
        a.add(@as(i32, @intCast(heads)));
        a.add(@as(i32, @intCast(d)));
        try launch_.launch(g.gate_fn, .{ .grid = grid(rows, heads * d), .block = .{ .x = threads } }, s, &a);
    }

    /// xs [rows, k / 64]: x's rows (stride ld) summed over each 64 values. k % 64 == 0.
    pub fn groupSums(g: *const ActRows, s: Stream, x: u64, ld: usize, xs: u64, rows: usize, k: usize) !void {
        if (k % 64 != 0 or !aligned(&.{x}, &.{ld})) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(ld)));
        a.add(xs);
        a.add(@as(i32, @intCast(k)));
        try launch_.launch(g.sums_fn, .{ .grid = grid(rows, k), .block = .{ .x = threads } }, s, &a);
    }
};
