//! qk_rows.cu's attention inputs for every CUDA family: QK norm, RoPE (rotate-half, partial) and the cache write.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// A packed row's heads: h query, hk key and hk value heads of `dim` (a multiple of 32, at most 256).
pub const Heads = struct { h: usize, hk: usize, dim: usize };

/// What a call does to q and k: norms fp32 [2, dim] (q, k) or none; RoPE on `rot` dims from fp32 [positions, rot].
pub const Shape = struct { norms: ?u64 = null, cos: ?u64 = null, sin: ?u64 = null, rot: usize = 0, eps: f32 = 1e-6 };

/// Where the rows go: q [n, h, dim], k and v [n, hk, dim]; k and v also into kc/vc at (slot * len + pos % len).
pub const Out = struct { q: u64, k: u64, v: u64, kc: ?u64 = null, vc: ?u64 = null, len: usize = 1 };

pub const QkRows = struct {
    mod: Module,
    fn_: Function,
    /// tf_qk_rows_w: dim 128 with no or full rotation, a warp a head.
    warp_fn: Function,

    pub fn load(d: *const Driver) !QkRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qk_rows);
        errdefer mod.unload();
        return .{ .mod = mod, .fn_ = try mod.function("tf_qk_rows"), .warp_fn = try mod.function("tf_qk_rows_w") };
    }

    pub fn unload(g: *QkRows) void {
        g.mod.unload();
    }

    /// n packed rows qkv [n, (h + 2 hk) dim] bf16 at positions pos (int32 [n]) in cache slots slot (int32 [n] or none).
    pub fn rows(g: *const QkRows, s: Stream, qkv: u64, hs: Heads, shape: Shape, pos: u64, slot: ?u64, out: Out, n: usize) !void {
        if (hs.dim % 32 != 0 or hs.dim > 256 or shape.rot % 2 != 0 or shape.rot > hs.dim) return error.HeadShape;
        if ((shape.cos == null) != (shape.sin == null) or (shape.cos != null and shape.rot == 0)) return error.HeadShape;
        if (n == 0) return;
        var a: launch_.Args = .{};
        a.add(qkv);
        a.add(shape.norms orelse 0);
        a.add(shape.cos orelse 0);
        a.add(shape.sin orelse 0);
        a.add(@as(i32, @intCast(if (shape.cos != null) shape.rot else 0)));
        a.add(pos);
        a.add(slot orelse 0);
        a.add(out.q);
        a.add(out.k);
        a.add(out.v);
        a.add(out.kc orelse 0);
        a.add(out.vc orelse 0);
        a.add(@as(i64, @intCast(out.len)));
        a.add(@as(i32, @intCast(hs.h)));
        a.add(@as(i32, @intCast(hs.hk)));
        a.add(shape.eps);
        const heads = hs.h + 2 * hs.hk;
        if (hs.dim == 128 and (shape.cos == null or shape.rot == 128)) {
            const grid: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast((heads + 3) / 4), .z = 1 };
            return launch_.launch(g.warp_fn, .{ .grid = grid, .block = .{ .x = 128 } }, s, &a);
        }
        const grid: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast(heads), .z = 1 };
        try launch_.launch(g.fn_, .{ .grid = grid, .block = .{ .x = @intCast(hs.dim) } }, s, &a);
    }
};
