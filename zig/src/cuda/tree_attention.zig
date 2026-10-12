//! tree_attention.cu for every CUDA family: drafted rows over committed keys and their paths, groups in order.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Keys a group (one partial), a path's most rows, and the part kernel's shared memory (four warps' K and V slots).
pub const span = 2048;
pub const max_depth = 128;
const part_smem: u32 = 4 * 2 * 32 * 16 * 16;

/// row_stream [n], streams [s, 4] (committed at [2]), offs [s, 2] (K, V from origin), paths [n, max_depth], depths.
pub const Rows = struct { row_stream: u64, streams: u64, offs: u64, origin: u64, paths: u64, depths: u64 };

pub const Scratch = struct { o: u64, m: u64, l: u64 };

pub const TreeAttention = struct {
    mod: Module,
    paths_fn: Function,
    part_fn: Function,
    merge_fn: Function,
    /// Blocks a (row, key head) at most; each takes every `group_slots`-th group (bits do not depend on it).
    group_slots: usize = 16,

    pub fn load(d: *const Driver) !TreeAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.tree_attention);
        errdefer mod.unload();
        const part = try mod.function("tf_tree_part");
        try part.allowDynamicShared(part_smem);
        return .{ .mod = mod, .paths_fn = try mod.function("tf_tree_paths"), .part_fn = part, .merge_fn = try mod.function("tf_tree_merge") };
    }

    pub fn unload(g: *TreeAttention) void {
        g.mod.unload();
    }

    /// Groups a row with `keys` committed and path keys needs.
    pub fn groups(keys: usize) usize {
        return (keys + span - 1) / span;
    }

    /// The fp32 partials of `rows` rows, `hk` key heads and `ng` groups: o [.., 16, 128], then max and sum [.., 16].
    pub fn scratchBytes(rows: usize, hk: usize, ng: usize) usize {
        return rows * hk * ng * 16 * (128 + 2) * 4;
    }

    pub fn scratch(base: u64, rows: usize, hk: usize, ng: usize) Scratch {
        const parts = rows * hk * ng * 16;
        return .{ .o = base, .m = base + parts * 128 * 4, .l = base + parts * 128 * 4 + parts * 4 };
    }

    /// Each row's path (root first) and depth from `parents` int32 [n] (global window rows, -1 at a root).
    pub fn paths(g: *const TreeAttention, s: Stream, parents: u64, out_paths: u64, depths: u64, n: usize) !void {
        if (n == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ parents, out_paths, depths }) |v| a.add(v);
        a.add(@as(i32, @intCast(n)));
        try launch_.launch(g.paths_fn, .{ .grid = .{ .x = @intCast((n + 127) / 128), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    /// q [n, h, 128] over committed keys and paths (kn, vn: window rows) -> out; ng >= a row's groups.
    pub fn run(g: *const TreeAttention, s: Stream, q: u64, kn: u64, vn: u64, rows: Rows, sc: Scratch, out: u64, n: usize, h: usize, hk: usize, ng: usize, scale: f32) !void {
        if (hk == 0 or h % hk != 0 or h / hk > 16 or ng == 0) return error.TreeShape;
        if (n == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ q, kn, vn, rows.origin, rows.offs, rows.row_stream, rows.streams, rows.paths, rows.depths, sc.o, sc.m, sc.l }) |v| a.add(v);
        a.add(@as(i32, @intCast(h)));
        a.add(@as(i32, @intCast(hk)));
        a.add(scale);
        a.add(@as(i32, @intCast(ng)));
        const slots = @max(1, @min(ng, g.group_slots));
        try launch_.launch(g.part_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(hk), .z = @intCast(slots) }, .block = .{ .x = 128 }, .shared = part_smem }, s, &a);
        var b: launch_.Args = .{};
        for ([_]u64{ sc.o, sc.m, sc.l, rows.row_stream, rows.streams, rows.depths, out }) |v| b.add(v);
        b.add(@as(i32, @intCast(h)));
        b.add(@as(i32, @intCast(hk)));
        b.add(@as(i32, @intCast(ng)));
        try launch_.launch(g.merge_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(hk), .z = 1 }, .block = .{ .x = 256 } }, s, &b);
    }
};
