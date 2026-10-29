//! EXL3 rows: a window's rows through resident mul1 projections in one dispatch, both rotations folded in.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const projection = @import("exl3_projection.zig");
const frags = @import("frags.zig");

pub const Matrix = projection.Matrix;
pub const View = projection.View;

/// K a threadgroup splits over simdgroups (the kernel's KS); a threadgroup is 8 * splits simdgroups.
pub const splits = 1;
/// Rows a dense dispatch's threadgroup carries (the widest instantiation).
pub const max_rows = 8;

/// exl3_rows.metal's TfExl3Job: a matrix's resident pointers and its place in the output rows.
pub const Job = extern struct {
    trellis: u64,
    su: u64,
    sv: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
    halves: u32,
    n: u32,
    y_col: u32,
    first_block: u32,
};

const Args = extern struct {
    rows: u32,
    k: u32,
    jobs: u32,
    kinds: u32,
    repeat: u32 = 1,
    output_f32: u32,
    experts: u32 = 0,
    ranges: u32,
    blocks: u32,
    routed_slots: u32 = 0,
    shared_base: u32 = 0,
    pad: u32 = 0,
    x_stride: u64,
    y_stride: u64,
};

pub const Output = projection.Output;

pub const Pipelines = struct {
    dense: [4]mtl.Pipeline, // 1, 2, 4, 8 rows a threadgroup
    routed: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_exl3_rows, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        var made: usize = 0;
        errdefer for (p.dense[0..made]) |q| q.deinit();
        inline for (.{ 1, 2, 4, 8 }, 0..) |r, i| {
            p.dense[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_exl3_rows_r{d}", .{r}), false);
            made += 1;
        }
        p.routed = try mtl.Pipeline.init(device, lib, "tf_exl3_rows_routed", false);
        for (p.dense ++ [_]mtl.Pipeline{p.routed}) |q| if (q.simdWidth() != 32 or q.maxThreads() < threads) return error.UnsupportedExl3Simd;
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.dense) |q| q.deinit();
        p.routed.deinit();
    }
};

const threads = 8 * splits * 32;

/// A matrix's job at output column `y_col`; mul1 with fp16 scales and no bias is the one form the kernel reads.
pub fn job(m: Matrix, y_col: usize) !Job {
    try m.validate();
    if (m.codebook != .mul1 or m.suh_format != .f16 or m.svh_format != .f16 or m.bias != null) return error.UnsupportedExl3Rows;
    // whole bits 2 to 8, 7 excepted (the kernel's widths)
    if (m.width.halves % 2 != 0 or m.width.halves == 14) return error.UnsupportedExl3Rows;
    // the kernel reads tiles as words
    const k_stride = try m.kStride();
    if (m.trellis.offset % 4 != 0 or k_stride % 4 != 0 or m.nStride() % 4 != 0) return error.UnalignedExl3Rows;
    return .{
        .trellis = m.trellis.buffer.gpuAddress() + m.trellis.offset,
        .su = m.suh.buffer.gpuAddress() + m.suh.offset,
        .sv = m.svh.buffer.gpuAddress() + m.svh.offset,
        .tile_k_stride = k_stride / 4,
        .tile_n_stride = m.nStride() / 4,
        .halves = m.width.halves,
        .n = @intCast(m.n),
        .y_col = @intCast(y_col),
        .first_block = 0,
    };
}

/// Jobs sharing K and rows, their blocks numbered in order; the table is the caller's resident buffer.
pub fn number(jobs: []Job) usize {
    var blocks: usize = 0;
    for (jobs) |*j| {
        j.first_block = @intCast(blocks);
        blocks += j.n / 128;
    }
    return blocks;
}

/// K ranges for a shape (never the rows, so a row's sums keep one order): enough threadgroups to fill the GPU.
pub fn ranges(blocks: usize, k: usize) usize {
    const most = @max(1, k / 128 / splits);
    var r: usize = 1;
    while (r < most and blocks * r < 192) r *= 2;
    return @min(r, most);
}

/// The fp32 partials a call with `ranges` ranges needs (none for one range).
pub fn partialBytes(ranges_: usize, rows: usize, blocks: usize) usize {
    return if (ranges_ <= 1) 0 else ranges_ * rows * blocks * 128 * 4;
}

/// The arrival counters (u32, zeroed once; every call leaves them zeroed) a call over `rows` rows needs.
pub fn countBytes(rows: usize, blocks: usize) usize {
    return rows * blocks * 4;
}

/// Scratch a call reads and writes besides its rows: fp32 partials, then the arrival counters (zeroed at allocation).
pub const Scratch = struct { partial: View, counts: View };

pub const Dense = struct { jobs: View, count: usize, blocks: usize, k: usize };

/// `rows` rows of x through every job into y (each at its column) over `kt` K ranges, partials in `sc`.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, d: Dense, x: View, x_stride: usize, y: View, y_stride: usize, rows: usize, out: Output, kt: usize, sc: Scratch) void {
    const per: usize = if (rows >= 8) 8 else if (rows >= 4) 4 else if (rows >= 2) 2 else 1;
    const which: usize = switch (per) {
        1 => 0,
        2 => 1,
        4 => 2,
        else => 3,
    };
    const a = Args{ .rows = @intCast(rows), .k = @intCast(d.k), .jobs = @intCast(d.count), .kinds = 1, .output_f32 = @intFromBool(out == .f32), .ranges = @intCast(kt), .blocks = @intCast(d.blocks), .x_stride = x_stride, .y_stride = y_stride };
    e.setPipeline(p.dense[which]);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(d.jobs.buffer, d.jobs.offset, 1);
    e.setBuffer(y.buffer, y.offset, 2);
    e.setValue(a, 3);
    e.setBuffer(sc.partial.buffer, sc.partial.offset, 4);
    e.setBuffer(sc.counts.buffer, sc.counts.offset, 5);
    e.dispatchGroups(mtl.Size.of(d.blocks, (rows + per - 1) / per, kt), mtl.Size.of(threads, 1, 1));
}

/// Routed experts: experts * kinds jobs of one n (with `shared`, the shared expert's after); slot s: ids[s].
pub const Routed = struct { jobs: View, experts: usize, kinds: usize, n: usize, k: usize, shared: bool = false };

/// Slots [0, routed) as routed, and with r.shared slots [routed, slots) on the shared expert, x row slot - shared_base.
pub fn encodeRouted(p: *const Pipelines, e: mtl.ComputeEncoder, r: Routed, x: View, x_stride: usize, ids: View, repeat: usize, y: View, y_stride: usize, routed: usize, slots: usize, shared_base: usize, out: Output, kt: usize, sc: Scratch) void {
    std.debug.assert(r.shared or routed == slots);
    const blocks = r.kinds * r.n / 128;
    const a = Args{ .rows = @intCast(slots), .k = @intCast(r.k), .jobs = @intCast((r.experts + @intFromBool(r.shared)) * r.kinds), .kinds = @intCast(r.kinds), .repeat = @intCast(repeat), .output_f32 = @intFromBool(out == .f32), .experts = @intCast(r.experts), .ranges = @intCast(kt), .blocks = @intCast(blocks), .routed_slots = @intCast(routed), .shared_base = @intCast(shared_base), .x_stride = x_stride, .y_stride = y_stride };
    e.setPipeline(p.routed);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(r.jobs.buffer, r.jobs.offset, 1);
    e.setBuffer(y.buffer, y.offset, 2);
    e.setValue(a, 3);
    e.setBuffer(sc.partial.buffer, sc.partial.offset, 4);
    e.setBuffer(sc.counts.buffer, sc.counts.offset, 5);
    e.setBuffer(ids.buffer, ids.offset, 6);
    e.dispatchGroups(mtl.Size.of(blocks, slots, kt), mtl.Size.of(threads, 1, 1));
}

const MmArgs = extern struct {
    rows: u32,
    k: u32,
    jobs: u32,
    kinds: u32,
    experts: u32 = 0,
    blocks: u32,
    routed: u32 = 0,
    scatter: u32 = 0,
    x_stride: u64,
    y_stride: u64,
};

/// Rows a prompt threadgroup carries (exl3_mm.metal's BM).
pub const mm_rows = 64;

/// Prompt chunks on the tensor units (exl3_mm.metal): dense sets, and expert-sorted rows.
pub const MmPipelines = struct {
    dense: mtl.Pipeline,
    gather: mtl.Pipeline,

    pub fn load(device: mtl.Device, gpa: std.mem.Allocator) !MmPipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const text = try frags.source(device, gpa, sources.core_exl3_mm);
        defer gpa.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const dense = try mtl.Pipeline.init(device, lib, "tf_exl3_mm", false);
        errdefer dense.deinit();
        return .{ .dense = dense, .gather = try mtl.Pipeline.init(device, lib, "tf_exl3_mm_gather", false) };
    }

    pub fn deinit(p: *MmPipelines) void {
        p.dense.deinit();
        p.gather.deinit();
    }
};

/// A prompt chunk's `rows` rows through a dense set (bf16 out), each weight decoded once a 64-row tile.
pub fn encodeMm(p: *const MmPipelines, e: mtl.ComputeEncoder, d: Dense, x: View, x_stride: usize, y: View, y_stride: usize, rows: usize) void {
    const a = MmArgs{ .rows = @intCast(rows), .k = @intCast(d.k), .jobs = @intCast(d.count), .kinds = 1, .blocks = @intCast(d.blocks), .x_stride = x_stride, .y_stride = y_stride };
    e.setPipeline(p.dense);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(d.jobs.buffer, d.jobs.offset, 1);
    e.setBuffer(y.buffer, y.offset, 2);
    e.setValue(a, 3);
    e.dispatchGroups(mtl.Size.of(d.blocks, (rows + mm_rows - 1) / mm_rows, 1), mtl.Size.of(128, 1, 1));
}

/// Expert-sorted rows (`offsets`: each expert's first, i32) through its matrices; with `dst`, row i to dst[i].
pub fn encodeMmGather(p: *const MmPipelines, e: mtl.ComputeEncoder, r: Routed, x: View, x_stride: usize, offsets: View, dst: ?View, y: View, y_stride: usize, rows: usize) void {
    const blocks = r.kinds * r.n / 128;
    const a = MmArgs{ .rows = @intCast(rows), .k = @intCast(r.k), .jobs = @intCast(r.experts * r.kinds), .kinds = @intCast(r.kinds), .experts = @intCast(r.experts), .blocks = @intCast(blocks), .routed = 1, .scatter = @intFromBool(dst != null), .x_stride = x_stride, .y_stride = y_stride };
    e.setPipeline(p.gather);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(r.jobs.buffer, r.jobs.offset, 1);
    e.setBuffer(y.buffer, y.offset, 2);
    e.setValue(a, 3);
    e.setBuffer(offsets.buffer, offsets.offset, 4);
    const d = dst orelse offsets;
    e.setBuffer(d.buffer, d.offset, 5);
    // each expert's last tile may be partial
    const tiles = @min(rows, (rows + mm_rows - 1) / mm_rows + r.experts - 1);
    e.dispatchGroups(mtl.Size.of(blocks, tiles, 1), mtl.Size.of(128, 1, 1));
}

test "the job and argument layouts match exl3_rows.metal" {
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Job));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Args));
}
