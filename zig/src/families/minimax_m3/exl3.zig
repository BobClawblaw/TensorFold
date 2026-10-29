//! A published EXL3 pack (mul1) as resident projection jobs: one shared table pointing into the shard buffers.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const xr = @import("../../core/exl3_rows.zig");

pub const View = xr.View;

/// EXL3 matrices on the same rows and K, written side by side (q | k | v, gate | up): one dispatch.
pub const Set = struct { jobs: View, count: usize, blocks: usize, k: usize, stride: usize };
/// A layer's routed experts, `kinds` n-wide matrices each (gate, up; or down), then the shared expert's alike.
pub const Routed = struct { jobs: View, experts: usize, kinds: usize, n: usize, k: usize };
/// fp16 [n, k] matrices on the same rows, at their columns (the index q | index k projections).
pub const Half = struct { jobs: View, count: usize, cols: usize, k: usize };

/// exl3_glue.metal's mm_half_job.
pub const HalfJob = extern struct { w: u64, n: u32, y_col: u32, first: u32, pad: u32 = 0 };

/// The pack's own markers: quant_method exl3 with the mul1 codebook.
pub fn isExl3(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return false;
    const q = doc.value.object.get("quantization_config") orelse return false;
    if (q != .object) return false;
    const method = q.object.get("quant_method") orelse return false;
    if (method != .string or !std.mem.eql(u8, method.string, "exl3")) return false;
    const book = q.object.get("codebook") orelse return error.UnsupportedMiniMaxExl3;
    if (book != .string or !std.mem.eql(u8, book.string, "mul1")) return error.UnsupportedMiniMaxExl3;
    return true;
}

/// EXL3 jobs, then fp16 jobs, placed in one shared buffer at the end; `fix` rebases views made before that.
pub const Table = struct {
    gpa: std.mem.Allocator,
    jobs: std.ArrayList(xr.Job) = .empty,
    halves: std.ArrayList(HalfJob) = .empty,
    buffer: ?mtl.Buffer = null,

    pub fn deinit(t: *Table) void {
        t.jobs.deinit(t.gpa);
        t.halves.deinit(t.gpa);
    }

    pub fn place(t: *Table, device: mtl.Device) !mtl.Buffer {
        const ex_bytes = t.jobs.items.len * @sizeOf(xr.Job);
        const bytes = ex_bytes + t.halves.items.len * @sizeOf(HalfJob);
        const b = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
        @memcpy(b.contents()[0..ex_bytes], std.mem.sliceAsBytes(t.jobs.items));
        @memcpy(b.contents()[ex_bytes..bytes], std.mem.sliceAsBytes(t.halves.items));
        t.buffer = b;
        return b;
    }

    /// An EXL3 view onto the placed buffer.
    pub fn fix(t: *const Table, v: *View) void {
        v.buffer = t.buffer.?;
    }
    /// An fp16 view onto the placed buffer, after the EXL3 jobs.
    pub fn fixHalf(t: *const Table, v: *View) void {
        v.buffer = t.buffer.?;
        v.offset += t.jobs.items.len * @sizeOf(xr.Job);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// The EXL3 group `prefix`: trellis int16 [k/16, n/16, 16 * bits], fp16 suh [k] and svh [n], the mul1 marker.
pub fn matrix(c: *const ck.Checkpoint, prefix: []const u8, k: usize, n: usize) !xr.Matrix {
    var b: [256]u8 = undefined;
    const tr = try c.get(try name(&b, prefix, ".trellis"));
    const su = try c.get(try name(&b, prefix, ".suh"));
    const sv = try c.get(try name(&b, prefix, ".svh"));
    const mark = try c.get(try name(&b, prefix, ".mul1"));
    if (mark.dtype != .i32 or mark.bytes != 4 or mark.host(u32)[0] != 0x83DCD12D) return bad(prefix);
    if (tr.dtype != .i16 or tr.rank != 3 or tr.shape[0] != k / 16 or tr.shape[1] != n / 16 or tr.shape[2] % 16 != 0) return bad(prefix);
    if (su.dtype != .f16 or su.rank != 1 or su.shape[0] != k or sv.dtype != .f16 or sv.rank != 1 or sv.shape[0] != n) return bad(prefix);
    const bits = tr.shape[2] / 16;
    if (bits < 1 or bits > 8 or tr.bytes != (k / 16) * (n / 16) * tr.shape[2] * 2) return bad(prefix);
    const m = xr.Matrix{ .trellis = .{ .buffer = tr.buffer, .offset = tr.offset }, .suh = .{ .buffer = su.buffer, .offset = su.offset }, .svh = .{ .buffer = sv.buffer, .offset = sv.offset }, .k = k, .n = n, .width = .{ .halves = @intCast(2 * bits) } };
    m.validate() catch return bad(prefix);
    return m;
}

fn bad(prefix: []const u8) error{UnexpectedMiniMaxExl3} {
    std.log.err("minimax_m3: {s} is not an EXL3 mul1 group of the expected shape", .{prefix});
    return error.UnexpectedMiniMaxExl3;
}

/// Matrices `prefixes` (each [k, ns[i]]) as one set, side by side from column 0.
pub fn set(t: *Table, c: *const ck.Checkpoint, base: []const u8, prefixes: []const []const u8, k: usize, ns: []const usize) !Set {
    const first = t.jobs.items.len;
    var col: usize = 0;
    for (prefixes, ns) |p, n| {
        var b: [256]u8 = undefined;
        try t.jobs.append(t.gpa, try xr.job(try matrix(c, try name(&b, base, p), k, n), col));
        col += n;
    }
    const blocks = xr.number(t.jobs.items[first..]);
    return .{ .jobs = .{ .buffer = undefined, .offset = first * @sizeOf(xr.Job) }, .count = prefixes.len, .blocks = blocks, .k = k, .stride = col };
}

/// Experts `{base}{e}{kind}` at columns kind * n, then the shared expert's `{shared}{kind}` the same way.
pub fn routed(t: *Table, c: *const ck.Checkpoint, base: []const u8, kinds: []const []const u8, shared: []const u8, shared_kinds: []const []const u8, experts: usize, k: usize, n: usize) !Routed {
    const first = t.jobs.items.len;
    for (0..experts + 1) |e| for (kinds, shared_kinds, 0..) |kind, sk, ki| {
        var b: [256]u8 = undefined;
        const p = if (e < experts) try std.fmt.bufPrint(&b, "{s}{d}{s}", .{ base, e, kind }) else try std.fmt.bufPrint(&b, "{s}{s}", .{ shared, sk });
        try t.jobs.append(t.gpa, try xr.job(try matrix(c, p, k, n), ki * n));
    };
    return .{ .jobs = .{ .buffer = undefined, .offset = first * @sizeOf(xr.Job) }, .experts = experts, .kinds = kinds.len, .n = n, .k = k };
}

/// fp16 [n, k] matrices `{base}{p}.weight` side by side from column `col`.
pub fn half(t: *Table, c: *const ck.Checkpoint, base: []const u8, prefixes: []const []const u8, k: usize, ns: []const usize, col: usize) !Half {
    const first = t.halves.items.len;
    var at = col;
    var cols: usize = 0;
    for (prefixes, ns) |p, n| {
        var b: [256]u8 = undefined;
        const full = try std.fmt.bufPrint(&b, "{s}{s}.weight", .{ base, p });
        const w = try c.get(full);
        if (w.dtype != .f16 or w.rank != 2 or w.shape[0] != n or w.shape[1] != k or n % 8 != 0 or k % 4 != 0 or w.offset % 2 != 0) {
            std.log.err("minimax_m3: {s} is not fp16 [{d}, {d}]", .{ full, n, k });
            return error.UnexpectedMiniMaxExl3;
        }
        try t.halves.append(t.gpa, .{ .w = w.buffer.gpuAddress() + w.offset, .n = @intCast(n), .y_col = @intCast(at), .first = @intCast(cols) });
        at += n;
        cols += n;
    }
    return .{ .jobs = .{ .buffer = undefined, .offset = first * @sizeOf(HalfJob) }, .count = prefixes.len, .cols = cols, .k = k };
}

/// The largest call's partials and arrival counters: `rows` rows of every set, `slots` of every routed table.
pub fn scratchBytes(sets: []const Set, routeds: []const Routed, rows: usize, slots: usize) struct { partial: usize, counts: usize } {
    var partial: usize = 16;
    var counts: usize = 16;
    for (sets) |s| {
        partial = @max(partial, xr.partialBytes(xr.ranges(s.blocks, s.k, xr.dense_target), rows, s.blocks));
        counts = @max(counts, xr.countBytes(rows, s.blocks));
    }
    for (routeds) |r| {
        const blocks = r.kinds * r.n / 128;
        partial = @max(partial, xr.partialBytes(xr.ranges(blocks, r.k, xr.routed_target), slots, blocks));
        counts = @max(counts, xr.countBytes(slots, blocks));
    }
    return .{ .partial = partial, .counts = counts };
}

test "the fp16 job layout matches exl3_glue.metal" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(HalfJob));
}
