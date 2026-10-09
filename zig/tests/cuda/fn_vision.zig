//! Flash Next on CUDA: the vision tower (src/families/flashnext_cuda/vision.zig) against the Python engine's torch
//! tower on CUDA (flashnext-zig/tools/vision_ref.py ... cuda): each reference image's patches encoded here, the
//! features compared. The bf16 tower is not reproducible to the bit even within torch (its CPU and CUDA runs differ by
//! 2.5-11% relative error on these images), so the bound is that noise's: relative error at most 15% and the median
//! row's cosine at least 0.999.
//! fn-vision <model dir> <reference dir> [triton kernel dir]

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fw = @import("flashnext_weights");
const Gpu = check.Gpu;

pub fn run(gpu: Gpu, model: []const u8, refdir: []const u8, triton: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var kernels = try fw.api.Kernels.load(gpa, io, gpu.d, gpu.ctx.device, triton);
    defer kernels.deinit();
    var k = try fw.kern.K.init(gpa, gpu.d, gpu.ctx, stream, &kernels.triton, &kernels.ext);
    const t0 = std.Io.Timestamp.now(io, .awake);
    var tower = try fw.vision.Tower.load(gpa, io, gpu.d, model);
    defer tower.deinit();
    std.debug.print("vision tower loaded in {d:.1} s\n", .{@as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e9});
    var nb: [512]u8 = undefined;
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/index.json", .{refdir}), gpa, .limited(1 << 20));
    const index = try std.json.parseFromSlice(std.json.Value, gpa, index_text, .{});
    var worst_rel: f64 = 0;
    var worst_median: f64 = 1;
    for (index.value.array.items) |case| {
        const name = case.object.get("name").?.string;
        const g = case.object.get("grid").?.array.items[0].array.items;
        const grid = [3]i64{ g[0].integer, g[1].integer, g[2].integer };
        const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/{s}.patches.f32", .{ refdir, name }), gpa, .limited(1 << 30));
        const patches: []align(1) const f32 = std.mem.bytesAsSlice(f32, pbytes);
        const pcopy = try gpa.alloc(f32, patches.len);
        @memcpy(pcopy, patches);
        const rbytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/{s}.features.bf16", .{ refdir, name }), gpa, .limited(1 << 30));
        const rows: usize = @intCast(@divExact(grid[1] * grid[2], 4));
        var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * 2560 * 2);
        defer out.free();
        const te = std.Io.Timestamp.now(io, .awake);
        try tower.encode(&k, pcopy, &.{grid}, out.ptr);
        try stream.synchronize();
        const ms = @as(f64, @floatFromInt(te.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
        const got = try gpa.alloc(u16, rows * 2560);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(@ptrCast(got.ptr), out.ptr, got.len * 2), "features");
        const want: []align(1) const u16 = std.mem.bytesAsSlice(u16, rbytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "{s}/{s}.got.bf16", .{ refdir, name }), .data = std.mem.sliceAsBytes(got) });
        var max_abs: f32 = 0;
        var sum_abs: f64 = 0;
        var min_cos: f64 = 1;
        var diff2: f64 = 0;
        var ref2: f64 = 0;
        const cosines = try gpa.alloc(f64, rows);
        for (0..rows) |r| {
            var dot: f64 = 0;
            var na: f64 = 0;
            var nbv: f64 = 0;
            for (0..2560) |c| {
                const a = bf(got[r * 2560 + c]);
                const b = bf(want[r * 2560 + c]);
                max_abs = @max(max_abs, @abs(a - b));
                sum_abs += @abs(a - b);
                diff2 += (a - b) * (a - b);
                ref2 += b * b;
                dot += a * b;
                na += a * a;
                nbv += b * b;
            }
            cosines[r] = dot / (@sqrt(na) * @sqrt(nbv) + 1e-30);
            min_cos = @min(min_cos, cosines[r]);
        }
        std.mem.sort(f64, cosines, {}, std.sort.asc(f64));
        const median = cosines[rows / 2];
        const rel = @sqrt(diff2 / ref2);
        worst_rel = @max(worst_rel, rel);
        worst_median = @min(worst_median, median);
        std.debug.print("{s}: grid {any}, {d} features in {d:.1} ms: relative error {d:.4}, median row cosine {d:.5}, worst {d:.4}, max |diff| {d:.4}, mean |diff| {d:.5}\n",
            .{ name, grid, rows, ms, rel, median, min_cos, max_abs, sum_abs / @as(f64, @floatFromInt(rows * 2560)) });
    }
    try check.expect(worst_rel <= 0.15 and worst_median >= 0.999, "relative error {d:.4}, median row cosine {d:.5} against the torch tower", .{ worst_rel, worst_median });
    check.pass("vision tower: features within torch's own CPU/CUDA noise (relative error at most {d:.4}, median row cosine at least {d:.5})", .{ worst_rel, worst_median });
}

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Image preparation (zig/src/vision/qwen_image.zig) against the Python frontend's: each reference PNG decoded and
/// prepared here (one image's budget, 4,096 tokens), the grid and the patches compared value by value (a pixel byte
/// apart is 1/127.5 apart after normalization). vision-prep <reference dir>
pub fn prep(gpu: Gpu, refdir: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const qi = @import("qwen_image");
    var nb: [512]u8 = undefined;
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/index.json", .{refdir}), gpa, .limited(1 << 20));
    const index = try std.json.parseFromSlice(std.json.Value, gpa, index_text, .{});
    var worst_bytes: u32 = 0;
    var grids_equal = true;
    for (index.value.array.items) |case| {
        const name = case.object.get("name").?.string;
        const png = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/{s}.png", .{ refdir, name }), gpa, .limited(1 << 26));
        var im = try qi.decode(gpa, png, .{});
        defer im.deinit(gpa);
        var p = try qi.prepare(gpa, im, 4096);
        defer p.deinit(gpa);
        const g = case.object.get("grid").?.array.items[0].array.items;
        const same_grid = p.grid[0] == g[0].integer and p.grid[1] == g[1].integer and p.grid[2] == g[2].integer;
        grids_equal = grids_equal and same_grid;
        const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/{s}.patches.f32", .{ refdir, name }), gpa, .limited(1 << 30));
        var differ: usize = 0;
        var max_d: f32 = 0;
        if (same_grid and pbytes.len == p.patches.len * 4) {
            for (p.patches, 0..) |v, i| {
                const want: f32 = @bitCast(std.mem.readInt(u32, pbytes[i * 4 ..][0..4], .little));
                const dd = @abs(v - want);
                if (dd > 0) differ += 1;
                max_d = @max(max_d, dd);
            }
        }
        const bytes: u32 = @intFromFloat(@round(max_d * 127.5));
        worst_bytes = @max(worst_bytes, bytes);
        std.debug.print("{s}: {d}x{d} -> grid {any} (Python {d},{d},{d}); {d} of {d} values differ, at most {d} pixel levels\n",
            .{ name, im.w, im.h, p.grid, g[0].integer, g[1].integer, g[2].integer, differ, p.patches.len, bytes });
    }
    try check.expect(grids_equal and worst_bytes <= 1, "grids equal and patches within one pixel level", .{});
    check.pass("image preparation: grids equal Python's, patches within {d} pixel level(s)", .{worst_bytes});
}
