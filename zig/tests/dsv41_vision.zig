//! DeepSeek-V4.1-Flash's image tower on given patches (bf16 [n_h * n_w, 588], ref/vision_dump.py's), the aligner's
//! rows written as bf16 for the reference to compare: MODEL_DIR PATCHES N_H N_W OUT. Or MODEL_DIR IMAGE OUT: the
//! image prepared natively (families/deepseek_v41/image.zig), its patches also written (OUT.patches, bf16).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.deepseek_v41;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 6 and args.len != 4) return error.ExpectedModelPatchesGridOut;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var n_h: u32 = undefined;
    var n_w: u32 = undefined;
    var bytes: []u8 = undefined;
    const out_path = args[args.len - 1];
    if (args.len == 4) {
        const file = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(32 << 20));
        defer gpa.free(file);
        const p = try q.image.prepare(init.io, gpa, file, q.vision.max_tokens);
        defer gpa.free(p.pixels);
        n_h = p.gh;
        n_w = p.gw;
        bytes = try gpa.alloc(u8, p.pixels.len * 2);
        for (std.mem.bytesAsSlice(u16, bytes), p.pixels) |*d, v| d.* = q.image.bf16(v);
        const name = try std.fmt.allocPrint(gpa, "{s}.patches", .{out_path});
        defer gpa.free(name);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = name, .data = bytes });
        std.debug.print("prepared: {d} x {d} patches, {d} span tokens\n", .{ n_h, n_w, p.tokens });
    } else {
        n_h = try std.fmt.parseInt(u32, args[3], 10);
        n_w = try std.fmt.parseInt(u32, args[4], 10);
        bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    }
    defer gpa.free(bytes);
    if (bytes.len != @as(usize, n_h) * n_w * q.vision.patch_values * 2) return error.PatchBytesMismatch;
    const m = try q.Model.load(gpa, init.io, args[1], false);
    defer m.deinit();
    const v = &(m.vision orelse return error.NoVisionTower);
    const opts = mtl.ResourceOptions.shared;
    const pixels = try m.device.buffer(bytes.len, opts);
    defer pixels.deinit();
    @memcpy(pixels.contents()[0..bytes.len], bytes);
    const g = q.vision.tokenGrid(n_h, n_w);
    const rows = @as(usize, g[0]) * g[1];
    const out = try m.device.buffer(rows * q.config.hidden * 2, opts);
    defer out.deinit();
    for (0..2) |pass| { // the second is the timed one (pipelines warm)
        var t0: std.c.timespec = undefined;
        var t1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &t0);
        try q.vision.run(m, v, pixels, n_h, n_w, out);
        _ = std.c.clock_gettime(.MONOTONIC, &t1);
        if (pass == 1) std.debug.print("tower and aligner: {d} patches -> {d} rows in {d:.1} ms\n", .{ @as(usize, n_h) * n_w, rows, @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6 });
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = out.contents()[0 .. rows * q.config.hidden * 2] });
}
