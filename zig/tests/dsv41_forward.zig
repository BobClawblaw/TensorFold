//! DeepSeek-V4.1-Flash token ids through the native forward: every row's bf16 logits (logits.bin), and with one
//! chunk each block's streams (layerN.bin, bf16 [rows, 4, 5120]), for the CPU reference to compare
//! (ref/compare.py in the recipe). CHUNK rows a call (default: all of them; 1: decode steps).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.deepseek_v41;
const c = q.config;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4 and args.len != 5) return error.ExpectedModelIdsOutDirChunk;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(4 << 20));
    defer gpa.free(bytes);
    const a = try tf.npy.parse(bytes);
    if (!std.mem.eql(u8, a.descr, "<u4") or a.rank != 1 or a.count() == 0) return error.ExpectedU32TokenVector;
    const ids = try gpa.alloc(u32, a.count());
    defer gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), a.data);
    const chunk: usize = if (args.len == 5) try std.fmt.parseInt(usize, args[4], 10) else ids.len;
    const out_dir = args[3];
    try std.Io.Dir.cwd().createDirPath(init.io, out_dir);

    var t0: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t0);
    const m = try q.Model.load(gpa, init.io, args[1], false);
    defer m.deinit();
    var t1: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t1);
    std.debug.print("loaded in {d:.1}s\n", .{seconds(t0, t1)});
    var cache = try q.state.Cache.init(gpa, m.device, ids.len + 64);
    defer cache.deinit();
    var scratch = try q.state.Scratch.init(gpa, m.device, @min(chunk, q.state.max_rows));
    defer scratch.deinit();

    var dump_bufs: [c.layers]mtl.Buffer = undefined;
    const dumping = chunk >= ids.len;
    if (dumping) for (&dump_bufs) |*b| {
        b.* = try m.device.buffer(ids.len * c.hc * c.hidden * 2, mtl.ResourceOptions.shared);
    };
    defer if (dumping) for (dump_bufs) |b| b.deinit();

    var sections: [q.forward.sections.len]f64 = @splat(0);
    if (std.c.getenv("DS_PROFILE") != null) q.forward.profile = &sections;
    var calls: usize = 0;
    var wall: f64 = 0;
    const logits = try gpa.alloc(u8, ids.len * c.vocab * 2);
    defer gpa.free(logits);
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(chunk, ids.len - at);
        _ = std.c.clock_gettime(.MONOTONIC, &t0);
        try q.forward.run(m, &scratch, &cache, ids[at..][0..n], .all, if (dumping) .{ .layers = &dump_bufs } else null);
        _ = std.c.clock_gettime(.MONOTONIC, &t1);
        if (at == 0 or n > 1) std.debug.print("rows {d}..{d}: {d:.1} ms\n", .{ at, at + n, seconds(t0, t1) * 1e3 });
        if (at > 0) {
            calls += 1;
            wall += seconds(t0, t1);
        }
        @memcpy(logits[at * c.vocab * 2 ..][0 .. n * c.vocab * 2], scratch.logits.contents()[0 .. n * c.vocab * 2]);
        at += n;
    }
    var dir = try std.Io.Dir.cwd().openDir(init.io, out_dir, .{});
    defer dir.close(init.io);
    try dir.writeFile(init.io, .{ .sub_path = "logits.bin", .data = logits });
    // the last index source's (layer 36's) picks for every row, when the whole input was one call
    if (dumping) try dir.writeFile(init.io, .{ .sub_path = "picks.bin", .data = scratch.picks.contents()[0 .. ids.len * q.state.picks * 4] });
    if (dumping) for (dump_bufs, 0..) |b, i| {
        var name: [32]u8 = undefined;
        const n = try std.fmt.bufPrint(&name, "layer{d}.bin", .{i});
        try dir.writeFile(init.io, .{ .sub_path = n, .data = b.contents()[0 .. ids.len * c.hc * c.hidden * 2] });
    };
    std.debug.print("native DeepSeek-V4.1 forward: {d} rows, bf16 logits\n", .{ids.len});
    if (calls > 0) std.debug.print("after the first call: {d:.2} ms a call ({d} calls)\n", .{ wall * 1e3 / @as(f64, @floatFromInt(calls)), calls });
    if (q.forward.profile != null) {
        const k: f64 = @floatFromInt(calls + 1);
        std.debug.print("GPU ms a call:", .{});
        for (q.forward.sections, sections) |name, v| std.debug.print(" {s} {d:.2},", .{ name, v * 1e3 / k });
        std.debug.print("\n", .{});
    }
}

fn seconds(a: std.c.timespec, b: std.c.timespec) f64 {
    return @as(f64, @floatFromInt(b.sec - a.sec)) + @as(f64, @floatFromInt(b.nsec - a.nsec)) / 1e9;
}
