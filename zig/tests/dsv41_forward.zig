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

    if (std.c.getenv("DS_DRAFT")) |n| return drafts(gpa, init.io, args[1], ids, try std.fmt.parseInt(usize, std.mem.span(n), 10));
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
    var timing: [3]f64 = @splat(0);
    if (std.c.getenv("DS_TIMING") != null) q.forward.timing = &timing;
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
    if (q.forward.timing != null) {
        const k: f64 = @floatFromInt(calls + 1);
        std.debug.print("a call: encode {d:.2} ms, GPU {d:.2} ms, all {d:.2} ms\n", .{ timing[0] * 1e3 / k, timing[1] * 1e3 / k, timing[2] * 1e3 / k });
    }
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

fn argmax(logits: []const u16) u32 {
    var best: usize = 0;
    for (logits, 0..) |v, i| if (bf(v) > bf(logits[best])) {
        best = i;
    };
    return @intCast(best);
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// `n` greedy tokens after the prompt, one row a step and then with the DSpark head's drafts verified in windows of
/// six rows: the two replies must be the same tokens; prints the acceptance and both speeds.
fn drafts(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, ids: []const u32, n: usize) !void {
    const m = try q.Model.load(gpa, io, dir, true);
    defer m.deinit();
    var scratch = try q.state.Scratch.init(gpa, m.device, 256);
    defer scratch.deinit();
    var ds = try q.state.Scratch.init(gpa, m.device, 8);
    defer ds.deinit();
    var replies: [2][]u32 = undefined;
    var times: [2]f64 = undefined;
    var rounds: usize = 0;
    for (0..2) |mode| {
        var cache = try q.state.Cache.initWith(gpa, m.device, ids.len + n + 16, true);
        defer cache.deinit();
        var at: usize = 0;
        while (at < ids.len) : (at += 256) {
            const end = @min(ids.len, at + 256);
            try q.forward.run(m, &scratch, &cache, ids[at..end], .last, null);
        }
        const out = try gpa.alloc(u32, n);
        var got: usize = 0;
        var pending = argmax(cache.logits.slice(u16, c.vocab));
        var t0: std.c.timespec = undefined;
        var t1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &t0);
        while (got < n) {
            out[got] = pending;
            got += 1;
            if (got == n) break;
            if (mode == 0) {
                try q.forward.run(m, &scratch, &cache, &.{pending}, .last, null);
                pending = argmax(cache.logits.slice(u16, c.vocab));
                continue;
            }
            // draft from the last committed row, verify pending + drafts, keep the matching prefix
            var held: [c.draft_block]u32 = undefined;
            try q.forward.draft(m, &ds, &cache, pending, cache.len - 1, &held);
            var window: [1 + c.draft_block]u32 = undefined;
            window[0] = pending;
            @memcpy(window[1..], &held);
            const len0 = cache.len;
            try q.forward.run(m, &scratch, &cache, &window, .all, null);
            rounds += 1;
            var kept: usize = 1;
            var next = argmax(scratch.logits.slice(u16, c.vocab));
            const rows = scratch.logits.slice(u16, window.len * c.vocab);
            while (kept < window.len and window[kept] == next and got < n) : (kept += 1) {
                out[got] = window[kept];
                got += 1;
                next = argmax(rows[kept * c.vocab ..][0..c.vocab]);
            }
            cache.len = len0 + kept; // the rejected rows' cache entries are rewritten by the next call
            pending = next;
        }
        _ = std.c.clock_gettime(.MONOTONIC, &t1);
        replies[mode] = out;
        times[mode] = seconds(t0, t1);
    }
    var same: usize = 0;
    while (same < n and replies[0][same] == replies[1][same]) same += 1;
    std.debug.print("plain {d:.1} tok/s, drafted {d:.1} tok/s; {d} rounds, {d:.2} tokens a round; replies equal for {d}/{d} tokens\n", .{ @as(f64, @floatFromInt(n)) / times[0], @as(f64, @floatFromInt(n)) / times[1], rounds, @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(rounds)), same, n });
}
