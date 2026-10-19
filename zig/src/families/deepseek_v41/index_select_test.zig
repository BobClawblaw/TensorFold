//! The indexer's merges and finish a threadgroup a row against the serial ones: the same heaps, candidates and picks.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const index = @import("csa2_index.zig");
const ts = @import("test_support.zig");

/// The indexer's pass twice, parallel and serial, on the same inputs: every selection buffer byte for byte.
fn same(device: mtl.Device, queue: mtl.Queue, parallel: *const index.Pipelines, serial: *const index.Pipelines, rows: u32, keys: u32, ties: bool, seed: u64) !void {
    var b: ts.Buffers = .{ .device = device };
    defer b.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const q = try b.get(u16, @as(usize, rows) * 32 * 128);
    const w = try b.get(u16, @as(usize, rows) * 32);
    const k = try b.get(u8, @as(usize, keys) * 68);
    for (q.slice(u16, @as(usize, rows) * 32 * 128)) |*x| x.* = ts.bf(if (ties) @as(f32, @floatFromInt(r.uintLessThan(u8, 3))) - 1 else (r.float(f32) - 0.5) * 2);
    for (w.slice(u16, @as(usize, rows) * 32)) |*x| x.* = ts.bf(if (ties) 1 else (r.float(f32) - 0.5) * 2);
    for (0..keys) |i| {
        const key = k.contents()[i * 68 ..][0..68];
        if (ties) @memset(key[0..64], @intCast(0x11 * (i % 3))) else r.bytes(key[0..64]);
        for (key[64..]) |*e| e.* = 125 + r.uintLessThan(u8, 4);
    }
    var scratch: [2]index.Scratch = .{ try index.Scratch.init(device, rows), try index.Scratch.init(device, rows) };
    defer for (&scratch) |*x| x.deinit();
    // the source layer's pass makes the candidates, the next one scores them
    for ([_][2]u32{ .{ 1, 0 }, .{ 0, 1 } }) |mode| {
        for ([_]*const index.Pipelines{ parallel, serial }, &scratch) |p, sc| {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            try p.encode(enc, ts.view(q), ts.view(w), ts.view(k), sc, .{ .rows = rows, .q_stride = 32 * 128, .weight_stride = 32, .position = keys * 2 - rows, .ratio = 2, .keys = keys, .make_candidates = mode[0], .use_candidates = mode[1] });
            try ts.done(cb, enc);
        }
        const n = index.Scratch.bytes(rows);
        const scores = @as(usize, rows) * index.score_tile_width * 4; // the last tile's raw scores: work, not a result
        const got = scratch[0].buffer.contents()[scores..n];
        const want = scratch[1].buffer.contents()[scores..n];
        if (!std.mem.eql(u8, want, got)) {
            std.debug.print("rows {d} keys {d} ties {} mode {any}: the selections differ\n", .{ rows, keys, ties, mode });
            return error.ParallelSelectionDiffers;
        }
    }
}

test "the parallel merges and finish give the serial ones' heaps, candidates and picks: tiles, ties, few keys" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var parallel = try index.Pipelines.load(device);
    defer parallel.deinit();
    var serial = try index.Pipelines.load(device);
    defer serial.deinit();
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_kv ++ sources.deepseek_v41_index, options);
    defer lib.deinit();
    serial.merge.deinit();
    serial.merge = try mtl.Pipeline.init(device, lib, "dsv41_index_merge_serial", false);
    serial.finish.deinit();
    serial.finish = try mtl.Pipeline.init(device, lib, "dsv41_index_finish_serial", false);
    var seed: u64 = 1;
    for ([_]u32{ 1, 3, 17 }) |rows| for ([_]u32{ 37, 600, index.score_tile_width + 5, 20000 }) |keys| for ([_]bool{ false, true }) |ties| {
        try same(device, queue, &parallel, &serial, rows, keys, ties, seed);
        seed += 1;
    };
}
