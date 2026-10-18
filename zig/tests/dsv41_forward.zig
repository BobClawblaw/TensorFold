//! DeepSeek-V4.1-Flash ids through the native forward: bf16 logits.bin (streams too with one chunk), CHUNK a call.
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
    if (std.c.getenv("DS_DISPATCH") != null) return dispatchCost(gpa, init.io, args[1]);
    if (std.c.getenv("DS_BW") != null) return bandwidth(gpa, init.io, args[1]);
    if (std.c.getenv("DS_ABLATE") != null) return ablate(gpa, init.io, args[1], ids);
    if (std.c.getenv("DS_DEPTH")) |d| return depths(gpa, init.io, args[1], ids, std.mem.span(d));
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
    if (std.c.getenv("DS_SKIP")) |k| q.forward.skip = std.mem.span(k);
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
    // past 1 GiB, logits.N.bin parts of 1 GiB (a single write past 2 GiB fails on macOS)
    if (logits.len <= 1 << 30) try dir.writeFile(init.io, .{ .sub_path = "logits.bin", .data = logits });
    var part: usize = 0;
    while (logits.len > 1 << 30 and part * (1 << 30) < logits.len) : (part += 1) {
        var name: [32]u8 = undefined;
        const lo = part * (1 << 30);
        try dir.writeFile(init.io, .{ .sub_path = try std.fmt.bufPrint(&name, "logits.{d}.bin", .{part}), .data = logits[lo..@min(logits.len, lo + (1 << 30))] });
    }
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

/// `n` greedy tokens a row a step, then with DSpark drafts in windows of six: the same tokens; both speeds.
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

/// What a dispatch costs on its own: 2,000 one-thread dispatches in one serial encoder, then in a concurrent one.
fn dispatchCost(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const m = try q.Model.load(gpa, io, dir, false);
    defer m.deinit();
    const buf = try m.device.buffer(4096, mtl.ResourceOptions.shared);
    defer buf.deinit();
    for ([_]mtl.DispatchType{ .serial, .concurrent }) |kind| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(kind);
        e.setPipeline(m.kernels.get("ds_scale"));
        e.setBuffer(buf, 0, 0);
        e.setValue(@as(f32, 1.0), 1);
        for (0..2000) |_| e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        std.debug.print("{s}: {d:.2} us a dispatch on the GPU\n", .{ @tagName(kind), cb.gpuSeconds() * 1e6 / 2000.0 });
    }
    // concurrent, a barrier after each (a chain of dependent dispatches)
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.concurrent);
    e.setPipeline(m.kernels.get("ds_scale"));
    e.setBuffer(buf, 0, 0);
    e.setValue(@as(f32, 1.0), 1);
    for (0..2000) |_| {
        e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        e.barrier();
    }
    e.end();
    cb.commit();
    cb.wait();
    std.debug.print("concurrent, barrier each: {d:.2} us a dispatch on the GPU\n", .{cb.gpuSeconds() * 1e6 / 2000.0});
    var scratch = try q.state.Scratch.init(gpa, m.device, 8);
    defer scratch.deinit();
    var cache = try q.state.Cache.init(gpa, m.device, 16);
    defer cache.deinit();
    try q.forward.run(m, &scratch, &cache, &.{0}, .last, null);
    q.forward.dispatches = 0;
    q.forward.trace = true;
    try q.forward.run(m, &scratch, &cache, &.{1}, .last, null);
    q.forward.trace = false;
    std.debug.print("a decode step: {d} dispatches\n", .{q.forward.dispatches});
}

/// GPU ms a decode step with each kernel left out in turn (the outputs are wrong; only the time is read).
fn ablate(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, ids: []const u32) !void {
    const m = try q.Model.load(gpa, io, dir, false);
    defer m.deinit();
    var scratch = try q.state.Scratch.init(gpa, m.device, 8);
    defer scratch.deinit();
    const names = [_][]const u8{ "", "rows", "indexed", "ds_hc_expand_mix", "ds_hc_mix_part", "ds_hc_boundary", "ds_hc_expand", "ds_kv_finish", "ds_rms", "ds_rope", "ds_fp8_qdq", "ds_fp4_qdq", "ds_put_rows", "ds_put_rows_f32", "ds_attend_split", "ds_attend_merge", "ds_gemv_bf_bf", "ds_gemv_bf_f32", "ds_route", "ds_swiglu", "ds_index_scores_2x16", "ds_topk", "ds_pool2", "ds_scale", "ds_engram", "ds_embed" };
    var base: f64 = 0;
    const only_all = std.mem.eql(u8, std.mem.span(std.c.getenv("DS_ABLATE").?), "all");
    for (names[0..if (only_all) 1 else names.len]) |name| {
        q.forward.skip = if (name.len == 0) null else name;
        var timing: [3]f64 = @splat(0);
        q.forward.timing = &timing;
        var cache = try q.state.Cache.init(gpa, m.device, ids.len + 8);
        defer cache.deinit();
        const w: usize = if (std.c.getenv("DS_ROWS")) |r| try std.fmt.parseInt(usize, std.mem.span(r), 10) else 1;
        var calls: usize = 0;
        var at: usize = 0;
        while (at + w <= ids.len) : (at += w) {
            if (calls == 2) timing = @splat(0); // the first calls warm the pipelines
            try q.forward.run(m, &scratch, &cache, ids[at..][0..w], .last, null);
            calls += 1;
        }
        const ms = timing[1] * 1e3 / @as(f64, @floatFromInt(calls - 2));
        if (name.len == 0) base = ms;
        std.debug.print("{s:<18} {d:6.2} ms ({d:.2})\n", .{ if (name.len == 0) "(all)" else name, ms, ms - base });
    }
}

/// GB/s the row kernels reach on a decode row (head, wq_b, wo_b, 6 experts), and a blit copy for the bus.
fn bandwidth(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const row = tf.row_projection;
    const m = try q.Model.load(gpa, io, dir, false);
    defer m.deinit();
    const opts = mtl.ResourceOptions.shared;
    const x = try m.device.buffer(64 << 20, opts);
    defer x.deinit();
    const y = try m.device.buffer(64 << 20, opts);
    defer y.deinit();
    const ids = try m.device.buffer(64, opts);
    defer ids.deinit();
    for (ids.slice(u32, 6), 0..) |*v, i| v.* = @intCast(i * 61 + 7);
    const reps = 20;
    const W = struct { name: []const u8, w: row.Weights };
    const l = &m.weights.layers[12];
    const list = [_]W{ .{ .name = "head", .w = m.weights.head }, .{ .name = "wq_b", .w = l.attn.wq_b }, .{ .name = "wo_b", .w = l.attn.wo_b }, .{ .name = "shared w1", .w = l.moe.shared.w1 } };
    for (list) |it| {
        const call = try row.call(&m.kernels.rows, it.w, 1, false);
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| {
            e.setPipeline(call.pipeline);
            e.setBuffer(x, 0, 0);
            e.setBuffer(it.w.w, it.w.w_off, 1);
            e.setBuffer(it.w.scales, it.w.s_off, 2);
            e.setBuffer(it.w.biases, it.w.b_off, 3);
            e.setBytes(std.mem.asBytes(&call.dims), 4);
            e.setBuffer(y, 0, 5);
            e.dispatchThreads(mtl.Size.of(call.groups * call.threads, 1, 1), mtl.Size.of(call.threads, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        const bytes: f64 = @floatFromInt(it.w.n * it.w.k * it.w.bits / 8 + it.w.n * (it.w.k / it.w.group) * 4);
        const t = cb.gpuSeconds() / reps;
        std.debug.print("{s:<10} {d}-bit {d:7.1} MB {d:7.1} us {d:6.0} GB/s\n", .{ it.name, it.w.bits, bytes / 1e6, t * 1e6, bytes / t / 1e9 });
    }
    {
        const ex = l.moe.w1;
        const call = try row.callIndexed(&m.kernels.rows, ex, 6, false);
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| {
            e.setPipeline(call.pipeline);
            e.setBuffer(x, 0, 0);
            e.setBuffer(ex.one.w, ex.one.w_off, 1);
            e.setBuffer(ex.one.scales, ex.one.s_off, 2);
            e.setBuffer(ex.one.biases, ex.one.b_off, 3);
            e.setBytes(std.mem.asBytes(&call.dims), 4);
            e.setBuffer(y, 0, 5);
            e.setBuffer(ids, 0, 6);
            e.dispatchThreads(mtl.Size.of(call.groups * call.threads, call.slots, 1), mtl.Size.of(call.threads, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        const one = ex.one;
        const bytes: f64 = @floatFromInt(6 * (one.n * one.k * one.bits / 8 + one.n * (one.k / one.group) * 4));
        const t = cb.gpuSeconds() / reps;
        std.debug.print("experts6   {d}-bit {d:7.1} MB {d:7.1} us {d:6.0} GB/s\n", .{ one.bits, bytes / 1e6, t * 1e6, bytes / t / 1e9 });
    }
    {
        const h = m.weights.head;
        const n = h.n * h.k * h.bits / 8;
        const dst = try m.device.buffer(n, opts);
        defer dst.deinit();
        const cb = m.queue.commandBuffer();
        const b = cb.blit();
        for (0..4) |_| b.copy(h.w, h.w_off, dst, 0, n);
        b.end();
        cb.commit();
        cb.wait();
        const t = cb.gpuSeconds() / 4;
        std.debug.print("blit copy  {d:7.1} MB {d:7.1} us {d:6.0} GB/s read+write\n", .{ @as(f64, @floatFromInt(n)) / 1e6, t * 1e6, 2 * @as(f64, @floatFromInt(n)) / t / 1e9 });
    }
}

/// GPU ms by section of a 256-row chunk and a decode row at each depth in `list`, over a random-bf16 cache.
fn depths(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, ids: []const u32, list: []const u8) !void {
    const m = try q.Model.load(gpa, io, dir, false);
    defer m.deinit();
    var scratch = try q.state.Scratch.init(gpa, m.device, 256);
    defer scratch.deinit();
    if (ids.len < 256) return error.ExpectedAChunkOfIds;
    var sections: [q.forward.sections.len]f64 = @splat(0);
    if (std.c.getenv("DS_SKIP")) |k| q.forward.skip = std.mem.span(k);
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |field| {
        const depth = try std.fmt.parseInt(usize, field, 10);
        var cache = try q.state.Cache.init(gpa, m.device, depth + 4 * 256 + 64);
        defer cache.deinit();
        var rng: u64 = 0x9e3779b97f4a7c15 ^ depth;
        for (cache.memory.buffers.items) |b| for (std.mem.bytesAsSlice(u16, b.contents()[0 .. b.length() & ~@as(usize, 1)])) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            v.* = @truncate((rng & 0x807f) | 0x3c00); // +-[2^-7, 2^-6): small finite values
        };
        for (0..depth) |i| cache.engram_ids.appendAssumeCapacity(m.engram.compress(ids[i % ids.len], true));
        for ([_]usize{ 256, 1 }) |rows| {
            const reps: usize = if (rows == 1) 8 else 2;
            for (0..reps + 1) |r| {
                if (r == 1) sections = @splat(0); // the first call warms the pipelines
                cache.len = depth;
                q.forward.profile = &sections;
                try q.forward.run(m, &scratch, &cache, ids[0..rows], .last, null);
                q.forward.profile = null;
            }
            var total: f64 = 0;
            for (sections) |v| total += v;
            const k: f64 = @floatFromInt(reps);
            std.debug.print("depth {d} rows {d}: {d:.2} ms ({d:.0} tok/s);", .{ depth, rows, total * 1e3 / k, @as(f64, @floatFromInt(rows)) * k / total });
            for (q.forward.sections, sections) |name, v| std.debug.print(" {s} {d:.2},", .{ name, v * 1e3 / k });
            std.debug.print("\n", .{});
        }
    }
}
