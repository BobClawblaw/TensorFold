//! Qwen3.5 projection kernels timed by tile width and split count on a checkpoint's shapes, and the one-row step.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources").qwen35;
const q = tf.qwen35;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const Variant = struct { s: usize, nt: usize, pipeline: mtl.Pipeline };

fn compile(gpa: std.mem.Allocator, device: mtl.Device, n: usize, k: usize, s: usize, nt: usize) !mtl.Pipeline {
    var buf: [128]u8 = undefined;
    const header = try std.fmt.bufPrint(&buf, "#define K {d}\n#define N {d}\n#define S {d}\n#define NT {d}\n", .{ k, n, s, nt });
    const text = try std.mem.concat(gpa, u8, &.{ header, sources.qmm.source });
    defer gpa.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, sources.qmm.function, false);
}

/// One projection `reps` times in one command buffer, as forward.zig dispatches it; GPU microseconds a call.
fn time(m: *q.Model, p: mtl.Pipeline, lin: q.weights.Linear, nt: usize, x: mtl.Buffer, y: mtl.Buffer, rows: usize, reps: usize) f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    const columns = 8 * nt;
    for (0..reps) |_| {
        e.setPipeline(p);
        e.setBuffer(x, 0, 0);
        e.setValue([2]i32{ @intCast(rows), @intCast(lin.inputs) }, 1);
        e.setBuffer(lin.weight.buffer, lin.weight.offset, 2);
        e.setBuffer(lin.scales.buffer, lin.scales.offset, 3);
        e.setBuffer(lin.biases.buffer, lin.biases.offset, 4);
        e.setValue([1]f32{1}, 5);
        e.setBuffer(y, 0, 6);
        e.dispatchThreads(mtl.Size.of(256 * ((lin.outputs + columns - 1) / columns), (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
    }
    e.end();
    cb.commit();
    cb.wait();
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(reps));
}

/// The FP32 path, dispatched as forward.zig does: 128 threads a threadgroup, 16 output rows each.
fn timeQmv(m: *q.Model, p: mtl.Pipeline, lin: q.weights.Linear, x: mtl.Buffer, y: mtl.Buffer, rows: usize, reps: usize) f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..reps) |_| {
        e.setPipeline(p);
        e.setBuffer(x, 0, 0);
        e.setValue([2]i32{ @intCast(rows), @intCast(lin.inputs) }, 1);
        e.setBuffer(lin.weight.buffer, lin.weight.offset, 2);
        e.setBuffer(lin.scales.buffer, lin.scales.offset, 3);
        e.setBuffer(lin.biases.buffer, lin.biases.offset, 4);
        e.setValue([1]f32{1}, 5);
        e.setBuffer(y, 0, 6);
        e.dispatchThreads(mtl.Size.of(128 * ((lin.outputs + q.kernels.qmv_rows - 1) / q.kernels.qmv_rows), 1, 1), mtl.Size.of(128, 1, 1));
    }
    e.end();
    cb.commit();
    cb.wait();
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(reps));
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.ExpectedModelDir;
    const reps: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 30; // ms a cell
    const grid = args.len > 3 and std.mem.eql(u8, args[3], "grid"); // the split and tile alternatives too
    if (args.len > 5 and std.mem.eql(u8, args[3], "teacher")) return teacher(init, args[1], args[4], args[5]);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    const g = m.config.g;
    const w = m.weights;
    var attn: ?q.weights.Attention = null;
    for (w.blocks) |b| if (b.mixer == .attention) {
        attn = b.mixer.attention;
        break;
    };
    const d = w.blocks[0].mixer.delta;
    const cases = [_]struct { name: []const u8, lin: q.weights.Linear }{
        .{ .name = "down", .lin = w.blocks[0].down },
        .{ .name = "delta_out", .lin = d.out },
        .{ .name = "attn_out", .lin = attn.?.out },
        .{ .name = "gate", .lin = w.blocks[0].gate },
        .{ .name = "qkv", .lin = d.qkv },
        .{ .name = "z", .lin = d.z },
        .{ .name = "attn_q", .lin = attn.?.q },
        .{ .name = "attn_k", .lin = attn.?.k },
        .{ .name = "head", .lin = w.head() },
    };
    const max_rows = 32;
    var max_k: usize = 0;
    var max_n: usize = 0;
    for (cases) |c| {
        max_k = @max(max_k, c.lin.inputs);
        max_n = @max(max_n, c.lin.outputs);
    }
    const x = try m.device.buffer(max_rows * max_k * 2, opts);
    const y = try m.device.buffer(max_rows * max_n * 2, opts);
    const y_ref = try gpa.alloc(u8, max_rows * max_n * 2);
    defer gpa.free(y_ref);
    for (x.slice(u16, max_rows * max_k), 0..) |*v, i| v.* = @truncate(0x3c00 + (i * 7) % 96 + (i / 1000) % 16);
    const rows_list = [_]usize{ 1, 2, 4, 8, 16, 32 };
    const nts = [_]usize{ 1, 2, 4, 8 };
    const ss = [_]usize{ 8, 16, 32 };
    std.debug.print("geometry hidden {d} layers {d}; {d} ms bursts a cell; us a call; '=' same bytes as shipped, '!' differs\n", .{ g.hidden, g.layers, reps });
    for (cases) |c| {
        const n, const k = .{ c.lin.outputs, c.lin.inputs };
        const s0, const nt0 = .{ q.kernels.splits(n), q.kernels.tiles(n) };
        const shipped = m.kernels.projection(n, k).?;
        std.debug.print("{s} N {d} K {d}: shipped S {d} NT {d} ({d} threadgroups at 1 row)\n", .{ c.name, n, k, s0, nt0, (n + 8 * nt0 - 1) / (8 * nt0) });
        for (rows_list) |rows| {
            // Bursts of at least `reps` ms so the GPU clock is up; the shipped kernel again at the end of the line.
            const probe = time(m, shipped.pipeline, c.lin, nt0, x, y, rows, 20);
            const n_reps: usize = @max(40, @as(usize, @intFromFloat(@as(f64, @floatFromInt(reps)) * 1000 / probe)));
            _ = time(m, shipped.pipeline, c.lin, nt0, x, y, rows, n_reps / 2);
            const base = time(m, shipped.pipeline, c.lin, nt0, x, y, rows, n_reps);
            @memcpy(y_ref[0 .. rows * n * 2], y.contents()[0 .. rows * n * 2]);
            std.debug.print("  rows {d:>2} ({d} reps): shipped {d:>7.1}", .{ rows, n_reps, base });
            if (rows <= 8) if (if (rows == 1) shipped.qmv1 else if (rows == 2) shipped.qmv2 else if (rows <= 4) shipped.qmv4 else shipped.qmv8) |pq| {
                @memset(y.contents()[0 .. rows * n * 2], 0);
                _ = timeQmv(m, pq, c.lin, x, y, rows, n_reps / 2);
                const t = timeQmv(m, pq, c.lin, x, y, rows, n_reps);
                // numeric distance from the shipped bytes: the largest |difference| over the largest |value|, and how many differ
                const a = std.mem.bytesAsSlice(u16, y_ref[0 .. rows * n * 2]);
                const b = y.slice(u16, rows * n);
                var worst: f32 = 0;
                var scale: f32 = 0;
                var differ: usize = 0;
                for (a, b) |u, v| {
                    const diff = @abs(bf(u) - bf(v));
                    worst = @max(worst, diff);
                    scale = @max(scale, @abs(bf(u)));
                    if (u != v) differ += 1;
                }
                std.debug.print("  fp32 {d:.1} (max diff {e:.1} of {e:.1}, {d}/{d} differ)", .{ t, worst, scale, differ, rows * n });
                // Row independence: the same lane width built for more rows must give these rows the same bytes.
                if (rows <= 2) {
                    const vpl: usize = if (rows == 1) 32 else 16;
                    var hb: [128]u8 = undefined;
                    const hdr = try std.fmt.bufPrint(&hb, "#define K {d}\n#define N {d}\n#define RM {d}\n#define VPL {d}\n", .{ k, n, 4, vpl });
                    const txt = try std.mem.concat(gpa, u8, &.{ hdr, @import("kernel_sources").qwen35.qmv.source });
                    defer gpa.free(txt);
                    const lib = try mtl.Library.fromSource(m.device, txt, mtl.CompileOptions.mlx());
                    defer lib.deinit();
                    const p4 = try mtl.Pipeline.init(m.device, lib, "qwen35_qmv", false);
                    defer p4.deinit();
                    const mine = try gpa.alloc(u8, rows * n * 2);
                    defer gpa.free(mine);
                    @memcpy(mine, y.contents()[0 .. rows * n * 2]);
                    @memset(y.contents()[0 .. rows * n * 2], 0);
                    _ = timeQmv(m, p4, c.lin, x, y, rows, 3);
                    std.debug.print("  [RM4@VPL{d} {s}]", .{ vpl, if (std.mem.eql(u8, mine, y.contents()[0 .. rows * n * 2])) "same bytes" else "DIFFERENT" });
                }
            };
            for (ss) |s| for (nts) |nt| {
                if (!grid or s * nt * 64 * 4 > 32768 or (s == s0 and nt == nt0)) continue;
                if (n % (8 * nt) != 0) continue;
                const p = try compile(gpa, m.device, n, k, s, nt);
                defer p.deinit();
                @memset(y.contents()[0 .. rows * n * 2], 0);
                _ = time(m, p, c.lin, nt, x, y, rows, n_reps / 2);
                const t = time(m, p, c.lin, nt, x, y, rows, n_reps);
                const same = std.mem.eql(u8, y_ref[0 .. rows * n * 2], y.contents()[0 .. rows * n * 2]);
                std.debug.print("  S{d}/NT{d} {d:.1}{s}", .{ s, nt, t, if (same) "=" else "!" });
            };
            std.debug.print("  shipped again {d:.1}\n", .{time(m, shipped.pipeline, c.lin, nt0, x, y, rows, n_reps)});
        }
    }
    // Projections-only floor: every projection of a one-row step, in one command buffer, on the shipped kernels.
    {
        const pool2 = mtl.objc.Pool.push();
        defer pool2.pop();
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        var calls: usize = 0;
        const One = struct {
            fn go(enc: mtl.ComputeEncoder, mm: *q.Model, lin: q.weights.Linear, xx: mtl.Buffer, yy: mtl.Buffer, count: *usize) void {
                const p = mm.kernels.projection(lin.outputs, lin.inputs).?;
                enc.setPipeline(p.pipeline);
                enc.setBuffer(xx, 0, 0);
                enc.setValue([2]i32{ 1, @intCast(lin.inputs) }, 1);
                enc.setBuffer(lin.weight.buffer, lin.weight.offset, 2);
                enc.setBuffer(lin.scales.buffer, lin.scales.offset, 3);
                enc.setBuffer(lin.biases.buffer, lin.biases.offset, 4);
                enc.setValue([1]f32{1}, 5);
                enc.setBuffer(yy, 0, 6);
                enc.dispatchThreads(mtl.Size.of(p.threads * ((lin.outputs + p.columns - 1) / p.columns), 1, 1), mtl.Size.of(p.threads, 1, 1));
                count.* += 1;
            }
        };
        for (w.blocks) |b| {
            switch (b.mixer) {
                .delta => |dd| for ([_]q.weights.Linear{ dd.qkv, dd.z, dd.a, dd.b, dd.out }) |lin| One.go(e, m, lin, x, y, &calls),
                .attention => |aa| for ([_]q.weights.Linear{ aa.q, aa.k, aa.v, aa.out }) |lin| One.go(e, m, lin, x, y, &calls),
            }
            for ([_]q.weights.Linear{ b.gate, b.up, b.down }) |lin| One.go(e, m, lin, x, y, &calls);
        }
        One.go(e, m, w.head(), x, y, &calls);
        e.end();
        cb.commit();
        cb.wait();
        std.debug.print("projections-only one-row step: {d} calls, {d:.2} ms GPU\n", .{ calls, cb.gpuSeconds() * 1e3 });
    }
    // The whole one-row step on the shipped kernels: dependent decode steps in a fresh cache.
    var cache = try q.state.Cache.init(gpa, m.device, g, 256);
    defer cache.deinit();
    var scratch = try q.state.Scratch.init(gpa, m.device, g, 32, cache.capacity);
    defer scratch.deinit();
    const ids = [_]u32{ 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198 };
    try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = ids.len }}, &ids, false, .last);
    var tok: u32 = 785;
    for (0..5) |_| try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = 1 }}, &.{tok}, false, .last);
    const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const steps = 40;
    for (0..steps) |i| {
        tok = 785 + @as(u32, @intCast(i % 50));
        try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = 1 }}, &.{tok}, false, .last);
    }
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0)) / 1e6 / steps;
    std.debug.print("one-row step on shipped kernels: {d:.2} ms ({d:.1} tok/s) over {d} steps\n", .{ ms, 1000 / ms, steps });
}

/// Teacher-forced one-row steps over a token fixture: each position's logits (bf16) to `out`, for comparing paths.
fn teacher(init: std.process.Init, model: []const u8, tokens: []const u8, out_path: []const u8) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, tokens, gpa, .limited(4 << 20));
    defer gpa.free(bytes);
    const a = try tf.npy.parse(bytes);
    const ids = try gpa.alloc(u32, a.count());
    defer gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), a.data);
    const m = try q.Model.load(gpa, init.io, model);
    defer m.deinit();
    var cache = try q.state.Cache.init(gpa, m.device, m.config.g, ids.len + 8);
    defer cache.deinit();
    var scratch = try q.state.Scratch.init(gpa, m.device, m.config.g, 32, cache.capacity);
    defer scratch.deinit();
    const out = try gpa.alloc(u8, ids.len * q.config.vocab * 2);
    defer gpa.free(out);
    for (ids, 0..) |id, i| {
        try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = 1 }}, &.{id}, false, .last);
        @memcpy(out[i * q.config.vocab * 2 ..][0 .. q.config.vocab * 2], scratch.logits.contents()[0 .. q.config.vocab * 2]);
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = out });
    std.debug.print("teacher-forced {d} one-row steps, logits to {s}\n", .{ ids.len, out_path });
}
