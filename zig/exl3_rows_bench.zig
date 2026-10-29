//! EXL3 rows against the reference projection (exl3_projection) at MiniMax-M3.1-Flash's shapes, and their DRAM speeds.
const std = @import("std");
const mtl = @import("metal");
const ex = xr.projection;
const xr = @import("exl3_rows");

const Shape = struct { name: []const u8, k: usize, n: usize, halves: u16 };
const shapes = [_]Shape{
    .{ .name = "tiny", .k = 256, .n = 128, .halves = 6 },
    .{ .name = "q_proj", .k = 6144, .n = 8192, .halves = 12 },
    .{ .name = "k_proj", .k = 6144, .n = 512, .halves = 12 },
    .{ .name = "o_proj", .k = 8192, .n = 6144, .halves = 12 },
    .{ .name = "dense gate", .k = 6144, .n = 12288, .halves = 10 },
    .{ .name = "shared down", .k = 3072, .n = 6144, .halves = 10 },
    .{ .name = "expert w1", .k = 6144, .n = 3072, .halves = 6 },
    .{ .name = "expert w2", .k = 3072, .n = 6144, .halves = 6 },
    .{ .name = "lm_head", .k = 6144, .n = 200064, .halves = 16 },
};

/// `copies` matrices of one shape in one trellis buffer (DRAM-sized), sharing scales.
const Owned = struct {
    ms: []ex.Matrix,
    bufs: [3]mtl.Buffer,

    fn init(gpa: std.mem.Allocator, d: mtl.Device, s: Shape, copies: usize, seed: u32) !Owned {
        const one = (s.k / 16) * (s.n / 16) * @as(usize, s.halves) * 16; // 16 * bits u16 a tile
        const t = try d.buffer(one * copies, mtl.ResourceOptions.shared);
        const su = try d.buffer(s.k * 2, mtl.ResourceOptions.shared);
        const sv = try d.buffer(s.n * 2, mtl.ResourceOptions.shared);
        var state = seed | 1;
        for (t.slice(u32, one * copies / 4)) |*v| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            v.* = state;
        }
        for (su.slice(f16, s.k), 0..) |*v, i| v.* = if ((i * 7 + seed) % 3 == 0) -0.75 else 1.125;
        for (sv.slice(f16, s.n), 0..) |*v, i| v.* = if ((i * 3 + seed) % 5 == 0) -0.0125 else 0.0175;
        const ms = try gpa.alloc(ex.Matrix, copies);
        for (ms, 0..) |*m, c| m.* = .{ .trellis = .{ .buffer = t, .offset = c * one }, .suh = .{ .buffer = su }, .svh = .{ .buffer = sv }, .k = s.k, .n = s.n, .width = .{ .halves = s.halves } };
        return .{ .ms = ms, .bufs = .{ t, su, sv } };
    }
    fn deinit(o: Owned, gpa: std.mem.Allocator) void {
        gpa.free(o.ms);
        for (o.bufs) |b| b.deinit();
    }
};

fn run(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !f64 {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("GPU failure: {s}", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds();
}

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
fn f32of(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var rows: usize = 1;
    var only: ?usize = null;
    var mm = false;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const v = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--rows")) rows = v else if (std.mem.eql(u8, args[i], "--shape")) only = v else if (std.mem.eql(u8, args[i], "--mm")) mm = v == 1 else return error.UnknownOption;
    }
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ref = try ex.Pipelines.load(device);
    defer ref.deinit();
    var p = try xr.Pipelines.load(device);
    defer p.deinit();
    var pm = try xr.MmPipelines.load(device, gpa);
    defer pm.deinit();
    try routedCheck(gpa, device, queue, &ref, &p);
    for (shapes, 0..) |s, si| {
        if (only) |o| if (o != si) continue;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const one = (s.k / 16) * (s.n / 16) * @as(usize, s.halves) * 16;
        const copies: usize = @max(1, @min(64, (2 << 30) / one));
        const jobs = try device.buffer(copies * @sizeOf(xr.Job), mtl.ResourceOptions.shared);
        defer jobs.deinit();
        const w = try Owned.init(gpa, device, s, copies, @intCast(si + 1));
        defer w.deinit(gpa);
        for (jobs.slice(xr.Job, copies), w.ms) |*j, m| j.* = try xr.job(m, 0);
        const x = try device.buffer(rows * s.k * 2, mtl.ResourceOptions.shared);
        defer x.deinit();
        for (x.slice(u16, rows * s.k), 0..) |*v, q| v.* = bf16(@sin(@as(f32, @floatFromInt(q)) * 0.37) * 2.0);
        const y_ref = try device.buffer(rows * s.n * 2, mtl.ResourceOptions.shared);
        defer y_ref.deinit();
        const y = try device.buffer(rows * s.n * 2, mtl.ResourceOptions.shared);
        defer y.deinit();
        const kt = xr.ranges(s.n / 128, s.k);
        const partial = try device.buffer(@max(16, xr.partialBytes(kt, rows, s.n / 128)), mtl.ResourceOptions.shared);
        defer partial.deinit();
        const counts = try device.buffer(@max(16, xr.countBytes(rows, s.n / 128)), mtl.ResourceOptions.shared);
        defer counts.deinit();
        @memset(counts.contents()[0..counts.length()], 0);
        const sc = xr.Scratch{ .partial = .{ .buffer = partial }, .counts = .{ .buffer = counts } };
        const scratch = try device.buffer(try ex.scratchBytes(s.k, s.n, rows), mtl.ResourceOptions.shared);
        defer scratch.deinit();
        // agreement on copy 0
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            try ex.encode(&ref, e, w.ms[0], .{ .buffer = x }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = s.k, .y_stride = s.n });
            for (w.bufs) |bb| e.useResource(bb, mtl.ResourceUsage.read);
            const d = xr.Dense{ .jobs = .{ .buffer = jobs }, .count = 1, .blocks = s.n / 128, .k = s.k };
            if (mm) xr.encodeMm(&pm, e, d, .{ .buffer = x }, s.k, .{ .buffer = y }, s.n, rows) else xr.encode(&p, e, d, .{ .buffer = x }, s.k, .{ .buffer = y }, s.n, rows, .bf16, kt, sc);
            _ = try run(cb, e);
        }
        var worst: f32 = 0;
        var scale: f32 = 0;
        var differ: usize = 0;
        for (y_ref.slice(u16, rows * s.n), y.slice(u16, rows * s.n)) |a, bq| {
            const fa = f32of(a);
            const fb = f32of(bq);
            worst = @max(worst, @abs(fa - fb));
            scale = @max(scale, @abs(fa));
            if (a != bq) differ += 1;
        }
        // speed: every copy once, twice over, back to back
        const reps = 2;
        var t_ref: f64 = 0;
        var t_new: f64 = 0;
        for (0..2) |pass| {
            {
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                for (0..reps) |_| for (w.ms) |m| try ex.encode(&ref, e, m, .{ .buffer = x }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = s.k, .y_stride = s.n });
                const secs = try run(cb, e);
                if (pass == 1) t_ref = secs / @as(f64, @floatFromInt(reps * copies));
            }
            {
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                for (w.bufs) |bb| e.useResource(bb, mtl.ResourceUsage.read);
                for (0..reps) |_| for (0..copies) |c| {
                    const d = xr.Dense{ .jobs = .{ .buffer = jobs, .offset = c * @sizeOf(xr.Job) }, .count = 1, .blocks = s.n / 128, .k = s.k };
                    if (mm) xr.encodeMm(&pm, e, d, .{ .buffer = x }, s.k, .{ .buffer = y }, s.n, rows) else xr.encode(&p, e, d, .{ .buffer = x }, s.k, .{ .buffer = y }, s.n, rows, .bf16, kt, sc);
                };
                const secs = try run(cb, e);
                if (pass == 1) t_new = secs / @as(f64, @floatFromInt(reps * copies));
            }
        }
        const mb = @as(f64, @floatFromInt(one)) / 1e6;
        std.debug.print("{s:<12} k={d} n={d} bits={d} rows={d} ranges={d} copies={d} {d:.1} MB  ref {d:.1} us ({d:.0} GB/s)  rows {d:.1} us ({d:.0} GB/s)  max|diff| {e:.2} of {e:.2}, {d} of {d} bf16 differ\n", .{ s.name, s.k, s.n, s.halves / 2, rows, kt, copies, mb, t_ref * 1e6, mb / 1e3 / t_ref, t_new * 1e6, mb / 1e3 / t_new, worst, scale, differ, rows * s.n });
    }
}

/// The routed kernel against the reference, expert by expert: 8 experts with gate and up, 2 tokens' 4 picks each.
fn routedCheck(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ref: *const ex.Pipelines, p: *const xr.Pipelines) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const s = Shape{ .name = "routed", .k = 6144, .n = 3072, .halves = 6 };
    const experts = 8;
    const kinds = 2;
    const w = try Owned.init(gpa, device, s, experts * kinds, 99);
    defer w.deinit(gpa);
    const jobs = try device.buffer(experts * kinds * @sizeOf(xr.Job), mtl.ResourceOptions.shared);
    defer jobs.deinit();
    for (jobs.slice(xr.Job, experts * kinds), w.ms, 0..) |*j, m, i| j.* = try xr.job(m, (i % kinds) * s.n);
    const picks = [_]u32{ 3, 1, 7, 0, 2, 2, 5, 6 };
    const slots = picks.len;
    const ids = try device.buffer(slots * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, slots), &picks);
    const x = try device.buffer(2 * s.k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    for (x.slice(u16, 2 * s.k), 0..) |*v, q| v.* = bf16(@cos(@as(f32, @floatFromInt(q)) * 0.29) * 2.0);
    const width = kinds * s.n;
    const y_ref = try device.buffer(slots * width * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(slots * width * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(s.k, s.n, 1), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const blocks = kinds * s.n / 128;
    const kt = xr.ranges(blocks, s.k);
    const partial = try device.buffer(@max(16, xr.partialBytes(kt, slots, blocks)), mtl.ResourceOptions.shared);
    defer partial.deinit();
    const counts = try device.buffer(@max(16, xr.countBytes(slots, blocks)), mtl.ResourceOptions.shared);
    defer counts.deinit();
    @memset(counts.contents()[0..counts.length()], 0);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    for (picks, 0..) |pk, slot| for (0..kinds) |q| {
        try ex.encode(ref, e, w.ms[pk * kinds + q], .{ .buffer = x, .offset = (slot / 4) * s.k * 2 }, .{ .buffer = y_ref, .offset = (slot * width + q * s.n) * 2 }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = s.k, .y_stride = s.n });
    };
    for (w.bufs) |bb| e.useResource(bb, mtl.ResourceUsage.read);
    xr.encodeRouted(p, e, .{ .jobs = .{ .buffer = jobs }, .experts = experts, .kinds = kinds, .n = s.n, .k = s.k }, .{ .buffer = x }, s.k, .{ .buffer = ids }, 4, .{ .buffer = y }, width, slots, slots, 0, .bf16, kt, .{ .partial = .{ .buffer = partial }, .counts = .{ .buffer = counts } });
    _ = try run(cb, e);
    var worst: f32 = 0;
    var differ: usize = 0;
    for (y_ref.slice(u16, slots * width), y.slice(u16, slots * width)) |a, bq| {
        worst = @max(worst, @abs(f32of(a) - f32of(bq)));
        if (a != bq) differ += 1;
    }
    std.debug.print("routed       experts={d} kinds={d} slots={d} ranges={d} max|diff| {e:.2}, {d} of {d} bf16 differ\n", .{ experts, kinds, slots, kt, worst, differ, slots * width });
}
