//! fn-gathers <rank> <rank 0 address> <port> [rows ...]: the forward's partial-sum gathers (fp32, rows x 2560 a rank)
//! timed between the two ranks: back to back eager, captured in one CUDA graph, and eager between small graphs, each
//! with a memset between gathers (a forward's dependency chain). NCCL_* settings come from the environment.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const D = 2560;
const N = 96; // gathers a forward

pub fn run(gpu: Gpu, args: []const [:0]const u8) !void {
    const io = gpu.io;
    const rank = try std.fmt.parseInt(u8, args[0], 10);
    const port = try std.fmt.parseInt(u16, args[2], 10);
    var link = if (rank == 0) try cuda.tp_link.Link.lead(port, 300_000)
        else try cuda.tp_link.Link.follow(io, try std.Io.net.IpAddress.parse(args[1], port), 300_000);
    defer link.close();
    var nccl = try cuda.nccl.Library.open();
    defer nccl.close();
    const comm = try cuda.tp_link.communicator(&nccl, link);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const send = try cuda.DeviceBuffer.alloc(gpu.d, 16 * D * 4);
    const recv = try cuda.DeviceBuffer.alloc(gpu.d, 2 * 16 * D * 4);
    const pad = try cuda.DeviceBuffer.alloc(gpu.d, 64 * 1024);
    var rows_list: [8]usize = undefined;
    var nr: usize = 0;
    for (args[3..]) |a| {
        rows_list[nr] = try std.fmt.parseInt(usize, a, 10);
        nr += 1;
    }
    if (nr == 0) {
        rows_list[0..3].* = .{ 1, 6, 16 };
        nr = 3;
    }
    const Ctx = struct {
        nccl: *cuda.nccl.Library,
        comm: cuda.nccl.Comm,
        stream: cuda.Stream,
        d: *const cuda.Driver,
        send: u64,
        recv: u64,
        pad: u64,

        fn gather(c: @This(), count: usize) !void {
            try c.nccl.check(c.nccl.api.ncclAllGather(c.send, c.recv, count, .f32, c.comm, c.stream.handle), "gather");
        }
        fn touch(c: @This()) !void {
            try c.d.check(c.d.api.cuMemsetD32Async(c.pad, 0, 1024, c.stream.handle), "memset");
        }
    };
    const c: Ctx = .{ .nccl = &nccl, .comm = comm, .stream = stream, .d = gpu.d, .send = send.ptr, .recv = recv.ptr, .pad = pad.ptr };
    for (0..20) |_| try c.gather(D); // connect and warm
    try stream.synchronize();
    for (rows_list[0..nr]) |rows| {
        const count = rows * D;
        // eager, back to back with a memset between
        var best_e: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try c.gather(count);
            try stream.synchronize();
            const t = std.Io.Timestamp.now(io, .awake);
            for (0..N) |_| {
                try c.touch();
                try c.gather(count);
            }
            try stream.synchronize();
            best_e = @min(best_e, us(io, t) / N);
        }
        // one graph of N gathers (memsets between)
        try cuda.graph.beginCapture(stream, .thread_local);
        for (0..N) |_| {
            try c.touch();
            try c.gather(count);
        }
        var g = try cuda.graph.endCapture(stream);
        const ex = try g.instantiate();
        g.deinit();
        try ex.upload(stream);
        var best_g: f64 = std.math.inf(f64);
        for (0..6) |rep| {
            try c.gather(count);
            try stream.synchronize();
            const t = std.Io.Timestamp.now(io, .awake);
            try ex.launchOn(stream);
            try stream.synchronize();
            if (rep > 0) best_g = @min(best_g, us(io, t) / N);
        }
        // eager gathers between one-memset graphs (graphs split around the collectives)
        try cuda.graph.beginCapture(stream, .thread_local);
        try c.touch();
        var g2 = try cuda.graph.endCapture(stream);
        const ex2 = try g2.instantiate();
        g2.deinit();
        try ex2.upload(stream);
        var best_s: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try c.gather(count);
            try stream.synchronize();
            const t = std.Io.Timestamp.now(io, .awake);
            for (0..N) |_| {
                try ex2.launchOn(stream);
                try c.gather(count);
            }
            try stream.synchronize();
            best_s = @min(best_s, us(io, t) / N);
        }
        std.debug.print("rank {d}: {d} rows ({d} KiB a rank): us a gather: eager {d:.1}, one graph {d:.1}, split graphs {d:.1}\n", .{
            rank, rows, count * 4 / 1024, best_e, best_g, best_s });
    }
    check.pass("rank {d}: gathers timed", .{rank});
}

fn us(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e3;
}
