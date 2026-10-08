//! Flash Next on CUDA, M1: checks of the pieces the native engine is built from (the Triton kernel set as 1.0.2's
//! aot.Set loads it, the extension fatbins as the build embeds them).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// fn-aot <kernel dir>: every captured Triton variant loads as aot.Set loads Nemotron's.
pub fn aotLoad(gpu: Gpu, dir: []const u8) !void {
    var set = try cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, dir);
    defer set.deinit();
    var fns = std.StringHashMap(void).init(gpu.gpa);
    defer fns.deinit();
    for (set.variants) |v| try fns.put(v.spec.@"fn", {});
    check.pass("Flash Next Triton set: {d} variants of {d} functions loaded from {s}", .{ set.variants.len, fns.count(), dir });
}

/// fn-fatbins: every embedded Flash Next extension image loads and holds the kernels the engine launches.
pub fn fatbins(gpu: Gpu) !void {
    const images = [_]struct { name: []const u8, bytes: []const u8 }{
        .{ .name = "fn_experts", .bytes = cuda.kernels.fn_experts },
        .{ .name = "fn_experts_prefill", .bytes = cuda.kernels.fn_experts_prefill },
        .{ .name = "fn_gdn_prefill", .bytes = cuda.kernels.fn_gdn_prefill },
        .{ .name = "fn_qmm", .bytes = cuda.kernels.fn_qmm },
        .{ .name = "fn_qmm_prefill", .bytes = cuda.kernels.fn_qmm_prefill },
        .{ .name = "fn_gdn", .bytes = cuda.kernels.fn_gdn },
        .{ .name = "fn_gdn_io", .bytes = cuda.kernels.fn_gdn_io },
    };
    try check.expect(cuda.kernels.available, "this build embeds no fatbins (build with -Dnvcc or -Dfatbins)", .{});
    for (images) |im| {
        var m = try cuda.Module.load(gpu.d, im.bytes);
        m.unload();
    }
    check.pass("Flash Next extension fatbins: {d} images load", .{images.len});
}

/// fn-tp <rank> <rank 0 address> <port>: the two-rank link (TCP frames), the NCCL communicator it brings up, and an
/// all-gather across both GPUs.
pub fn tpLink(gpu: Gpu, rank_s: []const u8, host: []const u8, port_s: []const u8) !void {
    const rank = try std.fmt.parseInt(u8, rank_s, 10);
    const port = try std.fmt.parseInt(u16, port_s, 10);
    var link = if (rank == 0) try cuda.tp_link.Link.lead(port, 120_000)
        else try cuda.tp_link.Link.follow(gpu.io, try std.Io.net.IpAddress.parse(host, port), 120_000);
    defer link.close();
    // a framed request one way, an acknowledgement back
    if (rank == 0) {
        try link.send(42, "admit: prompt of 2695 tokens");
        const ack = try link.recv(gpu.gpa);
        try check.expect(ack.tag == 43 and std.mem.eql(u8, ack.bytes, "ok"), "rank 1's acknowledgement", .{});
    } else {
        const m = try link.recv(gpu.gpa);
        try check.expect(m.tag == 42, "rank 0's request frame", .{});
        try link.send(43, "ok");
    }
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try cuda.tp_link.communicator(&lib, link);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const buf = try cuda.DeviceBuffer.alloc(gpu.d, 3 * 4096);
    var mine: [1024]u32 = undefined;
    for (&mine, 0..) |*x, i| x.* = @as(u32, rank) * 1_000_000 + @as(u32, @intCast(i));
    try buf.upload(0, std.mem.asBytes(&mine));
    try lib.check(lib.api.ncclAllGather(buf.ptr, buf.ptr + 4096, 1024, .u32, comm, stream.handle), "ncclAllGather");
    try stream.synchronize();
    var all: [2048]u32 = undefined;
    try buf.download(4096, std.mem.asBytes(&all));
    for (all, 0..) |x, i| try check.expect(x == @as(u32, @intCast(i / 1024)) * 1_000_000 + @as(u32, @intCast(i % 1024)), "gathered word {d}", .{i});
    _ = lib.api.ncclCommDestroy(comm);
    check.pass("rank {d}: TCP link frames, NCCL communicator from the link, all-gather of both GPUs' words equal", .{rank});
}
