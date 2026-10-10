//! cuda.relocate on the GPU: a graph captured on one buffer replayed on another, kernel, copy and memset nodes.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

fn read(buf: cuda.DeviceBuffer, at: usize) !u64 {
    var v: u64 = 0;
    try buf.download(at, std.mem.asBytes(&v));
    return v;
}

/// Captured on `a`: zero word 0, add 1..4 to it, copy it to word 1, set word 2's bytes to 7; then replayed on `b`.
pub fn run(gpu: Gpu) !void {
    const d = gpu.d;
    if (d.api.cuFuncGetParamInfo == null) {
        check.pass("SKIP relocate: the driver has no cuFuncGetParamInfo (CUDA 12.4+)", .{});
        return;
    }
    var probe = try cuda.Module.load(d, cuda.kernels.probe);
    defer probe.unload();
    const step = try probe.function("tf_probe_step");
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var a = try cuda.DeviceBuffer.alloc(d, 64);
    defer a.free();
    var b = try cuda.DeviceBuffer.alloc(d, 64);
    defer b.free();
    try a.fill8(0xff, null);
    try b.fill8(0xff, null);
    const one: cuda.Config = .{ .grid = .{}, .block = .{} };

    try cuda.graph.beginCapture(stream, .thread_local);
    try d.check(d.api.cuMemsetD8Async(a.ptr, 0, 8, stream.handle), "cuMemsetD8Async");
    for (1..5) |i| {
        var args: cuda.Args = .{};
        args.add(a.ptr);
        args.add(@as(u64, i));
        try cuda.launch.launch(step, one, stream, &args);
    }
    try d.check(d.api.cuMemcpyDtoDAsync_v2(a.ptr + 8, a.ptr, 8, stream.handle), "cuMemcpyDtoDAsync");
    try d.check(d.api.cuMemsetD8Async(a.ptr + 16, 7, 8, stream.handle), "cuMemsetD8Async");
    var g = try cuda.graph.endCapture(stream);
    defer g.deinit();

    var regions: cuda.relocate.Regions = .{};
    regions.add(a.ptr, 64);
    var plan = (try cuda.relocate.read(gpu.gpa, g, &regions)) orelse return error.NoPlan;
    defer plan.deinit(gpu.gpa);
    try expect(plan.kernels.len == 4 and plan.copies.len == 1 and plan.sets.len == 2, "4 kernels, 1 copy and 2 memsets point into the buffer", .{});
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try read(a, 0) == 10 and try read(a, 8) == 10 and try read(a, 16) == 0x0707070707070707, "replayed where captured", .{});

    var to: cuda.relocate.Regions = .{};
    to.add(b.ptr, 64);
    try expect(try plan.bind(exec, gpu.ctx.handle, &to), "the graph moves to the second buffer", .{});
    try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try read(b, 0) == 10 and try read(b, 8) == 10 and try read(b, 16) == 0x0707070707070707, "replayed on the second buffer", .{});
    try a.fill8(0xff, null);
    try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try read(a, 0) == 0xffffffffffffffff, "the first buffer is left alone once moved", .{});
    try expect(!(try plan.bind(exec, gpu.ctx.handle, &to)), "binding where it points already does nothing", .{});
    try expect(try plan.bind(exec, gpu.ctx.handle, &regions), "and it moves back", .{});
    try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try read(a, 0) == 10 and try read(a, 16) == 0x0707070707070707, "replayed on the first buffer again", .{});
    check.pass("relocate: kernels, a copy and memsets captured on one buffer, replayed on another and back", .{});
}
