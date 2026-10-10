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

/// fn-weights <served folder> <rank> <pack dir>: the native loader's tensors against the Python engine's prepared
/// weights (<pack dir>/weights.json + weights.bin), every one byte for byte.
pub fn weightsCheck(gpu: Gpu, dir: []const u8, rank_s: []const u8, pack: []const u8) !void {
    const fw = @import("flashnext_weights");
    const rank = try std.fmt.parseInt(u8, rank_s, 10);
    const gpa = gpu.gpa;
    const t0 = std.Io.Timestamp.now(gpu.io, .awake);
    const planned = try fw.deviceBytes(gpu.io, dir);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const ctx: fw.api.Ctx = .{ .d = gpu.d, .ctx = gpu.ctx, .stream = stream, .nccl = undefined, .comm = null, .rank = rank };
    var kernels: fw.api.Kernels = undefined;
    var store = try fw.load(gpa, gpu.io, &ctx, &kernels, dir, rank);
    defer store.deinit();
    const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(gpu.io, .awake)).toNanoseconds())) / 1e9;
    std.debug.print("loaded {d} tensors, {d:.2} GiB (planned {d:.2} GiB) in {d:.1} s\n", .{ store.map.count(),
        @as(f64, @floatFromInt(store.ints.get("device_bytes").?)) / (1 << 30), @as(f64, @floatFromInt(planned)) / (1 << 30), secs });
    const jp = try std.fs.path.join(gpa, &.{ pack, "weights.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, jp, gpa, .limited(1 << 26));
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    const bp = try std.fs.path.join(gpa, &.{ pack, "weights.bin" });
    const bin = try std.Io.Dir.cwd().openFile(gpu.io, bp, .{});
    defer bin.close(gpu.io);
    var equal: usize = 0;
    var bad: usize = 0;
    var seen = std.StringHashMap(void).init(gpa);
    for (parsed.value.object.get("tensors").?.array.items) |tv| {
        const t = tv.object;
        const name = t.get("name").?.string;
        try seen.put(name, {});
        const n: usize = @intCast(t.get("bytes").?.integer);
        const got = store.map.get(name) orelse {
            bad += 1;
            std.debug.print("MISSING {s}\n", .{name});
            continue;
        };
        if (got.bytes() != n) {
            bad += 1;
            std.debug.print("SIZE {s}: {d} bytes, the pack {d}\n", .{ name, got.bytes(), n });
            continue;
        }
        const want = try gpa.alloc(u8, n);
        defer gpa.free(want);
        _ = try bin.readPositionalAll(gpu.io, want, @intCast(t.get("offset").?.integer));
        const have = try gpa.alloc(u8, n);
        defer gpa.free(have);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(have.ptr, got.ptr, n), "download");
        if (std.mem.eql(u8, want, have)) {
            equal += 1;
        } else {
            bad += 1;
            const at = std.mem.indexOfDiff(u8, want, have).?;
            var diff: usize = 0;
            for (want, have) |x, y| diff += @intFromBool(x != y);
            std.debug.print("DIFFERS {s} {s}: from byte {d}, {d} of {d} bytes\n", .{ name, @tagName(got.dtype), at, diff, n });
        }
    }
    var extra: usize = 0;
    var it = store.map.keyIterator();
    while (it.next()) |k| if (!seen.contains(k.*)) {
        extra += 1;
        std.debug.print("EXTRA {s} (not in the pack)\n", .{k.*});
    };
    try check.expect(bad == 0, "{d} tensors equal, {d} not", .{ equal, bad });
    // the host facts: the n-gram table (every field but the shard files' paths) and the draft ids
    const np_ = try std.fs.path.join(gpa, &.{ pack, "ngram.json" });
    if (std.Io.Dir.cwd().readFileAlloc(gpu.io, np_, gpa, .limited(1 << 24))) |want_text| {
        const want = (try std.json.parseFromSlice(std.json.Value, gpa, want_text, .{})).value.object;
        const have = (try std.json.parseFromSlice(std.json.Value, gpa, store.host.get("ngram.json").?, .{})).value.object;
        var it2 = want.iterator();
        while (it2.next()) |e| {
            const k = e.key_ptr.*;
            const h = have.get(k) orelse return check.expect(false, "ngram.json lacks {s}", .{k});
            if (std.mem.eql(u8, k, "shards")) {
                for (e.value_ptr.array.items, h.array.items) |x, y| {
                    try check.expect(x.object.get("offset").?.integer == y.object.get("offset").?.integer and
                        x.object.get("rows").?.integer == y.object.get("rows").?.integer and
                        std.mem.eql(u8, std.fs.path.basename(x.object.get("file").?.string), std.fs.path.basename(y.object.get("file").?.string)),
                        "ngram.json shard differs", .{});
                }
                continue;
            }
            var a1: std.Io.Writer.Allocating = .init(gpa);
            var a2: std.Io.Writer.Allocating = .init(gpa);
            try std.json.Stringify.value(e.value_ptr.*, .{}, &a1.writer);
            try std.json.Stringify.value(h, .{}, &a2.writer);
            try check.expect(std.mem.eql(u8, a1.written(), a2.written()), "ngram.json {s} differs", .{k});
        }
        std.debug.print("ngram.json: every field equals the capture's\n", .{});
    } else |_| {}
    const dp = try std.fs.path.join(gpa, &.{ pack, "draft_ids.bin" });
    if (std.Io.Dir.cwd().readFileAlloc(gpu.io, dp, gpa, .limited(1 << 24))) |ids| {
        try check.expect(std.mem.eql(u8, ids, store.host.get("draft_ids").?), "draft ids differ", .{});
        std.debug.print("draft ids: {d} equal the capture's\n", .{ids.len / 4});
    } else |_| {}
    check.pass("rank {d}: all {d} prepared tensors equal the Python engine's ({d} only ours)", .{ rank, equal, extra });
}
