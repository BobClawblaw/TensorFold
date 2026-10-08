//! Flash Next on CUDA, M1: the native forward (src/families/flashnext_cuda/forward.zig) through its engine API only,
//! on two ranks: the reference prompt and Python's MTP reply, token for token, with the same rounds and drafts.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fwd = @import("flashnext_weights").forward;
const Gpu = check.Gpu;

fn ints(gpa: std.mem.Allocator, v: std.json.Value) ![]i64 {
    const a = v.array.items;
    const out = try gpa.alloc(i64, a.len);
    for (a, 0..) |x, i| out[i] = x.integer;
    return out;
}

/// fn-native <rank> <rank 0 address> <port> <triton dir> <pack.bin> <pack.json> <ngram.json> <ngram root> <reference.json> [more prompt lengths ...]
pub fn native(gpu: Gpu, args: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
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
    { // NCCL connects (and registers its buffers) before the weights and tables take the memory
        const warm = try cuda.DeviceBuffer.alloc(gpu.d, 4096);
        try nccl.check(nccl.api.ncclAllGather(warm.ptr, warm.ptr + 2048, 256, .i32, comm, stream.handle), "warm gather");
        try stream.synchronize();
    }
    const t0 = std.Io.Timestamp.now(io, .awake);
    var kernels = try fwd.api_.Kernels.load(gpa, io, gpu.d, gpu.ctx.device, args[3]);
    var store = try fwd.devstore.load(gpa, io, gpu.d, args[4], args[5], args[6], args[7]);
    var ctx: fwd.api_.Ctx = .{ .d = gpu.d, .ctx = gpu.ctx, .stream = stream, .nccl = &nccl, .comm = comm, .rank = rank };
    const e = try fwd.init(gpa, io, &ctx, &kernels, &store, .{ .context = 16384, .max_rows = 16, .depth = 15 });
    try fwd.prefetchTables(e);
    e.k.sync_each = std.mem.indexOf(u8, args[1], "sync") != null or (args.len > 9 and std.mem.eql(u8, args[args.len - 1], "sync"));
    std.debug.print("rank {d}: engine up in {d:.1} s\n", .{ rank, @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e9 });

    std.debug.print("rank {d}: reading {s}\n", .{ rank, args[8] });
    const rtext = try std.Io.Dir.cwd().readFileAlloc(io, args[8], gpa, .limited(1 << 26));
    const ref = (try std.json.parseFromSlice(std.json.Value, gpa, rtext, .{})).value.object;
    const prompt64 = try ints(gpa, ref.get("prompt").?);
    const want = try ints(gpa, ref.get("tokens").?);
    const prompt = try gpa.alloc(u32, prompt64.len);
    for (prompt64, prompt) |x, *y| y.* = @intCast(x);

    const s = try fwd.newSeq(e);
    std.debug.print("rank {d}: sequence allocated, prefilling\n", .{rank});
    const tp = std.Io.Timestamp.now(io, .awake);
    const first = try fwd.prefill(e, s, prompt);
    const prefill_ms = @as(f64, @floatFromInt(tp.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
    std.debug.print("prefill {d} tokens in {d:.1} ms; first token {d} (Python {d})\n", .{ prompt.len, prefill_ms, first, want[0] });

    // decode.mtp_decode: depth 6, confidence 0.7, as the reference ran
    const depth: usize = 6;
    const count = want.len;
    var out: std.ArrayList(u32) = .empty;
    try out.append(gpa, first);
    var drafts: [16]u32 = undefined;
    var nd: usize = 0;
    var rounds: usize = 0;
    var drafted: usize = 0;
    var accepted: usize = 0;
    const ts = std.Io.Timestamp.now(io, .awake);
    nd = try fwd.draftUpTo(e, s, &[1]u32{first}, @intCast(@min(depth, count - out.items.len)), 0.7, &drafts);
    while (out.items.len < count) {
        var tokens: [16]u32 = undefined;
        tokens[0] = out.items[out.items.len - 1];
        @memcpy(tokens[1 .. 1 + nd], drafts[0..nd]);
        const R = 1 + nd;
        var sampled: [16]u32 = undefined;
        try fwd.verify(e, s, tokens[0..R], sampled[0..R]);
        var keep: usize = 1;
        for (drafts[0..nd], 0..) |d, i| {
            if (sampled[i] != d) break;
            keep += 1;
        }
        try fwd.keep(e, s, @intCast(R), @intCast(keep));
        rounds += 1;
        drafted += nd;
        accepted += keep - 1;
        try out.appendSlice(gpa, sampled[0..keep]);
        if (out.items.len >= count) break;
        const n = @min(depth, count - out.items.len);
        nd = try fwd.draftUpTo(e, s, sampled[0..keep], @intCast(n), 0.7, &drafts);
    }
    try stream.synchronize();
    const secs = @as(f64, @floatFromInt(ts.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e9;
    var same: usize = 0;
    for (out.items[0..@min(count, out.items.len)], want) |g, w| {
        if (g != w) break;
        same += 1;
    }
    std.debug.print("tokens: {any}\n", .{out.items[0..@min(12, out.items.len)]});
    std.debug.print("MTP decode (native forward, eager): {d} tokens in {d:.3} s = {d:.1} tok/s; {d} rounds, {d} drafted, {d} accepted (Python: {d}, {d}, {d})\n", .{
        count - 1, secs, @as(f64, @floatFromInt(count - 1)) / secs, rounds, drafted, accepted,
        ref.get("rounds").?.integer, ref.get("drafted").?.integer, ref.get("accepted").?.integer });
    fwd.freeSeq(e, s);
    // other prompt lengths: their first tokens (prefixes of the reference prompt), checked by the caller's list
    var i: usize = 9;
    while (i < args.len) : (i += 1) {
        const n = std.fmt.parseInt(usize, args[i], 10) catch continue;
        const s2 = try fwd.newSeq(e);
        const p = try gpa.alloc(u32, n); // the reference prompt repeated to n tokens (first_tokens.py builds the same)
        for (p, 0..) |*x, j| x.* = prompt[j % prompt.len];
        const t = try fwd.prefill(e, s2, p);
        std.debug.print("prefix of {d} tokens: first token {d}\n", .{ n, t });
        fwd.freeSeq(e, s2);
    }
    try check.expect(same == count, "{d} of {d} tokens equal Python's", .{ same, count });
    check.pass("EXACT rank {d}: native forward, prefill + {d} tokens with MTP drafts equal the Python engine's", .{ rank, count });
}
