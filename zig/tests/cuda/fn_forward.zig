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

/// fn-native <rank> <rank 0 address> <port> <triton dir> <pack.bin> <pack.json> <ngram.json> <ngram root> <reference.json> [more prompt lengths ...] [kv4]
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
    var ctx: fwd.api_.Ctx = .{ .d = gpu.d, .ctx = gpu.ctx, .stream = stream, .nccl = &nccl, .comm = comm, .rank = rank };
    // weights: "ckpt:<dir>" loads the converted checkpoint (weights.load), else a Python pack (bin, index)
    var store = if (std.mem.startsWith(u8, args[4], "ckpt:"))
        try @import("flashnext_weights").load(gpa, io, &ctx, &kernels, args[4][5..], rank)
    else
        try fwd.devstore.load(gpa, io, gpu.d, args[4], args[5], args[6], args[7]);
    // "kv4" among the trailing words: the int4 attention cache (Python's --kv-dtype int4; its reference then)
    var kv_bits: u8 = 8;
    for (args[@min(9, args.len)..]) |a| if (std.mem.eql(u8, a, "kv4")) {
        kv_bits = 4;
    };
    const e = try fwd.init(gpa, io, &ctx, &kernels, &store, .{ .context = 16384, .max_rows = rows: {
        // "rows=<n>": the shared rounds' rows at most (the server's batch_rows)
        for (args[@min(9, args.len)..]) |a| if (std.mem.startsWith(u8, a, "rows=")) break :rows try std.fmt.parseInt(u32, a[5..], 10);
        break :rows 16;
    }, .depth = 15, .kv_bits = kv_bits });
    try fwd.prefetchTables(e);
    e.k.sync_each = std.mem.indexOf(u8, args[1], "sync") != null or (args.len > 9 and std.mem.eql(u8, args[args.len - 1], "sync"));
    // "nographs" among the trailing words: verify and draft steps eager (the graphs' A/B)
    for (args[@min(9, args.len)..]) |a| if (std.mem.eql(u8, a, "nographs")) {
        e.use_graphs = false;
    };
    // "nomultigdn", "nomultiattn": shared rounds launch DeltaNet chains or attention stream by stream (the A/B)
    for (args[@min(9, args.len)..]) |a| {
        if (std.mem.eql(u8, a, "nomultigdn")) e.multi_gdn = false;
        if (std.mem.eql(u8, a, "nomultiattn")) e.multi_attn = false;
        if (std.mem.eql(u8, a, "split")) e.split_gathers = true; // graphs split at the gathers (eager between)
        if (std.mem.eql(u8, a, "nosplit")) e.split_gathers = false;
        if (std.mem.eql(u8, a, "noshare")) e.share_graphs = false; // each sequence captures its own graphs
    }
    // "multi=on|off" between benches below: both on or both off from there
    std.debug.print("rank {d}: shared rounds: DeltaNet {s}, attention {s}\n", .{ rank, if (e.multi_gdn) "one launch" else "per stream", if (e.multi_attn) "one launch" else "per stream" });
    std.debug.print("rank {d}: n-gram tables locked {d} of {d} GiB\n", .{ rank, e.ng.locked >> 30, blk: {
        var t: usize = 0;
        for (e.ng.tables) |x| t += x.len;
        break :blk t >> 30;
    } });
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
    const first = try fwd.prefill(e, s, prompt, 0);
    const prefill_ms = @as(f64, @floatFromInt(tp.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
    std.debug.print("prefill {d} tokens in {d:.1} ms; first token {d} (Python {d})\n", .{ prompt.len, prefill_ms, first, want[0] });

    const tw = std.Io.Timestamp.now(io, .awake);
    try fwd.warm(e, s, 7);
    std.debug.print("warmed graphs in {d:.0} ms\n", .{@as(f64, @floatFromInt(tw.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6});
    e.timing = true;
    e.stage_ms = 0;
    e.split = .{ 0, 0, 0 };
    // decode.mtp_decode: depth 6, confidence 0.7, as the reference ran
    const depth: usize = 6;
    const count = want.len;
    var out: std.ArrayList(u32) = .empty;
    try out.append(gpa, first);
    var drafts: [16]u32 = undefined;
    var nd: usize = 0;
    var t_verify: f64 = 0;
    var t_keep: f64 = 0;
    var t_draft: f64 = 0;
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
        var tq = std.Io.Timestamp.now(io, .awake);
        try fwd.verify(e, s, tokens[0..R], sampled[0..R]);
        t_verify += @as(f64, @floatFromInt(tq.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
        var keep: usize = 1;
        for (drafts[0..nd], 0..) |d, i| {
            if (sampled[i] != d) break;
            keep += 1;
        }
        tq = std.Io.Timestamp.now(io, .awake);
        try fwd.keep(e, s, @intCast(R), @intCast(keep));
        t_keep += @as(f64, @floatFromInt(tq.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
        rounds += 1;
        drafted += nd;
        accepted += keep - 1;
        try out.appendSlice(gpa, sampled[0..keep]);
        if (out.items.len >= count) break;
        const n = @min(depth, count - out.items.len);
        tq = std.Io.Timestamp.now(io, .awake);
        nd = try fwd.draftUpTo(e, s, sampled[0..keep], @intCast(n), 0.7, &drafts);
        t_draft += @as(f64, @floatFromInt(tq.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
    }
    try stream.synchronize();
    const secs = @as(f64, @floatFromInt(ts.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e9;
    var same: usize = 0;
    for (out.items[0..@min(count, out.items.len)], want) |g, w| {
        if (g != w) break;
        same += 1;
    }
    std.debug.print("tokens: {any}\n", .{out.items[0..@min(12, out.items.len)]});
    std.debug.print("MTP decode (native forward, {s}): {d} tokens in {d:.3} s = {d:.1} tok/s; {d} rounds, {d} drafted, {d} accepted (Python: {d}, {d}, {d})\n", .{
        if (e.use_graphs) "graphs" else "eager", count - 1, secs, @as(f64, @floatFromInt(count - 1)) / secs, rounds, drafted, accepted,
        ref.get("rounds").?.integer, ref.get("drafted").?.integer, ref.get("accepted").?.integer });
    std.debug.print("time: verify {d:.1} ms ({d:.2} a round, staging {d:.1} ms, GPU {d:.1} ms), keep {d:.1} ms, draft {d:.1} ms\n", .{ t_verify, t_verify / @as(f64, @floatFromInt(rounds)), e.stage_ms, e.gpu_ms, t_keep, t_draft });
    std.debug.print("staging split: ids upload {d:.1} ms, n-gram ids {d:.1} ms, table rows {d:.1} ms\n", .{ e.split[0], e.split[1], e.split[2] });
    fwd.freeSeq(e, s);
    // other prompt lengths: their first tokens (prefixes of the reference prompt), checked by the caller's list
    var i: usize = 9;
    while (i < args.len) : (i += 1) {
        const n = std.fmt.parseInt(usize, args[i], 10) catch continue;
        const s2 = try fwd.newSeq(e);
        const p = try gpa.alloc(u32, n); // the reference prompt repeated to n tokens (first_tokens.py builds the same)
        for (p, 0..) |*x, j| x.* = prompt[j % prompt.len];
        const tn = std.Io.Timestamp.now(io, .awake);
        const t = try fwd.prefill(e, s2, p, 0);
        const ms = @as(f64, @floatFromInt(tn.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
        std.debug.print("prefix of {d} tokens: first token {d} in {d:.1} ms\n", .{ n, t, ms });
        fwd.freeSeq(e, s2);
    }
    try check.expect(same == count, "{d} of {d} tokens equal Python's", .{ same, count });
    check.pass("EXACT rank {d}: native forward, prefill + {d} tokens with MTP drafts equal the Python engine's", .{ rank, count });
    for (args[@min(9, args.len)..]) |a| if (std.mem.eql(u8, a, "shared")) try sharedCheck(e, prompt, rank);
    for (args[@min(9, args.len)..]) |a| if (std.mem.eql(u8, a, "resume")) try resumeCheck(e, prompt, rank);
    const has_multi = e.multi_attn;
    for (args[@min(9, args.len)..]) |a| {
        if (std.mem.eql(u8, a, "multi=off")) {
            e.multi_gdn = false;
            e.multi_attn = false;
        }
        if (std.mem.eql(u8, a, "multi=gdn")) {
            e.multi_gdn = true;
            e.multi_attn = false;
        }
        if (std.mem.eql(u8, a, "multi=on")) {
            e.multi_gdn = true;
            e.multi_attn = has_multi;
        }
        if (std.mem.startsWith(u8, a, "bench=")) {
            std.debug.print("rank {d}: bench with DeltaNet {s}, attention {s}\n", .{ rank, if (e.multi_gdn) "one launch" else "per stream", if (e.multi_attn) "one launch" else "per stream" });
            try benchShared(e, prompt, rank, a["bench=".len..]);
        }
        if (std.mem.eql(u8, a, "admit")) try benchAdmit(e, prompt, rank);
        if (std.mem.eql(u8, a, "prefills")) try prefillsCheck(e, prompt, rank);
        // "exit": leave through libc's exit, whose handlers flush a profiler's trace buffers (nsys)
        if (std.mem.eql(u8, a, "exit")) {
            try e.k.stream.synchronize();
            std.c.exit(0);
        }
    }
    for (args[@min(9, args.len)..]) |a| if (std.mem.startsWith(u8, a, "vision=")) try visionCheck(e, gpu, rank, a["vision=".len..]);
    for (args[@min(9, args.len)..]) |a| if (std.mem.startsWith(u8, a, "probes=")) try probesDecode(e, gpu, rank, a["probes=".len..]);
}

/// A next turn resumed from the kept point of the last (one token before its prompt's end) decodes as the same turn
/// prefilled fresh: the prompt, then a longer one sharing all but its last token, 32 greedy tokens each way.
fn resumeCheck(e: *fwd.Engine, prompt: []const u32, rank: u8) !void {
    const gpa = e.gpa;
    const k = prompt.len - 1;
    const next = try gpa.alloc(u32, k + 300);
    defer gpa.free(next);
    @memcpy(next[0..k], prompt[0..k]);
    for (next[k..], 0..) |*x, j| x.* = prompt[(j * 7 + 3) % prompt.len];
    var runs: [3][32]u32 = undefined; // fresh; resumed; resumed after a repeat of the kept prompt (a regenerate)
    for (0..3) |way| {
        const s = try fwd.newSeq(e);
        defer fwd.freeSeq(e, s);
        var tok: u32 = undefined;
        if (way == 0) {
            s.cut_at = @intCast(next.len - 1);
            tok = try fwd.prefill(e, s, next, 0);
        } else {
            s.cut_at = @intCast(k);
            _ = try fwd.prefill(e, s, prompt, 0);
            try check.expect(s.snap.at == @as(i64, @intCast(k)), "the kept point is at {d} (want {d})", .{ s.snap.at, k });
            if (way == 2) { // the same prompt again, resumed at its own kept point: the point must stay kept
                s.cut_at = @intCast(k);
                _ = try fwd.prefill(e, s, prompt, k);
                try check.expect(s.snap.at == @as(i64, @intCast(k)), "a repeat keeps the point at {d} (want {d})", .{ s.snap.at, k });
            }
            s.cut_at = @intCast(next.len - 1);
            tok = try fwd.prefill(e, s, next, k);
        }
        for (&runs[way]) |*o| {
            o.* = tok;
            var out: [1]u32 = undefined;
            try fwd.verify(e, s, &.{tok}, &out);
            try fwd.keep(e, s, 1, 1);
            tok = out[0];
        }
    }
    std.debug.print("fresh:   {any}\nresumed: {any}\nrepeat:  {any}\n", .{ runs[0][0..16], runs[1][0..16], runs[2][0..16] });
    try check.expect(std.mem.eql(u32, &runs[0], &runs[1]), "the resumed turn's 32 tokens equal the fresh turn's", .{});
    try check.expect(std.mem.eql(u32, &runs[0], &runs[2]), "after a repeat, the resumed turn's 32 tokens equal the fresh turn's", .{});
    check.pass("EXACT rank {d}: a turn resumed from its kept point ({d} of {d} prompt tokens kept), also after a repeat, decodes as prefilled fresh", .{ rank, k, next.len });
}

/// Each probe prompt of <dir> (vision_ref.py's tokens and grid) decoded greedily (40 tokens, one row a step) with
/// each feature set present: <name>.features.bf16 (torch on CUDA), <name>.got.bf16 (this tower), and <dir>_cpu's
/// (torch on the CPU); the ids written to <dir>/decoded.rank<r>.json for a tokenizer to read.
fn probesDecode(e: *fwd.Engine, gpu: Gpu, rank: u8, dir: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const vision = @import("flashnext_weights").vision;
    var nb: [512]u8 = undefined;
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/index.json", .{dir}), gpa, .limited(1 << 20));
    const index = try std.json.parseFromSlice(std.json.Value, gpa, index_text, .{});
    var out: std.Io.Writer.Allocating = .init(gpa);
    try out.writer.writeAll("{");
    for (index.value.array.items, 0..) |case, ci| {
        const name = case.object.get("name").?.string;
        const meta_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/{s}.json", .{ dir, name }), gpa, .limited(1 << 22));
        const meta = try std.json.parseFromSlice(std.json.Value, gpa, meta_text, .{});
        const g = meta.value.object.get("grid").?.array.items[0].array.items;
        const grid = [3]i64{ g[0].integer, g[1].integer, g[2].integer };
        const p64 = try ints(gpa, meta.value.object.get("tokens").?);
        const prompt = try gpa.alloc(u32, p64.len);
        for (prompt, p64) |*d, x| d.* = @intCast(x);
        var pos = try vision.mediaPositions(gpa, prompt, &.{grid}, .{});
        defer pos.deinit(gpa);
        try out.writer.print("{s}\"{s}\": {{", .{ if (ci > 0) ", " else "", name });
        const sources = [_][]const u8{ "gpu", "ours", "cpu" };
        for (sources, 0..) |src, si| {
            const path = if (si == 0) try std.fmt.bufPrint(&nb, "{s}/{s}.features.bf16", .{ dir, name }) else if (si == 1)
                try std.fmt.bufPrint(&nb, "{s}/{s}.got.bf16", .{ dir, name }) else try std.fmt.bufPrint(&nb, "{s}_cpu/{s}.features.bf16", .{ dir[0 .. dir.len - "_cuda".len], name });
            const fbytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28)) catch continue;
            var fdev = try cuda.DeviceBuffer.alloc(gpu.d, fbytes.len);
            defer fdev.free();
            try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(fdev.ptr, fbytes.ptr, fbytes.len), "features");
            const s = try fwd.newSeq(e);
            defer fwd.freeSeq(e, s);
            try fwd.attach(e, s, pos.rows, fdev.ptr, pos.pos, pos.delta);
            var tok = try fwd.prefill(e, s, prompt, 0);
            try out.writer.print("{s}\"{s}\": [{d}", .{ if (si > 0) ", " else "", src, tok });
            for (0..39) |_| {
                var o: [1]u32 = undefined;
                try fwd.verify(e, s, &.{tok}, &o);
                try fwd.keep(e, s, 1, 1);
                tok = o[0];
                try out.writer.print(", {d}", .{tok});
            }
            try out.writer.writeAll("]");
        }
        try out.writer.writeAll("}");
    }
    try out.writer.writeAll("}");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "{s}/decoded.rank{d}.json", .{ dir, rank }), .data = out.written() });
    std.debug.print("rank {d}: probes decoded into {s}/decoded.rank{d}.json\n", .{ rank, dir, rank });
}

/// Image prompts against the Python engine (flashnext-zig/tools/capture_vision.py): the rotary positions computed
/// here (and checked against Python's where it saved them), Python's tower features attached, prefill and an MTP
/// decode (depth 15, confidence 0.7, as the capture ran): the same tokens, rounds, drafts and acceptances.
fn visionCheck(e: *fwd.Engine, gpu: Gpu, rank: u8, dir: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const vision = @import("flashnext_weights").vision;
    var nb: [512]u8 = undefined;
    const refs_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/rank{d}/references.json", .{ dir, rank }), gpa, .limited(1 << 24));
    const refs = try std.json.parseFromSlice(std.json.Value, gpa, refs_text, .{});
    var all_same: usize = 0;
    for (refs.value.array.items) |ref| {
        const o = ref.object;
        const name = o.get("name").?.string;
        const meta_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/ref/{s}.json", .{ dir, name }), gpa, .limited(1 << 22));
        const meta = try std.json.parseFromSlice(std.json.Value, gpa, meta_text, .{});
        const g = meta.value.object.get("grid").?.array.items[0].array.items;
        const grid = [3]i64{ g[0].integer, g[1].integer, g[2].integer };
        const p64 = try ints(gpa, o.get("prompt").?);
        const prompt = try gpa.alloc(u32, p64.len);
        for (prompt, p64) |*d, x| d.* = @intCast(x);
        var pos = try vision.mediaPositions(gpa, prompt, &.{grid}, .{});
        defer pos.deinit(gpa);
        if (o.get("extra").?.integer == 0) { // Python saved this prompt's positions: compare
            const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/ref/{s}.positions.i32", .{ dir, name }), gpa, .limited(1 << 24));
            const want = try gpa.alloc(i32, pbytes.len / 4);
            @memcpy(std.mem.sliceAsBytes(want), pbytes[0 .. want.len * 4]);
            try check.expect(std.mem.eql(i32, pos.pos, want), "{s}: rotary positions equal Python's", .{name});
        }
        try check.expect(pos.delta == o.get("rope_delta").?.integer, "{s}: delta {d} (Python {d})", .{ name, pos.delta, o.get("rope_delta").?.integer });
        const fbytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}/ref/{s}.features.bf16", .{ dir, name }), gpa, .limited(1 << 28));
        var fdev = try cuda.DeviceBuffer.alloc(gpu.d, fbytes.len);
        defer fdev.free();
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(fdev.ptr, fbytes.ptr, fbytes.len), "features");
        const s = try fwd.newSeq(e);
        defer fwd.freeSeq(e, s);
        try fwd.attach(e, s, pos.rows, fdev.ptr, pos.pos, pos.delta);
        const want = try ints(gpa, o.get("tokens").?);
        const count = want.len;
        var out: std.ArrayList(u32) = .empty;
        const first = try fwd.prefill(e, s, prompt, 0);
        try out.append(gpa, first);
        var drafts: [16]u32 = undefined;
        const depth: usize = 15;
        var nd = try fwd.draftUpTo(e, s, &[1]u32{first}, @intCast(@min(depth, count - out.items.len)), 0.7, &drafts);
        var rounds: usize = 0;
        var drafted: usize = 0;
        var accepted: usize = 0;
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
            nd = try fwd.draftUpTo(e, s, sampled[0..keep], @intCast(@min(depth, count - out.items.len)), 0.7, &drafts);
        }
        var same: usize = 0;
        for (out.items[0..count], want) |a, b| {
            if (a == b) same += 1 else break;
        }
        std.debug.print("rank {d}: {s}+{d} ({d} tokens): {d} of {d} tokens equal Python's; rounds {d} drafted {d} accepted {d} (Python {d}, {d}, {d})\n", .{
            rank, name, o.get("extra").?.integer, prompt.len, same, count, rounds, drafted, accepted,
            o.get("rounds").?.integer, o.get("drafted").?.integer, o.get("accepted").?.integer });
        if (same == count) all_same += 1;
    }
    try check.expect(all_same == refs.value.array.items.len, "{d} of {d} image prompts equal Python's", .{ all_same, refs.value.array.items.len });
    check.pass("EXACT rank {d}: {d} image prompts (Python's features, positions computed here) equal the Python engine's replies", .{ rank, all_same });
}

/// Shared rounds against lone ones: three prompts of different lengths decode 16 greedy tokens each alone, then
/// together (every round one forward over the three windows): one-row windows, then four-row windows whose drafts
/// are the lone run's own later tokens (all kept). Every stream's tokens must equal its lone run's.
fn sharedCheck(e: *fwd.Engine, prompt: []const u32, rank: u8) !void {
    const lens = [_]usize{ 2695, 1500, 333 };
    const steps = 16;
    var alone: [lens.len][steps + 1]u32 = undefined;
    for (lens, 0..) |n, k| {
        const s = try fwd.newSeq(e);
        defer fwd.freeSeq(e, s);
        alone[k][0] = try fwd.prefill(e, s, prompt[0..n], 0);
        for (0..steps) |t| {
            var o: [1]u32 = undefined;
            try fwd.verify(e, s, alone[k][t .. t + 1], &o);
            try fwd.keep(e, s, 1, 1);
            alone[k][t + 1] = o[0];
        }
    }
    for ([_]u32{ 1, 4 }) |w| {
        var seqs: [lens.len]*fwd.Seq = undefined;
        var got: [lens.len][steps + 1]u32 = undefined;
        for (lens, 0..) |n, k| {
            seqs[k] = try fwd.newSeq(e);
            got[k][0] = try fwd.prefill(e, seqs[k], prompt[0..n], 0);
        }
        defer for (seqs) |s| fwd.freeSeq(e, s);
        var t: usize = 0;
        while (t < steps) : (t += w) {
            const rows = @min(w, steps - t);
            var parts: [lens.len]fwd.Part = undefined;
            for (&parts, 0..) |*p, k| p.* = .{ .s = seqs[k], .ids = alone[k][t .. t + rows] }; // pending + the lone run's next tokens
            var out: [lens.len * 4]u32 = undefined;
            try fwd.verifyShared(e, &parts, out[0 .. lens.len * rows]);
            for (0..lens.len) |k| {
                for (0..rows) |r| got[k][t + 1 + r] = out[k * rows + r];
                try fwd.keep(e, seqs[k], rows, rows);
            }
        }
        var same: usize = 0;
        for (0..lens.len) |k| for (0..steps + 1) |j| {
            if (got[k][j] == alone[k][j]) same += 1;
        };
        std.debug.print("rank {d}: shared rounds of {d}-row windows over {d} streams: {d} of {d} tokens equal the lone runs\n", .{ rank, w, lens.len, same, lens.len * (steps + 1) });
        try check.expect(same == lens.len * (steps + 1), "shared {d}-row rounds: {d} of {d} tokens equal", .{ w, same, lens.len * (steps + 1) });
    }
    check.pass("EXACT rank {d}: shared rounds over {d} streams equal each stream alone", .{ rank, lens.len });
    try draftsCheck(e, prompt, rank);
    try decodeCheck(e, prompt, rank);
}

/// MTP decoding together against alone: each stream drafts (depth 6, confidence 0.7), verifies, keeps the agreeing
/// prefix (the DeltaNet replays) and drafts again; alone (verify, keep, draftUpTo) and together (verifyShared,
/// keepMany, draftBatch) every stream's tokens are the same.
fn decodeCheck(e: *fwd.Engine, prompt: []const u32, rank: u8) !void {
    const lens = [_]usize{ 2695, 1500, 333, 97, 1200 };
    const N = lens.len;
    const want = 48;
    const depth = 6;
    var alone: [N][want + 16]u32 = undefined;
    for (lens, 0..) |n, k| {
        const s = try fwd.newSeq(e);
        defer fwd.freeSeq(e, s);
        var got: usize = 1;
        alone[k][0] = try fwd.prefill(e, s, prompt[0..n], 0);
        var drafts: [depth]u32 = undefined;
        var nd = try fwd.draftUpTo(e, s, alone[k][0..1], depth, 0.7, &drafts);
        while (got < want) {
            var win: [depth + 1]u32 = undefined;
            win[0] = alone[k][got - 1];
            @memcpy(win[1..][0..nd], drafts[0..nd]);
            var o: [depth + 1]u32 = undefined;
            try fwd.verify(e, s, win[0 .. nd + 1], o[0 .. nd + 1]);
            var kp: usize = 1;
            while (kp < nd + 1 and o[kp - 1] == win[kp]) kp += 1;
            try fwd.keep(e, s, @intCast(nd + 1), @intCast(kp));
            @memcpy(alone[k][got..][0..kp], o[0..kp]);
            got += kp;
            nd = try fwd.draftUpTo(e, s, o[0..kp], depth, 0.7, &drafts);
        }
    }
    var seqs: [N]*fwd.Seq = undefined;
    var together: [N][want + 16]u32 = undefined;
    var got: [N]usize = @splat(1);
    var drafts: [N][depth]u32 = undefined;
    var probs: [N][depth]f64 = undefined;
    var follows: [N][depth + 1]u32 = undefined;
    var reqs: [N]fwd.DraftReq = undefined;
    for (lens, 0..) |n, k| {
        seqs[k] = try fwd.newSeq(e);
        together[k][0] = try fwd.prefill(e, seqs[k], prompt[0..n], 0);
        follows[k][0] = together[k][0];
        reqs[k] = .{ .s = seqs[k], .follow = follows[k][0..1], .depth = depth, .out = &drafts[k], .probs = &probs[k] };
    }
    defer for (seqs) |s| fwd.freeSeq(e, s);
    try fwd.draftBatch(e, &reqs, 0.7);
    var rounds: usize = 0;
    var partial: usize = 0;
    while (true) {
        var parts: [N]fwd.Part = undefined;
        var wins: [N][depth + 1]u32 = undefined;
        var live: [N]usize = undefined;
        var nl: usize = 0;
        var R: usize = 0;
        for (0..N) |k| if (got[k] < want) {
            wins[k][0] = together[k][got[k] - 1];
            @memcpy(wins[k][1..][0..reqs[k].got], drafts[k][0..reqs[k].got]);
            parts[nl] = .{ .s = seqs[k], .ids = wins[k][0 .. 1 + reqs[k].got] };
            live[nl] = k;
            nl += 1;
            R += 1 + reqs[k].got;
        };
        if (nl == 0) break;
        var out: [N * (depth + 1)]u32 = undefined;
        if (nl == 1) try fwd.verify(e, parts[0].s, parts[0].ids, out[0..R]) else try fwd.verifyShared(e, parts[0..nl], out[0..R]);
        var keeps: [N]fwd.Keep = undefined;
        var dreqs: [N]fwd.DraftReq = undefined;
        var r0: usize = 0;
        for (parts[0..nl], live[0..nl], 0..) |p, k, j| {
            const w = p.ids.len;
            const o = out[r0..][0..w];
            var kp: usize = 1;
            while (kp < w and o[kp - 1] == p.ids[kp]) kp += 1;
            if (kp < w) partial += 1;
            @memcpy(together[k][got[k]..][0..kp], o[0..kp]);
            @memcpy(follows[k][0..kp], o[0..kp]);
            got[k] += kp;
            keeps[j] = .{ .s = seqs[k], .rows = @intCast(w), .kept = @intCast(kp) };
            reqs[k] = .{ .s = seqs[k], .follow = follows[k][0..kp], .depth = depth, .out = &drafts[k], .probs = &probs[k] };
            dreqs[j] = reqs[k];
            r0 += w;
        }
        try fwd.keepMany(e, keeps[0..nl]);
        if (nl == 1) {
            const k = live[0];
            reqs[k].got = try fwd.draftUpTo(e, seqs[k], reqs[k].follow, depth, 0.7, &drafts[k]);
        } else {
            try fwd.draftBatch(e, dreqs[0..nl], 0.7);
            for (live[0..nl], 0..) |k, j| reqs[k].got = dreqs[j].got;
        }
        rounds += 1;
    }
    var same: usize = 0;
    for (0..N) |k| {
        if (std.mem.eql(u32, alone[k][0..want], together[k][0..want])) same += 1;
    }
    std.debug.print("rank {d}: MTP decoding over {d} streams together: {d} of {d} streams equal alone ({d} rounds, {d} windows cut short)\n", .{ rank, N, same, N, rounds, partial });
    try check.expect(same == N, "decoding together: {d} of {d} streams equal", .{ same, N });
    check.pass("EXACT rank {d}: MTP decoding over {d} streams together equals each stream alone", .{ rank, N });
}

/// Batched drafts against lone ones: each stream drafts after its prompt, verifies the drafts, keeps the agreeing
/// prefix and drafts again; alone (draftUpTo, verify) and together (draftBatch, verifyShared): same drafts.
fn draftsCheck(e: *fwd.Engine, prompt: []const u32, rank: u8) !void {
    const lens = [_]usize{ 2695, 1500, 333 };
    const depth = 6;
    var alone: [lens.len][2][depth]u32 = undefined;
    var alone_n: [lens.len][2]usize = undefined;
    for (lens, 0..) |n, k| {
        const s = try fwd.newSeq(e);
        defer fwd.freeSeq(e, s);
        const first = try fwd.prefill(e, s, prompt[0..n], 0);
        alone_n[k][0] = try fwd.draftUpTo(e, s, &.{first}, depth, 0.7, &alone[k][0]);
        var win: [depth + 1]u32 = undefined;
        win[0] = first;
        @memcpy(win[1..][0..alone_n[k][0]], alone[k][0][0..alone_n[k][0]]);
        const R = 1 + alone_n[k][0];
        var o: [depth + 1]u32 = undefined;
        try fwd.verify(e, s, win[0..R], o[0..R]);
        var kept: usize = 1;
        while (kept < R and o[kept - 1] == win[kept]) kept += 1;
        try fwd.keep(e, s, @intCast(R), @intCast(kept));
        alone_n[k][1] = try fwd.draftUpTo(e, s, o[0..kept], depth, 0.7, &alone[k][1]);
    }
    var seqs: [lens.len]*fwd.Seq = undefined;
    var firsts: [lens.len]u32 = undefined;
    for (lens, 0..) |n, k| {
        seqs[k] = try fwd.newSeq(e);
        firsts[k] = try fwd.prefill(e, seqs[k], prompt[0..n], 0);
    }
    defer for (seqs) |s| fwd.freeSeq(e, s);
    var got: [lens.len][2][depth]u32 = undefined;
    var probs: [lens.len][depth]f64 = undefined;
    var reqs: [lens.len]fwd.DraftReq = undefined;
    for (&reqs, 0..) |*r, k| r.* = .{ .s = seqs[k], .follow = firsts[k .. k + 1], .depth = depth, .out = &got[k][0], .probs = &probs[k] };
    try fwd.draftBatch(e, &reqs, 0.7);
    var same: usize = 0;
    var total: usize = 0;
    var wins: [lens.len][depth + 1]u32 = undefined;
    var parts: [lens.len]fwd.Part = undefined;
    for (0..lens.len) |k| {
        total += 1;
        if (reqs[k].got == alone_n[k][0] and std.mem.eql(u32, got[k][0][0..reqs[k].got], alone[k][0][0..alone_n[k][0]])) same += 1;
        wins[k][0] = firsts[k];
        @memcpy(wins[k][1..][0..reqs[k].got], got[k][0][0..reqs[k].got]);
        parts[k] = .{ .s = seqs[k], .ids = wins[k][0 .. 1 + reqs[k].got] };
    }
    var out: [lens.len * (depth + 1)]u32 = undefined;
    var nrows: usize = 0;
    for (parts) |p| nrows += p.ids.len;
    try fwd.verifyShared(e, &parts, out[0..nrows]);
    var follows: [lens.len][depth + 1]u32 = undefined;
    var r0: usize = 0;
    for (parts, 0..) |p, k| {
        const R = p.ids.len;
        const o = out[r0..][0..R];
        var kept: usize = 1;
        while (kept < R and o[kept - 1] == p.ids[kept]) kept += 1;
        @memcpy(follows[k][0..kept], o[0..kept]);
        reqs[k] = .{ .s = seqs[k], .follow = follows[k][0..kept], .depth = depth, .out = &got[k][1], .probs = &probs[k] };
        r0 += R;
    }
    // a shared round drafts before it keeps: the drafts read the verify's rows, then each keeps its prefix
    for (parts, 0..) |p, k| try fwd.keep(e, seqs[k], @intCast(p.ids.len), @intCast(reqs[k].follow.len));
    try fwd.draftBatch(e, &reqs, 0.7);
    for (0..lens.len) |k| {
        total += 1;
        if (reqs[k].got == alone_n[k][1] and std.mem.eql(u32, got[k][1][0..reqs[k].got], alone[k][1][0..alone_n[k][1]])) same += 1;
    }
    std.debug.print("rank {d}: batched drafts over {d} streams: {d} of {d} draft runs equal the lone runs\n", .{ rank, lens.len, same, total });
    try check.expect(same == total, "batched drafts: {d} of {d} equal", .{ same, total });
    check.pass("EXACT rank {d}: batched drafts over {d} streams equal each stream's own", .{ rank, lens.len });
}

/// Shared rounds at load: ``streams`` streams (prompts of different lengths from the reference prompt) decoding
/// together as the Python engine's rounds do (every stream's chain to ``depth`` drafts, confidence 0.7, one verify
/// over every window, keeps, one batched draft), timed by phase. bench=<streams>x<depth>[x<rounds>]
fn benchShared(e: *fwd.Engine, prompt: []const u32, rank: u8, spec: []const u8) !void {
    const io = e.io;
    var it = std.mem.splitScalar(u8, spec, 'x');
    const n = try std.fmt.parseInt(usize, it.next().?, 10);
    const depth = try std.fmt.parseInt(u32, it.next().?, 10);
    const rounds = if (it.next()) |r| try std.fmt.parseInt(usize, r, 10) else 40;
    const M = 16;
    if (n > M or depth > 15) return error.BadBench;
    var seqs: [M]*fwd.Seq = undefined;
    var pend: [M]u32 = undefined;
    var t = std.Io.Timestamp.now(io, .awake);
    for (0..n) |k| {
        seqs[k] = try fwd.newSeq(e);
        const len = @min(prompt.len, 200 + 131 * k);
        pend[k] = try fwd.prefill(e, seqs[k], prompt[0..len], 0);
    }
    defer for (seqs[0..n]) |s| fwd.freeSeq(e, s);
    const ms = struct {
        fn since(io_: std.Io, t0: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io_, .awake)).toNanoseconds())) / 1e6;
        }
    }.since;
    std.debug.print("rank {d}: bench {d} prompts prefilled in {d:.0} ms\n", .{ rank, n, ms(io, t) });
    var drafts: [M][16]u32 = undefined;
    var probs: [M][16]f64 = undefined;
    var follows: [M][16]u32 = undefined;
    var reqs: [M]fwd.DraftReq = undefined;
    for (0..n) |k| {
        follows[k][0] = pend[k];
        reqs[k] = .{ .s = seqs[k], .follow = follows[k][0..1], .depth = depth, .out = drafts[k][0..depth], .probs = probs[k][0..depth] };
    }
    if (depth > 0) try fwd.draftBatch(e, reqs[0..n], 0.7);
    var t_verify: f64 = 0;
    var t_keep: f64 = 0;
    var t_draft: f64 = 0;
    var rows_total: usize = 0;
    var tokens: usize = 0;
    const tw = std.Io.Timestamp.now(io, .awake);
    for (0..rounds) |_| {
        var wins: [M][16]u32 = undefined;
        var parts: [M]fwd.Part = undefined;
        var R: usize = 0;
        for (0..n) |k| {
            const got = if (depth > 0) reqs[k].got else 0;
            wins[k][0] = pend[k];
            @memcpy(wins[k][1..][0..got], drafts[k][0..got]);
            parts[k] = .{ .s = seqs[k], .ids = wins[k][0 .. 1 + got] };
            R += 1 + got;
        }
        var out: [M * 16]u32 = undefined;
        t = std.Io.Timestamp.now(io, .awake);
        if (n == 1) try fwd.verify(e, seqs[0], parts[0].ids, out[0..R]) else try fwd.verifyShared(e, parts[0..n], out[0..R]);
        t_verify += ms(io, t);
        rows_total += R;
        var r0: usize = 0;
        var kept: [M]usize = undefined;
        for (parts[0..n], 0..) |p, k| {
            const w = p.ids.len;
            const o = out[r0..][0..w];
            var kp: usize = 1;
            while (kp < w and o[kp - 1] == p.ids[kp]) kp += 1;
            kept[k] = kp;
            @memcpy(follows[k][0..kp], o[0..kp]);
            pend[k] = o[kp - 1];
            tokens += kp;
            r0 += w;
        }
        t = std.Io.Timestamp.now(io, .awake);
        var keeps: [M]fwd.Keep = undefined;
        for (parts[0..n], 0..) |p, k| keeps[k] = .{ .s = seqs[k], .rows = @intCast(p.ids.len), .kept = @intCast(kept[k]) };
        try fwd.keepMany(e, keeps[0..n]);
        try e.k.stream.synchronize();
        t_keep += ms(io, t);
        if (depth > 0) {
            for (0..n) |k| reqs[k] = .{ .s = seqs[k], .follow = follows[k][0..kept[k]], .depth = depth, .out = drafts[k][0..depth], .probs = probs[k][0..depth] };
            t = std.Io.Timestamp.now(io, .awake);
            if (n == 1) {
                reqs[0].got = try fwd.draftUpTo(e, seqs[0], reqs[0].follow, depth, 0.7, reqs[0].out);
            } else try fwd.draftBatch(e, reqs[0..n], 0.7);
            t_draft += ms(io, t);
        }
    }
    const wall = ms(io, tw);
    const fr: f64 = @floatFromInt(rounds);
    std.debug.print("rank {d}: bench {d} streams depth {d}: {d} rounds, {d:.1} rows and {d:.1} tokens a round; {d:.1} ms a round " ++
        "(verify {d:.1}, keep {d:.1}, draft {d:.1}); {d:.0} tok/s\n", .{ rank, n, depth, rounds, @as(f64, @floatFromInt(rows_total)) / fr,
        @as(f64, @floatFromInt(tokens)) / fr, wall / fr, t_verify / fr, t_keep / fr, t_draft / fr, @as(f64, @floatFromInt(tokens)) / (wall / 1000) });
    std.debug.print("rank {d}: shared attention layers so far: {d} in one launch a kernel, {d} stream by stream (no variant)\n", .{ rank, e.multi_layers[0], e.multi_layers[1] });
}

/// What a request's admission costs: a short prompt's prefill and its first drafts, on a new sequence and on a
/// reused one (reset, its graphs kept), and the prefill alone at several short lengths.
fn benchAdmit(e: *fwd.Engine, prompt: []const u32, rank: u8) !void {
    const io = e.io;
    const ms = struct {
        fn since(io_: std.Io, t0: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io_, .awake)).toNanoseconds())) / 1e6;
        }
    }.since;
    var drafts: [6]u32 = undefined;
    for ([_]usize{ 40, 120, 400 }) |n| {
        var t = std.Io.Timestamp.now(io, .awake);
        const s = try fwd.newSeq(e);
        try e.k.stream.synchronize();
        const t_new = ms(io, t);
        t = std.Io.Timestamp.now(io, .awake);
        const first = try fwd.prefill(e, s, prompt[0..n], 0);
        const t_pre = ms(io, t);
        t = std.Io.Timestamp.now(io, .awake);
        _ = try fwd.draftUpTo(e, s, &.{first}, 6, 0.7, &drafts);
        const t_draft = ms(io, t);
        t = std.Io.Timestamp.now(io, .awake);
        try fwd.resetSeq(e, s);
        try e.k.stream.synchronize();
        const t_reset = ms(io, t);
        t = std.Io.Timestamp.now(io, .awake);
        const first2 = try fwd.prefill(e, s, prompt[0..n], 0);
        const t_pre2 = ms(io, t);
        t = std.Io.Timestamp.now(io, .awake);
        _ = try fwd.draftUpTo(e, s, &.{first2}, 6, 0.7, &drafts);
        const t_draft2 = ms(io, t);
        fwd.freeSeq(e, s);
        std.debug.print("rank {d}: admit {d} tokens: new sequence {d:.1} ms, prefill {d:.1} ms, first drafts {d:.1} ms; reset {d:.1} ms, prefill {d:.1} ms, first drafts {d:.1} ms\n", .{ rank, n, t_new, t_pre, t_draft, t_reset, t_pre2, t_draft2 });
    }
}

/// Several short prompts in one pass (prefillMany) against each prompt alone: first tokens and 24 greedy tokens after
/// them, each prompt's kept point taken (one token before its end) and a next turn resumed from it; and the pass's
/// time against the prompts one by one.
fn prefillsCheck(e: *fwd.Engine, prompt: []const u32, rank: u8) !void {
    const io = e.io;
    const lens = [_]usize{ 40, 97, 333, 7, 200, 61, 120, 2 };
    const N = lens.len;
    const steps = 24;
    const ms = struct {
        fn since(io_: std.Io, t0: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io_, .awake)).toNanoseconds())) / 1e6;
        }
    }.since;
    var alone: [N][steps + 1]u32 = undefined;
    var t = std.Io.Timestamp.now(io, .awake);
    var seqs: [N]*fwd.Seq = undefined;
    for (lens, 0..) |n, k| {
        seqs[k] = try fwd.newSeq(e);
        seqs[k].cut_at = @intCast(@max(1, n - 1));
        alone[k][0] = try fwd.prefill(e, seqs[k], prompt[k * 50 ..][0..n], 0);
    }
    const t_alone = ms(io, t);
    for (lens, 0..) |_, k| {
        for (0..steps) |j| {
            var o: [1]u32 = undefined;
            try fwd.verify(e, seqs[k], alone[k][j .. j + 1], &o);
            try fwd.keep(e, seqs[k], 1, 1);
            alone[k][j + 1] = o[0];
        }
        try fwd.resetSeq(e, seqs[k]);
    }
    var ps: [N]fwd.Prompt = undefined;
    for (lens, 0..) |n, k| {
        seqs[k].cut_at = @intCast(@max(1, n - 1));
        ps[k] = .{ .s = seqs[k], .ids = prompt[k * 50 ..][0..n] };
    }
    var firsts: [N]u32 = undefined;
    t = std.Io.Timestamp.now(io, .awake);
    try fwd.prefillMany(e, &ps, &firsts);
    const t_many = ms(io, t);
    var same: usize = 0;
    var resumed: usize = 0;
    for (lens, 0..) |n, k| {
        var got: [steps + 1]u32 = undefined;
        got[0] = firsts[k];
        for (0..steps) |j| {
            var o: [1]u32 = undefined;
            try fwd.verify(e, seqs[k], got[j .. j + 1], &o);
            try fwd.keep(e, seqs[k], 1, 1);
            got[j + 1] = o[0];
        }
        if (std.mem.eql(u32, &got, &alone[k])) same += 1 else std.debug.print("rank {d}: prompt {d} ({d} tokens): together {any} alone {any}\n", .{ rank, k, n, got[0..8], alone[k][0..8] });
        // the kept point: a next turn (this prompt and 30 more tokens) resumed there equals it prefilled fresh
        if (seqs[k].snap.at > 0) {
            const at: usize = @intCast(seqs[k].snap.at);
            const longer = prompt[k * 50 ..][0 .. n + 30];
            const r1 = try fwd.prefill(e, seqs[k], longer, at);
            const f = try fwd.newSeq(e);
            defer fwd.freeSeq(e, f);
            const r2 = try fwd.prefill(e, f, longer, 0);
            if (r1 == r2) resumed += 1;
        }
    }
    for (seqs) |s| fwd.freeSeq(e, s);
    std.debug.print("rank {d}: {d} prompts in one pass: {d:.1} ms, one by one {d:.1} ms; {d} of {d} decode as alone; {d} kept points resume as fresh\n", .{ rank, N, t_many, t_alone, same, N, resumed });
    try check.expect(same == N, "prompts together: {d} of {d} as alone", .{ same, N });
    check.pass("rank {d}: {d} prompts in one pass decode as each alone", .{ rank, N });
}
