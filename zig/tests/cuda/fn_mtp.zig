//! Flash Next port, step 2: MTP decode from Zig. The prompt pass with the MTP head absorbing the prompt, then rounds
//! as the Python engine's mtp_decode runs them: verify the pending token and its drafts in one window, keep rows up
//! to the first mismatch, commit them, let the MTP head absorb the kept rows and chain up to six drafts, a chain
//! ending before a later draft the head gives under 70%. Programs come from capture_mtp.py traces (gen_mtp.py);
//! verify windows and MTP steps replay as CUDA graphs. Drafts never change the reply: it must equal Python's.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fn_ext = @import("fn_ext.zig");
const fn_ngram = @import("fn_ngram.zig");
const R = @import("fn_run.zig");
const Gpu = check.Gpu;

const DEPTH = 6;
const CONFIDENCE = 0.7;
const ROW = 10240 * 2;          // one residual-stream row (bf16), the MTP head's input

/// Upload sources by sequence number (the programs' ``seq``), at most eight a program.
const Inputs = struct {
    a: [8]?[]const u8 = @splat(null),
    pub fn get(self: *const Inputs, seq: usize) ?[]const u8 {
        return if (seq < self.a.len) self.a[seq] else null;
    }
};

const Prog = struct { c: R.Compiled, g: R.DecodeGraph = .{}, logits: u64 = 0, rows: usize = 0, cols: usize = 0 };

const Times = struct { verify: f64 = 0, commit: f64 = 0, mtp: f64 = 0, pick: f64 = 0, mtp_steps: usize = 0, stage: f64 = 0, gpu: f64 = 0, greedy: f64 = 0 };

fn since(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
}

const VL = 16;

fn bf(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn lanes(row: []const u16, i: usize) @Vector(VL, f32) {
    const v: @Vector(VL, u16) = row[i..][0..VL].*;
    return @bitCast(@as(@Vector(VL, u32), v) << @splat(16));
}

/// The first maximum of a bf16 row (argmax's choice) and its value.
fn rowMax(row: []const u16) struct { i: usize, v: f32 } {
    var m: @Vector(VL, f32) = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + VL <= row.len) : (i += VL) m = @max(m, lanes(row, i));
    var best = @reduce(.Max, m);
    while (i < row.len) : (i += 1) best = @max(best, bf(row[i]));
    const bv: @Vector(VL, f32) = @splat(best);
    i = 0;
    while (i + VL <= row.len) : (i += VL) {
        if (@reduce(.Or, lanes(row, i) == bv)) break;
    }
    while (i < row.len) : (i += 1) if (bf(row[i]) == best) return .{ .i = i, .v = best };
    unreachable;
}

/// log(sum(exp(row))) given the row's maximum, in f32 lanes (torch's logsumexp precision).
fn rowLse(row: []const u16, max: f32) f64 {
    var acc: @Vector(VL, f32) = @splat(0);
    const mv: @Vector(VL, f32) = @splat(max);
    var i: usize = 0;
    while (i + VL <= row.len) : (i += VL) acc += @exp(lanes(row, i) - mv);
    var sum: f64 = @reduce(.Add, acc);
    while (i < row.len) : (i += 1) sum += @exp(bf(row[i]) - max);
    return @as(f64, max) + @log(sum);
}

const Ctx = struct {
    run: *R.Run,
    stream: cuda.Stream,
    nccl: *cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    selects: []R.Select,
    graphs: bool,
    xbuf: cuda.DeviceBuffer,
    host: cuda.HostBuffer,                // logits rows read back here (pinned, reused)
    device_sampling: bool = false,        // candidates from fn_rows_top inside each program, exchanged there too
    gathered: u64 = 0,                    // [2 ranks][rows][4] words: id, max, log-sum-exp, 0

    fn launch(self: *Ctx, p: *Prog, keys: i64, in: *const Inputs) !void {
        _ = try R.refresh(self.run, &p.c, self.selects, keys);
        if (self.graphs) return p.g.launch(self.run, &p.c, self.stream, self.nccl, self.comm, in);
        try R.runCompiled(self.run, p.c.ops.items, self.stream, self.nccl, self.comm, in, -1);
    }
};

fn markAddr(run: *R.Run, prog: std.json.ObjectMap, name: []const u8) ?struct { addr: u64, rows: usize, cols: usize } {
    const m = prog.get("marks") orelse return null;
    const v = m.object.get(name) orelse return null;
    const shape = v.object.get("shape").?.array.items;
    return .{ .addr = run.addr(v), .rows = @intCast(shape[0].integer), .cols = @intCast(shape[1].integer) };
}

/// The exchanged candidates of the last program (both ranks, ``rows`` rows), read back once.
fn readCands(cx: *Ctx, rows: usize) ![]const i32 {
    const d = cx.run.gpu.d;
    const n = 2 * rows * 4;
    const w: []i32 = @alignCast(std.mem.bytesAsSlice(i32, cx.host.bytes[0 .. n * 4]));
    try d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(w.ptr), cx.gathered, n * 4, cx.stream.handle), "candidates");
    try cx.stream.synchronize();
    return w;
}

/// Greedy over both ranks' vocabulary shards for ``rows`` rows: each rank's first maximum per row (global id, value),
/// exchanged; the larger value wins, a tie the lower id.
fn greedyRows(cx: *Ctx, logits: u64, rows: usize, cols: usize, vocab_offset: i64, out: []i64) !void {
    if (cx.device_sampling) {
        const g = try readCands(cx, rows);
        for (0..rows) |r| {
            const a = g[4 * r ..][0..4];
            const b = g[4 * (rows + r) ..][0..4];
            const v0: f32 = @bitCast(a[1]);
            const v1: f32 = @bitCast(b[1]);
            out[r] = if (v1 > v0 or (v1 == v0 and b[0] < a[0])) b[0] else a[0];
        }
        return;
    }
    const d = cx.run.gpu.d;
    const buf: []u16 = @alignCast(std.mem.bytesAsSlice(u16, cx.host.bytes[0 .. rows * cols * 2]));
    try d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(buf.ptr), logits, rows * cols * 2, cx.stream.handle), "logits rows");
    try cx.stream.synchronize();
    var mine: [2 * 16]i32 = undefined;
    for (0..rows) |r| {
        const m = rowMax(buf[r * cols ..][0..cols]);
        mine[2 * r] = @intCast(@as(i64, @intCast(m.i)) + vocab_offset);
        mine[2 * r + 1] = @bitCast(m.v);
    }
    const n = 2 * rows;
    try cx.xbuf.upload(0, std.mem.sliceAsBytes(mine[0..n]));
    try cx.nccl.check(cx.nccl.api.ncclAllGather(cx.xbuf.ptr, cx.xbuf.ptr + 256, n, .i32, cx.comm, cx.stream.handle), "greedy exchange");
    try cx.stream.synchronize();
    var all: [4 * 16]i32 = undefined;
    try cx.xbuf.download(256, std.mem.sliceAsBytes(all[0 .. 2 * n]));
    for (0..rows) |r| {
        const id0 = all[2 * r];
        const id1 = all[n + 2 * r];
        const v0: f32 = @bitCast(all[2 * r + 1]);
        const v1: f32 = @bitCast(all[n + 2 * r + 1]);
        out[r] = if (v1 > v0 or (v1 == v0 and id1 < id0)) id1 else id0;
    }
}

/// The MTP head's draft from its logits row over this rank's share of the draft vocabulary: the best (value, id)
/// across ranks and its temperature-1 probability from both shards' log-sum-exps.
fn draftPick(cx: *Ctx, logits: u64, cols: usize, ids: []const i32) !struct { tok: i64, p: f64 } {
    if (cx.device_sampling) {
        const g = try readCands(cx, 1);
        const v0: f32 = @bitCast(g[1]);
        const v1: f32 = @bitCast(g[5]);
        const l0: f64 = @as(f32, @bitCast(g[2]));
        const l1: f64 = @as(f32, @bitCast(g[6]));
        const first = !(v1 > v0 or (v1 == v0 and g[4] < g[0]));
        const top = @max(l0, l1);
        const total = top + @log(@exp(l0 - top) + @exp(l1 - top));
        const v: f64 = if (first) v0 else v1;
        return .{ .tok = if (first) g[0] else g[4], .p = @exp(v - total) };
    }
    const d = cx.run.gpu.d;
    const row: []u16 = @alignCast(std.mem.bytesAsSlice(u16, cx.host.bytes[0 .. cols * 2]));
    try d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(row.ptr), logits, cols * 2, cx.stream.handle), "draft row");
    try cx.stream.synchronize();
    const m = rowMax(row);
    const best = m.i;
    const bv = m.v;
    const lse = rowLse(row, bv);
    var mine: [4]i32 = undefined;
    mine[0] = ids[best];
    mine[1] = @bitCast(bv);
    const lb: [2]i32 = @bitCast(lse);
    mine[2] = lb[0];
    mine[3] = lb[1];
    try cx.xbuf.upload(0, std.mem.asBytes(&mine));
    try cx.nccl.check(cx.nccl.api.ncclAllGather(cx.xbuf.ptr, cx.xbuf.ptr + 256, 4, .i32, cx.comm, cx.stream.handle), "draft exchange");
    try cx.stream.synchronize();
    var all: [8]i32 = undefined;
    try cx.xbuf.download(256, std.mem.asBytes(&all));
    const v0: f32 = @bitCast(all[1]);
    const v1: f32 = @bitCast(all[5]);
    const l0: f64 = @bitCast([2]i32{ all[2], all[3] });
    const l1: f64 = @bitCast([2]i32{ all[6], all[7] });
    const first = !(v1 > v0 or (v1 == v0 and all[4] < all[0]));
    const top = @max(l0, l1);
    const total = top + @log(@exp(l0 - top) + @exp(l1 - top));
    const v: f64 = if (first) v0 else v1;
    return .{ .tok = if (first) all[0] else all[4], .p = @exp(v - total) };
}

pub fn generate(gpu: Gpu, args: []const [:0]const u8, graphs: bool, device_sampling: bool) !void {
    // fn-mtp <program.json> <weights.bin> <pre_prefill.bin> <ngram.json> <reference.json> <draft_ids.bin> <rank> <uid file>
    const gpa = gpu.gpa;
    const io = gpu.io;
    const rank: c_int = try std.fmt.parseInt(c_int, args[6], 10);
    var run: R.Run = .{ .gpu = gpu, .kernels = .init(gpa) };
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[0], gpa, .limited(1 << 30));
    run.prog = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    const progs = run.prog.value.object.get("programs").?.object;
    // temporaries: one arena, each index sized for the largest program using it
    var sizes: std.ArrayList(usize) = .empty;
    var pit = progs.iterator();
    while (pit.next()) |e| for (e.value_ptr.object.get("temps").?.array.items, 0..) |t, i| {
        if (i >= sizes.items.len) try sizes.append(gpa, 0);
        sizes.items[i] = @max(sizes.items[i], @as(usize, @intCast(t.integer)));
    };
    var off: usize = 0;
    for (sizes.items) |sz| {
        try run.temp_off.append(gpa, off);
        off += std.mem.alignForward(usize, @max(sz, 1), 512);
    }
    run.temps = try cuda.DeviceBuffer.alloc(gpu.d, off);
    run.scratch = try cuda.DeviceBuffer.alloc(gpu.d, 256 << 20);
    var k = try fn_ext.Kernels.init(gpu);
    defer k.deinit();
    var nccl = try cuda.nccl.Library.open();
    defer nccl.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try nccl.check(nccl.api.ncclGetUniqueId(&uid), "ncclGetUniqueId");
        const tmp = try std.fmt.allocPrint(gpa, "{s}.tmp", .{args[7]});
        const f = try std.Io.Dir.cwd().createFile(io, tmp, .{});
        try f.writePositionalAll(io, &uid.internal, 0);
        f.close(io);
        try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), args[7], io);
    } else {
        const uf = try R.openRead(gpu, args[7]);
        _ = try uf.readPositionalAll(io, &uid.internal, 0);
        uf.close(io);
    }
    var ng = try fn_ngram.NGram.open(gpa, io, args[3]);
    const ref = (try std.json.parseFromSlice(std.json.Value, gpa, try std.Io.Dir.cwd().readFileAlloc(io, args[4], gpa, .limited(1 << 26)), .{})).value.object;
    const prompt = try fn_ngram.NGram.ints(gpa, ref.get("prompt").?);
    const want = try fn_ngram.NGram.ints(gpa, ref.get("tokens").?);
    const draw = try std.Io.Dir.cwd().readFileAlloc(io, args[5], gpa, .limited(1 << 24));
    const draft_ids: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, draw));
    try R.loadStorages(&run, args[1], args[2]);
    var comm: cuda.nccl.Comm = null;
    try nccl.check(nccl.api.ncclCommInitRank(&comm, 2, uid, rank), "ncclCommInitRank");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    {   // NCCL connects (and registers its buffers) before the tables take the memory
        const warm = try cuda.DeviceBuffer.alloc(gpu.d, 4096);
        try nccl.check(nccl.api.ncclAllGather(warm.ptr, warm.ptr + 2048, 256, .i32, comm, stream.handle), "warm gather");
        try stream.synchronize();
    }
    const tn = std.Io.Timestamp.now(io, .awake);
    try ng.map(true);
    std.debug.print("n-gram tables mapped and paged in: {d:.1} s\n", .{since(io, tn) / 1000});

    // compile every program once
    const tc = std.Io.Timestamp.now(io, .awake);
    var consts: R.Consts = .{};
    var table = std.StringHashMap(*Prog).init(gpa);
    var all: std.ArrayList(*R.Compiled) = .empty;
    pit = progs.iterator();
    while (pit.next()) |e| {
        const p = try gpa.create(Prog);
        p.* = .{ .c = try R.compile(&run, e.value_ptr.object.get("forward").?.array.items, &k, stream, &consts) };
        if (markAddr(&run, e.value_ptr.object, "logits")) |m| { p.logits = m.addr; p.rows = m.rows; p.cols = m.cols; }
        try table.put(e.key_ptr.*, p);
        try all.append(gpa, &p.c);
    }
    try R.finishConsts(&run, &consts, all.items);
    var cx: Ctx = .{ .run = &run, .stream = stream, .nccl = &nccl, .comm = comm, .selects = try R.selectTable(&run),
                     .graphs = graphs, .xbuf = try cuda.DeviceBuffer.alloc(gpu.d, 4096),
                     .host = try cuda.HostBuffer.alloc(gpu.d, 8 * 131072 * 2) };
    if (device_sampling) {
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, "flashnext-zig/cubins/fn_sample.cubin", gpa, .limited(1 << 24));
        const img = try gpa.alignedAlloc(u8, .@"16", raw.len);
        @memcpy(img, raw);
        const mod = try cuda.Module.load(gpu.d, img);
        const top = try mod.function("fn_rows_top");
        const cand = try cuda.DeviceBuffer.alloc(gpu.d, 16 * 16);
        const gath = try cuda.DeviceBuffer.alloc(gpu.d, 2 * 16 * 16);
        const idmap = try cuda.DeviceBuffer.alloc(gpu.d, draw.len);
        try idmap.upload(0, draw);
        cx.device_sampling = true;
        cx.gathered = gath.ptr;
        var tit = table.iterator();
        while (tit.next()) |e| {
            const nm = e.key_ptr.*;
            const p = e.value_ptr.*;
            const head = std.mem.startsWith(u8, nm, "fwd_");
            if (!head and !std.mem.startsWith(u8, nm, "mtp_")) continue;
            var a: cuda.Args = .{};
            a.add(p.logits); a.add(@as(i32, @intCast(p.cols))); a.add(@as(i32, @intCast(p.cols)));
            a.add(@as(u64, if (head) 0 else idmap.ptr)); a.add(@as(i32, @intCast(if (head) ng.vocab_offset else 0))); a.add(cand.ptr);
            try p.c.ops.append(gpa, .{ .launch = .{ .f = top, .cfg = .{ .grid = .{ .x = @intCast(p.rows) }, .block = .{ .x = 1024 } }, .args = a, .name = "fn_rows_top" } });
            try p.c.ops.append(gpa, .{ .gather = .{ .send = cand.ptr, .recv = gath.ptr, .count = 4 * p.rows, .dt = .i32 } });
        }
    }
    std.debug.print("compiled {d} programs, {d:.0} ms\n", .{ table.count(),
        @as(f64, @floatFromInt(tc.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6 });
    var name_buf_init: [64]u8 = undefined;
    var mbuf_streams: u64 = 0;
    var mtp_pos: u64 = 0;
    for (run.prog.value.object.get("storages").?.array.items, 0..) |sv, i| {
        const nm = sv.object.get("name").?.string;
        if (std.mem.eql(u8, nm, "mbuf.streams")) mbuf_streams = run.store.items[i].ptr;
        if (std.mem.eql(u8, nm, "st.mtp_pos")) mtp_pos = run.store.items[i].ptr;
    }
    try check.expect(mbuf_streams != 0 and mtp_pos != 0, "mbuf.streams and st.mtp_pos storages", .{});
    // every MTP program stages its rows with uploads (ids, last) and a copy into its input: the copy's traced source
    // is the main window's streams (buf.streams), where kept rows come from
    for (1..DEPTH + 2) |n| {
        const p = table.get(try std.fmt.bufPrint(&name_buf_init, "mtp_n{d}", .{n})).?;
        try check.expect(p.c.lead == 3 and p.c.ops.items[2] == .copy, "mtp_n{d}: staged rows", .{n});
    }
    const main_streams = table.get("mtp_n1").?.c.ops.items[2].copy.src;

    // the prompt: per chunk its ids, n-gram rows, the MTP head's next tokens and last row
    const pre = table.get("prefill").?;
    var keep_alive: std.ArrayList([]const u8) = .empty;
    var pin: Inputs = .{};
    var chunks: usize = 0;
    var at: usize = 0;
    while (at < prompt.len) : (chunks += 1) {
        const end = @min(at + 2048, prompt.len);
        const toks = prompt[at..end];
        const idb = try gpa.alloc(i32, toks.len);
        for (toks, 0..) |t, i| idb[i] = @intCast(t);
        const rows = try gpa.alloc(i64, toks.len * ng.heads);
        try ng.ids(toks, rows);
        const vals = try gpa.alloc(u16, rows.len * ng.width);
        try ng.gather(rows, vals);
        const nxt = prompt[at + 1 .. @min(end + 1, prompt.len)];
        const nb = try gpa.alloc(i32, nxt.len);
        for (nxt, 0..) |t, i| nb[i] = @intCast(t);
        const last = try gpa.alloc(i32, 1);
        last[0] = @intCast(nxt.len - 1);
        pin.a[4 * chunks] = std.mem.sliceAsBytes(idb);
        pin.a[4 * chunks + 1] = std.mem.sliceAsBytes(vals);
        pin.a[4 * chunks + 2] = std.mem.sliceAsBytes(nb);
        pin.a[4 * chunks + 3] = std.mem.sliceAsBytes(last);
        try keep_alive.append(gpa, std.mem.sliceAsBytes(rows));
        ng.advance(toks);
        at = end;
    }
    const t0 = std.Io.Timestamp.now(io, .awake);
    try R.runCompiled(&run, pre.c.ops.items, stream, &nccl, comm, &pin, -1);
    try stream.synchronize();
    const t1 = std.Io.Timestamp.now(io, .awake);
    // the prompt's last residual streams (a temporary later programs reuse): kept for the first absorb
    const lastbuf = try cuda.DeviceBuffer.alloc(gpu.d, ROW);
    const ls = markAddr(&run, progs.get("prefill").?.object, "last_streams").?;
    try gpu.d.check(gpu.d.api.cuMemcpyDtoD_v2(lastbuf.ptr, ls.addr, ROW), "last streams");
    var first: [1]i64 = undefined;
    const dev = cx.device_sampling;
    cx.device_sampling = false;               // the prompt's program has no candidate step: its row is read here
    try greedyRows(&cx, pre.logits, 1, pre.cols, ng.vocab_offset, &first);
    cx.device_sampling = dev;
    std.debug.print("prefill {d} tokens in {d:.1} ms; first token {d} (Python {d})\n", .{ prompt.len,
        @as(f64, @floatFromInt(t0.durationTo(t1).toNanoseconds())) / 1e6, first[0], want[0] });

    // rounds
    var pos: i64 = @intCast(prompt.len);
    var mtp_len: i64 = @intCast(prompt.len - 1);
    var chained: i64 = 0;                         // MTP entries of chained drafts, trimmed by the next absorb
    var cur: usize = chunks % 2;                  // the DeltaNet state half (flips each prompt chunk and commit)
    const count = want.len;
    var out: std.ArrayList(i64) = .empty;
    try out.append(gpa, first[0]);
    var rounds: usize = 0;
    var drafted: usize = 0;
    var accepted: usize = 0;
    var name_buf: [64]u8 = undefined;
    var drafts: std.ArrayList(i64) = .empty;

    const Propose = struct {
        fn mtpStep(c: *Ctx, tbl: *std.StringHashMap(*Prog), nbuf: []u8, src: u64, next: []const i64, len_at: i64, posbuf: u64) !*Prog {
            const p = tbl.get(try std.fmt.bufPrint(nbuf, "mtp_n{d}", .{next.len})).?;
            var ids: [8]i32 = undefined;
            for (next, 0..) |t, i| ids[i] = @intCast(t);
            var last = [1]i32{@intCast(next.len - 1)};
            var in: Inputs = .{};
            in.a[0] = std.mem.sliceAsBytes(ids[0..next.len]);
            in.a[1] = std.mem.asBytes(&last);
            p.c.ops.items[2].copy.src = src;          // the head's input rows: kept main streams or its own last row
            const d = c.run.gpu.d;
            try d.check(d.api.cuMemsetD32Async(posbuf, @bitCast(@as(i32, @intCast(len_at))), 1, c.stream.handle), "mtp_pos");
            try c.launch(p, len_at + @as(i64, @intCast(next.len)), &in);
            return p;
        }
    };

    // the graphs a round can use, captured before the clock starts (Python's engine warms its graphs at startup)
    if (graphs) {
        const tw = std.Io.Timestamp.now(io, .awake);
        var empty: Inputs = .{};
        var captured: usize = 0;
        for (1..DEPTH + 2) |rows| {
            for (0..2) |par| {
                const fp = table.get(try std.fmt.bufPrint(&name_buf, "fwd_R{d}_p{d}", .{ rows, par })).?;
                _ = try R.refresh(&run, &fp.c, cx.selects, pos + @as(i64, @intCast(rows)));
                try fp.g.prepare(&run, &fp.c, stream, &nccl, comm, &empty);
                captured += 1;
            }
            const mp = table.get(try std.fmt.bufPrint(&name_buf, "mtp_n{d}", .{rows})).?;
            _ = try R.refresh(&run, &mp.c, cx.selects, mtp_len + @as(i64, @intCast(rows)));
            try mp.g.prepare(&run, &mp.c, stream, &nccl, comm, &empty);
            captured += 1;
        }
        std.debug.print("warmed {d} graphs in {d:.0} ms\n", .{ captured, since(io, tw) });
    }
    var tm: Times = .{};
    try stream.synchronize();
    const ts = std.Io.Timestamp.now(io, .awake);
    // propose(streams, next tokens, n): absorb the kept rows, then chain drafts while the head is confident
    var src: u64 = lastbuf.ptr;
    var next: std.ArrayList(i64) = .empty;
    try next.append(gpa, first[0]);
    var n_want: usize = @min(DEPTH, count - out.items.len);
    while (true) {
        drafts.clearRetainingCapacity();
        if (n_want > 0) {
            mtp_len -= chained;
            chained = 0;
            var tq = std.Io.Timestamp.now(io, .awake);
            var p = try Propose.mtpStep(&cx, &table, &name_buf, src, next.items, mtp_len, mtp_pos);
            tm.mtp_steps += 1;
            mtp_len += @intCast(next.items.len);
            var j: usize = 0;
            while (j < n_want) : (j += 1) {
                try stream.synchronize();
                const tpk = std.Io.Timestamp.now(io, .awake);
                const pick = try draftPick(&cx, p.logits, p.cols, draft_ids);
                tm.pick += since(io, tpk);
                tm.mtp += since(io, tq);
                const low = pick.p < CONFIDENCE;
                if (low and j > 0) break;
                try drafts.append(gpa, pick.tok);
                if (low) break;
                if (j + 1 < n_want) {
                    const prev = mbuf_streams + (if (j == 0) next.items.len - 1 else 0) * ROW;
                    tq = std.Io.Timestamp.now(io, .awake);
                    p = try Propose.mtpStep(&cx, &table, &name_buf, prev, &[1]i64{pick.tok}, mtp_len, mtp_pos);
                    tm.mtp_steps += 1;
                    mtp_len += 1;
                    chained += 1;
                }
            }
        }
        if (out.items.len >= count) break;
        // verify the pending token and its drafts
        const tv = std.Io.Timestamp.now(io, .awake);
        var tokens: [8]i64 = undefined;
        tokens[0] = out.items[out.items.len - 1];
        for (drafts.items, 0..) |dd, i| tokens[1 + i] = dd;
        const rows = 1 + drafts.items.len;
        var idb: [8]i32 = undefined;
        for (tokens[0..rows], 0..) |t, i| idb[i] = @intCast(t);
        var rids: [8 * 64]i64 = undefined;
        try ng.ids(tokens[0..rows], rids[0 .. rows * ng.heads]);
        const vals = try gpa.alloc(u16, rows * ng.heads * ng.width);
        defer gpa.free(vals);
        try ng.gather(rids[0 .. rows * ng.heads], vals);
        var in: Inputs = .{};
        in.a[0] = std.mem.sliceAsBytes(idb[0..rows]);
        in.a[1] = std.mem.sliceAsBytes(vals);
        tm.stage += since(io, tv);
        const fp = table.get(try std.fmt.bufPrint(&name_buf, "fwd_R{d}_p{d}", .{ rows, cur })).?;
        const tg = std.Io.Timestamp.now(io, .awake);
        try cx.launch(fp, pos + @as(i64, @intCast(rows)), &in);
        try stream.synchronize();
        tm.gpu += since(io, tg);
        const tgr = std.Io.Timestamp.now(io, .awake);
        var sampled: [8]i64 = undefined;
        try greedyRows(&cx, fp.logits, rows, fp.cols, ng.vocab_offset, sampled[0..rows]);
        tm.greedy += since(io, tgr);
        var keep: usize = 1;
        for (drafts.items, 0..) |dd, i| {
            if (sampled[i] != dd) break;
            keep += 1;
        }
        tm.verify += since(io, tv);
        const tk = std.Io.Timestamp.now(io, .awake);
        const cp = table.get(try std.fmt.bufPrint(&name_buf, "commit_R{d}_k{d}_p{d}", .{ rows, keep, cur })).?;
        try R.runCompiled(&run, cp.c.ops.items, stream, &nccl, comm, &in, pos + @as(i64, @intCast(keep)));
        cur ^= 1;
        tm.commit += since(io, tk);
        ng.advance(tokens[0..keep]);
        pos += @intCast(keep);
        rounds += 1;
        drafted += drafts.items.len;
        accepted += keep - 1;
        try out.appendSlice(gpa, sampled[0..keep]);
        if (out.items.len >= count) break;
        n_want = @min(DEPTH, count - out.items.len);
        src = main_streams;
        next.clearRetainingCapacity();
        try next.appendSlice(gpa, sampled[0..keep]);
    }
    try stream.synchronize();
    const secs = @as(f64, @floatFromInt(ts.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e9;
    var same: usize = 0;
    for (out.items[0..@min(count, out.items.len)], want) |g, w| {
        if (g != w) break;
        same += 1;
    }
    std.debug.print("tokens: {any}\n", .{out.items[0..@min(12, out.items.len)]});
    std.debug.print("MTP decode ({s}): {d} tokens in {d:.3} s = {d:.1} tok/s; {d} rounds, {d} drafted, {d} accepted (Python: {d}, {d}, {d})\n", .{
        if (graphs) "graphs" else "eager", count - 1, secs, @as(f64, @floatFromInt(count - 1)) / secs, rounds, drafted, accepted,
        ref.get("rounds").?.integer, ref.get("drafted").?.integer, ref.get("accepted").?.integer });
    std.debug.print("time: verify {d:.1} ms ({d:.2} a round), commit {d:.1} ms, MTP head {d:.1} ms over {d} steps ({d:.2} a step, sampling included)\n", .{
        tm.verify, tm.verify / @as(f64, @floatFromInt(rounds)), tm.commit, tm.mtp, tm.mtp_steps, tm.mtp / @as(f64, @floatFromInt(tm.mtp_steps)) });
    std.debug.print("verify split: staging {d:.1} ms, GPU {d:.1} ms, greedy {d:.1} ms; draft sampling {d:.1} ms\n", .{ tm.stage, tm.gpu, tm.greedy, tm.pick });
    try check.expect(same == count, "{d} of {d} tokens equal Python's", .{ same, count });
    check.pass("EXACT rank {d}: prefill + {d} tokens with MTP drafts equal the Python engine's", .{ rank, count });
}
