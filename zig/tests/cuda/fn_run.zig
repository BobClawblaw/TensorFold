//! Flash Next port, step 2 (stage 0): run a rank's decode program (gen_program.py, from the Python engine's full
//! trace) natively: storages from the weight pack and the reset dump, Triton cubins, our extension cubins, NCCL
//! gathers; teacher-forced steps with the captured inputs, each step's logits compared byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fn_ext = @import("fn_ext.zig");
const Gpu = check.Gpu;

const CACHE = ".cache/tensorfold-qwen38-int4mixed/cb2ebf0540f42604e2759b2ddef497861e928248";
const SIZE = std.StaticStringMap(usize).initComptime(.{ .{ "bfloat16", 2 }, .{ "float16", 2 }, .{ "float32", 4 },
    .{ "int32", 4 }, .{ "int64", 8 }, .{ "uint8", 1 }, .{ "int8", 1 }, .{ "bool", 1 }, .{ "int16", 2 }, .{ "float8_e4m3fn", 1 } });

fn openRead(gpu: Gpu, path: []const u8) !std.Io.File {
    return std.Io.Dir.cwd().openFile(gpu.io, path, .{});
}

pub fn writeUid(gpu: Gpu, path: []const u8) !void {
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var uid: cuda.nccl.UniqueId = undefined;
    try lib.check(lib.api.ncclGetUniqueId(&uid), "ncclGetUniqueId");
    const f = try std.Io.Dir.cwd().createFile(gpu.io, path, .{});
    defer f.close(gpu.io);
    try f.writePositionalAll(gpu.io, &uid.internal, 0);
    check.pass("NCCL unique id written to {s}", .{path});
}

const Run = struct {
    gpu: Gpu,
    store: std.ArrayList(cuda.DeviceBuffer) = .empty,
    temps: cuda.DeviceBuffer = undefined,
    temp_off: std.ArrayList(usize) = .empty,
    kernels: std.StringHashMap(cuda.triton.Kernel),
    images: std.ArrayList([]align(16) u8) = .empty,
    scratch: cuda.DeviceBuffer = undefined,
    prog: std.json.Parsed(std.json.Value) = undefined,
    fill_value: i32 = 0,
    fill64: u64 = 0,
    check_dir: []const u8 = "",
    checks_ok: usize = 0,
    checks_bad: usize = 0,

    fn addr(self: *const Run, v: std.json.Value) u64 {
        const o = v.object;
        const off: u64 = @intCast(o.get("off").?.integer);
        if (o.get("s")) |s| return self.store.items[@intCast(s.integer)].ptr + off;
        return self.temps.ptr + self.temp_off.items[@intCast(o.get("tmp").?.integer)] + off;
    }
};

fn extent(o: std.json.ObjectMap) usize {
    const shape = o.get("shape").?.array.items;
    const stride = o.get("stride").?.array.items;
    var e: usize = 1;
    for (shape, stride) |d, s| {
        if (d.integer == 0) return 0;
        e += @as(usize, @intCast(d.integer - 1)) * @as(usize, @intCast(s.integer));
    }
    return e * SIZE.get(o.get("dtype").?.string).?;
}

/// An extension op's arguments by position, as fn_ext.launchExt reads them.
const OpArgs = struct {
    run: *const Run,
    items: []std.json.Value,

    fn o(self: OpArgs, i: usize) std.json.ObjectMap {
        return self.items[i].object;
    }
    pub fn ptr(self: OpArgs, i: usize) u64 {
        if (self.numel(i) == 0) return 0;
        return self.run.addr(self.items[i]);
    }
    pub fn int(self: OpArgs, i: usize) i64 {
        return switch (self.items[i]) {
            .integer => |x| x,
            .bool => |b| @intFromBool(b),
            else => unreachable,
        };
    }
    pub fn float(self: OpArgs, i: usize) f32 {
        return @bitCast(@as(u32, @intCast(self.o(i).get("bits").?.integer)));
    }
    pub fn dim(self: OpArgs, i: usize, d: usize) i64 {
        return self.o(i).get("shape").?.array.items[d].integer;
    }
    pub fn stride(self: OpArgs, i: usize, d: usize) i64 {
        return self.o(i).get("stride").?.array.items[d].integer;
    }
    pub fn numel(self: OpArgs, i: usize) i64 {
        var n: i64 = 1;
        for (self.o(i).get("shape").?.array.items) |s| n *= s.integer;
        return n;
    }
    pub fn dtype(self: OpArgs, i: usize) []const u8 {
        return self.o(i).get("dtype").?.string;
    }
};

fn kernelFor(run: *Run, hash: []const u8) !cuda.triton.Kernel {
    if (run.kernels.get(hash)) |k| return k;
    const gpa = run.gpu.gpa;
    const info = run.prog.value.object.get("kernels").?.object.get(hash).?.object;
    var cubin_path: ?[]const u8 = null;
    var json_path: ?[]const u8 = null;
    var it = info.get("files").?.object.iterator();
    while (it.next()) |e| {
        if (std.mem.endsWith(u8, e.key_ptr.*, ".cubin")) cubin_path = e.value_ptr.string;
        if (std.mem.endsWith(u8, e.key_ptr.*, ".json") and !std.mem.startsWith(u8, e.key_ptr.*, "__grp__")) json_path = e.value_ptr.string;
    }
    const cp = try std.fs.path.join(gpa, &.{ CACHE, cubin_path.? });
    defer gpa.free(cp);
    const jp = try std.fs.path.join(gpa, &.{ CACHE, json_path.? });
    defer gpa.free(jp);
    const raw = try std.Io.Dir.cwd().readFileAlloc(run.gpu.io, cp, gpa, .limited(1 << 28));
    defer gpa.free(raw);
    const img = try gpa.alignedAlloc(u8, .@"16", raw.len);
    @memcpy(img, raw);
    try run.images.append(gpa, img);
    const mtext = try std.Io.Dir.cwd().readFileAlloc(run.gpu.io, jp, gpa, .limited(1 << 22));
    defer gpa.free(mtext);
    const meta = try cuda.triton.parseMeta(gpa, mtext);       // kept alive with the kernel (its name)
    const name_z = try gpa.dupeSentinel(u8, meta.value.name, 0);
    const k = try cuda.triton.Kernel.load(run.gpu.d, run.gpu.ctx.device, img, meta.value, name_z);
    try run.kernels.put(try gpa.dupe(u8, hash), k);
    return k;
}

fn loadStorages(run: *Run, pack_path: []const u8, reset_path: []const u8) !void {
    const gpa = run.gpu.gpa;
    const io = run.gpu.io;
    const pack = try openRead(run.gpu, pack_path);
    defer pack.close(io);
    const reset = try openRead(run.gpu, reset_path);
    defer reset.close(io);
    var host: []u8 = try gpa.alloc(u8, 1 << 30);
    defer gpa.free(host);
    var total: usize = 0;
    for (run.prog.value.object.get("storages").?.array.items) |sv| {
        const s = sv.object;
        const bytes: usize = @intCast(s.get("bytes").?.integer);
        var buf = try cuda.DeviceBuffer.alloc(run.gpu.d, @max(bytes, 1));
        try run.store.append(gpa, buf);
        total += bytes;
        const init = s.get("init").?.string;
        if (std.mem.eql(u8, init, "pack")) {
            for (s.get("fills").?.array.items) |fv| {
                const f = fv.object;
                var left: usize = @intCast(f.get("bytes").?.integer);
                var src: u64 = @intCast(f.get("pack_offset").?.integer);
                var at: usize = @intCast(f.get("at").?.integer);
                while (left > 0) {
                    const n = @min(left, host.len);
                    if (try pack.readPositionalAll(io, host[0..n], src) != n) return error.ShortRead;
                    try buf.upload(at, host[0..n]);
                    left -= n; src += n; at += n;
                }
            }
        } else if (std.mem.eql(u8, init, "reset")) {
            var left = bytes;
            var src: u64 = @intCast(s.get("reset_offset").?.integer);
            var at: usize = 0;
            while (left > 0) {
                const n = @min(left, host.len);
                if (try reset.readPositionalAll(io, host[0..n], src) != n) return error.ShortRead;
                try buf.upload(at, host[0..n]);
                left -= n; src += n; at += n;
            }
        } else {
            try run.gpu.d.check(run.gpu.d.api.cuMemsetD8_v2(buf.ptr, 0, bytes), "cuMemsetD8");
        }
    }
    check.pass("storages: {d} on the device, {d} MiB", .{ run.store.items.len, total >> 20 });
}

fn runOps(run: *Run, ops: []std.json.Value, k: *fn_ext.Kernels, stream: cuda.Stream, nccl: *cuda.nccl.Library,
          comm: cuda.nccl.Comm, inputs: anytype, pos: i64) !void {
    return runOpsKeys(run, ops, k, stream, nccl, comm, inputs, pos, -1);
}

/// ``keys``: the attended keys of a decode step (position + 1), which size two attention grids; -1 keeps the trace's.
fn runOpsKeys(run: *Run, ops: []std.json.Value, k: *fn_ext.Kernels, stream: cuda.Stream, nccl: *cuda.nccl.Library,
          comm: cuda.nccl.Comm, inputs: anytype, pos: i64, keys: i64) !void {
    const d = run.gpu.d;
    for (ops) |ov| {
        const o = ov.object;
        const kind = o.get("kind").?.string;
        const name = o.get("name").?.string;
        if (std.mem.eql(u8, kind, "triton")) {
            var hash = o.get("hash").?.string;
            if (keys >= 0 and std.mem.eql(u8, name, "_select")) {
                // attention._launch_select: BLOCK = next power of two of the step's key blocks (4 keys a block)
                const blocks: u64 = @intCast(@max(1, @divFloor(keys + 3, 4)));
                const width = std.math.ceilPowerOfTwo(u64, blocks) catch unreachable;
                var it = run.prog.value.object.get("kernels").?.object.iterator();
                var found: ?[]const u8 = null;
                while (it.next()) |e| {
                    const kv = e.value_ptr.object;
                    if (!std.mem.eql(u8, kv.get("name").?.string, "_select")) continue;
                    const b = kv.get("consts").?.object.get("BLOCK") orelse continue;
                    if (b.integer == @as(i64, @intCast(width))) found = e.key_ptr.*;
                }
                hash = found orelse return check.expect(false, "no _select compiled for BLOCK {d}", .{width});
            }
            const kern = try kernelFor(run, hash);
            var a: cuda.Args = .{};
            for (o.get("args").?.array.items) |av| {
                const ty = av.object.get("type").?.string;
                const v = av.object.get("v").?;
                if (ty[0] == '*') {
                    a.add(@as(u64, if (v == .null) 0 else run.addr(v)));
                } else if (std.mem.eql(u8, ty, "i32")) {
                    a.add(@as(i32, @intCast(switch (v) { .integer => |x| x, .bool => |b| @as(i64, @intFromBool(b)), else => unreachable })));
                } else if (std.mem.eql(u8, ty, "fp32")) {
                    a.add(@as(f32, @bitCast(@as(u32, @intCast(v.object.get("bits").?.integer)))));
                } else return check.expect(false, "{s}: argument type {s}", .{ name, ty });
            }
            const g = o.get("grid").?.array.items;
            var dims: cuda.Dim3 = .{ .x = @intCast(g[0].integer), .y = @intCast(g[1].integer), .z = @intCast(g[2].integer) };
            if (keys >= 0) {
                // attention.qsa_rows: blocks of 4 keys (index_ratio), 64 blocks a program; attention: 512-key chunks
                // over at most (budget / ratio + 1) * ratio - 1 = 2051 keys (index_budget 2048)
                if (std.mem.eql(u8, name, "_scores")) dims.y = @intCast(@divFloor(@max(1, @divFloor(keys + 3, 4)) + 63, 64));
                if (std.mem.eql(u8, name, "_chunks")) dims.z = @intCast(@divFloor(@min(keys, 2051) + 511, 512));
            }
            try check.expect(kern.globalScratchBytes(dims) <= run.scratch.len, "{s}: scratch", .{name});
            try kern.launchOn(dims, stream, &a, .{ .global = if (kern.meta.global_scratch_size > 0) run.scratch.ptr else 0,
                .profile = if (kern.meta.profile_scratch_size > 0) run.scratch.ptr else 0 }, &.{});
        } else if (std.mem.eql(u8, kind, "ext")) {
            const items = o.get("args").?.array.items;
            const short = name[std.mem.indexOfScalar(u8, name, '.').? + 1 ..];
            const full = try std.fmt.allocPrint(run.gpu.gpa, "tensorfold_{s}", .{
                if (std.mem.startsWith(u8, name, "experts.")) try std.fmt.allocPrint(run.gpu.gpa, "experts_v7.{s}", .{short})
                else if (std.mem.startsWith(u8, name, "gdn_v2.")) name
                else if (std.mem.startsWith(u8, name, "qmm.")) try std.fmt.allocPrint(run.gpu.gpa, "qmm_v5.{s}", .{short})
                else if (std.mem.startsWith(u8, name, "gdn_io.")) try std.fmt.allocPrint(run.gpu.gpa, "qwen4_exp_gdn_io.{s}", .{short})
                else name });
            const ret: u64 = if (o.get("ret")) |r| run.addr(r) else 0;
            try fn_ext.launchExt(k, run.gpu, full, OpArgs{ .run = run, .items = items }, ret, stream);
        } else if (std.mem.eql(u8, kind, "comm")) {
            const items = o.get("args").?.array.items;
            try check.expect(std.mem.eql(u8, name, "all_gather"), "comm {s}", .{name});
            const part = items[0].object;
            var count: usize = 1;
            for (part.get("shape").?.array.items) |s| count *= @intCast(s.integer);
            const dt: cuda.nccl.DataType = if (std.mem.eql(u8, part.get("dtype").?.string, "float32")) .f32 else .bf16;
            try nccl.check(nccl.api.ncclAllGather(run.addr(items[0]), run.addr(items[1]), count, dt, comm, stream.handle), "ncclAllGather");
        } else if (std.mem.eql(u8, kind, "upload")) {
            const dst = o.get("args").?.array.items[0];
            const seq: usize = @intCast(o.get("seq").?.integer);
            const bytes = inputs.get(seq) orelse return check.expect(false, "no input {d} ({s})", .{ seq, o.get("input").?.string });
            const n = extent(dst.object);
            try d.check(d.api.cuMemcpyHtoDAsync_v2(run.addr(dst), bytes.ptr, n, stream.handle), "upload");
        } else if (std.mem.eql(u8, name, "aten.copy_.default")) {
            const items = o.get("args").?.array.items;
            const n = extent(items[0].object);
            try d.check(d.api.cuMemcpyDtoDAsync_v2(run.addr(items[0]), run.addr(items[1]), n, stream.handle), "copy");
        } else if (std.mem.eql(u8, name, "aten.fill_.Scalar")) {
            const items = o.get("args").?.array.items;
            const t = items[0].object;
            const dt = t.get("dtype").?.string;
            const dst = run.addr(items[0]);
            const count = extent(t) / SIZE.get(dt).?;
            if (o.get("fill_ptr")) |fp| {
                run.fill64 = run.addr(fp);
                try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, &run.fill64, 8, stream.handle), "fill pointer");
            } else if (std.mem.eql(u8, dt, "int64")) {
                try check.expect(count == 1, "an int64 fill of {d} values", .{count});
                run.fill64 = @bitCast(items[1].integer);
                try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, &run.fill64, 8, stream.handle), "fill int64");
            } else {
                var bits: u32 = 0;
                const pos_fill = pos >= 0 and o.get("fill_name") != null and o.get("fill_name").? == .string and
                    std.mem.eql(u8, o.get("fill_name").?.string, "st.pos_dev");
                if (std.mem.eql(u8, dt, "int32")) {
                    const v: i32 = if (pos_fill) @intCast(pos) else @intCast(items[1].integer);
                    bits = @bitCast(v);
                } else if (std.mem.eql(u8, dt, "float32")) {
                    const v: f32 = switch (items[1]) { .integer => |x| @floatFromInt(x), .object => |ob| @bitCast(@as(u32, @intCast(ob.get("bits").?.integer))), else => unreachable };
                    bits = @bitCast(v);
                } else return check.expect(false, "a fill of {s}", .{dt});
                try d.check(d.api.cuMemsetD32Async(dst, bits, count, stream.handle), "fill");
            }
        } else if (std.mem.eql(u8, name, "aten.zero_.default")) {
            const items = o.get("args").?.array.items;
            try d.check(d.api.cuMemsetD8Async(run.addr(items[0]), 0, extent(items[0].object), stream.handle), "zero");
        } else if (std.mem.eql(u8, kind, "check")) {
            try stream.synchronize();
            const ref = o.get("args").?.array.items[0];
            const n = extent(ref.object);
            const f = try std.fmt.allocPrint(run.gpu.gpa, "{s}/{s}.bin", .{ run.check_dir, name });
            const want = std.Io.Dir.cwd().readFileAlloc(run.gpu.io, f, run.gpu.gpa, .limited(1 << 31)) catch {
                std.debug.print("check {s}: no dump\n", .{name});
                continue;
            };
            const got = try run.gpu.gpa.alloc(u8, n);
            try d.check(d.api.cuMemcpyDtoH_v2(got.ptr, run.addr(ref), n), "check");
            const m = @min(n, want.len);
            if (std.mem.eql(u8, got[0..m], want[0..m])) {
                run.checks_ok += 1;
            } else {
                run.checks_bad += 1;
                if (run.checks_bad <= 12) std.debug.print("check {s}: DIFFERS from byte {d} of {d}\n", .{ name, std.mem.indexOfDiff(u8, got[0..m], want[0..m]).?, m });
            }
        } else return check.expect(false, "no runner for {s} {s}", .{ kind, name });
    }
}

pub fn decode(gpu: Gpu, args: []const [:0]const u8) !void {
    // fn-run <program.json> <weights.bin> <reset_storages.bin> <steps dir> <rank> <uid file> <steps>
    const gpa = gpu.gpa;
    const rank: c_int = try std.fmt.parseInt(c_int, args[4], 10);
    const nsteps = try std.fmt.parseInt(usize, args[6], 10);
    var run: Run = .{ .gpu = gpu, .kernels = .init(gpa) };
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, args[0], gpa, .limited(1 << 28));
    run.prog = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    // temporaries: one arena, each 512-aligned
    var off: usize = 0;
    const progs = run.prog.value.object.get("programs").?.object;
    var maxn: usize = 0;
    for ([_][]const u8{ "odd", "even" }) |p| maxn = @max(maxn, progs.get(p).?.object.get("temps").?.array.items.len);
    for (0..maxn) |i| {
        var sz: usize = 0;
        for ([_][]const u8{ "odd", "even" }) |p| {
            const t = progs.get(p).?.object.get("temps").?.array.items;
            if (i < t.len) sz = @max(sz, @as(usize, @intCast(t[i].integer)));
        }
        try run.temp_off.append(gpa, off);
        off += std.mem.alignForward(usize, @max(sz, 1), 512);
    }
    run.temps = try cuda.DeviceBuffer.alloc(gpu.d, off);
    run.scratch = try cuda.DeviceBuffer.alloc(gpu.d, 256 << 20);
    var k = try fn_ext.Kernels.init(gpu);
    defer k.deinit();
    // NCCL: rank 0 makes the unique id (its bootstrap root lives in this process) and writes it; rank 1 reads it
    var nccl = try cuda.nccl.Library.open();
    defer nccl.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try nccl.check(nccl.api.ncclGetUniqueId(&uid), "ncclGetUniqueId");
        const tmp = try std.fmt.allocPrint(gpa, "{s}.tmp", .{args[5]});
        const f = try std.Io.Dir.cwd().createFile(gpu.io, tmp, .{});
        try f.writePositionalAll(gpu.io, &uid.internal, 0);
        f.close(gpu.io);
        try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), args[5], gpu.io);
    } else {
        const uf = try openRead(gpu, args[5]);
        _ = try uf.readPositionalAll(gpu.io, &uid.internal, 0);
        uf.close(gpu.io);
    }
    var comm: cuda.nccl.Comm = null;
    try loadStorages(&run, args[1], args[2]);
    try nccl.check(nccl.api.ncclCommInitRank(&comm, 2, uid, rank), "ncclCommInitRank");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var equal: usize = 0;
    for (0..nsteps) |step| {
        const prog = progs.get(if (step % 2 == 1) "odd" else "even").?.object;
        var inputs = std.AutoHashMap(usize, []u8).init(gpa);
        for ([_][]const u8{ "ids", "ple_v" }, 0..) |nm, seq| {
            const f = try std.fmt.allocPrint(gpa, "{s}/s{d:0>2}_{s}.bin", .{ args[3], step, nm });
            try inputs.put(seq, try std.Io.Dir.cwd().readFileAlloc(gpu.io, f, gpa, .limited(1 << 26)));
        }
        const fwd = prog.get("forward").?.array.items;
        try stream.synchronize();
        const t0 = std.Io.Timestamp.now(gpu.io, .awake);
        try runOps(&run, fwd, &k, stream, &nccl, comm, inputs, -1);
        const t1 = std.Io.Timestamp.now(gpu.io, .awake);
        try stream.synchronize();
        const t2 = std.Io.Timestamp.now(gpu.io, .awake);
        std.debug.print("step {d}: host {d:.2} ms, forward {d:.2} ms\n", .{ step, @as(f64, @floatFromInt(t0.durationTo(t1).toNanoseconds())) / 1e6, @as(f64, @floatFromInt(t0.durationTo(t2).toNanoseconds())) / 1e6 });
        // the head's logits: the last op's OUT (the q8 matmul's fourth runtime argument), row 0
        const outv = fwd[fwd.len - 1].object.get("args").?.array.items[3].object.get("v").?;
        const lf = try std.fmt.allocPrint(gpa, "{s}/s{d:0>2}_logits.bin", .{ args[3], step });
        const want = try std.Io.Dir.cwd().readFileAlloc(gpu.io, lf, gpa, .limited(1 << 26));
        const got = try gpa.alloc(u8, want.len);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(got.ptr, run.addr(outv), want.len), "logits");
        if (std.mem.eql(u8, got, want)) {
            equal += 1;
            std.debug.print("step {d}: logits equal ({d} bytes)\n", .{ step, want.len });
        } else {
            const first = std.mem.indexOfDiff(u8, got, want).?;
            std.debug.print("step {d}: logits DIFFER from byte {d}\n", .{ step, first });
        }
        try runOps(&run, prog.get("commit").?.array.items, &k, stream, &nccl, comm, inputs, @intCast(step + 1));
        try stream.synchronize();
    }
    try check.expect(equal == nsteps, "{d} of {d} steps equal", .{ equal, nsteps });
    check.pass("BITEXACT rank {d}: {d} teacher-forced decode steps, logits equal to Python's", .{ rank, nsteps });
}

const fn_ngram = @import("fn_ngram.zig");

fn lastHeadOut(ops: []std.json.Value) std.json.Value {
    var i = ops.len;
    while (i > 0) : (i -= 1) {
        const o = ops[i - 1].object;
        if (std.mem.eql(u8, o.get("kind").?.string, "triton") and std.mem.eql(u8, o.get("name").?.string, "_q8mm"))
            return o.get("args").?.array.items[3].object.get("v").?;
    }
    unreachable;
}

/// Greedy over both ranks' vocabulary shards: each rank's best (global id, value), exchanged; the larger value wins,
/// a tie goes to the lower id.
fn greedy(run: *Run, outv: std.json.Value, vocab_offset: i64, xbuf: cuda.DeviceBuffer, stream: cuda.Stream,
          nccl: *cuda.nccl.Library, comm: cuda.nccl.Comm) !i64 {
    const gpa = run.gpu.gpa;
    const vr: usize = @intCast(outv.object.get("shape").?.array.items[1].integer);
    const row = try gpa.alloc(u16, vr);
    defer gpa.free(row);
    try stream.synchronize();                // the head ran on the decode stream; the copy below does not wait for it
    try run.gpu.d.check(run.gpu.d.api.cuMemcpyDtoH_v2(@ptrCast(row.ptr), run.addr(outv), vr * 2), "logits row");
    var best: usize = 0;
    var bv: f32 = -std.math.inf(f32);
    for (row, 0..) |b, i| {
        const v: f32 = @bitCast(@as(u32, b) << 16);
        if (v > bv) { bv = v; best = i; }
    }
    var mine = [2]i32{ @intCast(@as(i64, @intCast(best)) + vocab_offset), @bitCast(bv) };
    try xbuf.upload(0, std.mem.asBytes(&mine));
    try nccl.check(nccl.api.ncclAllGather(xbuf.ptr, xbuf.ptr + 8, 2, .i32, comm, stream.handle), "greedy exchange");
    try stream.synchronize();
    var all: [4]i32 = undefined;
    try xbuf.download(8, std.mem.asBytes(&all));
    const v0: f32 = @bitCast(all[1]);
    const v1: f32 = @bitCast(all[3]);
    if (v1 > v0 or (v1 == v0 and all[2] < all[0])) return all[2];
    return all[0];
}

pub const Mode = enum { interp, compiled, graph, profile };

pub fn generate(gpu: Gpu, args: []const [:0]const u8, mode: Mode) !void {
    // fn-gen <program.json> <weights.bin> <reset_storages.bin> <ngram.json> <reference.json> <rank> <uid file> [prefill_inputs dir]
    const gpa = gpu.gpa;
    const rank: c_int = try std.fmt.parseInt(c_int, args[5], 10);
    var run: Run = .{ .gpu = gpu, .kernels = .init(gpa) };
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, args[0], gpa, .limited(1 << 28));
    run.prog = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    const progs = run.prog.value.object.get("programs").?.object;
    var maxn: usize = 0;
    for ([_][]const u8{ "odd", "even", "prefill" }) |p| maxn = @max(maxn, progs.get(p).?.object.get("temps").?.array.items.len);
    var off: usize = 0;
    for (0..maxn) |i| {
        var sz: usize = 0;
        for ([_][]const u8{ "odd", "even", "prefill" }) |p| {
            const t = progs.get(p).?.object.get("temps").?.array.items;
            if (i < t.len) sz = @max(sz, @as(usize, @intCast(t[i].integer)));
        }
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
        const tmp = try std.fmt.allocPrint(gpa, "{s}.tmp", .{args[6]});
        const f = try std.Io.Dir.cwd().createFile(gpu.io, tmp, .{});
        try f.writePositionalAll(gpu.io, &uid.internal, 0);
        f.close(gpu.io);
        try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), args[6], gpu.io);
    } else {
        const uf = try openRead(gpu, args[6]);
        _ = try uf.readPositionalAll(gpu.io, &uid.internal, 0);
        uf.close(gpu.io);
    }
    var ng = try fn_ngram.NGram.open(gpa, gpu.io, args[3]);
    const rtext = try std.Io.Dir.cwd().readFileAlloc(gpu.io, args[4], gpa, .limited(1 << 24));
    const ref = (try std.json.parseFromSlice(std.json.Value, gpa, rtext, .{})).value.object;
    const prompt = try fn_ngram.NGram.ints(gpa, ref.get("prompt").?);
    const want = try fn_ngram.NGram.ints(gpa, ref.get("tokens").?);
    try loadStorages(&run, args[1], args[2]);
    var comm: cuda.nccl.Comm = null;
    try nccl.check(nccl.api.ncclCommInitRank(&comm, 2, uid, rank), "ncclCommInitRank");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const xbuf = try cuda.DeviceBuffer.alloc(gpu.d, 64);
    var consts: Consts = .{};
    var cp_pre: Compiled = .{};
    var cp_dec: [2]Compiled = .{ .{}, .{} };       // even, odd
    var graphs: [2]DecodeGraph = .{ .{}, .{} };
    var selects: []Select = &.{};
    var prof: Profile = .{ .totals = .init(gpa) };
    if (mode != .interp) {
        const tc = std.Io.Timestamp.now(gpu.io, .awake);
        cp_pre = try compile(&run, progs.get("prefill").?.object.get("forward").?.array.items, &k, stream, &consts);
        for ([_][]const u8{ "even", "odd" }, 0..) |p, i| cp_dec[i] = try compile(&run, progs.get(p).?.object.get("forward").?.array.items, &k, stream, &consts);
        try finishConsts(&run, &consts, &.{ &cp_pre, &cp_dec[0], &cp_dec[1] });
        selects = try selectTable(&run);
        std.debug.print("compiled: prefill {d} ops, decode {d} + {d} ops ({d} keyed), {d} constants, {d} selector widths, {d:.0} ms\n", .{
            cp_pre.ops.items.len, cp_dec[0].ops.items.len, cp_dec[1].ops.items.len, cp_dec[0].keyed.items.len, consts.host.items.len, selects.len,
            @as(f64, @floatFromInt(tc.durationTo(std.Io.Timestamp.now(gpu.io, .awake)).toNanoseconds())) / 1e6 });
    }

    // the prompt: chunks of 2048 rows, each chunk's ids and n-gram rows staged here
    var inputs = std.AutoHashMap(usize, []u8).init(gpa);
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
        try inputs.put(2 * chunks, std.mem.sliceAsBytes(idb));
        try inputs.put(2 * chunks + 1, std.mem.sliceAsBytes(vals));
        if (args.len > 7) {               // the Python engine's staging of the same chunk
            const pf = try std.fmt.allocPrint(gpa, "{s}/c{d}_ple_v.bin", .{ args[7], chunks });
            const py = try std.Io.Dir.cwd().readFileAlloc(gpu.io, pf, gpa, .limited(1 << 30));
            const mine = std.mem.sliceAsBytes(vals);
            try check.expect(std.mem.eql(u8, py[0..mine.len], mine), "chunk {d}: n-gram rows differ from Python's", .{chunks});
            std.debug.print("chunk {d}: {d} rows, n-gram rows equal to Python's staging\n", .{ chunks, toks.len });
        }
        ng.advance(toks);
        at = end;
    }
    const pre = progs.get("prefill").?.object.get("forward").?.array.items;
    const t0 = std.Io.Timestamp.now(gpu.io, .awake);
    if (mode == .interp) try runOps(&run, pre, &k, stream, &nccl, comm, inputs, -1)
    else try runCompiled(&run, cp_pre.ops.items, stream, &nccl, comm, inputs, -1);
    try stream.synchronize();
    const t1 = std.Io.Timestamp.now(gpu.io, .awake);
    if (args.len > 8) {                  // the state after Python's prefill, storage by storage (args[8]: path prefix)
        const jf = try std.fmt.allocPrint(gpa, "{s}.json", .{args[8]});
        const bf = try std.fmt.allocPrint(gpa, "{s}.bin", .{args[8]});
        const ents = (try std.json.parseFromSlice(std.json.Value, gpa, try std.Io.Dir.cwd().readFileAlloc(gpu.io, jf, gpa, .limited(1 << 24)), .{})).value.array.items;
        const file = try openRead(gpu, bf);
        var bad: usize = 0;
        for (ents) |ev| {
            const e = ev.object;
            const base = e.get("base").?.integer;
            var sid: ?usize = null;
            for (run.prog.value.object.get("storages").?.array.items, 0..) |sv, i| {
                if (sv.object.get("base").?.integer == base) { sid = i; break; }
            }
            const nm = e.get("name").?.string;
            if (sid == null) continue;
            const n: usize = @intCast(e.get("bytes").?.integer);
            const want_b = try gpa.alloc(u8, n);
            defer gpa.free(want_b);
            _ = try file.readPositionalAll(gpu.io, want_b, @intCast(e.get("offset").?.integer));
            const got_b = try gpa.alloc(u8, n);
            defer gpa.free(got_b);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(got_b.ptr, run.store.items[sid.?].ptr, n), "state");
            if (!std.mem.eql(u8, want_b, got_b)) {
                bad += 1;
                if (std.mem.startsWith(u8, nm, "st.")) std.debug.print("after prefill: {s} DIFFERS from byte {d} of {d}\n", .{ nm, std.mem.indexOfDiff(u8, want_b, got_b).?, n });
            }
        }
        std.debug.print("after prefill: {d} storages differ (buf scratch may)\n", .{bad});
    }
    var got: std.ArrayList(i64) = .empty;
    try got.append(gpa, try greedy(&run, lastHeadOut(pre), ng.vocab_offset, xbuf, stream, &nccl, comm));
    std.debug.print("prefill {d} tokens in {d:.1} ms; first token {d} (Python {d})\n", .{ prompt.len,
        @as(f64, @floatFromInt(t0.durationTo(t1).toNanoseconds())) / 1e6, got.items[0], want[0] });
    // decode: one token a step, greedy over both ranks
    var dec_ms: f64 = 0;
    var step: usize = 0;
    while (got.items.len < want.len) : (step += 1) {
        const prog = progs.get(if ((chunks + step) % 2 == 1) "odd" else "even").?.object;
        const tok = if (args.len > 9) want[got.items.len - 1] else got.items[got.items.len - 1];
        var din = std.AutoHashMap(usize, []u8).init(gpa);
        var idv = [1]i32{@intCast(tok)};
        const rows = try gpa.alloc(i64, ng.heads);
        try ng.ids(&[1]i64{tok}, rows);
        const vals = try gpa.alloc(u16, rows.len * ng.width);
        try ng.gather(rows, vals);
        try din.put(0, std.mem.asBytes(&idv));
        try din.put(1, std.mem.sliceAsBytes(vals));
        const fwd = prog.get("forward").?.array.items;
        const a = std.Io.Timestamp.now(gpu.io, .awake);
        const keys: i64 = @intCast(prompt.len + step + 1);
        const par = (chunks + step) % 2;
        switch (mode) {
            .interp => try runOpsKeys(&run, fwd, &k, stream, &nccl, comm, din, -1, keys),
            .profile => {
                _ = try refresh(&run, &cp_dec[par], selects, keys);
                if (step >= 8 and step < 16) try prof.step(&run, cp_dec[par].ops.items, stream, &nccl, comm, din)
                else try runCompiled(&run, cp_dec[par].ops.items, stream, &nccl, comm, din, -1);
            },
            .compiled => {
                _ = try refresh(&run, &cp_dec[par], selects, keys);
                try runCompiled(&run, cp_dec[par].ops.items, stream, &nccl, comm, din, -1);
            },
            .graph => {
                _ = try refresh(&run, &cp_dec[par], selects, keys);
                try graphs[par].launch(&run, &cp_dec[par], stream, &nccl, comm, din);
            },
        }
        const next = try greedy(&run, lastHeadOut(fwd), ng.vocab_offset, xbuf, stream, &nccl, comm);
        if (args.len > 9) {
            const lf = try std.fmt.allocPrint(gpa, "{s}/d{d:0>2}_logits.bin", .{ args[9], step });
            if (std.Io.Dir.cwd().readFileAlloc(gpu.io, lf, gpa, .limited(1 << 26))) |wl| {
                const gl = try gpa.alloc(u8, wl.len);
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(gl.ptr, run.addr(lastHeadOut(fwd)), wl.len), "dec logits");
                const pf = try std.fmt.allocPrint(gpa, "{s}/d{d:0>2}_ple_v.bin", .{ args[9], step });
                const wp = try std.Io.Dir.cwd().readFileAlloc(gpu.io, pf, gpa, .limited(1 << 26));
                const mine = std.mem.sliceAsBytes(vals);
                std.debug.print("decode step {d} (token {d}): logits {s}, staged rows {s}\n", .{ step, tok,
                    if (std.mem.eql(u8, gl, wl)) "equal" else "DIFFER", if (std.mem.eql(u8, wp[0..mine.len], mine)) "equal" else "DIFFER" });
            } else |_| {}
        }
        dec_ms += @as(f64, @floatFromInt(a.durationTo(std.Io.Timestamp.now(gpu.io, .awake)).toNanoseconds())) / 1e6;
        try runOps(&run, prog.get("commit").?.array.items, &k, stream, &nccl, comm, din, @intCast(prompt.len + step + 1));
        ng.advance(&[1]i64{tok});
        try got.append(gpa, next);
    }
    try stream.synchronize();
    var same: usize = 0;
    for (got.items, want) |g, w| {
        if (g != w) break;
        same += 1;
    }
    std.debug.print("tokens: {any}\n", .{got.items[0..@min(12, got.items.len)]});
    std.debug.print("decode ({s}) {d} steps, {d:.2} ms a step ({d:.1} tok/s)\n", .{ @tagName(mode), step, dec_ms / @as(f64, @floatFromInt(step)), 1000.0 * @as(f64, @floatFromInt(step)) / dec_ms });
    if (mode == .profile) try prof.report(gpa);
    if (mode == .graph) std.debug.print("graphs: {d} captures, {d:.1} ms capturing\n", .{ graphs[0].captures + graphs[1].captures, graphs[0].capture_ms + graphs[1].capture_ms });
    try check.expect(same == want.len, "{d} of {d} tokens equal Python's", .{ same, want.len });
    check.pass("EXACT rank {d}: prefill + {d} greedy tokens equal the Python engine's", .{ rank, want.len });
}

pub fn checkChunk(gpu: Gpu, args: []const [:0]const u8) !void {
    // fn-check <program.json> <weights.bin> <reset_storages.bin> <dump dir (chunk)> <rank> <uid file>
    const gpa = gpu.gpa;
    const rank: c_int = try std.fmt.parseInt(c_int, args[4], 10);
    var run: Run = .{ .gpu = gpu, .kernels = .init(gpa), .check_dir = args[3] };
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, args[0], gpa, .limited(1 << 28));
    run.prog = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    const prog = run.prog.value.object.get("programs").?.object.get("chunk").?.object;
    var off: usize = 0;
    for (prog.get("temps").?.array.items) |t| {
        try run.temp_off.append(gpa, off);
        off += std.mem.alignForward(usize, @max(@as(usize, @intCast(t.integer)), 1), 512);
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
        const tmp = try std.fmt.allocPrint(gpa, "{s}.tmp", .{args[5]});
        const f = try std.Io.Dir.cwd().createFile(gpu.io, tmp, .{});
        try f.writePositionalAll(gpu.io, &uid.internal, 0);
        f.close(gpu.io);
        try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), args[5], gpu.io);
    } else {
        const uf = try openRead(gpu, args[5]);
        _ = try uf.readPositionalAll(gpu.io, &uid.internal, 0);
        uf.close(gpu.io);
    }
    try loadStorages(&run, args[1], args[2]);
    var comm: cuda.nccl.Comm = null;
    try nccl.check(nccl.api.ncclCommInitRank(&comm, 2, uid, rank), "ncclCommInitRank");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var inputs = std.AutoHashMap(usize, []u8).init(gpa);
    for ([_][]const u8{ "ids", "staged_ple_v" }, 0..) |nm, seq| {
        const f = try std.fmt.allocPrint(gpa, "{s}/{s}.bin", .{ args[3], nm });
        try inputs.put(seq, try std.Io.Dir.cwd().readFileAlloc(gpu.io, f, gpa, .limited(1 << 31)));
    }
    try runOps(&run, prog.get("forward").?.array.items, &k, stream, &nccl, comm, inputs, -1);
    try stream.synchronize();
    std.debug.print("checks: {d} equal, {d} differ\n", .{ run.checks_ok, run.checks_bad });
    try check.expect(run.checks_bad == 0, "{d} checks differ", .{run.checks_bad});
    check.pass("BITEXACT rank {d}: the {d}-check prompt chunk equals Python's at every dump", .{ rank, run.checks_ok });
}

// ---------------------------------------------------------------------------------------------------------------
// The compiled program: every traced op resolved once into what the driver takes (a function, its geometry and its
// packed arguments; a gather's addresses; a copy's extent), so a step does no JSON, hashing or symbol lookups. The
// three launches whose geometry follows the context length keep their source so ``refresh`` can re-derive it.

const Keyed = enum { none, scores, chunks, select };

const Launch = struct {
    f: cuda.Function,
    cfg: cuda.Config,
    args: cuda.Args,
    key: Keyed = .none,
    kern: ?cuda.triton.Kernel = null,     // Triton launches: the kernel (its config follows the grid)
    dims: cuda.Dim3 = .{},
    name: []const u8 = "",
};

const COp = union(enum) {
    launch: Launch,
    gather: struct { send: u64, recv: u64, count: usize, dt: cuda.nccl.DataType },
    upload: struct { dst: u64, seq: usize, n: usize },
    copy: struct { dst: u64, src: u64, n: usize },
    set32: struct { dst: u64, bits: u32, count: usize, pos: bool },
    zero: struct { dst: u64, n: usize },
};

const Select = struct { block: u64, kern: cuda.triton.Kernel };

const Compiled = struct {
    ops: std.ArrayList(COp) = .empty,
    keyed: std.ArrayList(usize) = .empty,     // indices of the launches ``refresh`` re-derives
    lead: usize = 0,                          // the leading uploads (the step's inputs), run before a graph
    sig: [3]u64 = .{ 0, 0, 0 },               // the geometry ``refresh`` last set
};

/// 64-bit constants (int64 fills, pointer fills) staged once on the device; a fill becomes an 8-byte copy from here.
const Consts = struct {
    host: std.ArrayList(u64) = .empty,
    dev: cuda.DeviceBuffer = undefined,

    fn put(self: *Consts, gpa: std.mem.Allocator, v: u64) !u64 {
        try self.host.append(gpa, v);
        return (self.host.items.len - 1) * 8;      // an offset until ``finish`` knows the buffer
    }
};

fn extName(gpa: std.mem.Allocator, name: []const u8) ![]const u8 {
    const short = name[std.mem.indexOfScalar(u8, name, '.').? + 1 ..];
    return std.fmt.allocPrint(gpa, "tensorfold_{s}", .{
        if (std.mem.startsWith(u8, name, "experts.")) try std.fmt.allocPrint(gpa, "experts_v7.{s}", .{short})
        else if (std.mem.startsWith(u8, name, "gdn_v2.")) name
        else if (std.mem.startsWith(u8, name, "qmm.")) try std.fmt.allocPrint(gpa, "qmm_v5.{s}", .{short})
        else if (std.mem.startsWith(u8, name, "gdn_io.")) try std.fmt.allocPrint(gpa, "qwen4_exp_gdn_io.{s}", .{short})
        else name });
}

fn tritonLaunch(run: *Run, kern: cuda.triton.Kernel, dims: cuda.Dim3, args: cuda.Args, key: Keyed, name: []const u8) !Launch {
    var a = args;
    try check.expect(kern.globalScratchBytes(dims) <= run.scratch.len, "{s}: scratch", .{name});
    if (kern.meta.num_ctas == 16) try kern.function.setAttribute(.non_portable_cluster_size_allowed, 1);
    a.add(@as(u64, if (kern.meta.global_scratch_size > 0) run.scratch.ptr else 0));
    a.add(@as(u64, if (kern.meta.profile_scratch_size > 0) run.scratch.ptr else 0));
    return .{ .f = kern.function, .cfg = kern.config(dims), .args = a, .key = key, .kern = kern, .dims = dims, .name = name };
}

fn compile(run: *Run, ops: []std.json.Value, k: *fn_ext.Kernels, stream: cuda.Stream, consts: *Consts) !Compiled {
    const gpa = run.gpu.gpa;
    var c: Compiled = .{};
    var leading = true;
    var pending64: std.ArrayList(usize) = .empty;      // copy ops whose source is a Consts offset
    defer pending64.deinit(gpa);
    for (ops) |ov| {
        const o = ov.object;
        const kind = o.get("kind").?.string;
        const name = o.get("name").?.string;
        if (!std.mem.eql(u8, kind, "upload")) leading = false;
        if (std.mem.eql(u8, kind, "triton")) {
            const kern = try kernelFor(run, o.get("hash").?.string);
            var a: cuda.Args = .{};
            for (o.get("args").?.array.items) |av| {
                const ty = av.object.get("type").?.string;
                const v = av.object.get("v").?;
                if (ty[0] == '*') {
                    a.add(@as(u64, if (v == .null) 0 else run.addr(v)));
                } else if (std.mem.eql(u8, ty, "i32")) {
                    a.add(@as(i32, @intCast(switch (v) { .integer => |x| x, .bool => |b| @as(i64, @intFromBool(b)), else => unreachable })));
                } else if (std.mem.eql(u8, ty, "fp32")) {
                    a.add(@as(f32, @bitCast(@as(u32, @intCast(v.object.get("bits").?.integer)))));
                } else { try check.expect(false, "{s}: argument type {s}", .{ name, ty }); unreachable; }
            }
            const g = o.get("grid").?.array.items;
            const dims: cuda.Dim3 = .{ .x = @intCast(g[0].integer), .y = @intCast(g[1].integer), .z = @intCast(g[2].integer) };
            const key: Keyed = if (std.mem.eql(u8, name, "_scores")) .scores else if (std.mem.eql(u8, name, "_chunks")) .chunks
                else if (std.mem.eql(u8, name, "_select")) .select else .none;
            if (key != .none) try c.keyed.append(gpa, c.ops.items.len);
            try c.ops.append(gpa, .{ .launch = try tritonLaunch(run, kern, dims, a, key, name) });
        } else if (std.mem.eql(u8, kind, "ext")) {
            var recs: std.ArrayList(fn_ext.Rec) = .empty;
            defer recs.deinit(gpa);
            fn_ext.record = .{ .gpa = gpa, .list = &recs };
            defer fn_ext.record = null;
            const ret: u64 = if (o.get("ret")) |r| run.addr(r) else 0;
            try fn_ext.launchExt(k, run.gpu, try extName(gpa, name), OpArgs{ .run = run, .items = o.get("args").?.array.items }, ret, stream);
            for (recs.items) |r| try c.ops.append(gpa, .{ .launch = .{ .f = r.f, .cfg = r.cfg, .args = r.args, .name = name } });
        } else if (std.mem.eql(u8, kind, "comm")) {
            const items = o.get("args").?.array.items;
            try check.expect(std.mem.eql(u8, name, "all_gather"), "comm {s}", .{name});
            var count: usize = 1;
            for (items[0].object.get("shape").?.array.items) |s| count *= @intCast(s.integer);
            try c.ops.append(gpa, .{ .gather = .{ .send = run.addr(items[0]), .recv = run.addr(items[1]), .count = count,
                .dt = if (std.mem.eql(u8, items[0].object.get("dtype").?.string, "float32")) .f32 else .bf16 } });
        } else if (std.mem.eql(u8, kind, "upload")) {
            const dst = o.get("args").?.array.items[0];
            try c.ops.append(gpa, .{ .upload = .{ .dst = run.addr(dst), .seq = @intCast(o.get("seq").?.integer), .n = extent(dst.object) } });
            if (leading) c.lead = c.ops.items.len;
        } else if (std.mem.eql(u8, name, "aten.copy_.default")) {
            const items = o.get("args").?.array.items;
            try c.ops.append(gpa, .{ .copy = .{ .dst = run.addr(items[0]), .src = run.addr(items[1]), .n = extent(items[0].object) } });
        } else if (std.mem.eql(u8, name, "aten.fill_.Scalar")) {
            const items = o.get("args").?.array.items;
            const t = items[0].object;
            const dt = t.get("dtype").?.string;
            const dst = run.addr(items[0]);
            const count = extent(t) / SIZE.get(dt).?;
            if (o.get("fill_ptr") != null or std.mem.eql(u8, dt, "int64")) {
                try check.expect(count == 1, "an int64 fill of {d} values", .{count});
                const v: u64 = if (o.get("fill_ptr")) |fp| run.addr(fp) else @bitCast(items[1].integer);
                try pending64.append(gpa, c.ops.items.len);
                try c.ops.append(gpa, .{ .copy = .{ .dst = dst, .src = try consts.put(gpa, v), .n = 8 } });
            } else {
                const pos_fill = o.get("fill_name") != null and o.get("fill_name").? == .string and
                    std.mem.eql(u8, o.get("fill_name").?.string, "st.pos_dev");
                var bits: u32 = 0;
                if (std.mem.eql(u8, dt, "int32")) {
                    bits = @bitCast(@as(i32, @intCast(items[1].integer)));
                } else if (std.mem.eql(u8, dt, "float32")) {
                    const v: f32 = switch (items[1]) { .integer => |x| @floatFromInt(x), .object => |ob| @bitCast(@as(u32, @intCast(ob.get("bits").?.integer))), else => unreachable };
                    bits = @bitCast(v);
                } else { try check.expect(false, "a fill of {s}", .{dt}); unreachable; }
                try c.ops.append(gpa, .{ .set32 = .{ .dst = dst, .bits = bits, .count = count, .pos = pos_fill } });
            }
        } else if (std.mem.eql(u8, name, "aten.zero_.default")) {
            const items = o.get("args").?.array.items;
            try c.ops.append(gpa, .{ .zero = .{ .dst = run.addr(items[0]), .n = extent(items[0].object) } });
        } else { try check.expect(false, "no compiler for {s} {s}", .{ kind, name }); unreachable; }
    }
    // the constants' sources become device addresses once the buffer exists (``finishConsts``)
    for (pending64.items) |i| c.ops.items[i].copy.src |= 1 << 63;
    return c;
}

/// Uploads the constants and rewrites every compiled program's constant sources (marked by bit 63) to addresses.
fn finishConsts(run: *Run, consts: *Consts, progs: []const *Compiled) !void {
    consts.dev = try cuda.DeviceBuffer.alloc(run.gpu.d, @max(8, consts.host.items.len * 8));
    if (consts.host.items.len > 0) try consts.dev.upload(0, std.mem.sliceAsBytes(consts.host.items));
    for (progs) |p| for (p.ops.items) |*op| switch (op.*) {
        .copy => |*cp| if (cp.src & (1 << 63) != 0) { cp.src = consts.dev.ptr + (cp.src & ~(@as(u64, 1) << 63)); },
        else => {},
    };
}

fn selectTable(run: *Run) ![]Select {
    const gpa = run.gpu.gpa;
    var out: std.ArrayList(Select) = .empty;
    var it = run.prog.value.object.get("kernels").?.object.iterator();
    while (it.next()) |e| {
        const kv = e.value_ptr.object;
        if (!std.mem.eql(u8, kv.get("name").?.string, "_select")) continue;
        const b = kv.get("consts").?.object.get("BLOCK") orelse continue;
        const kern = try kernelFor(run, e.key_ptr.*);
        if (kern.meta.num_ctas == 16) try kern.function.setAttribute(.non_portable_cluster_size_allowed, 1);
        try out.append(gpa, .{ .block = @intCast(b.integer), .kern = kern });
    }
    return out.items;
}

/// The geometry a decode step at ``keys`` attended keys needs: the indexer's score programs (4 keys a block, 64
/// blocks a program), the attention's 512-key chunks over at most 2051 keys, and the selector's power-of-two width.
fn keyedSig(keys: i64) [3]u64 {
    const blocks: u64 = @intCast(@max(1, @divFloor(keys + 3, 4)));
    return .{ (blocks + 63) / 64, @intCast(@divFloor(@min(keys, 2051) + 511, 512)), std.math.ceilPowerOfTwo(u64, blocks) catch unreachable };
}

/// Re-derives the keyed launches for ``keys``; returns whether anything changed.
fn refresh(run: *Run, c: *Compiled, selects: []const Select, keys: i64) !bool {
    const sig = keyedSig(keys);
    if (std.mem.eql(u64, &sig, &c.sig)) return false;
    c.sig = sig;
    for (c.keyed.items) |i| {
        const l = &c.ops.items[i].launch;
        switch (l.key) {
            .scores => l.dims.y = @intCast(sig[0]),
            .chunks => l.dims.z = @intCast(sig[1]),
            .select => {
                var found: ?cuda.triton.Kernel = null;
                for (selects) |s| if (s.block == sig[2]) { found = s.kern; };
                l.kern = found orelse {
                    try check.expect(false, "no _select compiled for BLOCK {d}", .{sig[2]});
                    unreachable;
                };
                l.f = l.kern.?.function;
            },
            .none => unreachable,
        }
        try check.expect(l.kern.?.globalScratchBytes(l.dims) <= run.scratch.len, "keyed launch: scratch", .{});
        l.cfg = l.kern.?.config(l.dims);
    }
    return true;
}

fn runCompiled(run: *Run, ops: []COp, stream: cuda.Stream, nccl: *cuda.nccl.Library, comm: cuda.nccl.Comm, inputs: anytype, pos: i64) !void {
    const d = run.gpu.d;
    for (ops) |*op| switch (op.*) {
        .launch => |*l| try cuda.launch.launch(l.f, l.cfg, stream, &l.args),
        .gather => |g| try nccl.check(nccl.api.ncclAllGather(g.send, g.recv, g.count, g.dt, comm, stream.handle), "ncclAllGather"),
        .upload => |u| {
            const bytes = inputs.get(u.seq) orelse return check.expect(false, "no input {d}", .{u.seq});
            try d.check(d.api.cuMemcpyHtoDAsync_v2(u.dst, bytes.ptr, u.n, stream.handle), "upload");
        },
        .copy => |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.n, stream.handle), "copy"),
        .set32 => |s| {
            const bits: u32 = if (s.pos and pos >= 0) @bitCast(@as(i32, @intCast(pos))) else s.bits;
            try d.check(d.api.cuMemsetD32Async(s.dst, bits, s.count, stream.handle), "fill");
        },
        .zero => |z| try d.check(d.api.cuMemsetD8Async(z.dst, 0, z.n, stream.handle), "zero"),
    };
}

/// A decode forward as a CUDA graph (everything after its input uploads), recaptured when the keyed geometry moves.
const DecodeGraph = struct {
    exec: ?cuda.graph.Exec = null,
    sig: [3]u64 = .{ 0, 0, 0 },
    captures: usize = 0,
    capture_ms: f64 = 0,

    fn launch(self: *DecodeGraph, run: *Run, c: *Compiled, stream: cuda.Stream, nccl: *cuda.nccl.Library, comm: cuda.nccl.Comm, inputs: anytype) !void {
        try runCompiled(run, c.ops.items[0..c.lead], stream, nccl, comm, inputs, -1);
        if (self.exec == null or !std.mem.eql(u64, &self.sig, &c.sig)) {
            const t0 = std.Io.Timestamp.now(run.gpu.io, .awake);
            try cuda.graph.beginCapture(stream, .thread_local);
            runCompiled(run, c.ops.items[c.lead..], stream, nccl, comm, inputs, -1) catch |e| {
                if (cuda.graph.endCapture(stream)) |g| { var gg = g; gg.deinit(); } else |_| {}
                return e;
            };
            var g = try cuda.graph.endCapture(stream);
            defer g.deinit();
            var reused = false;
            if (self.exec) |ex| {
                reused = (ex.update(g) catch .@"error") == .success;
                if (!reused) { var old = ex; old.deinit(); self.exec = null; }
            }
            if (!reused) {
                self.exec = try g.instantiate();
                try self.exec.?.upload(stream);
            }
            self.sig = c.sig;
            self.captures += 1;
            self.capture_ms += @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(run.gpu.io, .awake)).toNanoseconds())) / 1e6;
        }
        try self.exec.?.launchOn(stream);
    }
};

/// GPU time per op kind over a few decode steps: an event after every op, summed by label (a launch's kernel name,
/// a gather, a copy).
const Profile = struct {
    totals: std.StringHashMap(struct { ms: f64 = 0, n: usize = 0 }),
    steps: usize = 0,
    step_ms: f64 = 0,

    fn step(self: *Profile, run: *Run, ops: []COp, stream: cuda.Stream, nccl: *cuda.nccl.Library, comm: cuda.nccl.Comm, inputs: anytype) !void {
        const gpa = run.gpu.gpa;
        const evs = try gpa.alloc(cuda.Event, ops.len + 1);
        defer gpa.free(evs);
        for (evs) |*e| e.* = try cuda.Event.init(run.gpu.d, true);
        defer for (evs) |*e| e.deinit();
        try evs[0].record(stream);
        for (ops, 0..) |_, i| {
            try runCompiled(run, ops[i .. i + 1], stream, nccl, comm, inputs, -1);
            try evs[i + 1].record(stream);
        }
        try evs[ops.len].synchronize();
        for (ops, 0..) |op, i| {
            const label: []const u8 = switch (op) {
                .launch => |l| l.name,
                else => @tagName(op),
            };
            const ms: f64 = try cuda.Event.elapsedMs(evs[i], evs[i + 1]);
            const gop = try self.totals.getOrPut(label);
            if (!gop.found_existing) gop.value_ptr.* = .{};
            gop.value_ptr.ms += ms;
            gop.value_ptr.n += 1;
        }
        if (self.steps == 7) {            // the last profiled step, op by op, for offline attribution
            const f = try std.Io.Dir.cwd().createFile(run.gpu.io, "flashnext-zig/prof_ops.txt", .{});
            defer f.close(run.gpu.io);
            var at: u64 = 0;
            for (ops, 0..) |op, i| {
                const line = try std.fmt.allocPrint(gpa, "{d} {s} {d:.4}\n", .{ i, switch (op) { .launch => |l| l.name, else => @tagName(op) }, try cuda.Event.elapsedMs(evs[i], evs[i + 1]) });
                try f.writePositionalAll(run.gpu.io, line, at);
                at += line.len;
            }
        }
        self.step_ms += try cuda.Event.elapsedMs(evs[0], evs[ops.len]);
        self.steps += 1;
    }

    fn report(self: *Profile, gpa: std.mem.Allocator) !void {
        const E = struct { name: []const u8, ms: f64, n: usize };
        var list: std.ArrayList(E) = .empty;
        var it = self.totals.iterator();
        while (it.next()) |e| try list.append(gpa, .{ .name = e.key_ptr.*, .ms = e.value_ptr.ms, .n = e.value_ptr.n });
        std.mem.sort(E, list.items, {}, struct { fn lt(_: void, a: E, b: E) bool { return a.ms > b.ms; } }.lt);
        const st: f64 = @floatFromInt(self.steps);
        std.debug.print("profile: {d} steps, {d:.2} ms a step on the GPU (events after every op)\n", .{ self.steps, self.step_ms / st });
        for (list.items[0..@min(30, list.items.len)]) |e|
            std.debug.print("  {d:7.3} ms a step  {d:4} ops  {d:7.1} us each  {s}\n", .{ e.ms / st, e.n / self.steps, 1000 * e.ms / @as(f64, @floatFromInt(e.n)), e.name });
    }
};
