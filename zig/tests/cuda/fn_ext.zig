//! Flash Next port, step 1: our .cu kernels (the exact cubins inside the Python engine's extensions) launched the way
//! each extension's C++ launcher launches them, checked against one captured call's memory byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const CUBINS = "flashnext-zig/cubins";

fn readPath(gpu: Gpu, path: []const u8, limit: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(limit));
}

fn readIn(gpu: Gpu, dir: []const u8, name: []const u8, limit: usize) ![]u8 {
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(path);
    return readPath(gpu, path, limit);
}

/// The kernels: every extension cubin loaded once; a function found by two fragments of its mangled name.
pub const Kernels = struct {
    gpu: Gpu,
    symbols: std.json.Parsed(std.json.Value),
    modules: std.StringHashMap(cuda.Module),
    images: std.ArrayList([]align(16) u8),

    pub fn init(gpu: Gpu) !Kernels {
        const text = try readPath(gpu, CUBINS ++ "/symbols.json", 1 << 22);
        defer gpu.gpa.free(text);
        return .{ .gpu = gpu, .symbols = try std.json.parseFromSlice(std.json.Value, gpu.gpa, text, .{}),
                  .modules = .init(gpu.gpa), .images = .empty };
    }

    pub fn deinit(self: *Kernels) void {
        var it = self.modules.valueIterator();
        while (it.next()) |m| m.unload();
        self.modules.deinit();
        for (self.images.items) |i| self.gpu.gpa.free(i);
        self.images.deinit(self.gpu.gpa);
        self.symbols.deinit();
    }

    fn find(self: *Kernels, file: []const u8, tmpl: []const u8) !cuda.Function {
        var found: ?struct { cubin: []const u8, sym: []const u8 } = null;
        var it = self.symbols.value.object.iterator();
        while (it.next()) |e| for (e.value_ptr.array.items) |s| {
            if (std.mem.indexOf(u8, s.string, file) != null and std.mem.indexOf(u8, s.string, tmpl) != null) {
                if (found != null) {
                    try check.expect(false, "two kernels match {s} {s}", .{ file, tmpl });
                    unreachable;
                }
                found = .{ .cubin = e.key_ptr.*, .sym = s.string };
            }
        };
        const f = found orelse {
            try check.expect(false, "no kernel matches {s} {s}", .{ file, tmpl });
            unreachable;
        };
        const m = self.modules.get(f.cubin) orelse blk: {
            const path = try std.fs.path.join(self.gpu.gpa, &.{ CUBINS, f.cubin });
            defer self.gpu.gpa.free(path);
            const raw = try readPath(self.gpu, path, 1 << 28);
            defer self.gpu.gpa.free(raw);
            const img = try self.gpu.gpa.alignedAlloc(u8, .@"16", raw.len);
            @memcpy(img, raw);
            try self.images.append(self.gpu.gpa, img);
            const mod = try cuda.Module.load(self.gpu.d, img);
            try self.modules.put(f.cubin, mod);
            break :blk mod;
        };
        const z = try self.gpu.gpa.dupeSentinel(u8, f.sym, 0);
        defer self.gpu.gpa.free(z);
        return m.function(z);
    }
};

/// One captured call: its regions on the device at the captured alignment, and its arguments by position.
const Call = struct {
    gpu: Gpu,
    dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),
    regions: std.StringHashMap(struct { buf: cuda.DeviceBuffer, pad: usize, bytes: usize }),
    ret: ?cuda.DeviceBuffer = null,

    fn open(gpu: Gpu, dir: []const u8) !Call {
        const text = try readIn(gpu, dir, "manifest.json", 1 << 24);
        defer gpu.gpa.free(text);
        var c: Call = .{ .gpu = gpu, .dir = dir, .parsed = try std.json.parseFromSlice(std.json.Value, gpu.gpa, text, .{}), .regions = .init(gpu.gpa) };
        var it = c.parsed.value.object.get("arrays").?.object.iterator();
        while (it.next()) |e| {
            const bytes: usize = @intCast(e.value_ptr.object.get("bytes").?.integer);
            const a256: usize = @intCast(e.value_ptr.object.get("align256").?.integer);
            var buf = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 512);
            const pad = (a256 + 256 - @as(usize, @intCast(buf.ptr % 256))) % 256;
            const file = try std.fmt.allocPrint(gpu.gpa, "{s}.bin", .{e.key_ptr.*});
            defer gpu.gpa.free(file);
            const host = try readIn(gpu, dir, file, 1 << 32);
            defer gpu.gpa.free(host);
            try buf.upload(pad, host);
            try c.regions.put(e.key_ptr.*, .{ .buf = buf, .pad = pad, .bytes = bytes });
        }
        // pointer tables: entries rewritten to the regions they point into here
        if (c.parsed.value.object.get("relocate")) |rel| for (rel.array.items) |r| {
            const table = c.regions.get(r.object.get("table").?.string).?;
            const target = c.regions.get(r.object.get("array").?.string).?;
            const off: usize = @intCast(r.object.get("offset").?.integer);
            const addr: u64 = target.buf.ptr + target.pad;
            try table.buf.upload(table.pad + off, std.mem.asBytes(&addr));
        };
        if (c.parsed.value.object.get("returned")) |r| if (r == .object) {
            c.ret = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(r.object.get("bytes").?.integer));
        };
        return c;
    }

    fn deinit(self: *Call) void {
        var it = self.regions.valueIterator();
        while (it.next()) |r| r.buf.free();
        self.regions.deinit();
        if (self.ret) |*r| r.free();
        self.parsed.deinit();
    }

    fn arg(self: Call, i: usize) std.json.ObjectMap {
        return self.parsed.value.object.get("args").?.array.items[i].object;
    }
    /// A tensor's address, or 0 for an empty one (the launchers pass nullptr then).
    fn ptr(self: Call, i: usize) u64 {
        const a = self.arg(i);
        if (numel(self, i) == 0) return 0;
        const r = self.regions.get(a.get("array").?.string).?;
        return r.buf.ptr + r.pad + @as(u64, @intCast(a.get("offset").?.integer));
    }
    fn int(self: Call, i: usize) i64 {
        const v = self.arg(i).get("value").?;
        return switch (v) {
            .integer => |x| x,
            .bool => |b| @intFromBool(b),
            else => unreachable,
        };
    }
    fn float(self: Call, i: usize) f32 {
        return @bitCast(@as(u32, @intCast(self.arg(i).get("f32_bits").?.integer)));
    }
    fn dim(self: Call, i: usize, d: usize) i64 {
        return self.arg(i).get("shape").?.array.items[d].integer;
    }
    fn stride(self: Call, i: usize, d: usize) i64 {
        return self.arg(i).get("stride").?.array.items[d].integer;
    }
    fn numel(self: Call, i: usize) i64 {
        var n: i64 = 1;
        for (self.arg(i).get("shape").?.array.items) |s| n *= s.integer;
        return n;
    }
    fn dtype(self: Call, i: usize) []const u8 {
        return self.arg(i).get("dtype").?.string;
    }

    fn compare(self: *Call) !usize {
        // relocated pointer tables hold this run's addresses: put the captured bytes back before comparing
        if (self.parsed.value.object.get("relocate")) |rel| for (rel.array.items) |r| {
            const name = r.object.get("table").?.string;
            const t = self.regions.get(name).?;
            const file = try std.fmt.allocPrint(self.gpu.gpa, "{s}.bin", .{name});
            defer self.gpu.gpa.free(file);
            const host = try readIn(self.gpu, self.dir, file, 1 << 32);
            defer self.gpu.gpa.free(host);
            try t.buf.upload(t.pad, host);
        };
        var written: usize = 0;
        var it = self.regions.iterator();
        while (it.next()) |e| {
            const want_file = try std.fmt.allocPrint(self.gpu.gpa, "expected_{s}.bin", .{e.key_ptr.*});
            defer self.gpu.gpa.free(want_file);
            const want = try readIn(self.gpu, self.dir, want_file, 1 << 32);
            defer self.gpu.gpa.free(want);
            const got = try self.gpu.gpa.alloc(u8, e.value_ptr.bytes);
            defer self.gpu.gpa.free(got);
            try e.value_ptr.buf.download(e.value_ptr.pad, got);
            try check.sameBytes(e.key_ptr.*, got, want);
            if (self.parsed.value.object.get("arrays").?.object.get(e.key_ptr.*).?.object.get("changed").?.bool) written += 1;
        }
        if (self.ret) |r| {
            const want = try readIn(self.gpu, self.dir, "returned.bin", 1 << 32);
            defer self.gpu.gpa.free(want);
            const got = try self.gpu.gpa.alloc(u8, r.len);
            defer self.gpu.gpa.free(got);
            try r.download(0, got);
            try check.sameBytes("returned", got, want);
            written += 1;
        }
        return written;
    }
};

fn go(f: cuda.Function, grid: cuda.Dim3, block: u32, shared: u32, stream: cuda.Stream, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = block }, .shared = shared }, stream, args);
}

fn d1(x: anytype) cuda.Dim3 {
    return .{ .x = @intCast(x) };
}

fn i32of(x: i64) i32 {
    return @intCast(x);
}

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// One extension call launched as its C++ launcher would: ``c`` gives the arguments by position (ptr, int, float,
/// dim, stride, numel, dtype); ``ret_ptr`` is the buffer a call that returns a tensor writes.
pub fn launchExt(k: *Kernels, gpu: Gpu, name: []const u8, c: anytype, ret_ptr: u64, stream: cuda.Stream) !void {
    const sms: i64 = try gpu.ctx.attribute(.multiprocessor_count);
    var a: cuda.Args = .{};
    var tmpl_buf: [256]u8 = undefined;

    if (std.mem.endsWith(u8, name, "experts_v7.plan")) {
        const P = c.int(1);
        const E = i32of(c.int(2));
        const T = i32of(c.int(3));
        if (P <= 1024) {
            a.add(c.ptr(0)); a.add(i32of(P)); a.add(E); a.add(T); a.add(c.ptr(4)); a.add(c.ptr(5)); a.add(c.ptr(6));
            try go(try k.find("_10_experts_cu_", "11plan_kernelE"), d1(1), 1024, 0, stream, &a);
        } else {
            const nblk = cdiv(P, 1024);
            a.add(c.ptr(0)); a.add(i32of(P)); a.add(E); a.add(c.ptr(7)); a.add(c.ptr(8));
            try go(try k.find("_10_experts_cu_", "9plan_rankE"), d1(nblk), 1024, 0, stream, &a);
            var b: cuda.Args = .{};
            b.add(i32of(nblk)); b.add(E); b.add(T); b.add(c.ptr(8)); b.add(c.ptr(5)); b.add(c.ptr(6));
            try go(try k.find("_10_experts_cu_", "12plan_offsetsE"), d1(1), 1024, 0, stream, &b);
            var s: cuda.Args = .{};
            s.add(c.ptr(0)); s.add(i32of(P)); s.add(E); s.add(c.ptr(7)); s.add(c.ptr(8)); s.add(c.ptr(4));
            try go(try k.find("_10_experts_cu_", "12plan_scatterE"), d1(cdiv(P, 256)), 256, 0, stream, &s);
        }
    } else if (std.mem.endsWith(u8, name, "experts_v7.run") or std.mem.endsWith(u8, name, "experts_v7.prefill")) {
        const prefill = std.mem.endsWith(u8, name, "prefill");
        const gs = c.int(0);
        const epi = c.int(1);
        const m: i64 = if (epi == 2) 2 else 1;
        const nb = c.int(6);
        a.add(c.ptr(2)); a.add(i32of(c.stride(2, 0))); a.add(i32of(c.int(3))); a.add(c.ptr(4)); a.add(i32of(c.int(5)));
        a.add(i32of(nb)); a.add(c.ptr(7)); a.add(c.ptr(8)); a.add(c.ptr(9)); a.add(c.ptr(10)); a.add(i32of(c.int(11)));
        a.add(c.float(12));
        if (prefill) {
            const t = try std.fmt.bufPrint(&tmpl_buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi2ELi2ELi4E", .{ gs, m, epi });
            const f = try k.find("_18_experts_prefill_cu_", t);
            // Pre<GS, M, 2, 2, 4>: X rows (64 x 8 uint4) and WN weight blocks of 64 / GS groups a stage, 48 KiB at most
            const block: i64 = 32 * @divExact(4 * gs, 128) + 4 * 2;
            const su: i64 = 64 * 8 + 4 * @divExact(64, gs) * m * block;
            const sb = su * 16;
            const stages: i64 = if (sb * 4 <= 49152) 4 else if (sb * 3 <= 49152) 3 else 2;
            const smem: u32 = @intCast(stages * sb);
            try f.allowDynamicShared(smem);
            const grid = c.int(13) * cdiv(nb, 4);
            if (grid >= 1) try go(f, d1(grid), 256, smem, stream, &a);
        } else {
            const t = try std.fmt.bufPrint(&tmpl_buf, "13expert_kernelILi{d}ELi{d}ELi{d}ELi2ELi4E", .{ gs, m, epi });
            const f = try k.find("_10_experts_cu_", t);
            const per_sm: i64 = @max(1, try f.occupancy(128, 0));
            const grid = @min(cdiv(c.int(13), 4), per_sm * sms);
            if (grid >= 1) try go(f, d1(grid), 128, 0, stream, &a);
        }
    } else if (std.mem.endsWith(u8, name, "gdn_v2.prefill")) {
        const W = c.dim(0, 0);
        const hk = c.dim(0, 1);
        const hv = c.dim(2, 1);
        const rows: i64 = if (hv >= sms) 128 else 64;
        const qk = if (std.mem.eql(u8, c.dtype(0), "float32")) "f" else "13__nv_bfloat16";
        const t = try std.fmt.bufPrint(&tmpl_buf, "12chain_kernelI{s}Li{d}ELi32E", .{ qk, rows });
        const f = try k.find("_14_gdn_prefill_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4, 5, 6 }) |i| a.add(c.ptr(i));
        a.add(ret_ptr); a.add(i32of(W)); a.add(i32of(hk)); a.add(i32of(hv));
        if (W > 0) try go(f, .{ .x = @intCast(hv), .y = @intCast(@divExact(128, rows)) }, @intCast(2 * rows), 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qwen4_exp_gdn.chain")) {
        const nv = c.numel(4);
        const rows = c.int(8);
        const t = try std.fmt.bufPrint(&tmpl_buf, "12chain_kernelILi{d}ELi{d}ELb{d}E", .{ @divExact(nv, 3), nv, @intFromBool(rows > 1) });
        const f = try k.find("_6_gdn_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4, 5, 6 }) |i| a.add(c.ptr(i));
        a.add(c.float(7)); a.add(i32of(rows));
        for ([_]usize{ 9, 10, 11, 12, 13, 14, 15 }) |i| a.add(c.ptr(i));
        try go(f, d1(nv), 1024, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qwen4_exp_gdn.replay")) {
        const nv = c.dim(3, 1);
        const t = try std.fmt.bufPrint(&tmpl_buf, "13replay_kernelILi{d}ELi{d}E", .{ @divExact(nv, 3), nv });
        const f = try k.find("_6_gdn_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4 }) |i| a.add(c.ptr(i));
        a.add(i32of(c.int(5))); a.add(c.ptr(6));
        try go(f, d1(nv), 1024, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "gdn_io.front")) {
        const nv = c.numel(5);
        const nk = @divExact(nv, 3);
        const t = try std.fmt.bufPrint(&tmpl_buf, "12front_kernelILi{d}ELi{d}E", .{ nk, nv });
        const f = try k.find("_9_gdn_io_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }) |i| a.add(c.ptr(i));
        try go(f, .{ .x = @intCast(c.dim(3, 0)), .y = @intCast(nk + nv) }, 128, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "gdn_io.back")) {
        const nv = c.dim(0, 1);
        const t = try std.fmt.bufPrint(&tmpl_buf, "11back_kernelILi{d}ELi{d}E", .{ @divExact(nv, 3), nv });
        const f = try k.find("_9_gdn_io_cu_", t);
        a.add(c.ptr(0)); a.add(c.ptr(1)); a.add(c.ptr(2)); a.add(c.float(3)); a.add(c.ptr(4)); a.add(c.ptr(5));
        try go(f, .{ .x = @intCast(c.dim(0, 0)), .y = @intCast(nv) }, 128, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qmm_v5.qmm")) {
        const M = c.dim(0, 0);
        const K = c.dim(0, 1);
        const N = c.int(7);
        const sk = c.int(8);
        const gs = c.int(9);
        const bm = c.int(10);
        const fp32 = c.int(11) != 0;
        const reduce = c.int(12) != 0;
        const cluster = sk > 1 and sk <= 8 and reduce;
        try check.expect(sk == 1, "qmm with K slices is not ported yet (sk {d})", .{sk});
        const t = try std.fmt.bufPrint(&tmpl_buf, "10qmm_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb{d}ELb0E", .{ gs, bm, @intFromBool(fp32), @intFromBool(cluster) });
        const f = try k.find("_6_qmm_cu_", t);
        const stage = bm * gs * 2 + 64 * @divExact(gs, 2) + 2 * 64 * 2 + bm * 4;
        const partials = @divExact(bm, 16) * 2 * 4 * 128 * 4;
        const smem: u32 = @intCast(@max(4 * stage, partials));
        try f.allowDynamicShared(smem);
        const rows_t = cdiv(M, bm);
        const group = @max(1, @min(rows_t, @divFloor(@as(i64, 12 << 20), bm * K * 2)));
        a.add(c.ptr(0)); a.add(c.ptr(1)); a.add(c.ptr(2)); a.add(c.ptr(3)); a.add(c.ptr(4)); a.add(c.ptr(5));
        a.add(@as(u64, 0)); a.add(i32of(M)); a.add(i32of(N)); a.add(i32of(K)); a.add(i32of(sk)); a.add(i32of(c.stride(3, 0)));
        a.add(i32of(if (M == 1) K else c.stride(0, 0))); a.add(i32of(group));
        try go(f, .{ .x = @intCast(rows_t * cdiv(N, 64)), .y = 1, .z = @intCast(sk) }, 128, smem, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qmm_v5.qmm_prefill")) {
        const M = c.dim(0, 0);
        const K = c.dim(0, 1);
        const N = c.int(5);
        const gs = c.int(6);
        const fp32 = c.int(7) != 0;
        const tiles = [_][5]i64{ .{ 128, 128, 2, 4, 3 }, .{ 64, 128, 1, 4, 4 }, .{ 128, 64, 2, 2, 4 }, .{ 64, 64, 1, 4, 4 },
            .{ 128, 128, 2, 4, 4 }, .{ 128, 128, 2, 2, 3 }, .{ 128, 128, 2, 2, 4 }, .{ 128, 256, 2, 4, 3 },
            .{ 64, 256, 1, 4, 4 }, .{ 128, 128, 2, 2, 2 }, .{ 64, 128, 1, 2, 2 }, .{ 128, 256, 2, 4, 2 } };
        const tl = tiles[@intCast(c.int(8))];
        const bm = tl[0];
        const bn = tl[1];
        const t = try std.fmt.bufPrint(&tmpl_buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELb{d}E", .{ gs, bm, bn, tl[2], tl[3], tl[4], @intFromBool(fp32) });
        const f = try k.find("_14_qmm_prefill_cu_", t);
        const smem: u32 = @intCast(tl[4] * (bm * gs * 2 + bn * @divExact(gs, 2) + 2 * bn * 2));
        try f.allowDynamicShared(smem);
        const rows_t = cdiv(M, bm);
        const group = @max(1, @min(rows_t, @divFloor(@as(i64, 12 << 20), bm * K * 2)));
        a.add(c.ptr(0)); a.add(c.ptr(1)); a.add(c.ptr(2)); a.add(c.ptr(3)); a.add(c.ptr(4));
        a.add(i32of(M)); a.add(i32of(N)); a.add(i32of(K)); a.add(i32of(c.dim(2, 1)));
        a.add(i32of(if (M == 1) K else c.stride(0, 0))); a.add(i32of(group));
        try go(f, d1(rows_t * cdiv(N, bn)), @intCast(tl[2] * tl[3] * 32), smem, stream, &a);
    } else {
        return check.expect(false, "no launcher for {s}", .{name});
    }
}

pub fn extCall(gpu: Gpu, dir: []const u8) !void {
    var k = try Kernels.init(gpu);
    defer k.deinit();
    var c = try Call.open(gpu, dir);
    defer c.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const name = c.parsed.value.object.get("function").?.string;
    const sms: i64 = try gpu.ctx.attribute(.multiprocessor_count);
    var a: cuda.Args = .{};
    var tmpl_buf: [256]u8 = undefined;

    if (std.mem.endsWith(u8, name, "experts_v7.plan")) {
        const P = c.int(1);
        const E = i32of(c.int(2));
        const T = i32of(c.int(3));
        if (P <= 1024) {
            a.add(c.ptr(0)); a.add(i32of(P)); a.add(E); a.add(T); a.add(c.ptr(4)); a.add(c.ptr(5)); a.add(c.ptr(6));
            try go(try k.find("_10_experts_cu_", "11plan_kernelE"), d1(1), 1024, 0, stream, &a);
        } else {
            const nblk = cdiv(P, 1024);
            a.add(c.ptr(0)); a.add(i32of(P)); a.add(E); a.add(c.ptr(7)); a.add(c.ptr(8));
            try go(try k.find("_10_experts_cu_", "9plan_rankE"), d1(nblk), 1024, 0, stream, &a);
            var b: cuda.Args = .{};
            b.add(i32of(nblk)); b.add(E); b.add(T); b.add(c.ptr(8)); b.add(c.ptr(5)); b.add(c.ptr(6));
            try go(try k.find("_10_experts_cu_", "12plan_offsetsE"), d1(1), 1024, 0, stream, &b);
            var s: cuda.Args = .{};
            s.add(c.ptr(0)); s.add(i32of(P)); s.add(E); s.add(c.ptr(7)); s.add(c.ptr(8)); s.add(c.ptr(4));
            try go(try k.find("_10_experts_cu_", "12plan_scatterE"), d1(cdiv(P, 256)), 256, 0, stream, &s);
        }
    } else if (std.mem.endsWith(u8, name, "experts_v7.run") or std.mem.endsWith(u8, name, "experts_v7.prefill")) {
        const prefill = std.mem.endsWith(u8, name, "prefill");
        const gs = c.int(0);
        const epi = c.int(1);
        const m: i64 = if (epi == 2) 2 else 1;
        const nb = c.int(6);
        a.add(c.ptr(2)); a.add(i32of(c.stride(2, 0))); a.add(i32of(c.int(3))); a.add(c.ptr(4)); a.add(i32of(c.int(5)));
        a.add(i32of(nb)); a.add(c.ptr(7)); a.add(c.ptr(8)); a.add(c.ptr(9)); a.add(c.ptr(10)); a.add(i32of(c.int(11)));
        a.add(c.float(12));
        if (prefill) {
            const t = try std.fmt.bufPrint(&tmpl_buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi2ELi2ELi4E", .{ gs, m, epi });
            const f = try k.find("_18_experts_prefill_cu_", t);
            // Pre<GS, M, 2, 2, 4>: X rows (64 x 8 uint4) and WN weight blocks of 64 / GS groups a stage, 48 KiB at most
            const block: i64 = 32 * @divExact(4 * gs, 128) + 4 * 2;
            const su: i64 = 64 * 8 + 4 * @divExact(64, gs) * m * block;
            const sb = su * 16;
            const stages: i64 = if (sb * 4 <= 49152) 4 else if (sb * 3 <= 49152) 3 else 2;
            const smem: u32 = @intCast(stages * sb);
            try f.allowDynamicShared(smem);
            const grid = c.int(13) * cdiv(nb, 4);
            if (grid >= 1) try go(f, d1(grid), 256, smem, stream, &a);
        } else {
            const t = try std.fmt.bufPrint(&tmpl_buf, "13expert_kernelILi{d}ELi{d}ELi{d}ELi2ELi4E", .{ gs, m, epi });
            const f = try k.find("_10_experts_cu_", t);
            const per_sm: i64 = @max(1, try f.occupancy(128, 0));
            const grid = @min(cdiv(c.int(13), 4), per_sm * sms);
            if (grid >= 1) try go(f, d1(grid), 128, 0, stream, &a);
        }
    } else if (std.mem.endsWith(u8, name, "gdn_v2.prefill")) {
        const W = c.dim(0, 0);
        const hk = c.dim(0, 1);
        const hv = c.dim(2, 1);
        const rows: i64 = if (hv >= sms) 128 else 64;
        const qk = if (std.mem.eql(u8, c.dtype(0), "float32")) "f" else "13__nv_bfloat16";
        const t = try std.fmt.bufPrint(&tmpl_buf, "12chain_kernelI{s}Li{d}ELi32E", .{ qk, rows });
        const f = try k.find("_14_gdn_prefill_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4, 5, 6 }) |i| a.add(c.ptr(i));
        a.add(c.ret.?.ptr); a.add(i32of(W)); a.add(i32of(hk)); a.add(i32of(hv));
        if (W > 0) try go(f, .{ .x = @intCast(hv), .y = @intCast(@divExact(128, rows)) }, @intCast(2 * rows), 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qwen4_exp_gdn.chain")) {
        const nv = c.numel(4);
        const rows = c.int(8);
        const t = try std.fmt.bufPrint(&tmpl_buf, "12chain_kernelILi{d}ELi{d}ELb{d}E", .{ @divExact(nv, 3), nv, @intFromBool(rows > 1) });
        const f = try k.find("_6_gdn_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4, 5, 6 }) |i| a.add(c.ptr(i));
        a.add(c.float(7)); a.add(i32of(rows));
        for ([_]usize{ 9, 10, 11, 12, 13, 14, 15 }) |i| a.add(c.ptr(i));
        try go(f, d1(nv), 1024, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qwen4_exp_gdn.replay")) {
        const nv = c.dim(3, 1);
        const t = try std.fmt.bufPrint(&tmpl_buf, "13replay_kernelILi{d}ELi{d}E", .{ @divExact(nv, 3), nv });
        const f = try k.find("_6_gdn_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4 }) |i| a.add(c.ptr(i));
        a.add(i32of(c.int(5))); a.add(c.ptr(6));
        try go(f, d1(nv), 1024, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "gdn_io.front")) {
        const nv = c.numel(5);
        const nk = @divExact(nv, 3);
        const t = try std.fmt.bufPrint(&tmpl_buf, "12front_kernelILi{d}ELi{d}E", .{ nk, nv });
        const f = try k.find("_9_gdn_io_cu_", t);
        for ([_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }) |i| a.add(c.ptr(i));
        try go(f, .{ .x = @intCast(c.dim(3, 0)), .y = @intCast(nk + nv) }, 128, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "gdn_io.back")) {
        const nv = c.dim(0, 1);
        const t = try std.fmt.bufPrint(&tmpl_buf, "11back_kernelILi{d}ELi{d}E", .{ @divExact(nv, 3), nv });
        const f = try k.find("_9_gdn_io_cu_", t);
        a.add(c.ptr(0)); a.add(c.ptr(1)); a.add(c.ptr(2)); a.add(c.float(3)); a.add(c.ptr(4)); a.add(c.ptr(5));
        try go(f, .{ .x = @intCast(c.dim(0, 0)), .y = @intCast(nv) }, 128, 0, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qmm_v5.qmm")) {
        const M = c.dim(0, 0);
        const K = c.dim(0, 1);
        const N = c.int(7);
        const sk = c.int(8);
        const gs = c.int(9);
        const bm = c.int(10);
        const fp32 = c.int(11) != 0;
        const reduce = c.int(12) != 0;
        const cluster = sk > 1 and sk <= 8 and reduce;
        try check.expect(sk == 1, "qmm with K slices is not ported yet (sk {d})", .{sk});
        const t = try std.fmt.bufPrint(&tmpl_buf, "10qmm_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb{d}ELb0E", .{ gs, bm, @intFromBool(fp32), @intFromBool(cluster) });
        const f = try k.find("_6_qmm_cu_", t);
        const stage = bm * gs * 2 + 64 * @divExact(gs, 2) + 2 * 64 * 2 + bm * 4;
        const partials = @divExact(bm, 16) * 2 * 4 * 128 * 4;
        const smem: u32 = @intCast(@max(4 * stage, partials));
        try f.allowDynamicShared(smem);
        const rows_t = cdiv(M, bm);
        const group = @max(1, @min(rows_t, @divFloor(@as(i64, 12 << 20), bm * K * 2)));
        a.add(c.ptr(0)); a.add(c.ptr(1)); a.add(c.ptr(2)); a.add(c.ptr(3)); a.add(c.ptr(4)); a.add(c.ptr(5));
        a.add(@as(u64, 0)); a.add(i32of(M)); a.add(i32of(N)); a.add(i32of(K)); a.add(i32of(sk)); a.add(i32of(c.stride(3, 0)));
        a.add(i32of(if (M == 1) K else c.stride(0, 0))); a.add(i32of(group));
        try go(f, .{ .x = @intCast(rows_t * cdiv(N, 64)), .y = 1, .z = @intCast(sk) }, 128, smem, stream, &a);
    } else if (std.mem.endsWith(u8, name, "qmm_v5.qmm_prefill")) {
        const M = c.dim(0, 0);
        const K = c.dim(0, 1);
        const N = c.int(5);
        const gs = c.int(6);
        const fp32 = c.int(7) != 0;
        const tiles = [_][5]i64{ .{ 128, 128, 2, 4, 3 }, .{ 64, 128, 1, 4, 4 }, .{ 128, 64, 2, 2, 4 }, .{ 64, 64, 1, 4, 4 },
            .{ 128, 128, 2, 4, 4 }, .{ 128, 128, 2, 2, 3 }, .{ 128, 128, 2, 2, 4 }, .{ 128, 256, 2, 4, 3 },
            .{ 64, 256, 1, 4, 4 }, .{ 128, 128, 2, 2, 2 }, .{ 64, 128, 1, 2, 2 }, .{ 128, 256, 2, 4, 2 } };
        const tl = tiles[@intCast(c.int(8))];
        const bm = tl[0];
        const bn = tl[1];
        const t = try std.fmt.bufPrint(&tmpl_buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELb{d}E", .{ gs, bm, bn, tl[2], tl[3], tl[4], @intFromBool(fp32) });
        const f = try k.find("_14_qmm_prefill_cu_", t);
        const smem: u32 = @intCast(tl[4] * (bm * gs * 2 + bn * @divExact(gs, 2) + 2 * bn * 2));
        try f.allowDynamicShared(smem);
        const rows_t = cdiv(M, bm);
        const group = @max(1, @min(rows_t, @divFloor(@as(i64, 12 << 20), bm * K * 2)));
        a.add(c.ptr(0)); a.add(c.ptr(1)); a.add(c.ptr(2)); a.add(c.ptr(3)); a.add(c.ptr(4));
        a.add(i32of(M)); a.add(i32of(N)); a.add(i32of(K)); a.add(i32of(c.dim(2, 1)));
        a.add(i32of(if (M == 1) K else c.stride(0, 0))); a.add(i32of(group));
        try go(f, d1(rows_t * cdiv(N, bn)), @intCast(tl[2] * tl[3] * 32), smem, stream, &a);
    } else {
        return check.expect(false, "no launcher for {s}", .{name});
    }
    try stream.synchronize();
    const written = try c.compare();
    check.pass("BITEXACT {s}: {d} regions equal ({d} written)", .{ name, c.regions.count() + @intFromBool(c.ret != null), written });
}
