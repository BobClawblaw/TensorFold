//! Flash Next's prepared weights for one rank of two, read from the converted checkpoint (affine-experts v2: int8
//! dense linears with group scales, affine 4-bit routed experts in groups of 64, bf16 vectors, e4m3 n-gram shards),
//! as the Python engine prepares them (families/qwen4_exp/cuda/weights.py, its affine-experts path): every tensor
//! under the name, shape and bytes of the Python engine's prepared-weight pack, uploaded to this rank's GPU.
//!
//! Every derivation runs on the host with the Python arithmetic's float32 steps in the same order (int8 requant of
//! the MTP layer's bf16 linears, the draft head's 4-bit copy, YaRN's norm-scale fold), so each tensor is bit-equal;
//! integer layouts (the experts' blocks, the draft head's fragments) are the Python kernels' index maps.

const std = @import("std");
const cuda = @import("cuda");
pub const api = @import("api.zig");
/// The forward and its engine API (tf-cuda-test fn-native reaches it through this module).
pub const forward = @import("forward.zig");

const BASE = "model.language_model.";
const WORLD = 2;
const DRAFT_VOCAB = @embedFile("draft_vocab_default.txt");

// ---------------------------------------------------------------------------------------------------------------
// The checkpoint: its shards' headers, files opened once, tensors read whole or as row ranges.

const Entry = struct { dtype: []const u8, shape: []const i64, begin: u64, end: u64, file: usize };

const Ckpt = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    files: std.ArrayList(std.Io.File) = .empty,
    paths: std.ArrayList([]const u8) = .empty,
    entries: std.StringHashMap(Entry),

    /// ``dir``: the served folder. A shard (or the index) that is a symlink to a container path under /cache/tf/
    /// resolves against the folder's parent directory, so the same folder serves inside the container and on the host.
    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Ckpt {
        var ck: Ckpt = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .entries = .init(gpa) };
        errdefer ck.deinit();
        const a = ck.arena.allocator();
        const index_path = try resolve(a, io, dir, "model.safetensors.index.json");
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(1 << 26));
        const idx = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var shard_of = std.StringHashMap(usize).init(a);
        var it = idx.object.get("weight_map").?.object.iterator();
        while (it.next()) |e| {
            const shard = e.value_ptr.string;
            if (shard_of.contains(shard)) continue;
            const path = try resolve(a, io, dir, shard);
            try shard_of.put(shard, ck.files.items.len);
            try ck.files.append(gpa, try std.Io.Dir.cwd().openFile(io, path, .{}));
            try ck.paths.append(gpa, path);
            try ck.readHeader(ck.files.items.len - 1);
        }
        return ck;
    }

    fn readHeader(ck: *Ckpt, f: usize) !void {
        const a = ck.arena.allocator();
        var head: [8]u8 = undefined;
        _ = try ck.files.items[f].readPositionalAll(ck.io, &head, 0);
        const n = std.mem.readInt(u64, &head, .little);
        const text = try a.alloc(u8, n);
        _ = try ck.files.items[f].readPositionalAll(ck.io, text, 8);
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var it = v.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = e.value_ptr.object;
            const shp = o.get("shape").?.array.items;
            const shape = try a.alloc(i64, shp.len);
            for (shp, 0..) |x, i| shape[i] = x.integer;
            const offs = o.get("data_offsets").?.array.items;
            try ck.entries.put(e.key_ptr.*, .{ .dtype = o.get("dtype").?.string, .shape = shape,
                .begin = 8 + n + @as(u64, @intCast(offs[0].integer)), .end = 8 + n + @as(u64, @intCast(offs[1].integer)), .file = f });
        }
    }

    fn deinit(ck: *Ckpt) void {
        for (ck.files.items) |f| f.close(ck.io);
        ck.files.deinit(ck.gpa);
        ck.paths.deinit(ck.gpa);
        ck.entries.deinit();
        ck.arena.deinit();
    }

    fn has(ck: *const Ckpt, name: []const u8) bool {
        return ck.entries.contains(name);
    }

    fn get(ck: *const Ckpt, name: []const u8) !Entry {
        return ck.entries.get(name) orelse {
            std.log.err("flash next weights: the checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    /// ``len`` bytes at ``off`` into the tensor's data.
    fn readAt(ck: *const Ckpt, e: Entry, off: u64, out: []u8) !void {
        if (off + out.len > e.end - e.begin) return error.ShortRead;
        const got = try ck.files.items[e.file].readPositionalAll(ck.io, out, e.begin + off);
        if (got != out.len) return error.ShortRead;
    }

    fn readAll(ck: *const Ckpt, gpa: std.mem.Allocator, name: []const u8) ![]align(16) u8 {
        const e = try ck.get(name);
        const buf = try gpa.alignedAlloc(u8, .@"16", e.end - e.begin);
        errdefer gpa.free(buf);
        try ck.readAt(e, 0, buf);
        return buf;
    }
};

fn resolve(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    if (std.Io.Dir.cwd().openFile(io, path, .{})) |f| {
        f.close(io);
        return path;
    } else |_| {}
    var buf: [4096]u8 = undefined;
    const n = std.Io.Dir.cwd().readLink(io, path, &buf) catch return path;
    const target = buf[0..n];
    const container = "/cache/tf/";
    if (std.mem.startsWith(u8, target, container))
        return std.fs.path.join(a, &.{ std.fs.path.dirname(dir) orelse ".", target[container.len..] });
    if (std.fs.path.isAbsolute(target)) return a.dupe(u8, target);
    return std.fs.path.join(a, &.{ dir, target });
}

// ---------------------------------------------------------------------------------------------------------------
// Float steps as the Python engine takes them (torch float32 on CUDA: IEEE ops, round-to-nearest-even casts).

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7FC0;
    return @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// torch.round: half to even.
fn rne(x: f32) f32 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2.0) == 0) f else f + 1;
}

fn readF32(ck: *const Ckpt, gpa: std.mem.Allocator, name: []const u8) ![]f32 {
    const e = try ck.get(name);
    const raw = try ck.readAll(gpa, name);
    defer gpa.free(raw);
    const n = numelOf(e.shape);
    const out = try gpa.alloc(f32, n);
    if (std.mem.eql(u8, e.dtype, "F32")) {
        @memcpy(std.mem.sliceAsBytes(out), raw[0 .. n * 4]);
    } else if (std.mem.eql(u8, e.dtype, "BF16")) {
        const h = std.mem.bytesAsSlice(u16, raw);
        for (out, 0..) |*o, i| o.* = bf16ToF32(h[i]);
    } else if (std.mem.eql(u8, e.dtype, "F16")) {
        const h = std.mem.bytesAsSlice(f16, raw);
        for (out, 0..) |*o, i| o.* = h[i];
    } else return error.UnsupportedDType;
    return out;
}

fn numelOf(shape: []const i64) usize {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    return n;
}

// ---------------------------------------------------------------------------------------------------------------
// Linears as the Python q8 faces: pieces (int8 rows with group scales, or bf16 rows), adjacent ones merged.

const Rows = union(enum) { all, range: [2]usize, index: []const usize };

const Piece = struct {
    q8: bool,
    n: usize,
    k: usize,
    gs: usize = 0,
    w: ?[]u8 = null, // int8 [n, k] or bf16 [n, k]
    s: ?[]f32 = null, // [n, k / gs]

    fn free(p: *Piece, gpa: std.mem.Allocator) void {
        if (p.w) |w| gpa.free(w);
        if (p.s) |s| gpa.free(s);
        p.w = null;
        p.s = null;
    }
};

fn rowCount(rows: Rows, n: usize) usize {
    return switch (rows) {
        .all => n,
        .range => |r| r[1] - r[0],
        .index => |ix| ix.len,
    };
}

fn rowAt(rows: Rows, i: usize) usize {
    return switch (rows) {
        .all => i,
        .range => |r| r[0] + i,
        .index => |ix| ix[i],
    };
}

/// Selected rows and input columns of a row-major [n, k] array of ``elem``-byte values.
fn select(gpa: std.mem.Allocator, src: []const u8, n: usize, k: usize, elem: usize, rows: Rows, cols: ?[2]usize) ![]u8 {
    const c0 = if (cols) |c| c[0] else 0;
    const c1 = if (cols) |c| c[1] else k;
    const nr = if (rows == .all) n else rowCount(rows, n);
    const width = (c1 - c0) * elem;
    const out = try gpa.alloc(u8, nr * width);
    for (0..nr) |i| {
        const r = rowAt(rows, i);
        @memcpy(out[i * width ..][0..width], src[(r * k + c0) * elem ..][0..width]);
    }
    return out;
}

const Builder = struct {
    gpa: std.mem.Allocator,
    ck: *const Ckpt,
    dry: bool,
    rank: usize,
    around_one: bool = true,
    rope_factor: f32 = 1.0,
    rotary_dim: usize = 64,
    // output
    store: ?*api.Store = null,
    d: ?*const cuda.Driver = null,
    bytes: u64 = 0,

    // -- emitting

    fn put(b: *Builder, name: []const u8, dtype: api.DType, shape: []const i64, data: ?[]const u8) !void {
        var t: api.Tensor = .{ .ptr = 0, .dtype = dtype, .nd = @intCast(shape.len) };
        for (shape, 0..) |x, i| t.shape[i] = x;
        const n = t.bytes();
        b.bytes += n;
        if (b.dry) return;
        const store = b.store.?;
        var buf = try cuda.DeviceBuffer.alloc(b.d.?, @max(n, 1));
        errdefer buf.free();
        if (n > 0) try buf.upload(0, data.?[0..n]);
        t.ptr = buf.ptr;
        try store.owned.append(store.gpa, buf);
        try store.put(name, t);
    }

    fn putf(b: *Builder, comptime fmt: []const u8, args: anytype, dtype: api.DType, shape: []const i64, data: ?[]const u8) !void {
        var nb: [256]u8 = undefined;
        try b.put(try std.fmt.bufPrint(&nb, fmt, args), dtype, shape, data);
    }

    // -- pieces

    fn isQ8(b: *const Builder, name: []const u8) bool {
        var nb: [256]u8 = undefined;
        return b.ck.has(std.fmt.bufPrint(&nb, "{s}.qweight", .{name}) catch return false);
    }

    /// The MTP layer's bf16 linears on an affine-experts checkpoint: int8 at load (they only draft).
    fn draftQ8(b: *const Builder, name: []const u8) bool {
        return std.mem.startsWith(u8, name, "mtp.") and !b.isQ8(name);
    }

    fn q8Part(b: *Builder, name: []const u8, rows: Rows, cols: ?[2]usize) !Piece {
        var nb: [256]u8 = undefined;
        const qn = try std.fmt.bufPrint(&nb, "{s}.qweight", .{name});
        const qe = try b.ck.get(qn);
        var nb2: [256]u8 = undefined;
        const sn = try std.fmt.bufPrint(&nb2, "{s}.qscale", .{name});
        const se = try b.ck.get(sn);
        const n: usize = @intCast(qe.shape[0]);
        const k: usize = @intCast(qe.shape[1]);
        const gs = k / @as(usize, @intCast(se.shape[1]));
        const c0 = if (cols) |c| c[0] else 0;
        const c1 = if (cols) |c| c[1] else k;
        if (c0 % gs != 0 or c1 % gs != 0) return error.ColumnsCutGroups;
        const nr = if (rows == .all) n else rowCount(rows, n);
        var p: Piece = .{ .q8 = true, .n = nr, .k = c1 - c0, .gs = gs };
        if (b.dry) return p;
        const w = try b.ck.readAll(b.gpa, qn);
        defer b.gpa.free(w);
        p.w = try select(b.gpa, w, n, k, 1, rows, .{ c0, c1 });
        const s = try readF32(b.ck, b.gpa, sn);
        defer b.gpa.free(s);
        const sb = try select(b.gpa, std.mem.sliceAsBytes(s), n, k / gs, 4, rows, .{ c0 / gs, c1 / gs });
        defer b.gpa.free(sb);
        // select's bytes are u8-aligned: an f32 copy, so Piece.free frees what it allocated
        const sf = try b.gpa.alloc(f32, sb.len / 4);
        @memcpy(std.mem.sliceAsBytes(sf), sb);
        p.s = sf;
        return p;
    }

    /// A linear's weight as bf16 rows (selected rows and columns).
    fn bf16Piece(b: *Builder, name: []const u8, rows: Rows, cols: ?[2]usize) !Piece {
        var nb: [256]u8 = undefined;
        const wn = try std.fmt.bufPrint(&nb, "{s}.weight", .{name});
        const e = try b.ck.get(wn);
        const n: usize = @intCast(e.shape[0]);
        const k: usize = numelOf(e.shape) / n;
        const c0 = if (cols) |c| c[0] else 0;
        const c1 = if (cols) |c| c[1] else k;
        const nr = if (rows == .all) n else rowCount(rows, n);
        var p: Piece = .{ .q8 = false, .n = nr, .k = c1 - c0 };
        if (b.dry) return p;
        if (std.mem.eql(u8, e.dtype, "BF16")) {
            const w = try b.ck.readAll(b.gpa, wn);
            defer b.gpa.free(w);
            p.w = try select(b.gpa, w, n, k, 2, rows, .{ c0, c1 });
        } else {
            const f = try readF32(b.ck, b.gpa, wn);
            defer b.gpa.free(f);
            const h = try b.gpa.alloc(u16, f.len);
            defer b.gpa.free(h);
            for (h, f) |*o, x| o.* = f32ToBf16(x);
            p.w = try select(b.gpa, std.mem.sliceAsBytes(h), n, k, 2, rows, .{ c0, c1 });
        }
        return p;
    }

    /// q8.quantize_rows: bf16 [n, k] -> int8 with an absmax/127 scale per row and group of 64 inputs.
    fn quantizeRows(b: *Builder, p: Piece) !Piece {
        const gs: usize = 64;
        var q: Piece = .{ .q8 = true, .n = p.n, .k = p.k, .gs = gs };
        if (b.dry) return q;
        const src = std.mem.bytesAsSlice(u16, p.w.?);
        const w = try b.gpa.alloc(u8, p.n * p.k);
        const s = try b.gpa.alloc(f32, p.n * (p.k / gs));
        for (0..p.n) |r| for (0..p.k / gs) |g| {
            const row = src[r * p.k + g * gs ..][0..gs];
            var amax: f32 = 0;
            for (row) |x| amax = @max(amax, @abs(bf16ToF32(x)));
            // torch divides by a host scalar as a product with its fp32 reciprocal
            var scale: f32 = amax * (@as(f32, 1.0) / @as(f32, 127.0));
            scale = @max(scale, @as(f32, 1e-12));
            s[r * (p.k / gs) + g] = scale;
            for (row, 0..) |x, i| {
                const v = std.math.clamp(rne(bf16ToF32(x) / scale), -127.0, 127.0);
                w[r * p.k + g * gs + i] = @bitCast(@as(i8, @intFromFloat(v)));
            }
        };
        q.w = w;
        q.s = s;
        return q;
    }

    /// One linear's face piece by its storage: int8 rows (``qweight``), the MTP layer's bf16 requantized, or bf16.
    fn piece(b: *Builder, name: []const u8, rows: Rows, cols: ?[2]usize) !Piece {
        if (b.isQ8(name)) return b.q8Part(name, rows, cols);
        var p = try b.bf16Piece(name, rows, cols);
        if (!b.draftQ8(name)) return p;
        defer p.free(b.gpa);
        return b.quantizeRows(p);
    }

    /// q8.stack: adjacent int8 pieces of one group size merge, adjacent bf16 pieces merge; then each part emitted.
    fn face(b: *Builder, prefix: []const u8, pieces: []Piece) !void {
        var parts: std.ArrayList(Piece) = .empty;
        defer {
            for (parts.items) |*p| p.free(b.gpa);
            parts.deinit(b.gpa);
        }
        for (pieces) |p| {
            if (parts.items.len > 0) {
                const last = &parts.items[parts.items.len - 1];
                if (last.q8 == p.q8 and (!p.q8 or last.gs == p.gs) and last.k == p.k) {
                    const esz: usize = if (p.q8) 1 else 2;
                    if (!b.dry) {
                        const w = try b.gpa.alloc(u8, (last.n + p.n) * p.k * esz);
                        @memcpy(w[0 .. last.n * p.k * esz], last.w.?);
                        @memcpy(w[last.n * p.k * esz ..], p.w.?);
                        b.gpa.free(last.w.?);
                        last.w = w;
                        if (p.q8) {
                            const s = try b.gpa.alloc(f32, (last.n + p.n) * (p.k / p.gs));
                            @memcpy(s[0..last.s.?.len], last.s.?);
                            @memcpy(s[last.s.?.len..], p.s.?);
                            b.gpa.free(last.s.?);
                            last.s = s;
                        }
                        var pp = p;
                        pp.free(b.gpa);
                    }
                    last.n += p.n;
                    continue;
                }
            }
            try parts.append(b.gpa, p);
        }
        for (parts.items, 0..) |p, j| {
            const shape = [_]i64{ @intCast(p.n), @intCast(p.k) };
            try b.putf("{s}.parts.{d}.weight", .{ prefix, j }, if (p.q8) .i8 else .bf16, &shape,
                if (p.w) |w| w else null);
            if (p.q8) {
                const ss = [_]i64{ @intCast(p.n), @intCast(p.k / p.gs) };
                try b.putf("{s}.parts.{d}.scale", .{ prefix, j }, .f32, &ss, if (p.s) |s| std.mem.sliceAsBytes(s) else null);
            }
        }
    }

    // -- vectors

    /// A norm's scales: as stored when the checkpoint keeps them around one, else 1 + w (fp32).
    fn cscale(b: *Builder, out: []const u8, name: []const u8, rotary: bool) !void {
        var nb: [256]u8 = undefined;
        const wn = try std.fmt.bufPrint(&nb, "{s}.weight", .{name});
        const e = try b.ck.get(wn);
        const shape = [_]i64{@intCast(numelOf(e.shape))};
        if (b.dry) return b.put(out, .f32, &shape, null);
        const v = try readF32(b.ck, b.gpa, wn);
        defer b.gpa.free(v);
        if (!b.around_one) for (v) |*x| {
            x.* = x.* + 1.0;
        };
        // scale_rotary: the YaRN attention factor on every attention head's rotated dims
        if (rotary and b.rope_factor != 1.0) for (v[0..b.rotary_dim]) |*x| {
            x.* = x.* * b.rope_factor;
        };
        try b.put(out, .f32, &shape, std.mem.sliceAsBytes(v));
    }

    fn vecF32(b: *Builder, out: []const u8, name: []const u8, lo: usize, hi: usize) !void {
        const shape = [_]i64{@intCast(hi - lo)};
        if (b.dry) return b.put(out, .f32, &shape, null);
        const v = try readF32(b.ck, b.gpa, name);
        defer b.gpa.free(v);
        try b.put(out, .f32, &shape, std.mem.sliceAsBytes(v[lo..hi]));
    }

    fn rawBf16(b: *Builder, out: []const u8, name: []const u8, shape: []const i64) !void {
        if (b.dry) return b.put(out, .bf16, shape, null);
        const e = try b.ck.get(name);
        if (!std.mem.eql(u8, e.dtype, "BF16")) return error.UnsupportedDType;
        const w = try b.ck.readAll(b.gpa, name);
        defer b.gpa.free(w);
        try b.put(out, .bf16, shape, w);
    }

    // -- blocks

    fn hc(b: *Builder, out: []const u8, name: []const u8, inject: bool) !void {
        var nb: [256]u8 = undefined;
        var nb2: [256]u8 = undefined;
        var down: [2]Piece = undefined;
        down[0] = try b.piece(try std.fmt.bufPrint(&nb, "{s}.input_mix_weight_down", .{name}), .all, null);
        if (inject) down[1] = try b.piece(try std.fmt.bufPrint(&nb, "{s}.block_inject_weight", .{name}), .all, null);
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.down", .{out}), down[0 .. @as(usize, 1) + @intFromBool(inject)]);
        var up = [_]Piece{try b.piece(try std.fmt.bufPrint(&nb, "{s}.input_mix_weight_up", .{name}), .all, null)};
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.up", .{out}), &up);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.scale", .{out}), try std.fmt.bufPrint(&nb, "{s}.hc_norm", .{name}), false);
    }

    fn gdn(b: *Builder, out: []const u8, name: []const u8, c: *const Cfg) !void {
        const kl = c.nk / WORLD;
        const vl = c.nv / WORLD;
        const r = b.rank;
        // this rank's q, k and v channels of the fused projection (and of the conv)
        const channels = try b.gpa.alloc(usize, 2 * kl * c.dk + vl * c.dv);
        defer b.gpa.free(channels);
        for (0..kl * c.dk) |i| {
            channels[i] = r * kl * c.dk + i;
            channels[kl * c.dk + i] = c.nk * c.dk + r * kl * c.dk + i;
        }
        for (0..vl * c.dv) |i| channels[2 * kl * c.dk + i] = 2 * c.nk * c.dk + r * vl * c.dv + i;
        var nb: [256]u8 = undefined;
        var nb2: [256]u8 = undefined;
        var proj = [_]Piece{
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.in_proj_qkv", .{name}), .{ .index = channels }, null),
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.in_proj_z", .{name}), .{ .range = .{ r * vl * c.dv, (r + 1) * vl * c.dv } }, null),
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.in_proj_b", .{name}), .{ .range = .{ r * vl, (r + 1) * vl } }, null),
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.in_proj_a", .{name}), .{ .range = .{ r * vl, (r + 1) * vl } }, null),
        };
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.proj", .{out}), &proj);
        // conv: [conv_dim, taps] bf16, this rank's channels
        {
            const cn = try std.fmt.bufPrint(&nb, "{s}.conv1d.weight", .{name});
            const shape = [_]i64{ @intCast(channels.len), @intCast(c.conv_kernel) };
            if (b.dry) {
                try b.putf("{s}.conv", .{out}, .bf16, &shape, null);
            } else {
                const w = try b.ck.readAll(b.gpa, cn);
                defer b.gpa.free(w);
                const sel = try select(b.gpa, w, c.conv_dim, c.conv_kernel, 2, .{ .index = channels }, null);
                defer b.gpa.free(sel);
                try b.putf("{s}.conv", .{out}, .bf16, &shape, sel);
            }
        }
        try b.vecF32(try std.fmt.bufPrint(&nb2, "{s}.a_log", .{out}), try std.fmt.bufPrint(&nb, "{s}.A_log", .{name}), r * vl, (r + 1) * vl);
        try b.vecF32(try std.fmt.bufPrint(&nb2, "{s}.dt_bias", .{out}), try std.fmt.bufPrint(&nb, "{s}.dt_bias", .{name}), r * vl, (r + 1) * vl);
        try b.rawBf16(try std.fmt.bufPrint(&nb2, "{s}.norm", .{out}), try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{name}), &.{@intCast(c.dv)});
        var o = [_]Piece{try b.piece(try std.fmt.bufPrint(&nb, "{s}.out_proj", .{name}), .all, .{ r * vl * c.dv, (r + 1) * vl * c.dv })};
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.out", .{out}), &o);
    }

    fn attention(b: *Builder, out: []const u8, name: []const u8, c: *const Cfg) !void {
        const hd = c.head_dim;
        const hl = c.heads / WORLD;
        const kl = c.kv_heads / WORLD;
        const r = b.rank;
        var nb: [256]u8 = undefined;
        var nb2: [256]u8 = undefined;
        var proj = [_]Piece{
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.q_proj", .{name}), .{ .range = .{ r * hl * 2 * hd, (r + 1) * hl * 2 * hd } }, null),
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.k_proj", .{name}), .{ .range = .{ r * kl * hd, (r + 1) * kl * hd } }, null),
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.v_proj", .{name}), .{ .range = .{ r * kl * hd, (r + 1) * kl * hd } }, null),
            try b.piece(try std.fmt.bufPrint(&nb, "{s}.indexer.index_qk_proj", .{name}), .all, null),
        };
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.proj", .{out}), &proj);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.q_scale", .{out}), try std.fmt.bufPrint(&nb, "{s}.q_norm", .{name}), true);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.k_scale", .{out}), try std.fmt.bufPrint(&nb, "{s}.k_norm", .{name}), true);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.iq_scale", .{out}), try std.fmt.bufPrint(&nb, "{s}.indexer.q_layernorm", .{name}), true);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.ik_scale", .{out}), try std.fmt.bufPrint(&nb, "{s}.indexer.k_layernorm", .{name}), true);
        var o = [_]Piece{try b.piece(try std.fmt.bufPrint(&nb, "{s}.o_proj", .{name}), .all, .{ r * hl * hd, (r + 1) * hl * hd })};
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.o", .{out}), &o);
    }

    fn moe(b: *Builder, out: []const u8, name: []const u8, c: *const Cfg) !void {
        var nb: [256]u8 = undefined;
        var nb2: [256]u8 = undefined;
        // router: the experts' gate rows, then the shared expert's gate row
        {
            const shape = [_]i64{ @intCast(c.experts + 1), @intCast(c.hidden) };
            if (b.dry) {
                try b.putf("{s}.router", .{out}, .bf16, &shape, null);
            } else {
                const g = try b.ck.readAll(b.gpa, try std.fmt.bufPrint(&nb, "{s}.gate.weight", .{name}));
                defer b.gpa.free(g);
                const sg = try b.ck.readAll(b.gpa, try std.fmt.bufPrint(&nb, "{s}.shared_expert_gate.weight", .{name}));
                defer b.gpa.free(sg);
                const all = try b.gpa.alloc(u8, g.len + sg.len);
                defer b.gpa.free(all);
                @memcpy(all[0..g.len], g);
                @memcpy(all[g.len..], sg);
                try b.putf("{s}.router", .{out}, .bf16, &shape, all);
            }
        }
        const lo = b.rank * c.moe_width / WORLD;
        const hi = (b.rank + 1) * c.moe_width / WORLD;
        try b.routed(try std.fmt.bufPrint(&nb2, "{s}.experts.routed_experts", .{out}), try std.fmt.bufPrint(&nb, "{s}.switch_mlp", .{name}), c, lo, hi);
        const se_gate = try std.fmt.allocPrint(b.gpa, "{s}.shared_expert.gate_proj", .{name});
        defer b.gpa.free(se_gate);
        const se_up = try std.fmt.allocPrint(b.gpa, "{s}.shared_expert.up_proj", .{name});
        defer b.gpa.free(se_up);
        const se_down = try std.fmt.allocPrint(b.gpa, "{s}.shared_expert.down_proj", .{name});
        defer b.gpa.free(se_down);
        if (b.isQ8(se_gate)) {
            var gu = [_]Piece{ try b.q8Part(se_gate, .{ .range = .{ lo, hi } }, null), try b.q8Part(se_up, .{ .range = .{ lo, hi } }, null) };
            try b.face(try std.fmt.bufPrint(&nb2, "{s}.experts.shared_gu", .{out}), &gu);
            var dn = [_]Piece{try b.q8Part(se_down, .all, .{ lo, hi })};
            try b.face(try std.fmt.bufPrint(&nb2, "{s}.experts.shared_down", .{out}), &dn);
        } else {
            // the MTP layer's shared expert: [gate | up] rows and the down columns requantized (drafts only)
            var g = try b.bf16Piece(se_gate, .{ .range = .{ lo, hi } }, null);
            defer g.free(b.gpa);
            var u = try b.bf16Piece(se_up, .{ .range = .{ lo, hi } }, null);
            defer u.free(b.gpa);
            var cat: Piece = .{ .q8 = false, .n = g.n + u.n, .k = g.k };
            if (!b.dry) {
                const w = try b.gpa.alloc(u8, g.w.?.len + u.w.?.len);
                @memcpy(w[0..g.w.?.len], g.w.?);
                @memcpy(w[g.w.?.len..], u.w.?);
                cat.w = w;
            }
            defer cat.free(b.gpa);
            var gu = [_]Piece{try b.quantizeRows(cat)};
            try b.face(try std.fmt.bufPrint(&nb2, "{s}.experts.shared_gu", .{out}), &gu);
            var d = try b.bf16Piece(se_down, .all, .{ lo, hi });
            defer d.free(b.gpa);
            var dn = [_]Piece{try b.quantizeRows(d)};
            try b.face(try std.fmt.bufPrint(&nb2, "{s}.experts.shared_down", .{out}), &dn);
        }
    }

    /// grouped.make: gate and up rows [lo, hi) and down input columns [lo, hi), packed into the experts' blocks.
    fn routed(b: *Builder, out: []const u8, name: []const u8, c: *const Cfg, lo: usize, hi: usize) !void {
        const gs = c.group_size;
        const h = gs / 32;
        const block = 32 * 4 * h + 32;
        const e = c.experts;
        const d = c.hidden;
        const ni = hi - lo;
        const up_shape = [_]i64{ @intCast(e), @intCast(ni / 32), @intCast(d / gs), 2, @intCast(block) };
        const dn_shape = [_]i64{ @intCast(e), @intCast(d / 32), @intCast(ni / gs), 1, @intCast(block) };
        var nb: [256]u8 = undefined;
        if (b.dry) {
            try b.putf("{s}.up", .{out}, .i32, &up_shape, null);
            try b.putf("{s}.down", .{out}, .i32, &dn_shape, null);
            return;
        }
        // gate and up: rows [lo, hi) of every expert ([E, N, K/8] words, [E, N, K/gs] scales and biases)
        const up = try b.gpa.alloc(u32, numelOf(&up_shape));
        defer b.gpa.free(up);
        for (0..2) |m| {
            const proj = if (m == 0) "gate_proj" else "up_proj";
            const t = try readTriple(b, try std.fmt.bufPrint(&nb, "{s}.{s}", .{ name, proj }), lo, hi, null);
            defer t.free(b.gpa);
            packExperts(t, gs, up, 2, m);
        }
        try b.putf("{s}.up", .{out}, .i32, &up_shape, std.mem.sliceAsBytes(up));
        // down: input columns [lo, hi) (whole groups) of every expert's rows
        const dn = try b.gpa.alloc(u32, numelOf(&dn_shape));
        defer b.gpa.free(dn);
        const t = try readTriple(b, try std.fmt.bufPrint(&nb, "{s}.down_proj", .{name}), 0, d, .{ lo, hi });
        defer t.free(b.gpa);
        packExperts(t, gs, dn, 1, 0);
        try b.putf("{s}.down", .{out}, .i32, &dn_shape, std.mem.sliceAsBytes(dn));
    }
};

/// One stacked-experts MLX array triple, rows [r0, r1) and input columns ``cols`` (whole groups) of every expert.
const Triple = struct {
    e: usize,
    n: usize,
    k8: usize,
    kg: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,

    fn free(t: Triple, gpa: std.mem.Allocator) void {
        gpa.free(t.words);
        gpa.free(t.scales);
        gpa.free(t.biases);
    }
};

fn readTriple(b: *Builder, name: []const u8, r0: usize, r1: usize, cols: ?[2]usize) !Triple {
    var nb: [256]u8 = undefined;
    const we = try b.ck.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{name}));
    const e: usize = @intCast(we.shape[0]);
    const n: usize = @intCast(we.shape[1]);
    const k8: usize = @intCast(we.shape[2]);
    var nb2: [256]u8 = undefined;
    const se = try b.ck.get(try std.fmt.bufPrint(&nb2, "{s}.scales", .{name}));
    const kg: usize = @intCast(se.shape[2]);
    const gs = k8 * 8 / kg;
    const c0 = if (cols) |c| c[0] else 0;
    const c1 = if (cols) |c| c[1] else k8 * 8;
    const nr = r1 - r0;
    const ok8 = (c1 - c0) / 8;
    const okg = (c1 - c0) / gs;
    var t: Triple = .{ .e = e, .n = nr, .k8 = ok8, .kg = okg, .words = try b.gpa.alloc(u32, e * nr * ok8),
        .scales = try b.gpa.alloc(u16, e * nr * okg), .biases = try b.gpa.alloc(u16, e * nr * okg) };
    errdefer t.free(b.gpa);
    var nb3: [256]u8 = undefined;
    const be = try b.ck.get(try std.fmt.bufPrint(&nb3, "{s}.biases", .{name}));
    // one expert's rows at a time (whole rows read, the columns cut on the host)
    const rowbuf = try b.gpa.alloc(u8, nr * k8 * 4);
    defer b.gpa.free(rowbuf);
    const sbuf = try b.gpa.alloc(u8, nr * kg * 2);
    defer b.gpa.free(sbuf);
    for (0..e) |x| {
        try b.ck.readAt(we, ((x * n) + r0) * k8 * 4, rowbuf);
        const w32 = std.mem.bytesAsSlice(u32, rowbuf);
        for (0..nr) |r| @memcpy(t.words[(x * nr + r) * ok8 ..][0..ok8], w32[r * k8 + c0 / 8 ..][0..ok8]);
        for ([_]Entry{ se, be }, 0..) |ent, which| {
            try b.ck.readAt(ent, ((x * n) + r0) * kg * 2, sbuf);
            const s16 = std.mem.bytesAsSlice(u16, sbuf);
            const dst = if (which == 0) t.scales else t.biases;
            for (0..nr) |r| @memcpy(dst[(x * nr + r) * okg ..][0..okg], s16[r * kg + c0 / gs ..][0..okg]);
        }
    }
    return t;
}

/// experts_pack.cu's index map: each (group, column block, expert) block holds 32 * 4 * H weight words (nibbles
/// reordered i0 i2 i4 i6 i1 i3 i5 i7) then 32 scale/bias words; written as slot ``m`` of ``mcount`` per block.
fn packExperts(t: Triple, gs: usize, out: []u32, mcount: usize, m: usize) void {
    const h = gs / 32;
    const Work = struct {
        fn run(tt: Triple, hh: usize, o: []u32, mc: usize, mm: usize, e0: usize, e1: usize) void {
            const wp = 32 * 4 * hh;
            const blk = wp + 32;
            const nbb = tt.n / 32;
            for (e0..e1) |e| for (0..nbb) |bb| for (0..tt.kg) |g| {
                const dst = o[(((e * nbb + bb) * tt.kg + g) * mc + mm) * blk ..][0..blk];
                for (0..wp) |i| {
                    const row = i / (4 * hh);
                    const kk = i % (4 * hh);
                    const tq = row / 8;
                    const r = row % 8;
                    const q = kk / hh;
                    const j = kk % hh;
                    const w = tt.words[(e * tt.n + bb * 32 + row) * tt.k8 + g * 4 * hh + kk];
                    const f = ((tq * hh + j) * 8 + r) * 4 + q;
                    dst[(f / 128) * 128 + (f % 32) * 4 + (f / 32) % 4] = shuffle(w);
                }
                for (0..32) |s| {
                    const p = s / 8;
                    const tt4 = s % 4;
                    const src = if ((s / 4) % 2 == 1) tt.biases else tt.scales;
                    const at = (e * tt.n + bb * 32 + tt4 * 8 + p * 2) * tt.kg + g;
                    dst[wp + s] = @as(u32, src[at]) | (@as(u32, src[at + tt.kg]) << 16);
                }
            };
        }
    };
    const threads = 16;
    var pool: [threads]?std.Thread = @splat(null);
    const per = (t.e + threads - 1) / threads;
    for (0..threads) |i| {
        const e0 = i * per;
        const e1 = @min(t.e, e0 + per);
        if (e0 >= e1) break;
        pool[i] = std.Thread.spawn(.{}, Work.run, .{ t, h, out, mcount, m, e0, e1 }) catch blk: {
            Work.run(t, h, out, mcount, m, e0, e1);
            break :blk null;
        };
    }
    for (pool) |th| if (th) |x| x.join();
}

fn shuffle(w: u32) u32 {
    var even = w & 0x0F0F0F0F;
    var odd = (w >> 4) & 0x0F0F0F0F;
    even = (even | (even >> 4)) & 0x00FF00FF;
    odd = (odd | (odd >> 4)) & 0x00FF00FF;
    even = (even | (even >> 8)) & 0x0000FFFF;
    odd = (odd | (odd >> 8)) & 0x0000FFFF;
    return even | (odd << 16);
}

// ---------------------------------------------------------------------------------------------------------------
// The text config (the fields this checkpoint's preparation reads).

const Cfg = struct {
    hidden: usize,
    layers: usize,
    linear: [64]bool = @splat(false),
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    rotary_dim: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    conv_kernel: usize,
    conv_dim: usize,
    experts: usize,
    moe_width: usize,
    streams: usize,
    ple_layers: [8]usize = @splat(0),
    ple_count: usize = 0,
    ple_dim: usize,
    ngram_size: usize,
    per_ngram: usize,
    divisor: usize,
    shards: usize,
    ple_eos: i64,
    group_size: usize,
    yarn: ?Yarn = null,

    const Yarn = struct { factor: f64, original: f64, beta_fast: f64, beta_slow: f64, truncate: bool, attention: ?f64, mscale: ?f64, mscale_all: ?f64 };

    fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Cfg {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const t = (root.object.get("text_config") orelse root).object;
        const q = (root.object.get("quantization_config") orelse return error.NotAffineExperts).object;
        if (!std.mem.eql(u8, q.get("quant_method").?.string, "affine-experts")) return error.NotAffineExperts;
        const rope = t.get("rope_parameters").?.object;
        const int = struct {
            fn f(o: std.json.ObjectMap, k: []const u8) usize {
                return @intCast(o.get(k).?.integer);
            }
            fn fl(v: std.json.Value) f64 {
                return switch (v) {
                    .integer => |x| @floatFromInt(x),
                    .float => |x| x,
                    else => 0,
                };
            }
        };
        const head_dim = int.f(t, "head_dim");
        const partial = int.fl(rope.get("partial_rotary_factor") orelse std.json.Value{ .float = 0.25 });
        var c: Cfg = .{
            .hidden = int.f(t, "hidden_size"), .layers = int.f(t, "num_hidden_layers"), .vocab = int.f(t, "vocab_size"),
            .heads = int.f(t, "num_attention_heads"), .kv_heads = int.f(t, "num_key_value_heads"), .head_dim = head_dim,
            .rope_theta = int.fl(rope.get("rope_theta").?), .rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * partial),
            .nk = int.f(t, "linear_num_key_heads"), .nv = int.f(t, "linear_num_value_heads"),
            .dk = int.f(t, "linear_key_head_dim"), .dv = int.f(t, "linear_value_head_dim"),
            .conv_kernel = int.f(t, "linear_conv_kernel_dim"), .conv_dim = 0, .experts = int.f(t, "num_experts"),
            .moe_width = int.f(t, "moe_intermediate_size"), .streams = int.f(t, "hc_count"),
            .ple_dim = int.f(t, "ple_embed_dim"), .ngram_size = int.f(t, "ngram_size"),
            .per_ngram = int.f(t, "heads_per_ngram"), .divisor = int.f(t, "make_ngram_vocab_size_divisible_by"),
            .shards = int.f(t, "split_ngram_parts"), .ple_eos = t.get("eos_token_id").?.integer,
            .group_size = @intCast(q.get("group_size").?.integer),
        };
        c.conv_dim = 2 * c.nk * c.dk + c.nv * c.dv;
        for (t.get("layer_types").?.array.items, 0..) |v, i| c.linear[i] = std.mem.eql(u8, v.string, "linear_attention");
        for (t.get("ple_layer_ids").?.array.items) |v| {
            c.ple_layers[c.ple_count] = @intCast(v.integer - 1);
            c.ple_count += 1;
        }
        const kind = if (rope.get("rope_type")) |v| v.string else "default";
        if (std.mem.eql(u8, kind, "yarn")) {
            c.yarn = .{ .factor = int.fl(rope.get("factor").?), .original = int.fl(rope.get("original_max_position_embeddings").?),
                .beta_fast = if (rope.get("beta_fast")) |v| int.fl(v) else 32, .beta_slow = if (rope.get("beta_slow")) |v| int.fl(v) else 1,
                .truncate = if (rope.get("truncate")) |v| v.bool else true,
                .attention = if (rope.get("attention_factor")) |v| int.fl(v) else null,
                .mscale = if (rope.get("mscale")) |v| int.fl(v) else null, .mscale_all = if (rope.get("mscale_all_dim")) |v| int.fl(v) else null };
        }
        return c;
    }

    fn isPle(c: *const Cfg, i: usize) bool {
        for (c.ple_layers[0..c.ple_count]) |x| if (x == i) return true;
        return false;
    }
};

/// weight_types.rotary_inv_freq (float64, then fp32) and its attention factor.
fn rotaryInv(c: *const Cfg, out: []f32) f64 {
    const dim: f64 = @floatFromInt(c.rotary_dim);
    const half = c.rotary_dim / 2;
    var plain: [256]f64 = undefined;
    for (0..half) |i| plain[i] = 1.0 / std.math.pow(f64, c.rope_theta, @as(f64, @floatFromInt(2 * i)) / dim);
    const y = c.yarn orelse {
        for (0..half) |i| out[i] = @floatCast(plain[i]);
        return 1.0;
    };
    const mscaleOf = struct {
        fn f(scale: f64, m: f64) f64 {
            return if (scale <= 1) 1.0 else 0.1 * m * @log(scale) + 1.0;
        }
    };
    const attention = y.attention orelse if (y.mscale != null and y.mscale_all != null and y.mscale.? != 0 and y.mscale_all.? != 0)
        mscaleOf.f(y.factor, y.mscale.?) / mscaleOf.f(y.factor, y.mscale_all.?) else mscaleOf.f(y.factor, 1.0);
    const corr = struct {
        fn f(d: f64, original: f64, base: f64, rot: f64) f64 {
            return (d * @log(original / (rot * 2 * std.math.pi))) / (2 * @log(base));
        }
    };
    var low = corr.f(dim, y.original, c.rope_theta, y.beta_fast);
    var high = corr.f(dim, y.original, c.rope_theta, y.beta_slow);
    if (y.truncate) {
        low = @floor(low);
        high = @ceil(high);
    }
    low = @max(low, 0);
    high = @min(high, dim - 1);
    if (low == high) high += 0.001;
    for (0..half) |i| {
        const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / (high - low), 0, 1);
        const extrapolation = 1 - ramp;
        const inv = (plain[i] / y.factor) * (1 - extrapolation) + plain[i] * extrapolation;
        out[i] = @floatCast(inv);
    }
    return attention;
}

// ---------------------------------------------------------------------------------------------------------------
// The whole preparation (``dry``: shapes only, for the device bytes).

fn build(b: *Builder, c: *const Cfg) !void {
    var nb: [256]u8 = undefined;
    // norms stored around one, or centred (add one): the attention hyper-connections' mean decides
    if (!b.dry) {
        var ones: usize = 0;
        var seen: usize = 0;
        for (0..c.layers) |i| {
            const name = try std.fmt.bufPrint(&nb, BASE ++ "layers.{d}.attn_hyper_connection.hc_norm.weight", .{i});
            if (!b.ck.has(name)) continue;
            const v = try readF32(b.ck, b.gpa, name);
            defer b.gpa.free(v);
            var s: f64 = 0;
            for (v) |x| s += x;
            if (s / @as(f64, @floatFromInt(v.len)) > 0.5) ones += 1;
            seen += 1;
        }
        b.around_one = seen == 0 or ones * 10 >= seen * 9;
    }
    var inv: [128]f32 = undefined;
    b.rotary_dim = c.rotary_dim;
    b.rope_factor = @floatCast(rotaryInv(c, &inv));

    try b.rawBf16("embed.0", BASE ++ "embed_tokens.weight", &.{ @intCast(c.vocab), @intCast(c.hidden) });
    for (0..c.layers) |i| {
        const base = try std.fmt.allocPrint(b.gpa, BASE ++ "layers.{d}", .{i});
        defer b.gpa.free(base);
        const out = try std.fmt.allocPrint(b.gpa, "layers.{d}", .{i});
        defer b.gpa.free(out);
        try layer(b, c, out, base, c.linear[i], c.isPle(i));
    }
    try b.hc("mixer", BASE ++ "hyper_connection_mixer", false);
    const vl = c.vocab / WORLD;
    var head = [_]Piece{try b.q8Part("lm_head", .{ .range = .{ b.rank * vl, (b.rank + 1) * vl } }, null)};
    try b.face("head", &head);
    try b.put("inv_freq", .f32, &.{@intCast(c.rotary_dim / 2)}, std.mem.sliceAsBytes(inv[0 .. c.rotary_dim / 2]));
    // the MTP head
    try b.cscale("mtp.norm_e", "mtp.pre_fc_norm_embedding", false);
    try b.cscale("mtp.norm_h", "mtp.pre_fc_norm_hidden", false);
    var fe = [_]Piece{try b.piece("mtp.fc_embedding", .all, null)};
    try b.face("mtp.fc_e", &fe);
    var fh = [_]Piece{try b.piece("mtp.fc_hidden", .all, null)};
    try b.face("mtp.fc_h", &fh);
    try layer(b, c, "mtp.layer", "mtp.layers.0", false, false);
    try b.hc("mtp.mixer", "mtp.hyper_connection_mixer", false);
    try draftHead(b, c);
}

fn layer(b: *Builder, c: *const Cfg, out: []const u8, base: []const u8, linear: bool, ple: bool) !void {
    var nb: [256]u8 = undefined;
    var nb2: [256]u8 = undefined;
    try b.hc(try std.fmt.bufPrint(&nb2, "{s}.attn_hc", .{out}), try std.fmt.bufPrint(&nb, "{s}.attn_hyper_connection", .{base}), true);
    try b.hc(try std.fmt.bufPrint(&nb2, "{s}.mlp_hc", .{out}), try std.fmt.bufPrint(&nb, "{s}.mlp_hyper_connection", .{base}), true);
    if (linear) {
        try b.gdn(try std.fmt.bufPrint(&nb2, "{s}.gdn", .{out}), try std.fmt.bufPrint(&nb, "{s}.linear_attn", .{base}), c);
    } else {
        try b.attention(try std.fmt.bufPrint(&nb2, "{s}.attn", .{out}), try std.fmt.bufPrint(&nb, "{s}.self_attn", .{base}), c);
    }
    try b.moe(try std.fmt.bufPrint(&nb2, "{s}.moe", .{out}), try std.fmt.bufPrint(&nb, "{s}.mlp", .{base}), c);
    if (ple) {
        const p = try std.fmt.allocPrint(b.gpa, "{s}.ple", .{base});
        defer b.gpa.free(p);
        var key = [_]Piece{try b.piece(try std.fmt.bufPrint(&nb, "{s}.key_proj", .{p}), .all, null)};
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.ple.key", .{out}), &key);
        var val = [_]Piece{try b.piece(try std.fmt.bufPrint(&nb, "{s}.value_proj", .{p}), .all, null)};
        try b.face(try std.fmt.bufPrint(&nb2, "{s}.ple.value", .{out}), &val);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.ple.norm_key", .{out}), try std.fmt.bufPrint(&nb, "{s}.norm_key", .{p}), false);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.ple.norm_query", .{out}), try std.fmt.bufPrint(&nb, "{s}.norm_query", .{p}), false);
        try b.cscale(try std.fmt.bufPrint(&nb2, "{s}.ple.norm_conv", .{out}), try std.fmt.bufPrint(&nb, "{s}.norm_conv", .{p}), false);
        try b.rawBf16(try std.fmt.bufPrint(&nb2, "{s}.ple.conv", .{out}), try std.fmt.bufPrint(&nb, "{s}.conv1d.weight", .{p}),
            &.{ @intCast(c.streams * c.hidden), @intCast(c.conv_kernel) });
    }
}

/// This rank's share of the draft vocabulary (the sorted unique default ids below the vocabulary, split in two).
fn draftIds(gpa: std.mem.Allocator, vocab: usize, rank: usize) ![]i64 {
    var all: std.ArrayList(i64) = .empty;
    defer all.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, DRAFT_VOCAB, " \r\n\t");
    while (it.next()) |tok| try all.append(gpa, try std.fmt.parseInt(i64, tok, 10));
    std.mem.sort(i64, all.items, {}, std.sort.asc(i64));
    var uniq: std.ArrayList(i64) = .empty;
    defer uniq.deinit(gpa);
    for (all.items) |x| {
        if (x >= vocab) continue;
        if (uniq.items.len == 0 or uniq.items[uniq.items.len - 1] != x) try uniq.append(gpa, x);
    }
    // np.array_split: the first len % world parts take one more
    const n = uniq.items.len;
    const base = n / WORLD;
    const extra = n % WORLD;
    var start: usize = 0;
    for (0..rank) |r| start += base + @intFromBool(r < extra);
    const len = base + @intFromBool(rank < extra);
    return gpa.dupe(i64, uniq.items[start .. start + len]);
}

/// The MTP drafts' head: the int8 lm_head's draft rows dequantized to bf16, then MLX affine 4-bit in groups of 32
/// (bf16.quantize4), packed as kernels/qmm.py's pack lays its fragments out (rows padded to 128).
fn draftHead(b: *Builder, c: *const Cfg) !void {
    const ids = try draftIds(b.gpa, c.vocab, b.rank);
    defer b.gpa.free(ids);
    const n = ids.len;
    const k = c.hidden;
    const kg = k / 32;
    const npad = (n + 127) / 128 * 128;
    const wshape = [_]i64{ @intCast(npad / 64), @intCast(kg), 8, 32, 1 };
    const sshape = [_]i64{ @intCast(kg), @intCast(npad) };
    if (b.dry) {
        try b.put("draft_head.weight", .i32, &wshape, null);
        try b.put("draft_head.scales", .bf16, &sshape, null);
        try b.put("draft_head.biases", .bf16, &sshape, null);
        try b.put("draft_ids", .i64, &.{@intCast(n)}, null);
        return;
    }
    const qe = try b.ck.get("lm_head.qweight");
    const se = try b.ck.get("lm_head.qscale");
    const sk: usize = @intCast(se.shape[1]);
    const gs = k / sk;
    const qrow = try b.gpa.alloc(u8, k);
    defer b.gpa.free(qrow);
    const srow = try b.gpa.alloc(u8, sk * 4);
    defer b.gpa.free(srow);
    const words = try b.gpa.alloc(u32, n * (k / 8));
    defer b.gpa.free(words);
    const scales = try b.gpa.alloc(u16, kg * npad);
    defer b.gpa.free(scales);
    const biases = try b.gpa.alloc(u16, kg * npad);
    defer b.gpa.free(biases);
    @memset(scales, 0);
    @memset(biases, 0);
    var row: [8192]f32 = undefined;
    for (ids, 0..) |id, r| {
        const at: usize = @intCast(id);
        try b.ck.readAt(qe, at * k, qrow);
        try b.ck.readAt(se, at * sk * 4, srow);
        const s = std.mem.bytesAsSlice(f32, srow);
        // weight_bf16: q * s in fp32, then bf16
        for (0..k) |i| row[i] = bf16ToF32(f32ToBf16(@as(f32, @floatFromInt(@as(i8, @bitCast(qrow[i])))) * s[i / gs]));
        for (0..kg) |g| {
            const grp = row[g * 32 ..][0..32];
            var mn: f32 = grp[0];
            var mx: f32 = grp[0];
            for (grp) |x| {
                mn = @min(mn, x);
                mx = @max(mx, x);
            }
            var sc: f32 = (mx - mn) * (@as(f32, 1.0) / @as(f32, 15.0));
            sc = @max(sc, @as(f32, 1e-8));
            var q: [32]u32 = undefined;
            for (grp, 0..) |x, i| q[i] = @intFromFloat(std.math.clamp(rne((x - mn) / sc), 0.0, 15.0));
            for (0..4) |wd| {
                var w: u32 = 0;
                for (0..8) |nib| w |= q[wd * 8 + nib] << @intCast(4 * nib);
                words[r * (k / 8) + g * 4 + wd] = w;
            }
            scales[g * npad + r] = f32ToBf16(sc);
            biases[g * npad + r] = f32ToBf16(mn);
        }
    }
    // the fragments: out[T][g][j][r * 4 + c] = the 8 nibbles at OFFSETS of column T * 64 + j * 8 + r, pair c
    const offsets = [_]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };
    const out = try b.gpa.alloc(u32, numelOf(&wshape));
    defer b.gpa.free(out);
    for (0..npad / 64) |t| for (0..kg) |g| for (0..8) |j| for (0..8) |rr| for (0..4) |cc| {
        const col = t * 64 + j * 8 + rr;
        var v: u32 = 0;
        if (col < n) for (offsets, 0..) |off, o| {
            const e = g * 32 + 2 * cc + off;
            const nib = (words[col * (k / 8) + e / 8] >> @intCast(4 * (e % 8))) & 0xF;
            v |= nib << @intCast(4 * o);
        };
        out[(((t * kg + g) * 8 + j) * 32) + rr * 4 + cc] = v;
    };
    try b.put("draft_head.weight", .i32, &wshape, std.mem.sliceAsBytes(out));
    try b.put("draft_head.scales", .bf16, &sshape, std.mem.sliceAsBytes(scales));
    try b.put("draft_head.biases", .bf16, &sshape, std.mem.sliceAsBytes(biases));
    try b.put("draft_ids", .i64, &.{@intCast(n)}, std.mem.sliceAsBytes(ids));
    // host copies: the ids as i32 (the drafter maps its columns with them) and their count
    const store = b.store.?;
    const i32s = try store.gpa.alloc(u8, n * 4);
    for (ids, 0..) |id, i| std.mem.writeInt(i32, i32s[i * 4 ..][0..4], @intCast(id), .little);
    try store.host.put(try store.gpa.dupe(u8, "draft_ids"), i32s);
    try store.ints.put(try store.gpa.dupe(u8, "draft_count"), @intCast(n));
}

// ---------------------------------------------------------------------------------------------------------------
// The n-gram table on the host: its geometry, shard files and the e4m3 lookup table (fn_ngram's ngram.json).

fn e4m3(code: u8) f32 {
    const s: f32 = if (code & 0x80 != 0) -1 else 1;
    const e: i32 = (code >> 3) & 0xF;
    const m: f32 = @floatFromInt(code & 7);
    if (e == 0) return s * m / 8.0 * std.math.pow(f32, 2, -6);
    if (e == 15 and code & 7 == 7) return std.math.nan(f32);
    return s * (1 + m / 8.0) * std.math.pow(f32, 2, @floatFromInt(e - 7));
}

fn ngramJson(gpa: std.mem.Allocator, ck: *const Ckpt, c: *const Cfg, rank: usize) ![]u8 {
    if (c.ple_count == 0) return error.NoNgramLayer;
    const li = c.ple_layers[0];
    var nb: [256]u8 = undefined;
    const pe = try std.fmt.bufPrint(&nb, BASE ++ "layers.{d}.ple.ple_embedding.", .{li});
    const pfx = try gpa.dupe(u8, pe);
    defer gpa.free(pfx);
    const readI64 = struct {
        fn f(g: std.mem.Allocator, k: *const Ckpt, name: []const u8) ![]i64 {
            const raw = try k.readAll(g, name);
            defer g.free(raw);
            return g.dupe(i64, @alignCast(std.mem.bytesAsSlice(i64, raw)));
        }
    };
    var nb2: [256]u8 = undefined;
    const sizes = try readI64.f(gpa, ck, try std.fmt.bufPrint(&nb2, "{s}ngram_heads_vocab_sizes", .{pfx}));
    defer gpa.free(sizes);
    const offsets = try readI64.f(gpa, ck, try std.fmt.bufPrint(&nb2, "{s}ngram_heads_offsets", .{pfx}));
    defer gpa.free(offsets);
    const mult = try readI64.f(gpa, ck, try std.fmt.bufPrint(&nb2, "{s}layer_multipliers", .{pfx}));
    defer gpa.free(mult);
    const sv = try readF32(ck, gpa, try std.fmt.bufPrint(&nb2, "{s}ngram_embedding.weight_scale", .{pfx}));
    defer gpa.free(sv);
    const scale = sv[0];
    const heads = (c.ngram_size - 1) * c.per_ngram;
    var total: i64 = 0;
    for (sizes) |x| total += x;
    const div: i64 = @intCast(c.divisor);
    const rows = @divFloor(total + div - 1, div) * div;

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const w = &out.writer;
    try w.print("{{\"n\": {d}, \"per_ngram\": {d}, \"heads\": {d}, \"eos\": {d}, \"dims\": {d}, \"head_sizes\": [", .{
        c.ngram_size, c.per_ngram, heads, c.ple_eos, c.ple_dim / heads });
    for (sizes, 0..) |x, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", x });
    try w.writeAll("], \"head_offsets\": [");
    for (offsets, 0..) |x, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", x });
    try w.writeAll("], \"multipliers\": [");
    for (mult, 0..) |x, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", x });
    // shards: flat or nested spellings, in order
    var width: i64 = -1;
    var starts: std.ArrayList(i64) = .empty;
    defer starts.deinit(gpa);
    try starts.append(gpa, 0);
    var shard_text: std.Io.Writer.Allocating = .init(gpa);
    defer shard_text.deinit();
    for (0..c.shards) |i| {
        var nb3: [256]u8 = undefined;
        var key = try std.fmt.bufPrint(&nb3, "{s}ngram_embedding.shard_{d}.weight", .{ pfx, i });
        if (!ck.has(key)) key = try std.fmt.bufPrint(&nb3, "{s}ngram_embedding.shards.{d}.weight", .{ pfx, i });
        const e = try ck.get(key);
        if (!std.mem.eql(u8, e.dtype, "F8_E4M3")) return error.UnsupportedNgramShards;
        if (width < 0) width = e.shape[1];
        if (e.shape[1] != width) return error.UnevenNgramShards;
        try starts.append(gpa, starts.items[starts.items.len - 1] + e.shape[0]);
        try shard_text.writer.print("{s}{{\"file\": \"{s}\", \"offset\": {d}, \"rows\": {d}}}", .{
            if (i > 0) ", " else "", ck.paths.items[e.file], e.begin, e.shape[0] });
    }
    if (starts.items[starts.items.len - 1] != rows) return error.NgramRowsMismatch;
    try w.print("], \"width\": {d}, \"rows\": {d}, \"starts\": [", .{ width, rows });
    for (starts.items, 0..) |x, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", x });
    try w.writeAll("], \"lut\": [");
    for (0..256) |i| {
        var v: u16 = f32ToBf16(e4m3(@intCast(i)) * scale);
        if (i & 0x7F == 0x7F) v = 0x7FC0;
        try w.print("{s}{d}", .{ if (i > 0) ", " else "", v });
    }
    try w.print("], \"shards\": [{s}], \"initial_history\": [", .{shard_text.written()});
    for (0..c.ngram_size - 1) |i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", c.ple_eos });
    try w.print("], \"vocab_offset\": {d}}}", .{rank * (c.vocab / WORLD)});
    return out.toOwnedSlice();
}

// ---------------------------------------------------------------------------------------------------------------

/// This rank's prepared weights on its GPU, under the Python pack's names; host facts in ``ints`` and ``host``.
pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const api.Ctx, kernels: *api.Kernels, dir: []const u8, rank: u8) !api.Store {
    _ = kernels;
    var ck = try Ckpt.open(gpa, io, dir);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const c = try Cfg.read(arena.allocator(), io, dir);
    var store = api.Store.init(gpa);
    errdefer store.deinit();
    var b: Builder = .{ .gpa = gpa, .ck = &ck, .dry = false, .rank = rank, .store = &store, .d = ctx.d };
    try build(&b, &c);
    const vl = c.vocab / WORLD;
    const facts = [_]struct { []const u8, i64 }{
        .{ "rank", rank }, .{ "world", WORLD }, .{ "vocab", @intCast(c.vocab) }, .{ "vocab_shard", @intCast(vl) },
        .{ "vocab_offset", @intCast(rank * vl) }, .{ "around_one", @intFromBool(b.around_one) },
        .{ "rotary_dim", @intCast(c.rotary_dim) }, .{ "ple_layer", @intCast(c.ple_layers[0]) },
        .{ "ple_eos", c.ple_eos }, .{ "group_size", @intCast(c.group_size) }, .{ "device_bytes", @intCast(b.bytes) },
    };
    for (facts) |f| try store.ints.put(try gpa.dupe(u8, f[0]), f[1]);
    try store.host.put(try gpa.dupe(u8, "ngram.json"), try ngramJson(gpa, &ck, &c, rank));
    return store;
}

/// Device bytes this rank's prepared weights take (the n-gram tables stay on the host): the same preparation's
/// shapes, nothing read but headers and the config.
pub fn deviceBytes(io: std.Io, dir: []const u8) !u64 {
    const gpa = std.heap.page_allocator;
    var ck = try Ckpt.open(gpa, io, dir);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const c = try Cfg.read(arena.allocator(), io, dir);
    var b: Builder = .{ .gpa = gpa, .ck = &ck, .dry = true, .rank = 0 };
    try build(&b, &c);
    return b.bytes;
}
