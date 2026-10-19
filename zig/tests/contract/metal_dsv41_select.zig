//! DeepSeek V4.1's index selection under the contract: a tile's merge into the held picks and the finish's sort.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;

fn noSums(_: c.Shape) usize {
    return 0;
}

const tile = 4096;
const topk = 512;
const candidate_topk = 2048;
const block = 8;
const rule = "a NaN or infinite score is never picked; a block's maximum may be +inf, and the newest visible block is pinned";

const Params = extern struct {
    rows: u32,
    heads: u32 = 32,
    width: u32 = 128,
    q_stride: u32 = 32 * 128,
    weight_stride: u32 = 32,
    position: u32,
    ratio: u32 = 1,
    keys: u32,
    key_stride: u32 = 68,
    tile_start: u32,
    tile_width: u32 = tile,
    topk: u32 = topk,
    candidate_topk: u32 = candidate_topk,
    block_size: u32 = block,
    use_candidates: u32 = 0,
    make_candidates: u32 = 1,
    scale: f32 = 0.015625,
};
const Pick = extern struct { score: f32, position: i32 };

fn better(a: Pick, b: Pick) bool {
    return a.score > b.score or (a.score == b.score and a.position < b.position);
}
fn byPosition(_: void, a: Pick, b: Pick) bool {
    return @as(u32, @bitCast(a.position)) < @as(u32, @bitCast(b.position)); // -1 last
}

/// The held picks after a tile: the best `k` of the held and the new (as a set), then {-inf, -1}, sorted by position.
fn best(a: std.mem.Allocator, held: []const Pick, new: []const Pick, k: usize) ![]Pick {
    var all: std.ArrayList(Pick) = .empty;
    defer all.deinit(a);
    for (held) |p| if (p.position >= 0 and p.score > -std.math.inf(f32)) try all.append(a, p);
    for (new) |p| try all.append(a, p);
    std.mem.sort(Pick, all.items, {}, struct {
        fn f(_: void, x: Pick, y: Pick) bool {
            return better(x, y);
        }
    }.f);
    const out = try a.alloc(Pick, k);
    for (out, 0..) |*o, i| o.* = if (i < all.items.len) all.items[i] else .{ .score = -std.math.inf(f32), .position = -1 };
    std.mem.sort(Pick, out, {}, byPosition);
    return out;
}

/// Rows: query rows; n: the tile's scores that are finite (the rest -inf), k: unused. The second of two tiles.
pub fn mergeSpec() c.Spec {
    return .{
        .name = "metal deepseek_v41 index merge dsv41_index_merge",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{ .size = 8 }, .{} },
        .served = &.{ .{ .rows = 1, .n = tile, .k = 1 }, .{ .rows = 64, .n = tile, .k = 1 }, .{ .rows = 3, .n = 37, .k = 1 } },
    };
}

pub fn finishSpec() c.Spec {
    return .{
        .name = "metal deepseek_v41 index finish dsv41_index_finish",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .replaces = "metal deepseek_v41 index finish dsv41_index_finish_serial",
        .tiles = .{ .{}, .{}, .{} },
        .served = &.{ .{ .rows = 1, .n = topk, .k = 1 }, .{ .rows = 64, .n = topk, .k = 1 }, .{ .rows = 3, .n = 37, .k = 1 } },
    };
}

/// The merge (`finish` false) or the finish over prepared held picks, scores and candidates.
pub const Select = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    old: mtl.Pipeline,
    finish: bool,
    shape: c.Shape = undefined,
    scores: []f32 = &.{},
    top: []Pick = &.{},
    cand: []Pick = &.{},
    refs: [4]rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, finish: bool) !Select {
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise; // as csa2_index.zig compiles it
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_kv ++ sources.deepseek_v41_index, options);
        defer lib.deinit();
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try mtl.Pipeline.init(device, lib, if (finish) "dsv41_index_finish" else "dsv41_index_merge", false),
            .old = try mtl.Pipeline.init(device, lib, if (finish) "dsv41_index_finish_serial" else "dsv41_index_merge_serial", false),
            .finish = finish,
        };
    }

    pub fn deinit(t: *Select) void {
        t.free();
        t.pipe.deinit();
        t.old.deinit();
        t.rig.deinit();
    }

    fn free(t: *Select) void {
        t.a.free(t.scores);
        t.a.free(t.top);
        t.a.free(t.cand);
        t.scores = &.{};
        t.top = &.{};
        t.cand = &.{};
    }

    pub fn kernel(t: *Select) g.Kernel {
        return .{ .spec = if (t.finish) finishSpec() else mergeSpec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Select);

    fn args(t: *const Select) Params {
        // the second tile of 8,192 keys, every key visible to the last row: its last block is the pinned one
        const rows: u32 = @intCast(t.shape.rows);
        return .{ .rows = rows, .position = 2 * tile - rows, .keys = 2 * tile, .tile_start = tile };
    }

    pub fn prepare(t: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n > @as(usize, if (t.finish) topk else tile)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        // eighths: many ties, broken by position
        const score = struct {
            fn f(rng: std.Random) f32 {
                return @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 8 - 4;
            }
        }.f;
        t.scores = try t.a.alloc(f32, shape.rows * tile);
        for (t.scores, 0..) |*v, i| v.* = if (i % tile < shape.n) score(r) else -std.math.inf(f32);
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, i| t.scores[(i * 977) % shape.n] = v;
        }
        // the held picks from an earlier tile: the first n of them real (positions below the tile), the rest unset
        t.top = try t.a.alloc(Pick, shape.rows * topk);
        t.cand = try t.a.alloc(Pick, shape.rows * candidate_topk);
        for (t.top, 0..) |*p, i| p.* = if (i % topk < @min(shape.n, topk)) .{ .score = score(r), .position = @intCast((i % topk) * 7 % tile) } else .{ .score = -std.math.inf(f32), .position = -1 };
        // held blocks from the earlier tile (ids below its 512), as a real pass holds them
        for (t.cand, 0..) |*p, i| p.* = if (i % candidate_topk < @min(shape.n / 2, tile / block)) .{ .score = score(r), .position = @intCast(i % candidate_topk) } else .{ .score = -std.math.inf(f32), .position = -1 };
        if (t.finish) for (0..shape.rows) |row| r.shuffle(Pick, t.top[row * topk ..][0..topk]);
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.scores));
        t.refs[1] = try t.rig.output(t.top.len * 8);
        t.refs[2] = try t.rig.output(t.cand.len * 8);
        t.refs[3] = try t.rig.output(shape.rows * topk * 4);
        return true;
    }

    fn launch(t: *Select, pipe: mtl.Pipeline, serial: bool) !void {
        @memcpy(t.rig.bytes(t.refs[1], t.top.len * 8), std.mem.sliceAsBytes(t.top));
        @memcpy(t.rig.bytes(t.refs[2], t.cand.len * 8), std.mem.sliceAsBytes(t.cand));
        const p = t.rig.begin();
        p.e.setPipeline(pipe);
        if (t.finish) {
            for (t.refs[1..4], 0..) |ref, i| p.e.setBuffer(ref.buf, ref.off, i);
        } else {
            for (t.refs[0..3], 0..) |ref, i| p.e.setBuffer(ref.buf, ref.off, i);
        }
        p.e.setValue(t.args(), 3);
        const rows: u32 = @intCast(t.shape.rows);
        if (serial) p.e.dispatchThreads(mtl.Size.of(rows, 1, 1), mtl.Size.of(1, 1, 1)) else p.e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn run(t: *Select) !void {
        return t.launch(t.pipe, false);
    }

    pub fn replaced(t: *Select) ![]const u8 {
        try t.launch(t.old, true);
        return t.rig.outputBytes();
    }

    pub fn count(t: *Select) usize {
        return t.shape.rows * (topk + candidate_topk) * 8 + t.shape.rows * topk * 4;
    }

    pub fn reference(_: *Select, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The finish: each row sorted by position. The merge: the best picks as sets (its order is the finish's job).
    pub fn check(t: *Select, at: g.Gate) !?[]const u8 {
        const top = std.mem.bytesAsSlice(Pick, t.rig.bytes(t.refs[1], t.top.len * 8));
        const cand = std.mem.bytesAsSlice(Pick, t.rig.bytes(t.refs[2], t.cand.len * 8));
        var reached = false;
        for (0..t.shape.rows) |row| {
            const got_top = try t.a.alloc(Pick, topk);
            defer t.a.free(got_top);
            for (got_top, top[row * topk ..][0..topk]) |*d, v| d.* = v;
            const got_cand = try t.a.alloc(Pick, candidate_topk);
            defer t.a.free(got_cand);
            for (got_cand, cand[row * candidate_topk ..][0..candidate_topk]) |*d, v| d.* = v;
            std.mem.sort(Pick, got_top, {}, byPosition);
            std.mem.sort(Pick, got_cand, {}, byPosition);
            var new: std.ArrayList(Pick) = .empty;
            defer new.deinit(t.a);
            var blocks: std.ArrayList(Pick) = .empty;
            defer blocks.deinit(t.a);
            if (!t.finish) {
                const a = t.args();
                const s = t.scores[row * tile ..][0..tile];
                for (s, 0..) |v, i| {
                    if (!std.math.isFinite(v)) reached = true;
                    if (std.math.isFinite(v)) try new.append(t.a, .{ .score = v, .position = @intCast(a.tile_start + i) });
                }
                const visible = @min(a.keys, (a.position + row + 1) / a.ratio);
                var base: usize = 0;
                while (base < tile) : (base += block) {
                    var m = -std.math.inf(f32);
                    for (s[base..][0..block]) |v| m = @max(m, v);
                    const id = (a.tile_start + base) / block;
                    if (visible > 0 and id == (visible - 1) / block) m = std.math.inf(f32);
                    if (m > -std.math.inf(f32)) try blocks.append(t.a, .{ .score = m, .position = @intCast(id) });
                }
            }
            const want_top = try best(t.a, t.top[row * topk ..][0..topk], new.items, topk);
            defer t.a.free(want_top);
            const want_cand = if (t.finish) try t.a.dupe(Pick, t.cand[row * candidate_topk ..][0..candidate_topk]) else try best(t.a, t.cand[row * candidate_topk ..][0..candidate_topk], blocks.items, candidate_topk);
            defer t.a.free(want_cand);
            if (t.finish) std.mem.sort(Pick, want_cand, {}, byPosition);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(want_top), std.mem.sliceAsBytes(got_top)))
                return try std.fmt.bufPrint(&t.msg, "row {d}: the held picks differ from the oracle's", .{row});
            if (!t.finish and !std.mem.eql(u8, std.mem.sliceAsBytes(want_cand), std.mem.sliceAsBytes(got_cand)))
                return try std.fmt.bufPrint(&t.msg, "row {d}: the candidate blocks differ from the oracle's", .{row});
            if (t.finish) {
                const pos = std.mem.bytesAsSlice(i32, t.rig.bytes(t.refs[3], t.shape.rows * topk * 4))[row * topk ..][0..topk];
                for (pos, top[row * topk ..][0..topk]) |p, h| if (p != h.position) return try std.fmt.bufPrint(&t.msg, "row {d}: positions out of step with the sorted picks", .{row});
                for (top[row * topk ..][0..topk], want_top) |h, w| if (h.position != w.position) return try std.fmt.bufPrint(&t.msg, "row {d}: not sorted by position", .{row});
            }
        }
        if (at == .nonfinite and !t.finish and !reached) return try std.fmt.bufPrint(&t.msg, "no planted score in a launched row: no proof", .{});
        return null;
    }
};
