//! Unlimited-OCR-2's CUDA moves contract, part three: the MoE's row gather and scatter and the token embedding.
const std = @import("std");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_uocr_moves_kit.zig");

const c = kit.c;
const g = kit.g;
const Kernels = kit.Kernels;
const blocks = kit.blocks;
const planted = kit.planted;
const bits = kit.bits;
const val = kit.val;
const release = kit.release;
const upload = kit.upload;
const hidden = kit.hidden;
const chunk = kit.chunk;

// --- tf_uocr_gather and tf_uocr_scatter: an expert's rows by index, y[i] = x[rows[i]], slots[dest[i]] = y[i] ---

pub const Rows = struct {
    rig: *rig_.Rig,
    k: Kernels,
    scatter: bool,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    src: []u16 = &.{},
    ids: []u32 = &.{},
    /// tf_uocr_scatter's slot rows before the launch: the rows no index names keep them.
    bg: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: expert rows (1..2048); n: hidden (1280); k: indexed table (tokens or slots); indices distinct, shuffled.
    const gather_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = chunk }, .{ .rows = chunk, .n = hidden, .k = chunk }, .{ .rows = 1, .n = hidden, .k = 1 } };
    const scatter_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = kit.slots }, .{ .rows = chunk, .n = hidden, .k = kit.slots }, .{ .rows = 1, .n = hidden, .k = kit.topk } };

    pub fn spec(t: *const Rows) c.Spec {
        return .{
            .name = t.name,
            .accum = .none,
            .depth = kit.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 64 }, .{}, .{} },
            .served = if (t.scatter) &scatter_served else &gather_served,
        };
    }
    pub fn deinit(t: *Rows) void {
        release(t.rig.gpu.gpa, t, .{ "src", "ids", "bg" });
    }
    pub fn kernel(t: *Rows) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Rows);

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hidden or shape.rows > shape.k) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        release(a, t, .{ "src", "ids", "bg" });
        t.shape = shape;
        const n = hidden;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gather reads table rows by index; scatter reads its own rows in order
        const src_rows = if (t.scatter) shape.rows else shape.k;
        t.src = try a.alloc(u16, src_rows * n);
        for (t.src) |*v| v.* = kit.finiteBits(rng);
        t.ids = try kit.distinct(a, rng, shape.rows, shape.k);
        t.bg = try a.alloc(u16, if (t.scatter) shape.k * n else 0);
        for (t.bg) |*v| v.* = kit.finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 3) {
            // the rows the first four indices move: NaN, +Inf, -Inf, both infinities
            for (0..4) |x| {
                const row = if (t.scatter) x else t.ids[x];
                if (x < 3) t.src[row * n + 11 * x] = bits(planted[x]) else {
                    t.src[row * n + 40] = bits(planted[1]);
                    t.src[row * n + n - 1] = bits(planted[2]);
                }
            }
            // gather: a NaN in a row no index names must stay out
            if (!t.scatter and shape.k > shape.rows) {
                const seen = try a.alloc(bool, shape.k);
                defer a.free(seen);
                @memset(seen, false);
                for (t.ids) |i| seen[i] = true;
                const free_row = std.mem.indexOfScalar(bool, seen, false).?;
                t.src[free_row * n] = bits(planted[0]);
            }
        }
        t.ins[0] = try upload(t.rig, t.src);
        t.ins[1] = try upload(t.rig, t.ids);
        if (t.scatter) t.ins[2] = try upload(t.rig, t.bg);
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Rows) !void {
        const grid = .{ t.shape.rows, blocks(hidden), 1 };
        if (t.scatter) {
            // the slots' rows before the launch go back before each independent run
            try kit.preset(t.rig, t.out, t.ins[2], t.bg.len * 2);
            try t.k.run("scatter", t.rig.s, grid, 256, .{ t.ins[0], t.ins[1], t.out });
        } else {
            try t.k.run("gather", t.rig.s, grid, 256, .{ t.ins[0], t.ins[1], t.out });
        }
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Rows) usize {
        return (if (t.scatter) t.shape.k else t.shape.rows) * hidden;
    }

    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        const n = hidden;
        @memset(mag, 0);
        if (t.scatter) {
            for (want, t.bg) |*o, v| o.* = val(v);
            for (t.ids, 0..) |d, i| for (0..n) |j| {
                want[d * n + j] = val(t.src[i * n + j]);
            };
            return;
        }
        for (want, 0..) |*o, i| o.* = val(t.src[t.ids[i / n] * n + i % n]);
    }
};

// --- tf_uocr_embed: x[i] = embed[ids[i]], image tokens' rows left as they are (their pictures land there) ---------

const vocab = kit.vocab;
const image_token = kit.image_token;

pub const Embed = struct {
    /// The kernel's and its control's one rig: each prepare frees the other's 331 MB table first.
    rig: *rig_.Rig,
    k: Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    ids: []u32 = &.{},
    table: []u16 = &.{},
    /// The rows before the launch: image tokens' rows keep them.
    bg: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: a prompt chunk's tokens (2048, a 1-token tail); n: hidden; k: the table's rows (the vocabulary).
    const served = [_]c.Shape{ .{ .rows = chunk, .n = hidden, .k = vocab }, .{ .rows = 1, .n = hidden, .k = vocab } };

    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = kit.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 64 }, .{}, .{} },
            .served = &served,
        };
    }
    pub fn deinit(t: *Embed) void {
        release(t.rig.gpu.gpa, t, .{ "ids", "table", "bg" });
    }
    pub fn kernel(t: *Embed) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Embed);

    /// Every fifth row from row 2 is an image token; the rest are distinct text tokens.
    fn image(r: usize) bool {
        return r % 5 == 2;
    }

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hidden or shape.k <= image_token or shape.rows >= shape.k) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        release(a, t, .{ "ids", "table", "bg" });
        t.shape = shape;
        const n = hidden;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // distinct ids from the table less the image token's row, then the image rows marked
        t.ids = try kit.distinct(a, rng, shape.rows, shape.k - 1);
        for (t.ids, 0..) |*id, r| {
            if (id.* >= image_token) id.* += 1;
            if (image(r)) id.* = image_token;
        }
        t.table = try a.alloc(u16, shape.k * n);
        for (t.table) |*v| v.* = kit.finiteBits(rng);
        t.bg = try a.alloc(u16, shape.rows * n);
        for (t.bg) |*v| v.* = kit.finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 4) {
            // text rows 0, 1, 3 and 4 read NaN, +Inf, -Inf and both infinities from their table rows
            const text = [_]usize{ 0, 1, 3, 4 };
            for (text, 0..) |r, x| {
                const row = t.ids[r];
                if (x < 3) t.table[row * n + 11 * x] = bits(planted[x]) else {
                    t.table[row * n + 40] = bits(planted[1]);
                    t.table[row * n + n - 1] = bits(planted[2]);
                }
            }
            // the image row's NaN stays; the image token's own table row, and a row no id names, stay out
            t.bg[2 * n + 5] = bits(planted[0]);
            t.table[image_token * n + 6] = bits(planted[1]);
            const seen = try a.alloc(bool, shape.k);
            defer a.free(seen);
            @memset(seen, false);
            for (t.ids) |i| seen[i] = true;
            t.table[std.mem.indexOfScalar(bool, seen, false).? * n + 7] = bits(planted[0]);
        }
        t.ins[0] = try upload(t.rig, t.ids);
        t.ins[1] = try upload(t.rig, t.table);
        t.ins[2] = try upload(t.rig, t.bg);
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Embed) !void {
        // the image rows hold their pictures before the launch: they go back before each independent run
        try kit.preset(t.rig, t.out, t.ins[2], t.bg.len * 2);
        const args = .{ t.ins[0], t.ins[1], t.out, @as(u32, image_token) };
        try t.k.run("embed", t.rig.s, .{ t.shape.rows, blocks(hidden), 1 }, 256, args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Embed) usize {
        return t.shape.rows * hidden;
    }

    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        const n = hidden;
        @memset(mag, 0);
        for (want, 0..) |*o, i| {
            const id = t.ids[i / n];
            o.* = if (id == image_token) val(t.bg[i]) else val(t.table[id * n + i % n]);
        }
    }
};
