//! The keyed host sampler over a bf16 logits row, in one pass: the top-k candidates by an order-preserving integer key
//! of each bf16 word, then lanes.sampling.choose. The candidates, their order and the draw equal `reference`, which is
//! qwen3_5/backend.zig's draw without its mask (two passes in f64), kept here for wide top-k and the tests.
const std = @import("std");
const lanes = @import("lanes");

/// k at most this takes the one-pass path; a wider (or unlimited) top-k takes the reference's.
pub const MAX_K = 256;

/// A bf16 word's order as an unsigned key: larger floats give larger keys, -0 and +0 the same.
fn key(word: u16) u16 {
    const w = if (word == 0x8000) 0 else word;
    return if (w & 0x8000 != 0) ~w else w | 0x8000;
}

fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// qwen3_5/backend.zig's draw without a mask: the unchanged top-k candidates for the keyed sampler.
pub fn reference(gpa: std.mem.Allocator, logits: []const u16, p: lanes.Sampling, position: u64) !u32 {
    var best: ?usize = null;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteQwenLogits;
        if (best == null or x > value(logits[best.?])) best = id;
    }
    if (p.temperature <= 0) return @intCast(best.?);
    const allowed = logits.len;
    const k = if (p.top_k == 0) allowed else @min(allowed, p.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (k == allowed) {
            values[filled] = x;
            ids[filled] = id;
            filled += 1;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, p));
}

/// The draw for `logits` with `p` keyed at `position` (p.temperature above 0).
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, p: lanes.Sampling, position: u64) !u32 {
    const k: usize = p.top_k;
    if (k == 0 or k > MAX_K or k >= logits.len) return reference(gpa, logits, p, position);
    var keys: [MAX_K]u16 = undefined;
    var ids: [MAX_K]u64 = undefined;
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteQwenLogits; // exponent all ones: inf or nan
        const x = key(word);
        if (filled == k and x <= keys[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > keys[at - 1]) : (at -= 1) {
            keys[at] = keys[at - 1];
            ids[at] = ids[at - 1];
        }
        keys[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    var values: [MAX_K]f64 = undefined;
    for (ids[0..filled], values[0..filled]) |id, *v| v.* = value(logits[id]);
    return @intCast(try lanes.sampling.choose(gpa, values[0..filled], ids[0..filled], position, p));
}

test "the one-pass draw equals the reference on random rows with ties, signed zeros and every top-k" {
    const gpa = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    const row = try gpa.alloc(u16, 4096);
    defer gpa.free(row);
    for (0..40) |trial| {
        for (row) |*w| {
            const f: f32 = @floatCast(r.floatNorm(f64) * 3.0);
            w.* = @intCast(@as(u32, @bitCast(f)) >> 16);
        }
        for (0..64) |j| row[r.uintLessThan(usize, row.len)] = row[j]; // ties
        row[r.uintLessThan(usize, row.len)] = 0x8000;
        row[r.uintLessThan(usize, row.len)] = 0;
        for ([_]u32{ 1, 2, 20, 64, 256 }) |k| for ([_]f64{ 0.6, 1.0, 1.5 }) |t| {
            const p: lanes.Sampling = .{ .seed = trial * 31 + k, .temperature = t, .top_k = k, .top_p = 0.95 };
            const pos: u64 = trial * 7 + k;
            try std.testing.expectEqual(try reference(gpa, row, p, pos), try draw(gpa, row, p, pos));
        };
    }
}
