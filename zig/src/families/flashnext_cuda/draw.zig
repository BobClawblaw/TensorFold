//! A sampled row's token from both ranks' gathered candidates (fn_rows_topk): each rank's 64 largest logits, global
//! ids, its log-sum-exp and its log-sum-exp at the row's temperature. The merged 128 hold the vocabulary's top 64
//! exactly, so a rule within them draws as lanes.sampling.choose over the whole vocabulary would: top_k 1..64 by
//! choose itself (only the top k take part), a wider rule when its nucleus (top_p) or floor (min_p) ends within the
//! top 64, keyed the same way (seed, position, global id). A rule that reaches past them asks for the full row.

const std = @import("std");
const lanes = @import("lanes");
const smp = lanes.sampling;

pub const TOPK = 64;
pub const W = 2 * TOPK + 4; // a row's ints in the gathered block: ids, values, lse, lse at the temperature, spare

pub const Outcome = union(enum) { token: u32, full };

const Cand = struct { value: f64, id: u64 };

fn desc(_: void, a: Cand, b: Cand) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

fn logAddExp(a: f64, b: f64) f64 {
    const m = @max(a, b);
    if (m == -std.math.inf(f64)) return m;
    return m + @log(@exp(a - m) + @exp(b - m));
}

/// ``blocks``: the two ranks' W-int rows (rank 0 first). ``position``: the draw's key.
pub fn draw(gpa: std.mem.Allocator, blocks: [2][]const i32, s: smp.Sampling, position: u64) !Outcome {
    var cands: [2 * TOPK]Cand = undefined;
    var n: usize = 0;
    var lse_t: f64 = -std.math.inf(f64);
    for (blocks) |b| {
        for (0..TOPK) |k| {
            if (b[k] < 0) continue;
            cands[n] = .{ .value = @as(f32, @bitCast(b[TOPK + k])), .id = @intCast(b[k]) };
            n += 1;
        }
        lse_t = logAddExp(lse_t, @as(f32, @bitCast(b[2 * TOPK + 1])));
    }
    std.mem.sort(Cand, cands[0..n], {}, desc);
    const exact = @min(n, TOPK);
    if (s.top_k >= 1 and s.top_k <= TOPK) {
        const k = @min(@as(usize, s.top_k), exact);
        var values: [TOPK]f64 = undefined;
        var ids: [TOPK]u64 = undefined;
        for (cands[0..k], 0..) |c, i| {
            values[i] = c.value;
            ids[i] = c.id;
        }
        return .{ .token = @intCast(try smp.choose(gpa, values[0..k], ids[0..k], position, s)) };
    }
    // the whole vocabulary's rule (top_k 0 or past the candidates): its cut must end within the exact top 64
    const t = @max(s.temperature, 1e-6);
    var keep: usize = std.math.maxInt(usize);
    if (s.top_p > 0.0 and s.top_p < 1.0) {
        var cum: f64 = 0;
        var below: usize = 0;
        for (cands[0..exact]) |c| {
            cum += @exp(c.value / t - lse_t);
            if (cum < s.top_p) below += 1 else break;
        }
        if (below >= exact) return .full;
        keep = below + 1;
    }
    if (s.min_p > 0.0) {
        const floor = cands[0].value / t + s.minLog();
        var m: usize = 0;
        for (cands[0..exact]) |c| {
            if (c.value / t >= floor) m += 1;
        }
        if (m >= exact and keep > exact) return .full;
        keep = @min(keep, m);
    }
    if (s.top_k > TOPK) keep = @min(keep, @as(usize, s.top_k));
    if (keep > exact) return .full;
    var best: usize = 0;
    var best_score = -std.math.inf(f64);
    for (cands[0..keep], 0..) |c, j| {
        const gumbel = -@log(-@log(smp.uniform(s.seed, position, c.id)));
        const score = c.value / t + gumbel;
        if (j == 0 or score > best_score) {
            best = j;
            best_score = score;
        }
    }
    return .{ .token = @intCast(cands[best].id) };
}

/// The full rule over a whole gathered row (``values``: the vocabulary's logits by global id).
pub fn drawFull(gpa: std.mem.Allocator, values: []const f64, s: smp.Sampling, position: u64) !u32 {
    const ids = try gpa.alloc(u64, values.len);
    defer gpa.free(ids);
    for (ids, 0..) |*x, i| x.* = i;
    return @intCast(try smp.choose(gpa, values, ids, position, s));
}

test "a top_k rule draws the same token from the merged candidates as choose over a whole row" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    const V = 2000;
    var row: [V]f64 = undefined;
    for (&row) |*x| x.* = @floatCast(@as(f32, @floatCast(r.floatNorm(f64) * 3)));
    // each rank's half: its top 64 as the kernel writes them
    var blocks: [2][W]i32 = undefined;
    for (0..2) |rk| {
        var c: [V / 2]Cand = undefined;
        for (0..V / 2) |i| c[i] = .{ .value = row[rk * V / 2 + i], .id = rk * V / 2 + i };
        std.mem.sort(Cand, &c, {}, desc);
        for (0..TOPK) |k| {
            blocks[rk][k] = @intCast(c[k].id);
            blocks[rk][TOPK + k] = @bitCast(@as(f32, @floatCast(c[k].value)));
        }
        blocks[rk][2 * TOPK] = 0;
        blocks[rk][2 * TOPK + 1] = 0;
    }
    for ([_]u32{ 1, 5, 20, 64 }) |k| for (0..20) |pos| {
        const s: smp.Sampling = .{ .seed = 11, .temperature = 0.8, .top_k = k, .top_p = 0.95 };
        const got = try draw(gpa, .{ &blocks[0], &blocks[1] }, s, pos);
        const want = try drawFull(gpa, &row, s, pos);
        try std.testing.expectEqual(want, got.token);
    };
}
