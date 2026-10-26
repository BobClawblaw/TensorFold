//! MiniMax-M3.1-Flash's glue kernels on CUDA (minimax_m3.cu): the Metal family's glue.metal launches, one by one.

const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

pub const Kernels = struct {
    mod: cuda.Module,
    attn_mod: cuda.Module, // minimax_m3_attn: the indexer and the attention
    embed: Function,
    norm: Function,
    add_norm: Function,
    prep: Function,
    index_scores: Function,
    index_scores_tc: Function,
    index_select: Function,
    attn_decode: Function,
    attn_merge: Function,
    attn_prompt: Function,
    attn_prompt_tc: Function,
    swiglu_oai: Function,
    router: Function,
    route: Function,
    moe_partial: Function,
    top1: Function,
    put_words: Function,
    tap: Function,
    markov: Function,
    vit_rope: Function,
    splice: Function,
    top1f: Function,
    pick_merge: Function,
    f32_to_bf16: Function,
    bf16_to_f32: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.minimax_m3);
        errdefer mod.unload();
        var attn = try cuda.Module.load(d, cuda.kernels.minimax_m3_attn);
        errdefer attn.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        k.attn_mod = attn;
        inline for (@typeInfo(Kernels).@"struct".field_names, @typeInfo(Kernels).@"struct".field_types) |name, T| if (T == Function) {
            const sym = "tf_mm3_" ++ name;
            const in_attn = comptime std.mem.startsWith(u8, name, "attn_") or std.mem.startsWith(u8, name, "index_"); // minimax_m3_attn.cu
            @field(k, name) = try (if (in_attn) attn else mod).function(sym);
        };
        try k.index_scores.allowDynamicShared(indexScoresShared(4));
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.attn_mod.unload();
        k.mod.unload();
        k.* = undefined;
    }
};

/// tf_mm3_index_scores' dynamic shared bytes: 16 rows' index queries in fp32 and the warps' partial maxima.
pub fn indexScoresShared(heads: usize) u32 {
    return @intCast((16 * heads * 128 + 4 * 16 * heads) * 4);
}

fn go(f: Function, s: Stream, grid: cuda.Dim3, block: cuda.Dim3, shared: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = block, .shared = shared }, s, a);
}

/// The ids' bf16 embeddings as the fp32 residual rows.
pub fn embed(k: *const Kernels, s: Stream, ids: u64, table: u64, resid: u64, rows: usize, hidden: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ ids, table, resid }) |v| a.add(v);
    a.add(int(rows));
    a.add(int(hidden));
    try go(k.embed, s, .{ .x = ceil(hidden, 256), .y = @intCast(rows), .z = 1 }, .{ .x = 256 }, 0, &a);
}

/// Gemma's RMSNorm (1 + w) of the residual rows into bf16 rows.
pub fn norm(k: *const Kernels, s: Stream, resid: u64, w: u64, out: u64, rows: usize, hidden: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ resid, w, out }) |v| a.add(v);
    a.add(int(rows));
    a.add(int(hidden));
    a.add(eps);
    try go(k.norm, s, .{ .x = @intCast(rows) }, .{ .x = 1024 }, 0, &a);
}

/// The ranks' partials ([world][rows][slots][hidden] fp32) added into the residual, then the next norm (w 0: none).
pub fn addNorm(k: *const Kernels, s: Stream, resid: u64, parts: u64, world: usize, slots: usize, scale: f32, w: u64, out: u64, rows: usize, hidden: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    a.add(resid);
    a.add(parts);
    a.add(int(world));
    a.add(int(slots));
    a.add(scale);
    a.add(w);
    a.add(out);
    a.add(int(rows));
    a.add(int(hidden));
    a.add(eps);
    try go(k.add_norm, s, .{ .x = @intCast(rows) }, .{ .x = 1024 }, 0, &a);
}

/// The stacked projection's heads normed and rotated: q and index q rows, k, v and index k into the caches.
pub const Prep = struct { x: u64, stride: usize, qw: u64, kw: u64, iqw: u64, ikw: u64, q: u64, iq: u64, kc: u64, vc: u64, ik: u64, rows: usize, pos: usize, hq: usize, hk: usize, hi: usize, rotary: usize, eps: f32, theta: f32 };

pub fn prep(k: *const Kernels, s: Stream, p: Prep) !void {
    var a: cuda.Args = .{};
    a.add(p.x);
    a.add(int(p.stride));
    for ([_]u64{ p.qw, p.kw, p.iqw, p.ikw, p.q, p.iq, p.kc, p.vc, p.ik }) |v| a.add(v);
    for ([_]usize{ p.rows, p.pos, p.hq, p.hk, p.hi, p.rotary }) |v| a.add(int(v));
    a.add(p.eps);
    a.add(p.theta);
    const units = p.hq + p.hk + p.hi + 1;
    try go(k.prep, s, .{ .x = ceil(units, 8), .y = @intCast(p.rows), .z = 1 }, .{ .x = 256 }, 0, &a);
}

/// Index scores [rows][heads][blocks] fp32: each block's max dot of the row's index queries with its keys.
pub fn indexScores(k: *const Kernels, s: Stream, iq: u64, ik: u64, scores: u64, rows: usize, pos: usize, heads: usize, blocks: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ iq, ik, scores }) |v| a.add(v);
    for ([_]usize{ rows, pos, heads, blocks }) |v| a.add(int(v));
    // prompt rows on the tensor units (a decode window's few rows keep the row kernel: no 64-pair tile to fill)
    if (rows > 16) return go(k.index_scores_tc, s, .{ .x = @intCast(blocks), .y = ceil(rows * heads, 64), .z = 1 }, .{ .x = 128 }, 0, &a);
    try go(k.index_scores, s, .{ .x = @intCast(blocks), .y = ceil(rows, 16), .z = 1 }, .{ .x = 128 }, indexScoresShared(heads), &a);
}

/// Each (row, head)'s kept blocks ascending (-1 past the last) and, with `words` > 0, their bitmaps.
pub fn indexSelect(k: *const Kernels, s: Stream, scores: u64, sel: u64, bits: u64, rows: usize, pos: usize, heads: usize, blocks: usize, topk: usize, words: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ scores, sel, bits }) |v| a.add(v);
    for ([_]usize{ rows, pos, heads, blocks, topk, words }) |v| a.add(int(v));
    try go(k.index_select, s, .{ .x = @intCast(heads), .y = @intCast(rows), .z = 1 }, .{ .x = 32 }, 0, &a);
}

/// Decode attention partials by (row, head, quarter of a kept block), then their merge into bf16 rows.
pub const Decode = struct { q: u64, kc: u64, vc: u64, sel: u64, pm: u64, pl: u64, po: u64, o: u64, rows: usize, pos: usize, hq: usize, hk: usize, topk: usize, all: bool, scale: f32 };

pub fn attnDecode(k: *const Kernels, s: Stream, d: Decode) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ d.q, d.kc, d.vc, d.sel, d.pm, d.pl, d.po }) |v| a.add(v);
    for ([_]usize{ d.rows, d.pos, d.hq, d.hk, d.topk }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(d.all)));
    a.add(d.scale);
    const g: u32 = @intCast(d.hq / d.hk);
    try go(k.attn_decode, s, .{ .x = @intCast(d.hk), .y = @intCast(d.rows), .z = @intCast(d.topk * 4) }, .{ .x = 32, .y = g, .z = 1 }, 0, &a);
    var m: cuda.Args = .{};
    for ([_]u64{ d.pm, d.pl, d.po, d.o }) |v| m.add(v);
    m.add(int(d.rows));
    m.add(int(d.hq));
    m.add(int(d.topk * 4));
    try go(k.attn_merge, s, .{ .x = @intCast(d.hq), .y = @intCast(d.rows), .z = 1 }, .{ .x = 32 }, 0, &m);
}

/// Prompt attention: a block a (row, key head) over the row's kept blocks (its bitmap; words 0: every block).
pub fn attnPrompt(k: *const Kernels, s: Stream, q: u64, kc: u64, vc: u64, bits: u64, o: u64, rows: usize, pos: usize, hq: usize, hk: usize, words: usize, scale: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ q, kc, vc, bits, o }) |v| a.add(v);
    for ([_]usize{ rows, pos, hq, hk, words }) |v| a.add(int(v));
    a.add(scale);
    // the tensor units take 16 query heads a key head (MiniMax's at any TP); TF_MINIMAX_PROMPT_ATTN=rows: rows
    if (hq == 16 * hk and !scalarPrompt()) {
        a.add(@as(c_int, 0));
        a.add(@as(c_int, -1));
        return go(k.attn_prompt_tc, s, .{ .x = ceil(rows, 8), .y = @intCast(hk), .z = 1 }, .{ .x = 256 }, 0, &a);
    }
    try go(k.attn_prompt, s, .{ .x = @intCast(rows), .y = @intCast(hk), .z = 1 }, .{ .x = 128 }, 0, &a);
}

/// A draft block's attention: every row over keys [lo, hi] both ways (16 query heads a key head).
pub fn attnWindow(k: *const Kernels, s: Stream, q: u64, kc: u64, vc: u64, o: u64, rows: usize, hq: usize, hk: usize, lo: usize, hi: usize, scale: f32) !void {
    if (hq != 16 * hk) return error.UnsupportedMiniMaxAttention;
    var a: cuda.Args = .{};
    for ([_]u64{ q, kc, vc, 0, o }) |v| a.add(v);
    for ([_]usize{ rows, 0, hq, hk, 0 }) |v| a.add(int(v));
    a.add(scale);
    a.add(int(lo));
    a.add(int(hi));
    try go(k.attn_prompt_tc, s, .{ .x = ceil(rows, 8), .y = @intCast(hk), .z = 1 }, .{ .x = 256 }, 0, &a);
}

/// Markov scores of one block row over this rank's `n` vocabulary rows, the pick before at device `prev_at`.
pub fn markov(k: *const Kernels, s: Stream, logits: u64, w2: u64, w1: u64, prev_at: u64, out: u64, n: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ logits, w2, w1, prev_at, out }) |v| a.add(v);
    a.add(int(n));
    try go(k.markov, s, .{ .x = ceil(n, 8) }, .{ .x = 256 }, 0, &a);
}

/// Every rank's (value, id) at `stride` bytes a rank, merged into walk[j + 1] (the best; the low id on a tie).
pub fn pickMerge(k: *const Kernels, s: Stream, picks: u64, world: usize, stride: usize, walk: u64, j: usize) !void {
    var a: cuda.Args = .{};
    a.add(picks);
    a.add(int(world));
    a.add(int(stride));
    a.add(walk);
    a.add(int(j));
    try go(k.pick_merge, s, .{ .x = 1 }, .{ .x = 32 }, 0, &a);
}

/// The greedy pick of `n` fp32 scores (ids offset by `base`) into one value and id.
pub fn top1f(k: *const Kernels, s: Stream, scores: u64, n: usize, base: usize, val: u64, id: u64) !void {
    var a: cuda.Args = .{};
    a.add(scores);
    a.add(int(n));
    a.add(int(base));
    a.add(val);
    a.add(id);
    try go(k.top1f, s, .{ .x = 1 }, .{ .x = 1024 }, 0, &a);
}

/// The tower's axial RoPE on q and k of qkv [P][3][16][80] in place, from cos and sin [P][39].
pub fn vitRope(k: *const Kernels, s: Stream, qkv: u64, cs: u64, sn: u64, patches: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ qkv, cs, sn }) |v| a.add(v);
    a.add(int(patches));
    try go(k.vit_rope, s, .{ .x = @intCast(patches), .y = 16, .z = 2 }, .{ .x = 64 }, 0, &a);
}

/// `n` image rows (bf16) over residual rows (fp32).
pub fn splice(k: *const Kernels, s: Stream, rows: u64, resid: u64, n: usize, hidden: usize) !void {
    var a: cuda.Args = .{};
    a.add(rows);
    a.add(resid);
    a.add(int(n));
    a.add(int(hidden));
    try go(k.splice, s, .{ .x = ceil(hidden, 256), .y = @intCast(n), .z = 1 }, .{ .x = 256 }, 0, &a);
}

fn scalarPrompt() bool {
    const v = std.c.getenv("TF_MINIMAX_PROMPT_ATTN") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "rows");
}

/// SwiGLU-OAI of gate | up rows (`stride` apart) into `width`-wide rows.
pub fn swigluOai(k: *const Kernels, s: Stream, gu: u64, out: u64, rows: usize, width: usize, stride: usize, alpha: f32, limit: f32) !void {
    var a: cuda.Args = .{};
    a.add(gu);
    a.add(out);
    for ([_]usize{ rows, width, stride }) |v| a.add(int(v));
    a.add(alpha);
    a.add(limit);
    try go(k.swiglu_oai, s, .{ .x = ceil(width, 256), .y = @intCast(rows), .z = 1 }, .{ .x = 256 }, 0, &a);
}

/// The router's fp32 logits [rows][experts] of bf16 rows against the fp32 router.
pub fn router(k: *const Kernels, s: Stream, x: u64, w: u64, logits: u64, rows: usize, width: usize, experts: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, logits }) |v| a.add(v);
    for ([_]usize{ rows, width, experts }) |v| a.add(int(v));
    try go(k.router, s, .{ .x = ceil(experts, 8), .y = @intCast(rows), .z = 1 }, .{ .x = 256 }, 0, &a);
}

/// Each row's top-k picks (ascending expert id) and their normalized sigmoid weights.
pub fn route(k: *const Kernels, s: Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, experts: usize, topk: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ logits, bias, pick, wts }) |v| a.add(v);
    for ([_]usize{ rows, experts, topk }) |v| a.add(int(v));
    try go(k.route, s, .{ .x = ceil(rows, 8) }, .{ .x = 256 }, 0, &a);
}

/// This rank's MoE partial rows [rows][2][hidden] fp32: weighted routed rows (fp32 or bf16) and the shared expert's.
pub fn moePartial(k: *const Kernels, s: Stream, ey: u64, ey_bf16: bool, wts: u64, sy: u64, out: u64, rows: usize, topk: usize, hidden: usize) !void {
    var a: cuda.Args = .{};
    a.add(ey);
    a.add(@as(c_int, @intFromBool(ey_bf16)));
    for ([_]u64{ wts, sy, out }) |v| a.add(v);
    for ([_]usize{ rows, topk, hidden }) |v| a.add(int(v));
    try go(k.moe_partial, s, .{ .x = ceil(hidden, 256), .y = @intCast(rows), .z = 1 }, .{ .x = 256 }, 0, &a);
}

/// Each row's greedy pick over this rank's `n` logits (ids offset by `base`): values and ids.
pub fn top1(k: *const Kernels, s: Stream, logits: u64, rows: usize, n: usize, base: usize, vals: u64, ids: u64) !void {
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(int(rows));
    a.add(int(n));
    a.add(int(base));
    a.add(vals);
    a.add(ids);
    try go(k.top1, s, .{ .x = @intCast(rows) }, .{ .x = 1024 }, 0, &a);
}

/// A tapped layer's residual rows into a drafter's feature rows.
pub fn tap(k: *const Kernels, s: Stream, resid: u64, taps: u64, rows: usize, hidden: usize, ntaps: usize, which: usize) !void {
    var a: cuda.Args = .{};
    a.add(resid);
    a.add(taps);
    for ([_]usize{ rows, hidden, ntaps, which }) |v| a.add(int(v));
    try go(k.tap, s, .{ .x = ceil(hidden, 256), .y = @intCast(rows), .z = 1 }, .{ .x = 256 }, 0, &a);
}

/// fp32 values as bf16 (`n` of them), or back.
pub fn f32ToBf16(k: *const Kernels, s: Stream, x: u64, y: u64, n: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(y);
    a.add(@as(c_long, @intCast(n)));
    try go(k.f32_to_bf16, s, .{ .x = ceil(n, 256) }, .{ .x = 256 }, 0, &a);
}

pub fn bf16ToF32(k: *const Kernels, s: Stream, x: u64, y: u64, n: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(y);
    a.add(@as(c_long, @intCast(n)));
    try go(k.bf16_to_f32, s, .{ .x = ceil(n, 256) }, .{ .x = 256 }, 0, &a);
}

test "launch geometry: the shared bytes the index scores stage, and ceilings" {
    try std.testing.expectEqual(@as(u32, (16 * 4 * 128 + 256) * 4), indexScoresShared(4));
    try std.testing.expectEqual(@as(u32, 24), ceil(6144, 256));
    try std.testing.expectEqual(@as(u32, 1), ceil(1, 16));
}
