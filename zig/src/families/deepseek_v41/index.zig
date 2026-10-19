//! DeepSeek-V4.1-Flash's compressors and indexers: the KV sources' compressed latents and each row's picks.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Enc = fwd.Enc;
const Buffer = mtl.Buffer;

const PoolArgs = extern struct { first: i32, ring: i32 };
const ScoreArgs = extern struct { rows: i32, groups: i32, ratio: i32, first_pos: i32, stride: i32 };
const KeptArgs = extern struct { rows: i32, groups: i32, ratio: i32, first_pos: i32, block: i32, kept: i32 };
const BlockArgs = extern struct { rows: i32, groups: i32, block: i32, first_pos: i32, ratio: i32, stride: i32 };
const TopkArgs = extern struct { n: i32, k: i32, stride: i32 };

/// A KV source's compressor: completed groups pooled or projected, normed, index keys, RoPE'd, fp4 into each cache.
pub fn compress(x: Enc, si: usize, ratio: usize, cp: wts.Compressor, ix: wts.Indexer, freq: Buffer) !void {
    const s = x.s;
    const rows = x.rows;
    if (ratio > 1) {
        x.gemv(true, cp.wkv, s.normed, 0, c.hidden, s.cproj, 0, rows, c.head_dim, c.hidden, c.head_dim);
        x.gemv(true, cp.wgate.?, s.normed, 0, c.hidden, s.cscore, 0, rows, c.head_dim, c.hidden, c.head_dim);
    } else {
        x.gemv(false, cp.wkv, s.normed, 0, c.hidden, s.latent, 0, rows, c.head_dim, c.hidden, c.head_dim);
    }
    for (x.segs) |g| try compressOne(x.of(g), g.cache, si, ratio, cp, ix, freq);
}

fn compressOne(x: Enc, cache: *st.Cache, si: usize, ratio: usize, cp: wts.Compressor, ix: wts.Indexer, freq: Buffer) !void {
    const s = x.s;
    const src = &cache.sources[si];
    const span = fwd.completed(x.base, x.rows, ratio);
    const n = span[1] - span[0];
    if (span[1] > src.groups) return error.DeepseekCacheFull;
    const hd = c.head_dim * 2;
    if (ratio > 1) {
        x.putF32(s.cproj, s.pos, cache.proj_kv[si].?, x.rows, c.head_dim, st.ring);
        x.putF32(s.cscore, s.pos, cache.proj_score[si].?, x.rows, c.head_dim, st.ring);
        if (n == 0) return;
        x.pipe("ds_pool2");
        x.buf(cache.proj_kv[si].?, 0, 0);
        x.buf(cache.proj_score[si].?, 0, 1);
        x.at(s.latent, hd, 2);
        x.args(PoolArgs{ .first = @intCast(span[0]), .ring = st.ring }, 3);
        x.threads(.{ c.head_dim, n, 1 }, .{ 256, 1, 1 });
    }
    x.rms(s.latent, x.r0 * hd, cp.norm, s.latent, x.r0 * hd, n, c.head_dim);
    for (s.gpos[si].slice(i32, x.r0 + n)[x.r0..], s.gidx[si].slice(i32, x.r0 + n)[x.r0..], 0..) |*p, *g, j| {
        g.* = @intCast(span[0] + j);
        p.* = @intCast((span[0] + j) * ratio);
    }
    // index keys: k_norm(wk(latent)), RoPE'd at the group's first position, fp4 with e8m0 scales a 32
    const kd = c.index_dim * 2;
    x.gemv(false, ix.wk.?, s.latent, x.r0 * hd, c.head_dim, s.ik, x.r0 * kd, n, c.index_dim, c.head_dim, c.index_dim);
    x.rms(s.ik, x.r0 * kd, ix.k_norm.?, s.ik, x.r0 * kd, n, c.index_dim);
    x.rope(s.ik, n, 1, c.index_dim, s.gpos[si], freq, false);
    x.fp4(s.ik, n, c.index_dim, false);
    x.put(s.ik, s.gidx[si], src.keys, n, c.index_dim, 0);
    // the latents: RoPE'd at the same position, fp4 with e4m3 scales a 16
    x.rope(s.latent, n, 1, c.head_dim, s.gpos[si], freq, false);
    x.fp4(s.latent, n, c.head_dim, true);
    x.put(s.latent, s.gidx[si], src.latents, n, c.head_dim, 0);
}

/// An index source's picks: scores over visible keys, each row's 512 best in key order, rows in slices that fit.
pub fn index(x: Enc, si: usize, layer: usize, ratio: usize, ix: wts.Indexer, freq: Buffer) !void {
    const s = x.s;
    const rows = x.rows;
    x.linear(ix.wq_b, s.qr, 0, s.iq, 0, rows);
    x.rope(s.iq, rows, c.index_heads, c.index_dim, s.pos, freq, false);
    x.fp4(s.iq, rows, c.index_heads * c.index_dim, false);
    x.gemv(false, ix.weights_proj, s.normed, 0, c.hidden, s.iw, 0, rows, c.index_heads, c.hidden, c.index_heads);
    x.pipe("ds_scale");
    x.buf(s.iw, 0, 0);
    x.args(@as(f32, 1.0 / @sqrt(@as(f32, c.index_dim)) * (1.0 / @sqrt(@as(f32, c.index_heads)))), 1);
    x.threads(.{ rows * c.index_heads, 1, 1 }, .{ 256, 1, 1 });
    for (x.segs) |g| indexOne(x.of(g), &g.cache.sources[si], layer, ratio);
}

fn indexOne(x: Enc, src: *const st.Source, layer: usize, ratio: usize) void {
    const s = x.s;
    const rows = x.rows;
    const groups = (x.base + rows) / ratio;
    if (groups == 0) {
        x.pipe("ds_topk");
        x.buf(s.scores, 0, 0);
        x.at(s.picks, st.picks * 4, 1);
        x.args(TopkArgs{ .n = 0, .k = st.picks, .stride = 1 }, 2);
        x.groups(.{ rows, 1, 1 }, .{ 1024, 1, 1 });
        return;
    }
    if (layer > c.candidate_source) return indexKept(x, src, groups, ratio);
    const slice = @max(1, @min(rows, st.score_budget / 4 / groups));
    var r0: usize = 0;
    while (r0 < rows) : (r0 += slice) {
        const n = @min(slice, rows - r0);
        const r = x.r0 + r0;
        const tile: [2]usize = if (n == 1) .{ 1, 8 } else .{ 2, 16 };
        if (n == 1) x.pipe("ds_index_scores_1x8") else x.pipe("ds_index_scores_2x16");
        x.buf(s.iq, r * c.index_heads * c.index_dim * 2, 0);
        x.buf(s.iw, r * c.index_heads * 2, 1);
        x.buf(src.keys, 0, 2);
        x.buf(s.scores, 0, 3);
        x.args(ScoreArgs{ .rows = @intCast(n), .groups = @intCast(groups), .ratio = @intCast(ratio), .first_pos = @intCast(x.base + r0), .stride = @intCast(groups) }, 4);
        const row_tiles = (n + tile[0] - 1) / tile[0];
        x.threads(.{ row_tiles * 32, (groups + tile[1] - 1) / tile[1], 1 }, .{ @as(usize, @min(row_tiles, 8)) * 32, 1, 1 });
        const nb = (groups + c.candidate_block - 1) / c.candidate_block;
        if (layer == c.candidate_source) {
            x.pipe("ds_block_max");
            x.buf(s.scores, 0, 0);
            x.buf(s.blocks, 0, 1);
            x.args(BlockArgs{ .rows = @intCast(n), .groups = @intCast(groups), .block = c.candidate_block, .first_pos = @intCast(x.base + r0), .ratio = @intCast(ratio), .stride = @intCast(groups) }, 2);
            x.threads(.{ nb, n, 1 }, .{ @min(nb, 256), 1, 1 });
            x.pipe("ds_topk");
            x.buf(s.blocks, 0, 0);
            x.buf(s.keep, r * c.candidate_blocks * 4, 1);
            x.args(TopkArgs{ .n = @intCast(nb), .k = c.candidate_blocks, .stride = @intCast(groups) }, 2);
            x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
        }
        x.pipe("ds_topk");
        x.buf(s.scores, 0, 0);
        x.buf(s.picks, r * st.picks * 4, 1);
        x.args(TopkArgs{ .n = @intCast(groups), .k = st.picks, .stride = @intCast(groups) }, 2);
        x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
    }
}

/// Indexers after layer 20 score only its candidate blocks, in key order: the top-k equals the masked scan's.
fn indexKept(x: Enc, src: *const st.Source, groups: usize, ratio: usize) void {
    const s = x.s;
    const slots = c.candidate_blocks * c.candidate_block;
    const slice = @max(1, @min(x.rows, st.score_budget / 4 / slots));
    var r0: usize = 0;
    while (r0 < x.rows) : (r0 += slice) {
        const n = @min(slice, x.rows - r0);
        const r = x.r0 + r0;
        const args = KeptArgs{ .rows = @intCast(n), .groups = @intCast(groups), .ratio = @intCast(ratio), .first_pos = @intCast(x.base + r0), .block = c.candidate_block, .kept = c.candidate_blocks };
        x.pipe("ds_index_scores_kept");
        x.buf(s.iq, r * c.index_heads * c.index_dim * 2, 0);
        x.buf(s.iw, r * c.index_heads * 2, 1);
        x.buf(src.keys, 0, 2);
        x.buf(s.scores, 0, 3);
        x.buf(s.keep, r * c.candidate_blocks * 4, 4);
        x.args(args, 5);
        x.threads(.{ slots * 32, n, 1 }, .{ 256, 1, 1 }); // a simdgroup a slot
        x.pipe("ds_topk");
        x.buf(s.scores, 0, 0);
        x.buf(s.picks, r * st.picks * 4, 1);
        x.args(TopkArgs{ .n = @intCast(slots), .k = st.picks, .stride = @intCast(slots) }, 2);
        x.groups(.{ n, 1, 1 }, .{ 1024, 1, 1 });
        x.pipe("ds_kept_keys");
        x.buf(s.picks, r * st.picks * 4, 0);
        x.buf(s.keep, r * c.candidate_blocks * 4, 1);
        x.args(args, 2);
        x.threads(.{ st.picks, n, 1 }, .{ 256, 1, 1 });
    }
}
