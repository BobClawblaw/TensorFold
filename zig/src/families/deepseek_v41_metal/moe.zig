//! DeepSeek-V4.1-Flash's MoE: routing, the routed experts and the shared expert.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Enc = fwd.Enc;
const Buffer = mtl.Buffer;
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");

const RouteArgs = extern struct { experts: i32, topk: i32, scale: f32 };
const GluArgs = extern struct { width: i32, weighted: i32, limit: f32 };
const SortArgs = extern struct { slots: i32, experts: i32 };

/// Routed experts (row kernels by id) and the shared expert, summed in expert order as the reference does.
pub fn moe(x: Enc, mo: *const wts.Moe, topk: usize, experts: usize) !void {
    const s = x.s;
    const rows = x.rows;
    const slots = rows * topk;
    const side = !x.units; // decode-sized: the shared expert runs beside the routed path
    if (rows >= fwd.nax_rows) {
        router(x, mo.router, rows, experts);
    } else x.gemv(true, mo.router, s.normed, 0, c.hidden, s.router, 0, rows, experts, c.hidden, experts);
    if (side) {
        x.with();
        x.linear(mo.shared.w1, s.normed, 0, s.sh_gate, 0, rows);
        x.with();
        x.linear(mo.shared.w3, s.normed, 0, s.sh_up, 0, rows);
    }
    x.pipe("ds_route");
    x.buf(s.router, 0, 0);
    x.tensor(mo.bias, 1);
    x.buf(s.route_pick, 0, 2);
    x.buf(s.route_weight, 0, 3);
    x.args(RouteArgs{ .experts = @intCast(experts), .topk = @intCast(topk), .scale = c.route_scale }, 4);
    x.buf(s.img, 0, 5);
    x.tensor(mo.bias_vl, 6);
    x.groups(.{ rows, 1, 1 }, .{ (experts + 31) / 32 * 32, 1, 1 });
    if (side) {
        x.with();
        gluTo(x, rows, false, s.route_weight, s.sh_gate, s.sh_up, s.sh_act);
    }
    try x.cut(5);
    if (x.nax(mo.w1.one)) |p_bf| {
        // the experts' inputs as int8 when that form loaded (kernels.int8_experts), so the MMAs run int8 x int8
        const p8: ?am.Pipes = if (x.units) x.m.kernels.nax_i8[mo.w1.one.bits] else null;
        const p = p8 orelse p_bf;
        const int8 = p8 != null;
        // a prompt chunk: slots sorted by expert, each expert's rows through its weights once on the tensor units
        const blocks = (slots + st.sort_block - 1) / st.sort_block;
        const sort = SortArgs{ .slots = @intCast(slots), .experts = @intCast(experts) };
        std.debug.assert(experts <= 1024);
        x.pipe("ds_sort_count");
        x.buf(s.route_pick, 0, 0);
        x.buf(s.counts, 0, 1);
        x.args(sort, 2);
        x.groups(.{ blocks, 1, 1 }, .{ st.sort_block, 1, 1 });
        x.pipe("ds_sort_scan");
        x.buf(s.counts, 0, 0);
        x.buf(s.offs, 0, 1);
        x.args(sort, 2);
        x.groups(.{ 1, 1, 1 }, .{ 1024, 1, 1 });
        x.pipe("ds_sort_place");
        x.buf(s.route_pick, 0, 0);
        x.buf(s.route_weight, 0, 1);
        x.buf(s.counts, 0, 2);
        x.buf(s.order, 0, 3);
        x.buf(s.wsort, 0, 4);
        x.args(sort, 5);
        x.groups(.{ blocks, 1, 1 }, .{ st.sort_block, 1, 1 });
        const e = x.ep.*;
        const sums = fwd.Ref{ .buf = s.sums };
        const offs = fwd.Ref{ .buf = s.offs };
        const order = fwd.Ref{ .buf = s.order };
        const xq = fwd.Ref{ .buf = s.xq };
        const xa = fwd.Ref{ .buf = s.xa };
        x.fence();
        const normed = fwd.Ref{ .buf = s.normed };
        const act = fwd.Ref{ .buf = s.act };
        if (int8) am.quantRows(e, p, 64, normed, sums, xq, xa, @intCast(rows), c.hidden) else am.rowSums(e, p, 64, normed, sums, @intCast(rows), c.hidden);
        x.fence();
        if (int8) am.bindScales(e, xa);
        am.gatherGluWeighted(e, p, if (int8) xq else normed, sums, fwd.matrix(mo.w1.one), fwd.matrix(mo.w3.one), offs, order, fwd.Ref{ .buf = s.wsort }, act, @intCast(slots), @intCast(experts), @intCast(topk), c.swiglu_limit);
        x.fence();
        if (int8) am.quantRows(e, p, 64, act, sums, xq, xa, @intCast(slots), c.moe_inter) else am.rowSums(e, p, 64, act, sums, @intCast(slots), c.moe_inter);
        x.fence();
        if (int8) am.bindScales(e, xa);
        am.gatherTo(e, p, if (int8) xq else act, sums, fwd.matrix(mo.w2.one), offs, fwd.Ref{ .buf = s.experts }, order, @intCast(slots), @intCast(experts));
    } else {
        x.indexed(mo.w1, slots, s.normed, s.route_pick, s.gate);
        x.with();
        x.indexed(mo.w3, slots, s.normed, s.route_pick, s.up);
        if (side) {
            x.with();
            x.linear(mo.shared.w2, s.sh_act, 0, s.shared, 0, rows);
        }
        glu(x, slots, true);
        x.indexed(mo.w2, slots, s.act, s.route_pick, s.experts);
    }
    try x.cut(6);
    if (!side) {
        x.linear(mo.shared.w1, s.normed, 0, s.gate, 0, rows);
        x.linear(mo.shared.w3, s.normed, 0, s.up, 0, rows);
        glu(x, rows, false);
        x.linear(mo.shared.w2, s.act, 0, s.shared, 0, rows);
    }
    // the branch, routed rows plus the shared one, is summed where the expand that follows reads it
    fwd.pending_topk = @intCast(topk);
    try x.cut(7);
}

/// A chunk's router logits: `gemv`'s sums, 4 outputs a simdgroup so each input is read once for them.
fn router(x: Enc, w: wts.Tensor, rows: usize, experts: usize) void {
    x.pipe("ds_gemv_cols_bf_f32");
    x.buf(x.s.normed, 0, 0);
    x.tensor(w, 1);
    x.buf(x.s.router, 0, 2);
    x.args(fwd.GemvArgs{ .rows = @intCast(rows), .n = @intCast(experts), .k = c.hidden, .group_n = @intCast(experts), .x_stride = c.hidden, .y_stride = @intCast(experts) }, 3);
    x.groups(.{ (experts + 31) / 32, (rows + 7) / 8, 1 }, .{ 256, 1, 1 });
}

pub fn glu(x: Enc, n: usize, weighted: bool) void {
    gluTo(x, n, weighted, x.s.route_weight, x.s.gate, x.s.up, x.s.act);
}

pub fn gluTo(x: Enc, n: usize, weighted: bool, weights: Buffer, gate: Buffer, up: Buffer, act: Buffer) void {
    x.pipe("ds_swiglu");
    x.buf(gate, 0, 0);
    x.buf(up, 0, 1);
    x.buf(weights, 0, 2);
    x.buf(act, 0, 3);
    x.args(GluArgs{ .width = c.moe_inter, .weighted = @intFromBool(weighted), .limit = c.swiglu_limit }, 4);
    x.threads(.{ c.moe_inter, n, 1 }, .{ 256, 1, 1 });
}
