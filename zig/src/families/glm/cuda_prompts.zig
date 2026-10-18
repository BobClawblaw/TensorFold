//! GLM-5.3-Flash's prefill_many: fresh streams' whole prompts in one prompt pass, each left as its own pass leaves it.

const std = @import("std");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine_types.zig");
const Engine = @import("cuda_engine.zig").Engine;
const dflash = @import("dflash.zig");

comptime { // a pass's rows go to DFlash2 in one ingest, every prompt within its window
    std.debug.assert(fw.prefill_rows <= dflash.MAX_CTX and fw.prefill_rows <= dflash.WIN);
}

/// Whether `parts` go in one pass: fresh prompts, prompt_rows in all, under the capacity, hold rows from hold0 fit.
pub fn fits(parts: []const engine.Prompt, hold0: usize, capacity: usize) bool {
    if (parts.len == 0 or hold0 + parts.len > engine.max_together) return false;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len + fw.window_rows >= capacity) return false;
        for (parts[0..i]) |q| if (q.slot == p.slot) return false;
        rows += p.ids.len;
    }
    return rows <= fw.prefill_rows;
}

/// The prompts' pass (each begun at 0): prompt k's first token in firsts[k], draft source in hold row hold0 + k.
pub fn prefillMany(e: *Engine, parts: []const engine.Prompt, firsts: []u32, hold0: usize) !void {
    if (!fits(parts, hold0, e.capacity) or firsts.len < parts.len) return error.BadPrompts;
    var segs: [engine.max_together]fw.Seg = undefined;
    var ids: [fw.prefill_rows]u32 = undefined;
    var rows: usize = 0;
    for (parts, 0..) |p, k| {
        try e.bind(p.slot);
        if (e.cur.st.pos != 0 or e.cur.st.mtp_len != 0) return error.BadPrompts;
        segs[k] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
        @memcpy(ids[rows..][0..p.ids.len], p.ids);
        rows += p.ids.len;
    }
    try e.gate(null);
    var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = &e.bp, .segs = segs[0..parts.len] }};
    var f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = &e.pieces, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .taps = if (e.drafter) |dr| dr.taps(true) else null };
    try f.run(@ptrCast(ids[0..rows]));
    if (e.drafter) |dr| { // every prompt is within DFlash2's window (prompt_rows): all its rows into its slot's ring
        var at: [engine.max_together][3]usize = undefined;
        for (parts, segs[0..parts.len], at[0..parts.len]) |p, g, *a| a.* = .{ p.slot, g.row0, g.rows };
        try dr.prompts(e, at[0..parts.len]);
    }
    const D = e.c.hidden;
    for (parts, segs[0..parts.len], 0..) |p, g, k| {
        try e.bind(p.slot);
        try e.draw(e.bp.logits + k * e.share.vocab.len() * 2, 1, p.ids.len, firsts[k..][0..1]);
        if (!e.mtp) continue;
        try hold(e, hold0 + k, e.bp.fnormed + (g.row0 + g.rows - 1) * D * 2);
    }
    f.pieces = null;
    f.taps = null;
    for (parts, segs[0..parts.len]) |p, g| {
        one[0] = .{ .w = &e.w, .st = g.st, .b = &e.bp };
        // decode._absorb_rows; the head's pass writes final-norm row 0, which the hold rows above already copied
        if (e.mtp and g.rows > 1) {
            try f.mtp(@ptrCast(p.ids[1..]), &.{e.bp.fnormed + g.row0 * D * 2});
            try g.st.setMtpLen(e.stream, g.st.mtp_len + g.rows - 1);
        }
        try f.commit(g.rows);
    }
}

/// The bound stream's first-draft source: its prompt's last hidden row (bf16 at `src`) into hold row `row`.
pub fn hold(e: *Engine, row: usize, src: u64) !void {
    const at = e.hold.ptr + row * e.c.hidden * 2;
    try e.copy(at, src, e.c.hidden * 2);
    e.cur.held_at = at;
}
