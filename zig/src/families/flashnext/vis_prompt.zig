//! An image prompt in the replay engine: the tower's rows for its images, where they sit, and the positions its prompt
//! pass rotates at. transformers' get_rope_index gives text one position on all three axes and an image's merged
//! tokens (p, p + row, p + col), moving p on by max(rows, cols) past it: text after an image sits below its slot.
//! Rotary scores read only differences of positions, so this engine shifts the whole frame by the images' total:
//! text after the last image rotates at its own slot (decode, drafts and GPU-side rounds unchanged), and only prompt
//! rows up to the last image's end (`frame_end`) take a table position.
const std = @import("std");
const mtl = @import("metal");

/// An image's tokens: the prompt slot of its first, how many, its merged grid, and its first row of the tower's rows.
pub const Span = struct { at: u32, n: u32, mh: u32, mw: u32, row: u32 };

pub const VisPrompt = struct {
    rows: mtl.Buffer, // [tokens, 4, 2560] bf16: the tower's rows in prompt order, once a stream as the embedding
    spans: []const Span,
    pos3: []const i32, // [frame_end * 3]: each slot's t, h, w below frame_end
    frame_end: u32,
    prep: mtl.Pipeline, // fnv_attn_prep_mrope

    /// Rows [pos, pos + n): the recorded kernels' position (t) and the multimodal kernel's three.
    pub fn positions(v: *const VisPrompt, pos: usize, n: usize, out_pos: []i32, out3: []i32) void {
        for (0..n) |i| {
            const slot = pos + i;
            const p: [3]i32 = if (slot < v.frame_end) v.pos3[slot * 3 ..][0..3].* else @splat(@intCast(slot));
            out_pos[i] = p[0];
            @memcpy(out3[i * 3 ..][0..3], &p);
        }
    }

    /// The image rows among prompt rows [pos, pos + n): (row in the chunk, row in `rows`, count) a span.
    pub fn overlap(s: Span, pos: usize, n: usize) ?[3]usize {
        const lo = @max(pos, s.at);
        const hi = @min(pos + n, @as(usize, s.at) + s.n);
        if (lo >= hi) return null;
        return .{ lo - pos, s.row + (lo - s.at), hi - lo };
    }
};

/// Each slot's three positions below the last image's end, in the shifted frame (`spans` in prompt order).
pub fn frame(gpa: std.mem.Allocator, spans: []const Span) ![]i32 {
    if (spans.len == 0) return &.{};
    const last = spans[spans.len - 1];
    const end: usize = @as(usize, last.at) + last.n;
    var shift: i64 = 0; // the total the frame moves by: every image's tokens past max(rows, cols)
    for (spans) |s| shift += @as(i64, s.n) - @max(s.mh, s.mw);
    const out = try gpa.alloc(i32, end * 3);
    var cur: i64 = 0; // transformers' current_pos
    var slot: usize = 0;
    for (spans) |s| {
        if (s.at < slot or s.n != s.mh * s.mw) return error.BadImageSpans;
        while (slot < s.at) : (slot += 1) {
            @memset(out[slot * 3 ..][0..3], @intCast(cur + shift));
            cur += 1;
        }
        for (0..s.n) |k| {
            const t = cur + shift;
            const row: i64 = @intCast(k / s.mw);
            const col: i64 = @intCast(k % s.mw);
            out[(slot + k) * 3 ..][0..3].* = .{ @intCast(t), @intCast(t + row), @intCast(t + col) };
        }
        slot += s.n;
        cur += @max(s.mh, s.mw);
    }
    return out;
}

test "the shifted frame: text after the last image at its slot, the image's tokens on its grid" {
    const a = std.testing.allocator;
    // 3 text tokens, a 2x3 image (6 tokens), then text: transformers gives the image (3, 3+r, 3+c) and the next text 6
    const p = try frame(a, &.{.{ .at = 3, .n = 6, .mh = 2, .mw = 3, .row = 0 }});
    defer a.free(p);
    try std.testing.expectEqual(@as(usize, 9 * 3), p.len);
    // shift 6 - 3 = 3: text 0, 1, 2 at 3, 4, 5; the image from t = 6; the next text (slot 9) at 6 + 3 = 9, its slot
    try std.testing.expectEqualSlices(i32, &.{ 3, 3, 3, 4, 4, 4, 5, 5, 5 }, p[0..9]);
    try std.testing.expectEqualSlices(i32, &.{ 6, 6, 6, 6, 6, 7, 6, 6, 8, 6, 7, 6, 6, 7, 7, 6, 7, 8 }, p[9..27]);
}

test "the rows of a span a chunk holds" {
    const s: Span = .{ .at = 10, .n = 20, .mh = 4, .mw = 5, .row = 7 };
    try std.testing.expectEqual(@as(?[3]usize, null), VisPrompt.overlap(s, 0, 10));
    try std.testing.expectEqual(@as(?[3]usize, .{ 5, 7, 5 }), VisPrompt.overlap(s, 5, 10));
    try std.testing.expectEqual(@as(?[3]usize, .{ 0, 12, 15 }), VisPrompt.overlap(s, 15, 100));
}

/// `images` (each with pixels, gh, gw, at, tokens: lanes' Image) through `tower` on `queue`: the image prompt for a
/// prompt of `prompt_len` tokens, its rows on `device` (freed by the caller: `rows.deinit()`), the rest in `a`.
pub fn build(tower: anytype, device: mtl.Device, queue: mtl.Queue, a: std.mem.Allocator, images: anytype, prompt_len: usize) !*VisPrompt {
    const fnv = @import("vision.zig");
    var total: usize = 0;
    var most: usize = 0;
    const spans = try a.alloc(Span, images.len);
    for (images, spans, 0..) |img, *s, i| {
        if (img.gh % fnv.merge != 0 or img.gw % fnv.merge != 0 or img.tokens * fnv.patches_per_token != img.gh * img.gw) return error.ImageGrid;
        if (@as(usize, img.at) + img.tokens > prompt_len or (i > 0 and img.at < spans[i - 1].at + spans[i - 1].n)) return error.ImageOutOfPrompt;
        s.* = .{ .at = img.at, .n = img.tokens, .mh = img.gh / fnv.merge, .mw = img.gw / fnv.merge, .row = @intCast(total) };
        total += img.tokens;
        most = @max(most, img.tokens);
    }
    // the tower runs one image (a video's group) at a time
    if (most * fnv.patches_per_token > tower.max_patches) return error.ImageTooLarge;
    const row = fnv.out_hidden * 2;
    const rows = try device.buffer(total * 4 * row, mtl.ResourceOptions.shared);
    errdefer rows.deinit();
    const out = try device.buffer(most * row, mtl.ResourceOptions.shared);
    defer out.deinit();
    const t0 = std.c.mach_absolute_time();
    var gpu: f64 = 0;
    for (images, spans) |img, s| { // one image a command buffer: the tower's host-written scratch is shared
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        try tower.encode(enc, img.pixels, img.gh, img.gw, .{ .buf = out }, null);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("flash next vision: the tower's command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        gpu += cb.gpuSeconds();
        const src = out.contents();
        const dst = rows.contents();
        for (0..s.n) |k| for (0..4) |stream| @memcpy(dst[((s.row + k) * 4 + stream) * row ..][0..row], src[k * row ..][0..row]);
    }
    const ms = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e3; // 24 MHz ticks on Apple silicon
    std.log.info("flash next vision: {d} images, {d} tokens through the tower in {d:.1} ms ({d:.1} ms on the GPU)", .{ images.len, total, ms, gpu * 1e3 });
    const p = try a.create(VisPrompt);
    const last = spans[spans.len - 1];
    p.* = .{ .rows = rows, .spans = spans, .pos3 = try frame(a, spans), .frame_end = last.at + last.n, .prep = tower.k.attn_prep };
    return p;
}
