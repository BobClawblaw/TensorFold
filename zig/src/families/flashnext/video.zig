//! A video's frames to Flash Next's tower input, as transformers' Qwen3VLVideoProcessor prepares them (fps 2, 4 to 768
//! frames, smart_resize over the clip's pixel budget, bicubic, frames paired in time) and Qwen3VLProcessor times them
//! (each pair at the mean of its frames' times). Each pair is one group: an image to the tower and the engine.
const std = @import("std");
const glm_image = @import("../glm/image.zig");
const vision = @import("vision.zig");

pub const fps: f64 = 2;
pub const min_frames: u32 = 4;
pub const max_frames: u32 = 768;
/// The clip's pixels (frames times height times width) between these: the checkpoint's preprocessor_config.json size
/// (shortest_edge, longest_edge), which the video processor reads too (no video config; no per-frame cap).
pub const min_pixels: u64 = 65536;
pub const max_pixels: u64 = 16777216;
const factor: u64 = vision.patch * vision.merge; // 32

/// numpy's round(): halves to even.
fn roundEven(x: f64) f64 {
    if (@abs(x - @trunc(x)) == 0.5) return 2.0 * @round(x / 2.0);
    return @round(x);
}

/// sample_frames: the frame indices fps 2 picks from `total` frames at `rate` frames a second, at most `most`.
pub fn sample(a: std.mem.Allocator, total: u64, rate: f64, most: u32) ![]u32 {
    if (total == 0) return error.NoFrames;
    const r = if (rate > 0) rate else 24; // the processor's guess without metadata
    const want: u64 = @intFromFloat(@as(f64, @floatFromInt(total)) / r * fps);
    const n: u64 = @min(@max(want, min_frames), @min(max_frames, most), total);
    const out = try a.alloc(u32, n);
    for (out, 0..) |*o, i| { // np.linspace(0, total - 1, n).round()
        const x: f64 = if (n == 1) 0 else @as(f64, @floatFromInt(i)) * @as(f64, @floatFromInt(total - 1)) / @as(f64, @floatFromInt(n - 1));
        o.* = @intFromFloat(roundEven(x));
    }
    return out;
}

/// The video processor's smart_resize(num_frames, height, width, factor=32, temporal_factor=2) at its size bounds.
pub fn smartResize(frames: u32, h0: u32, w0: u32) ![2]u32 {
    if (frames < 2) return error.TooFewFrames;
    var h: f64 = @floatFromInt(h0);
    var w: f64 = @floatFromInt(w0);
    const f: f64 = @floatFromInt(factor);
    if (h < f or w < f) {
        const scale = @max(f / h, f / w);
        h = @trunc(h * scale);
        w = @trunc(w * scale);
    }
    if (@max(h, w) / @min(h, w) > 200) return error.BadAspect;
    const t: f64 = @floatFromInt(frames);
    var hb = roundEven(h / f) * f;
    var wb = roundEven(w / f) * f;
    const tb = roundEven(t / 2) * 2;
    if (tb * hb * wb > @as(f64, @floatFromInt(max_pixels))) {
        const beta = @sqrt(t * h * w / @as(f64, @floatFromInt(max_pixels)));
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (tb * hb * wb < @as(f64, @floatFromInt(min_pixels))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (t * h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ @intFromFloat(hb), @intFromFloat(wb) };
}

/// _calculate_timestamps: each pair's time, the mean of its two frames' (the last index repeated to an even count).
pub fn times(a: std.mem.Allocator, indices: []const u32, rate: f64) ![]f64 {
    const r = if (rate > 0) rate else 24;
    const pairs = (indices.len + 1) / 2;
    const out = try a.alloc(f64, pairs);
    for (out, 0..) |*o, p| {
        const first = indices[2 * p];
        const second = if (2 * p + 1 < indices.len) indices[2 * p + 1] else indices[indices.len - 1];
        o.* = (@as(f64, @floatFromInt(first)) / r + @as(f64, @floatFromInt(second)) / r) / 2;
    }
    return out;
}

/// The frames gathered at the resized size as the decoder hands them over.
pub const Frames = struct {
    a: std.mem.Allocator,
    w: u32,
    h: u32,
    px: [][]u8, // a frame's RGB at w x h, in sampled order

    pub fn take(ctx: *anyopaque, k: usize, rgb: []const u8, width: u32, height: u32, stride: usize) anyerror!void {
        const fr: *Frames = @ptrCast(@alignCast(ctx));
        if (k >= fr.px.len) return error.FrameOutOfRange;
        const tight = try fr.a.alloc(u8, @as(usize, width) * height * 3);
        defer fr.a.free(tight);
        for (0..height) |y| @memcpy(tight[y * width * 3 ..][0 .. width * 3], rgb[y * stride ..][0 .. width * 3]);
        const small = try glm_image.resize(fr.a, .{ .w = width, .h = height, .px = tight }, fr.w, fr.h);
        fr.px[k] = small.px;
    }
};

/// The frames in pairs as the tower reads them: each pair's [gh * gw, 1536] patches (channel, frame, row, column),
/// merge windows contiguous, scaled to [-1, 1].
pub fn group(a: std.mem.Allocator, f0: []const u8, f1: []const u8, w: u32, h: u32) ![]f32 {
    const gh = h / vision.patch;
    const gw = w / vision.patch;
    const P: usize = @as(usize, gh) * gw;
    const pixels = try a.alloc(f32, P * vision.patch_width);
    const pp = vision.patch * vision.patch;
    const frames = [2][]const u8{ f0, f1 };
    for (0..P) |i| {
        const at = vision.patchAt(@intCast(i), gw);
        const dst = pixels[i * vision.patch_width ..][0..vision.patch_width];
        for (0..3) |c| for (frames, 0..) |fr, t| for (0..vision.patch) |y| for (0..vision.patch) |x| {
            const py = at[0] * vision.patch + y;
            const px = at[1] * vision.patch + x;
            const v: f32 = @floatFromInt(fr[(py * w + px) * 3 + c]);
            dst[c * vision.temporal * pp + t * pp + y * vision.patch + x] = (v * (1.0 / 255.0) - 0.5) / 0.5;
        };
    }
    return pixels;
}

test "sampling: fps 2, at least 4 frames, linspace rounded half to even" {
    const a = std.testing.allocator;
    const s = try sample(a, 300, 30, 768); // 10 s at 30 fps: 20 frames
    defer a.free(s);
    try std.testing.expectEqual(@as(usize, 20), s.len);
    try std.testing.expectEqual(@as(u32, 0), s[0]);
    try std.testing.expectEqual(@as(u32, 299), s[19]);
    const short = try sample(a, 30, 30, 768); // 1 s: the floor of 4
    defer a.free(short);
    try std.testing.expectEqual(@as(usize, 4), short.len);
    try std.testing.expectEqualSlices(u32, &.{ 0, 10, 19, 29 }, short);
}

test "times: each pair at the mean of its frames" {
    const a = std.testing.allocator;
    const t = try times(a, &.{ 0, 10, 19, 29, 40 }, 10);
    defer a.free(t);
    for ([_]f64{ 0.5, 2.4, 4.0 }, t) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-12);
}
