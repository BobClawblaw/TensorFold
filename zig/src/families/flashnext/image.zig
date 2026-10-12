//! An image's bytes to Flash Next's tower input, as transformers' Qwen2VLImageProcessor prepares it: decoded (ImageIO,
//! EXIF orientation, alpha over white), resized to smart_resize's multiples of 32 pixels (bicubic, PIL's 8-bit
//! resampler, which torchvision's antialiased uint8 path follows), scaled to [-1, 1] and cut into 16x16 patches, each
//! duplicated in time, the patches of a 2x2 merge window contiguous.
const std = @import("std");
const glm_image = @import("../glm/image.zig");
const vision = @import("vision.zig");

pub const Rgb = glm_image.Rgb;
pub const Error = glm_image.Error;
pub const decode = glm_image.decode;

/// preprocessor_config.json's floor: 65,536 pixels (64 tokens).
pub const min_pixels: u64 = 65536;
const factor: u64 = vision.patch * vision.merge; // 32
const pixels_per_token: u64 = factor * factor;

pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// Python's round(): halves to even.
fn roundHalfEven(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return 2.0 * @round(x / 2.0);
    return r;
}

/// smart_resize(height, width, factor=32, min_pixels, max_pixels=max_tokens * 1024): the resized height and width.
pub fn smartResize(h: u32, w: u32, max_tokens: u32) Error![2]u32 {
    if (h == 0 or w == 0) return error.BadImage;
    if (@max(h, w) > 200 * @min(h, w)) return error.BadImage; // the processor refuses an aspect ratio past 200
    const H: f64 = @floatFromInt(h);
    const W: f64 = @floatFromInt(w);
    const f: f64 = @floatFromInt(factor);
    const max_px: f64 = @floatFromInt(@as(u64, max_tokens) * pixels_per_token);
    const min_px: f64 = @floatFromInt(@min(min_pixels, @as(u64, max_tokens) * pixels_per_token));
    var hb = roundHalfEven(H / f) * f;
    var wb = roundHalfEven(W / f) * f;
    if (hb * wb > max_px) {
        const beta = @sqrt(H * W / max_px);
        hb = @max(f, @floor(H / beta / f) * f);
        wb = @max(f, @floor(W / beta / f) * f);
    } else if (hb * wb < min_px) {
        const beta = @sqrt(min_px / (H * W));
        hb = @ceil(H * beta / f) * f;
        wb = @ceil(W * beta / f) * f;
    }
    if (hb < f or wb < f) return error.ImageTooSmall;
    return .{ @intFromFloat(hb), @intFromFloat(wb) };
}

/// An image's bytes to patches under `max_tokens` tokens (allocated in `gpa`).
pub fn prepare(gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32) Error!Prepared {
    const img = try decode(gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, std.hash.Wyhash.hash(0x0e55, bytes), max_tokens);
}

pub fn prepareRgb(gpa: std.mem.Allocator, img: Rgb, hash: u64, max_tokens: u32) Error!Prepared {
    const size = try smartResize(img.h, img.w, max_tokens);
    const rh = size[0];
    const rw = size[1];
    const resized = try glm_image.resize(gpa, img, rw, rh);
    defer resized.deinit(gpa);
    const gh = rh / vision.patch;
    const gw = rw / vision.patch;
    const P: usize = @as(usize, gh) * gw;
    const pixels = try gpa.alloc(f32, P * vision.patch_width);
    errdefer gpa.free(pixels);
    const pp = vision.patch * vision.patch;
    for (0..P) |i| {
        const at = vision.patchAt(@intCast(i), gw);
        const dst = pixels[i * vision.patch_width ..][0..vision.patch_width];
        for (0..3) |c| for (0..vision.patch) |y| for (0..vision.patch) |x| {
            const py = at[0] * vision.patch + y;
            const px = at[1] * vision.patch + x;
            const v: f32 = @floatFromInt(resized.px[(py * rw + px) * 3 + c]);
            // rescale (1/255) then normalize (mean 0.5, std 0.5), in fp32 as the processor's fused op
            const n = (v * (1.0 / 255.0) - 0.5) / 0.5;
            for (0..vision.temporal) |t| dst[c * vision.temporal * pp + t * pp + y * vision.patch + x] = n;
        };
    }
    return .{ .pixels = pixels, .gh = gh, .gw = gw, .tokens = @intCast(P / vision.patches_per_token), .hash = hash };
}

test "smart_resize: the reference's grids for the fixture sizes" {
    try std.testing.expectEqual([2]u32{ 224, 320 }, try smartResize(15, 20, 1024)); // tiny: up to the floor
    try std.testing.expectEqual([2]u32{ 384, 512 }, try smartResize(389, 517, 1024));
    try std.testing.expectEqual([2]u32{ 256, 256 }, try smartResize(96, 96, 1024));
    try std.testing.expectEqual([2]u32{ 224, 320 }, try smartResize(200, 300, 1024));
    try std.testing.expectEqual([2]u32{ 864, 1152 }, try smartResize(1500, 2000, 1024)); // down to the budget
}
