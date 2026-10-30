//! A picture an image engine makes from a prompt: the request, and what the engine serves.

const std = @import("std");

/// A picture for Engine.submitImage: caller-owned until its finished event, after the PNG, a cancel or a failure.
pub const ImageRequest = struct {
    prompt: []const u8,
    negative: []const u8 = "",
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    guidance: f32,
    /// The PNG's bytes, appended with `gpa` on the engine's thread.
    png: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
};

/// The largest picture, step count and prompt one request may ask for.
pub const ImageInfo = struct {
    max_pixels: u64,
    max_side: u32,
    max_steps: u32,
    max_prompt_bytes: u32,
    /// Requests that may wait behind the one running.
    queue: u32,
};
