//! The voices a speech server holds: Drama 3's --voices files, or a ZONOS3 Mini release's default voices.
const std = @import("std");
const api = @import("engine_api");
const drama_voice = @import("drama_voice.zig");
const zonos = @import("zonos_speech.zig");

pub const Voices = struct {
    drama: []const drama_voice.Voice = &.{},
    zonos: ?*zonos.State = null,
};

/// Loaded once at startup from the arena `a`; `voices_dir` needs a speech engine, and the default voice must exist.
pub fn load(a: std.mem.Allocator, io: std.Io, info: api.Info, model_dir: []const u8, voices_dir: ?[]const u8, default_voice: []const u8) !Voices {
    var out: Voices = .{};
    if (voices_dir) |dir| {
        if (info.speech == null) return error.VoicesRequireSpeech;
        out.drama = try drama_voice.load(a, io, dir);
    }
    if (info.speech != null) out.zonos = try zonos.load(a, io, model_dir, default_voice);
    if (out.zonos == null) _ = try drama_voice.select(out.drama, default_voice, "default");
    return out;
}
