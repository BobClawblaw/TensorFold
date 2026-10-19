//! Prefer foreground and short prompts, with bounded waiting for every admitted fill.
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;

pub const preference_rounds = 8;

/// After eight skipped choices, oldest wait wins; each fill runs within eight plus the admitted lane count.
pub fn next(h: *const LaneHost) usize {
    var at: usize = 0;
    for (h.filling.items, 0..) |j, i| {
        const best = h.filling.items[at];
        const aged = j.fill_wait >= preference_rounds;
        const best_aged = best.fill_wait >= preference_rounds;
        if (aged != best_aged) {
            if (aged) at = i;
        } else if (aged) {
            if (j.fill_wait > best.fill_wait) at = i;
        } else if (j.request.background != best.request.background) {
            if (!j.request.background) at = i;
        } else if (j.stream.prompt_len - j.stream.cached < best.stream.prompt_len - best.stream.cached) at = i;
    }
    for (h.filling.items, 0..) |j, i| j.fill_wait = if (i == at) 0 else j.fill_wait +| 1;
    return at;
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}

/// A chunk of `s`'s prompt, timed: the live streams are owed rounds for it under `Info.decode_share`.
pub fn timed(h: *LaneHost, s: *lanes.Stream, first: bool, hook: lanes.backend.Committed) !bool {
    const began = now(h);
    defer {
        const share: f64 = h.info_.decode_share;
        if (share > 0 and h.core.activeCount() > 0) h.fill_owed += @intFromFloat(@as(f64, @floatFromInt(now(h) - began)) * share / (1 - share));
    }
    return h.core.fillStreamCommitted(s, first, hook);
}

/// Whether the live streams are still owed rounds for the last chunk (with none live, nothing is owed).
pub fn owed(h: *LaneHost) bool {
    if (h.core.activeCount() == 0) h.fill_owed = 0;
    return h.fill_owed > 0;
}

/// The live streams' round, paid against what they are owed (never into credit for later chunks).
pub fn round(h: *LaneHost) !void {
    const began = now(h);
    defer h.fill_owed = @max(0, h.fill_owed - (now(h) - began));
    return h.core.step();
}

fn now(h: *const LaneHost) i96 {
    return std.Io.Clock.awake.now(h.io).toNanoseconds();
}
