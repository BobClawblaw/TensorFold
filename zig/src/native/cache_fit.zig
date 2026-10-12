//! The prompt cache's budget from memory: what 70% of RAM leaves past the server's footprint once loaded, less a margin.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");

pub const GiB: u64 = 1 << 30;
/// The share of RAM a served model may hold: the rest stays with macOS and other processes.
pub const SHARE_PERCENT = 70;
/// Kept back from the cache for buffers that grow with long prompts.
pub const MARGIN: u64 = 2 * GiB;

/// The cache's budget, what fits under the cap, and the cap itself (bytes).
pub const Fit = struct { budget: u64, room: u64, cap: u64 };

/// The budget for `total` bytes of RAM and a footprint of `ready` bytes once loaded: `gib` when given and it fits (or `over`),
/// else all that fits; a given budget past the room is refused.
pub fn fit(total: u64, ready: u64, gib: ?f64, over: bool) error{CacheOverCap}!Fit {
    return fitUnder(total / 100 * SHARE_PERCENT, ready, gib, over);
}

/// `fit` under a given ceiling `cap` (bytes) instead of 70% of RAM: a load limit covers the cache as well.
pub fn fitUnder(cap: u64, ready: u64, gib: ?f64, over: bool) error{CacheOverCap}!Fit {
    const room = cap -| ready -| MARGIN;
    const want = if (gib) |g| (if (g > 0) std.math.lossyCast(u64, g * GiB) else 0) else room;
    if (want > room and !over) return error.CacheOverCap;
    return .{ .budget = want, .room = room, .cap = cap };
}

/// The ceiling a served model may hold: 70% of `total`, or `text` (a load limit in GB, as GLM_LOAD_LIMIT_GB) up to
/// `most` (the smaller of RAM and the GPU's recommended working set); text that is not a number above 0 is refused.
pub const Ceiling = struct { bytes: u64, default: u64, asked: bool = false, capped: bool = false };

pub fn ceiling(total: u64, most: u64, text: ?[]const u8) error{BadLoadLimit}!Ceiling {
    const default = total / 100 * SHARE_PERCENT;
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return .{ .bytes = default, .default = default };
    const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
    if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
    const top: f64 = @floatFromInt(@min(total, most));
    return .{ .bytes = @intFromFloat(@min(gb * 1e9, top)), .default = default, .asked = true, .capped = gb * 1e9 > top };
}

/// This machine's RAM (null where the OS does not say).
pub fn ram() ?u64 {
    return std.process.totalSystemMemory() catch null;
}

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

/// This process's physical footprint as macOS counts it, Metal buffers included (rusage_info_v4; null elsewhere).
pub fn footprint() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var info: [43]u64 = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return info[9];
}

/// The cache plan: what 70% of RAM leaves past this process, or `gib` if it fits (else refused unless `over`).
pub fn budget(gib: ?f64, over: bool, spare: u64, a: std.mem.Allocator, why: *[]const u8, rank: []const u8) !api.PromptCachePlan {
    return budgetUnder(null, gib, over, spare, a, why, rank);
}

/// `budget` under a load limit `limit` (bytes; null: 70% of RAM), so the cache and the rest stay under one ceiling.
pub fn budgetUnder(limit: ?u64, gib: ?f64, over: bool, spare: u64, a: std.mem.Allocator, why: *[]const u8, rank: []const u8) !api.PromptCachePlan {
    const total = ram() orelse 0;
    const ready = footprint() orelse 0;
    if (total == 0 or ready == 0) {
        const b = if (gib) |g| (if (g > 0) std.math.lossyCast(u64, g * GiB) else 0) else spare;
        std.log.info("prompt cache: {d:.1} GiB for kept prompt states{s} (no memory reading)", .{ gibs(b), rank });
        return .{ .source = if (gib != null) .explicit else .metal_working_set, .budget_bytes = b, .explicit_budget = gib != null };
    }
    const cap = limit orelse total / 100 * SHARE_PERCENT;
    const f = fitUnder(cap, ready, gib, over) catch |e| {
        const left = (fitUnder(cap, ready, null, false) catch unreachable).room; // without a given budget it never refuses
        if (limit != null) {
            why.* = try std.fmt.allocPrint(a, "--prompt-cache-gib {d} would take this server past its load limit: it holds {d:.1} GiB once loaded, the limit is {d:.1} GiB, and {d} GiB stays free for prompt buffers, so {d:.1} GiB is left for kept prompt states. Leave --prompt-cache-gib out to use that, pass a smaller one, raise the limit, or add --prompt-cache-over-cap to keep {d} GiB anyway.", .{ gib.?, gibs(ready), gibs(cap), MARGIN >> 30, gibs(left), gib.? });
        } else why.* = try std.fmt.allocPrint(a, "--prompt-cache-gib {d} would take this server past 70% of this Mac's memory: it holds {d:.1} GiB once loaded, 70% of {d:.0} GiB is {d:.1} GiB, and {d} GiB stays free for prompt buffers, so {d:.1} GiB is left for kept prompt states. Leave --prompt-cache-gib out to use that, pass a smaller one, or add --prompt-cache-over-cap to keep {d} GiB anyway.", .{ gib.?, gibs(ready), gibs(total), gibs(cap), MARGIN >> 30, gibs(left), gib.? });
        return e;
    };
    std.log.info("prompt cache: {d:.1} GiB from {d:.1} GiB free under the {s} ({d:.1} GiB in use once loaded, {d:.0} GiB of RAM, {d} GiB kept for prompt buffers){s}{s}", .{ gibs(f.budget), gibs(f.room), if (limit != null) "load limit" else "70% cap", gibs(ready), gibs(total), MARGIN >> 30, if (f.budget > f.room) ", past the cap by --prompt-cache-over-cap" else "", rank });
    return .{
        .source = .physical_footprint,
        .budget_bytes = f.budget,
        .explicit_budget = gib != null,
        .over_cap = f.budget > f.room,
        .ram_bytes = total,
        .ready_footprint_bytes = ready,
        .cap_bytes = f.cap,
        .room_bytes = f.room,
        .margin_bytes = MARGIN,
    };
}

/// GiB with one decimal, for log lines and messages.
pub fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / GiB;
}

test "the default budget is what 70% of RAM leaves past the footprint at ready and the margin" {
    const ram256: u64 = 256 * GiB;
    const f = try fit(ram256, 172 * GiB, null, false); // this tip on a 256 GB M5 Ultra
    try std.testing.expectEqual(ram256 / 100 * 70, f.cap);
    try std.testing.expectEqual(f.cap - 174 * GiB, f.budget);
    try std.testing.expect(gibs(f.budget) > 5.0 and gibs(f.budget) < 5.3);
    try std.testing.expectEqual(@as(u64, 0), (try fit(ram256, 179 * GiB, null, false)).budget); // nothing left: no cache
    try std.testing.expectEqual(@as(u64, 0), (try fit(192 * GiB, 172 * GiB, null, false)).budget); // a 192 GB Mac: past the cap already
}

test "a given budget that fits is kept, a larger one is refused unless the override is set" {
    const ram256: u64 = 256 * GiB;
    try std.testing.expectEqual(@as(u64, 4 * GiB), (try fit(ram256, 172 * GiB, 4, false)).budget);
    try std.testing.expectEqual(@as(u64, 0), (try fit(ram256, 172 * GiB, 0, false)).budget);
    try std.testing.expectError(error.CacheOverCap, fit(ram256, 172 * GiB, 8, false));
    try std.testing.expectError(error.CacheOverCap, fit(ram256, 172 * GiB, 16, false));
    try std.testing.expectEqual(@as(u64, 16 * GiB), (try fit(ram256, 172 * GiB, 16, true)).budget);
    try std.testing.expectEqual(@as(u64, 64 * GiB), (try fit(512 * GiB, 172 * GiB, 64, false)).budget); // a 512 GB Mac has room
}

test "under a load limit the automatic cache gets what the limit leaves, and a given one is held to the same limit" {
    // 100 GiB of RAM, a 60 GiB load limit, 50 GiB in use once the sessions are allocated (the case in review)
    const cap: u64 = 60 * GiB;
    const ready: u64 = 50 * GiB;
    const auto = try fitUnder(cap, ready, null, false);
    try std.testing.expectEqual(@as(u64, 8 * GiB), auto.budget); // 60 - 50 - 2, not the 18 GiB 70% of RAM leaves
    try std.testing.expect(ready + MARGIN + auto.budget <= cap);
    try std.testing.expectEqual(@as(u64, 18 * GiB), (try fit(100 * GiB, ready, null, false)).budget); // 70% of RAM: past the limit
    try std.testing.expectEqual(@as(u64, 8 * GiB), (try fitUnder(cap, ready, 8, false)).budget);
    try std.testing.expectError(error.CacheOverCap, fitUnder(cap, ready, 16, false));
    try std.testing.expectEqual(@as(u64, 16 * GiB), (try fitUnder(cap, ready, 16, true)).budget); // the override still passes it
    // a raised limit raises both: past 70% of RAM a given cache that fits under the limit needs no override
    const raised: u64 = 80 * GiB;
    try std.testing.expectEqual(@as(u64, 16 * GiB), (try fitUnder(raised, ready, 16, false)).budget);
    try std.testing.expectError(error.CacheOverCap, fit(100 * GiB, 60 * GiB, 16, false));
    try std.testing.expectEqual(@as(u64, 18 * GiB), (try fitUnder(raised, 60 * GiB, null, false)).budget);
}

test "the ceiling: 70% of RAM, or a load limit in GB up to RAM and the GPU's working set; bad text refused" {
    const ram256: u64 = 256 * GiB;
    const ws: u64 = 239_000_000_000;
    const d = try ceiling(ram256, ws, null);
    try std.testing.expectEqual(ram256 / 100 * 70, d.bytes);
    try std.testing.expect(!d.asked and !d.capped);
    const asked = try ceiling(ram256, ws, "215");
    try std.testing.expectEqual(@as(u64, 215_000_000_000), asked.bytes);
    try std.testing.expect(asked.asked and !asked.capped and asked.bytes > asked.default);
    try std.testing.expect((try ceiling(ram256, ws, " 300 ")).capped);
    try std.testing.expectEqual(ws, (try ceiling(ram256, ws, "300")).bytes);
    try std.testing.expectEqual(ram256 / 100 * 70, (try ceiling(ram256, ws, "  ")).bytes);
    try std.testing.expectError(error.BadLoadLimit, ceiling(ram256, ws, "lots"));
    try std.testing.expectError(error.BadLoadLimit, ceiling(ram256, ws, "0"));
    try std.testing.expectError(error.BadLoadLimit, ceiling(ram256, ws, "-4"));
}

test "the OS readings: RAM and this process's footprint" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const r = ram() orelse return error.NoRam;
    const p = footprint() orelse return error.NoFootprint;
    try std.testing.expect(r >= 8 * GiB and p > 0 and p < r);
}
