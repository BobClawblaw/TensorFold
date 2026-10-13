//! Opt-in single-Mac Flash Next serving through the lane core, sharing each target forward.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const Backend = tf.flashnext_backend.Backend;
const Session = tf.flashnext_session.Session;
const fx = tf.flashnext_engine;
const fit = @import("cache_fit.zig");

const Host = struct {
    gpa: std.mem.Allocator,
    eng: *fx.Engine,
    back: Backend,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null,
    warm: mtl.keepalive.Target,
};

/// What the server may hold in all, sessions and prompt cache: 70% of RAM, or FLASHNEXT_LOAD_LIMIT_GB (GB, as
/// GLM_LOAD_LIMIT_GB) up to the GPU's recommended working set; past the default the log names the risk, as GLM's does.
fn ceiling(total: u64, working_set: u64) error{BadLoadLimit}!u64 {
    const text: ?[]const u8 = if (std.c.getenv("FLASHNEXT_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const c = fit.ceiling(total, working_set, text) catch |err| {
        std.log.err("flash next: FLASHNEXT_LOAD_LIMIT_GB={s} is not a load limit: give a number of GB above 0, or unset it for 70% of RAM", .{text.?});
        return err;
    };
    if (c.capped) std.log.info("flash next: FLASHNEXT_LOAD_LIMIT_GB={s} passes this GPU's recommended working set; the limit is {d:.1} GB", .{ text.?, fit.gibs(c.bytes) * 1.073741824 });
    if (c.bytes > c.default) std.log.warn("flash next: FLASHNEXT_LOAD_LIMIT_GB lets the server (sessions and prompt cache) take this Mac to {d:.1} GiB, past the default {d:.1} GiB (70% of RAM): a load that wires most of the RAM can stall macOS until its watchdog kills the server", .{ fit.gibs(c.bytes), fit.gibs(c.default) });
    return c.bytes;
}

/// The sessions past the engine's own from FLASHNEXT_SLOT_CONTEXTS (as "131072,65536x2,16384x8"); null when unset.
fn slotContexts(gpa: std.mem.Allocator, capacity: usize) !?[]usize {
    const text = std.mem.trim(u8, std.mem.span(std.c.getenv("FLASHNEXT_SLOT_CONTEXTS") orelse return null), " \t");
    if (text.len == 0) return null;
    return try tf.flashnext_slots.parse(gpa, text, capacity, @intCast(fx.MARGIN), tf.flashnext_replay.MAXR - 1);
}

/// Reply tokens a session must hold past the prompt at admission (FLASHNEXT_SLOT_GROW, default 8192); past them a stream moves.
fn slotGrow() usize {
    const v = std.c.getenv("FLASHNEXT_SLOT_GROW") orelse return 8192;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 8192;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, dump: ?[]const u8, window: i64, streams: u32, cache_gib: ?f64, over: bool, a: std.mem.Allocator, why: *[]const u8) !api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const capacity: usize = @intCast(@min(tf.flashnext_replay.CAP, window + fx.MARGIN));
    const eng = try fx.Engine.load(gpa, io, dir, dump);
    errdefer eng.deinit();
    if (eng.r.tp != null) return error.BatchSpeedUpUnsupported;
    try eng.warm();
    eng.hostMode();
    const ready = fit.footprint() orelse return error.NoMemoryReading;
    const total = fit.ram() orelse return error.NoMemoryReading;
    const cap = try ceiling(total, eng.r.device.maxWorkingSet());
    const reserved = if (!over) (if (cache_gib) |g| (if (g > 0) std.math.lossyCast(u64, g * fit.GiB) else 0) else 0) else 0;
    const fixed = Session.overhead(tf.flashnext_replay.CAP) + tf.flashnext_replay.MAXR * tf.flashnext_replay.WIDE * 2;
    const available = cap -| ready -| fit.MARGIN -| reserved;
    if (available < fixed) return error.OverMemoryLimit;
    const room = available - fixed;
    const tiers = slotContexts(gpa, capacity) catch |err| {
        std.log.err("flash next: FLASHNEXT_SLOT_CONTEXTS is not a list of sessions: token counts of 1024 or more, CONTEXTxN for N alike, at most {d} sessions past the engine's own", .{tf.flashnext_replay.MAXR - 1});
        return err;
    };
    defer if (tiers) |t| gpa.free(t);
    var caps: std.ArrayList(usize) = .empty;
    defer caps.deinit(gpa);
    try caps.append(gpa, capacity); // the engine's own session: the whole window
    if (tiers) |t| {
        var bytes: u64 = 0;
        for (t) |c| bytes += Session.bytes(c);
        if (bytes > room) {
            std.log.err("flash next: the sessions FLASHNEXT_SLOT_CONTEXTS lists take {d:.1} GiB; {d:.1} GiB fits under the limit beside the prompt cache", .{ fit.gibs(bytes), fit.gibs(room) });
            return error.OverMemoryLimit;
        }
        try caps.appendSlice(gpa, t);
    } else {
        const n = try tf.flashnext_batch_meta.fit(streams, true, room, Session.bytes(capacity));
        for (1..n) |_| try caps.append(gpa, capacity);
    }
    const slots: u32 = @intCast(caps.items.len);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.eng = eng;
    h.back = try Backend.init(gpa, eng, caps.items);
    errdefer h.back.deinit();
    if (tiers != null) {
        h.back.sized = true;
        h.back.grow = slotGrow();
    }
    const allocated = fit.footprint() orelse return error.NoMemoryReading;
    if (allocated +| fit.MARGIN +| reserved > cap) return error.OverMemoryLimit;
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), tf.flashnext_replay.MAXR, tf.flashnext_replay.MAXR - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.clock.clock());
    errdefer h.core.deinit();
    const plan = try fit.budgetUnder(cap, cache_gib, over, 0, a, why, ""); // the cache under the same limit as the sessions
    eng.snap_pool.max = plan.budget_bytes;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "flashnext-zig", .lanes = slots, .context_window = @intCast(window), .prefill_step = @intCast(eng.pr.step), .prompt_cache_plan = plan, .admit = if (tiers != null) .{ .ctx = &h.back, .fits = Backend.fits } else null });
    h.cache = null;
    if (plan.budget_bytes > 0) {
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = &h.back, .vtable = &.{ .bytes = Backend.snapBytes, .save = Backend.snapSave, .restore = Backend.snapRestore, .drop = Backend.snapDrop, .charged = Backend.snapCharged, .spare = Backend.snapSpare, .trim = Backend.snapTrim, .reuses = Backend.snapReuses } }, .{ .lookahead = 1, .planned = true }, plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.host.explain = .{ .text = words };
    h.warm = .{ .queue = eng.r.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    var listed: [tf.flashnext_replay.MAXR * 8]u8 = undefined;
    var w: std.Io.Writer = .fixed(&listed);
    for (caps.items, 0..) |c, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", c }) catch break;
    std.log.info("Flash Next: {d} session slots ({s} tokens), shared target forwards, serial MTP drafts", .{ slots, w.buffered() });
    return .{ .engine = h.host.engine(), .close = close, .ctx = h };
}

fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        else => null,
    };
}

fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.stop();
    if (h.cache) |*store| store.deinit();
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    h.eng.deinit();
    h.gpa.destroy(h);
}
