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
const vh = @import("flashnext_vision_host.zig");

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
    vision: ?*vh.Vis,
};

/// What the sessions may take the server to: 70% of RAM, or FLASHNEXT_LOAD_LIMIT_GB (GB, as GLM_LOAD_LIMIT_GB) up to the
/// GPU's recommended working set; past the default the log names the risk, as GLM's loader does.
fn ceiling(total: u64, working_set: u64) error{BadLoadLimit}!u64 {
    const text: ?[]const u8 = if (std.c.getenv("FLASHNEXT_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const c = fit.ceiling(total, working_set, text) catch |err| {
        std.log.err("flash next: FLASHNEXT_LOAD_LIMIT_GB={s} is not a load limit: give a number of GB above 0, or unset it for 70% of RAM", .{text.?});
        return err;
    };
    if (c.capped) std.log.info("flash next: FLASHNEXT_LOAD_LIMIT_GB={s} passes this GPU's recommended working set; the limit is {d:.1} GB", .{ text.?, fit.gibs(c.bytes) * 1.073741824 });
    if (c.bytes > c.default) std.log.warn("flash next: FLASHNEXT_LOAD_LIMIT_GB lets the sessions take this Mac to {d:.1} GiB, past the default {d:.1} GiB (70% of RAM): a load that wires most of the RAM can stall macOS until its watchdog kills the server", .{ fit.gibs(c.bytes), fit.gibs(c.default) });
    return c.bytes;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, dump: ?[]const u8, window: i64, streams: u32, cache_gib: ?f64, over: bool, vision: ?api.VisionOpen, a: std.mem.Allocator, why: *[]const u8) !api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    // slot 0 holds the engine's own window (past the trained one with YaRN); the others the trained window at most
    const capacity: usize = @intCast(@min(@as(i64, @intCast(tf.flashnext_replay.TRAINED)), window + fx.MARGIN));
    const eng = try fx.Engine.load(gpa, io, dir, dump);
    errdefer eng.deinit();
    if (eng.r.tp != null) return error.BatchSpeedUpUnsupported;
    try eng.warm();
    eng.hostMode();
    const vis: ?*vh.Vis = if (vision) |lim| try vh.attach(gpa, eng, dir, lim) else null; // before the memory reading
    errdefer if (vis) |v| v.deinit();
    const ready = fit.footprint() orelse return error.NoMemoryReading;
    const total = fit.ram() orelse return error.NoMemoryReading;
    const cap = try ceiling(total, eng.r.device.maxWorkingSet());
    const reserved = if (!over) (if (cache_gib) |g| (if (g > 0) std.math.lossyCast(u64, g * fit.GiB) else 0) else 0) else 0;
    const fixed = Session.overhead(tf.flashnext_replay.CAP) + tf.flashnext_replay.MAXR * tf.flashnext_replay.WIDE * 2;
    const available = cap -| ready -| fit.MARGIN -| reserved;
    if (available < fixed) return error.OverMemoryLimit;
    const room = available - fixed;
    const slots = try tf.flashnext_batch_meta.fit(streams, true, room, Session.bytes(capacity));
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.eng = eng;
    h.back = try Backend.init(gpa, eng, slots, capacity);
    errdefer h.back.deinit();
    h.vision = vis;
    h.back.tower = if (vis) |v| v.tower else null;
    const allocated = fit.footprint() orelse return error.NoMemoryReading;
    if (allocated +| fit.MARGIN +| reserved > cap) return error.OverMemoryLimit;
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), tf.flashnext_replay.MAXR, tf.flashnext_replay.MAXR - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.clock.clock());
    errdefer h.core.deinit();
    const plan = try fit.budget(cache_gib, over, 0, a, why, "");
    eng.snap_pool.max = plan.budget_bytes;
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "flashnext-zig", .lanes = slots, .greedy_only = true, .context_window = @intCast(window), .prefill_step = @intCast(eng.pr.step), .prompt_cache_plan = plan, .vision = if (vis) |v| v.offer() else null, .admit = .{ .ctx = &h.back, .fits = Backend.fits } });
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
    std.log.info("Flash Next: {d} session slots, shared target forwards, serial MTP drafts", .{slots});
    return .{ .engine = h.host.engine(), .close = close, .ctx = h };
}

fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.GreedyOnly => "the native Flash Next engine decodes greedily only: send temperature 0",
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
    if (h.vision) |v| v.deinit();
    h.eng.deinit();
    h.gpa.destroy(h);
}
