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
const Resident = @import("resident.zig").Resident;

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
    resident: ?Resident = null, // the engine's, sessions' and tower's buffers, kept wired between requests
    learned: ?api.prompt_imprint.Imprint = null, // --learn: shared cuts' and evicted states on disk
};

/// --learn's states under `root`, by checkpoint, kernel sources and modes, OS build and chip: the shared sessions'
/// own (their prompt pass cuts its chunks as the single-reply engine's need not).
fn learnedStates(gpa: std.mem.Allocator, eng: *fx.Engine, root: []const u8, cap: u64) !api.prompt_imprint.Imprint {
    var h = std.hash.Wyhash.init(0x6662);
    const modes = [_]u64{ @intFromBool(eng.r.xnew), @intFromBool(eng.r.hc_up), @intFromBool(eng.r.device.tensorUnits()) };
    for ([_]u64{ eng.identity_hash, tf.flashnext_learned.sourceHash(), modes[0], modes[1], modes[2] }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(eng.r.device.name()));
    const m = try api.prompt_imprint.Imprint.open(gpa, root, h.final(), cap);
    std.log.info("flash next: prompt states on disk in {s} ({d} known, {d} of {d} MiB): shared cuts, and evicted states as a spill", .{ m.dir, m.metas.items.len, m.total() >> 20, cap >> 20 });
    return m;
}

/// macOS unwires GPU memory after 1-2 s idle and the next request re-wires it all (2 s before its first token): the
/// engine's buffers, every session's and the tower's join one residency set the idle keepalive uses.
fn holdResident(h: *Host) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    try list.appendSlice(h.gpa, h.eng.r.held.items);
    for (h.back.sessions) |s| try list.appendSlice(h.gpa, s.owned.items);
    try list.append(h.gpa, h.back.hidden.b);
    if (h.vision) |v| try list.appendSlice(h.gpa, &v.tower.buffers());
    h.resident = try Resident.initTicked(h.eng.r.device, h.eng.r.queue, list.items, true);
    h.warm = (&h.resident.?).target();
    std.log.info("flash next: {d} buffers ({d:.1} GiB) held resident between requests", .{ list.items.len, fit.gibs(h.resident.?.set.allocatedSize()) });
}

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

/// The sessions past the engine's own from FLASHNEXT_SLOT_CONTEXTS: token counts, `CONTEXTxN` for N sessions of one
/// size (as "131072,65536x2,16384x8"), each holding at most `capacity` rows; null when unset.
fn slotContexts(gpa: std.mem.Allocator, capacity: usize) !?[]usize {
    const text = std.mem.trim(u8, std.mem.span(std.c.getenv("FLASHNEXT_SLOT_CONTEXTS") orelse return null), " \t");
    if (text.len == 0) return null;
    var caps: std.ArrayList(usize) = .empty;
    errdefer caps.deinit(gpa);
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |entry| {
        const e = std.mem.trim(u8, entry, " \t");
        const x = std.mem.indexOfScalar(u8, e, 'x');
        const tokens = std.fmt.parseInt(usize, if (x) |i| e[0..i] else e, 10) catch return error.BadSlotContexts;
        const n = if (x) |i| std.fmt.parseInt(usize, e[i + 1 ..], 10) catch return error.BadSlotContexts else 1;
        if (tokens < 1024 or n == 0) return error.BadSlotContexts;
        for (0..n) |_| try caps.append(gpa, @min(capacity, tokens + @as(usize, @intCast(fx.MARGIN))));
    }
    if (caps.items.len == 0 or caps.items.len + 1 > tf.flashnext_replay.MAXR) return error.BadSlotContexts;
    return try caps.toOwnedSlice(gpa);
}

/// Reply tokens a stream's session must hold past its prompt at admission (FLASHNEXT_SLOT_GROW, default 8192): a
/// stream that outgrows it moves to a free larger session at a round's end, or waits for one.
fn slotGrow() usize {
    const v = std.c.getenv("FLASHNEXT_SLOT_GROW") orelse return 8192;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 8192;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, dump: ?[]const u8, window: i64, streams: u32, cache_gib: ?f64, over: bool, learn: ?[]const u8, learn_cap: u64, learn_min_free: u64, vision: ?api.VisionOpen, a: std.mem.Allocator, why: *[]const u8) !api.Opened {
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
    const tiers = slotContexts(gpa, capacity) catch |err| {
        std.log.err("flash next: FLASHNEXT_SLOT_CONTEXTS is not a list of sessions: token counts of 1024 or more, CONTEXTxN for N alike, at most {d} sessions past the engine's own", .{tf.flashnext_replay.MAXR - 1});
        return err;
    };
    defer if (tiers) |t| gpa.free(t);
    var caps: std.ArrayList(usize) = .empty;
    defer caps.deinit(gpa);
    try caps.append(gpa, capacity); // the engine's own session: its whole window
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
    if (tiers != null) h.back.grow = slotGrow();
    try h.back.timeCosts();
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
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "flashnext-zig", .lanes = slots, .context_window = @intCast(window), .prefill_step = @intCast(eng.pr.step), .prompt_cache_plan = plan, .vision = if (vis) |v| v.offer() else null, .admit = .{ .ctx = &h.back, .fits = Backend.fits } });
    // a lone text request runs on the engine's GPU-side rounds until a second one arrives (FLASHNEXT_NO_LONE=1: shared rounds)
    if (eng.r.gdn_kept == null and std.c.getenv("FLASHNEXT_NO_LONE") == null) h.host.lone = .{ .ctx = h, .sampled = true, .images = false, .run = loneRun };
    h.cache = null;
    if (plan.budget_bytes > 0) {
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = &h.back, .vtable = &.{ .bytes = Backend.snapBytes, .save = Backend.snapSave, .restore = Backend.snapRestore, .drop = Backend.snapDrop, .charged = Backend.snapCharged, .spare = Backend.snapSpare, .trim = Backend.snapTrim, .reuses = Backend.snapReuses, .write = Backend.snapWrite, .read = Backend.snapRead, .forget = Backend.snapForget, .forget_checked = api.prompt_cache.singleForgetChecked, .reclaim = api.prompt_cache.singleReclaim } }, .{ .lookahead = 1, .planned = true }, plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.learned = null;
    if (learn) |root| if (h.cache) |*store| {
        h.learned = try learnedStates(gpa, eng, root, learn_cap);
        h.learned.?.admission.floor = learn_min_free;
        store.imprint = &h.learned.?;
        store.spill = true;
    };
    errdefer if (h.learned) |*m| m.deinit();
    h.host.explain = .{ .text = words };
    h.warm = .{ .queue = eng.r.queue };
    h.resident = null;
    if (std.c.getenv("FLASHNEXT_NO_RESIDENCY") == null) holdResident(h) catch |err| std.log.warn("flash next: no residency set ({s}): the first request after an idle second re-wires the model", .{@errorName(err)});
    errdefer if (h.resident) |*r| r.deinit();
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    var listed: [tf.flashnext_replay.MAXR * 8]u8 = undefined;
    var w: std.Io.Writer = .fixed(&listed);
    for (caps.items, 0..) |c, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", c }) catch break;
    std.log.info("Flash Next: {d} session slots ({s} tokens), shared target forwards, {s}{s}", .{ slots, w.buffered(), if (h.back.serial_drafts) "serial MTP drafts" else "every stream's drafts in one command buffer", if (h.host.lone != null) "; a lone text request on the engine's own rounds" else "" });
    return .{ .engine = h.host.engine(), .close = close, .ctx = h };
}

fn loneRun(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
    const h: *Host = @ptrCast(@alignCast(ctx));
    return h.back.lone(s, hooks);
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
    if (h.resident) |*r| r.deinit();
    if (h.learned) |*m| m.deinit();
    h.back.deinit();
    if (h.vision) |v| v.deinit();
    h.eng.deinit();
    h.gpa.destroy(h);
}
