//! Captured graphs any sequence launches: moved onto its buffers, or captured again on them when a node cannot move.

const std = @import("std");
const cuda = @import("cuda");

pub const Replay = struct {
    exec: cuda.graph.Exec,
    plan: ?cuda.relocate.Plan, // null: captured again for each other sequence
    tmpl: ?cuda.graph.Graph, // the plan names this graph's nodes, so it lives as long as the plan

    pub fn deinit(r: *Replay, gpa: std.mem.Allocator) void {
        r.exec.deinit();
        if (r.plan) |*p| p.deinit(gpa);
        if (r.tmpl) |*g| g.deinit();
        r.* = undefined;
    }
};

/// A sequence's own graphs for the replays without a plan, by replay id.
pub const Recaptured = std.AutoHashMapUnmanaged(u32, cuda.graph.Exec);

pub fn freeRecaptured(gpa: std.mem.Allocator, m: *Recaptured) void {
    var it = m.valueIterator();
    while (it.next()) |x| x.deinit();
    m.deinit(gpa);
    m.* = .{};
}

/// Captures and moves since load, for TF_GRAPH_LOG's per-request line and the receipts.
pub const Stats = struct {
    captures: u64 = 0,
    capture_ns: u64 = 0,
    moves: u64 = 0,
    replays: u64 = 0,
    eager: u64 = 0,

    pub fn since(now: Stats, then: Stats) Stats {
        return .{ .captures = now.captures - then.captures, .capture_ns = now.capture_ns - then.capture_ns, .moves = now.moves - then.moves, .replays = now.replays - then.replays, .eager = now.eager - then.eager };
    }
};

/// One graph of `body(ctx)` captured and uploaded, with a plan to move it when `regions` (the capturer's) is set.
pub fn capture(gpa: std.mem.Allocator, io: std.Io, stream: cuda.Stream, regions: ?*const cuda.relocate.Regions, stats: *Stats, ctx: anytype, comptime body: fn (@TypeOf(ctx)) anyerror!void) !Replay {
    const t0 = std.Io.Clock.awake.now(io);
    try cuda.graph.beginCapture(stream, .thread_local);
    body(ctx) catch |err| {
        if (cuda.graph.endCapture(stream)) |g| {
            var x = g;
            x.deinit();
        } else |_| {}
        return err;
    };
    var g = try cuda.graph.endCapture(stream);
    var keep = false;
    defer if (!keep) g.deinit();
    var plan: ?cuda.relocate.Plan = if (regions) |r| try cuda.relocate.read(gpa, g, r) else null;
    errdefer if (plan) |*p| p.deinit(gpa);
    var exec = try g.instantiate();
    errdefer exec.deinit();
    try exec.upload(stream);
    try stream.synchronize();
    keep = plan != null;
    stats.captures += 1;
    stats.capture_ns += @intCast(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds());
    return .{ .exec = exec, .plan = plan, .tmpl = if (keep) g else null };
}

/// Whether TF_GRAPH_RELOCATE=0 asks every other sequence to capture its own graphs (the refusal path, on purpose).
pub fn relocateFromEnv() bool {
    const v = std.c.getenv("TF_GRAPH_RELOCATE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Whether TF_GRAPH_LOG=1 asks for a line a request: its captures, their time, moves and replays.
pub fn logFromEnv() bool {
    const v = std.c.getenv("TF_GRAPH_LOG") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}
