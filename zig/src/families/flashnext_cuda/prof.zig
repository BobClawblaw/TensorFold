//! TENSORFOLD_FN_PROFILE=<rounds>: rank 0's backend calls timed by kind, and the host's time between them, logged
//! every that many rounds (0 or unset: off). A development measure; it changes nothing it times.
const std = @import("std");

pub const Kind = enum { prefill, prefills, verify, shared, keep, draft, drafts, gap };
const N = @typeInfo(Kind).@"enum".field_names.len;

pub const Prof = struct {
    every: u64 = 0,
    ms: [N]f64 = @splat(0),
    n: [N]u64 = @splat(0),
    rows: u64 = 0, // verify rows (lone and shared)
    parts: u64 = 0, // streams in shared rounds
    last: ?std.Io.Timestamp = null,
    enq: ?*f64 = null, // the engine's verify host time until read-back, and its wait there (forward.zig)
    wait: ?*f64 = null,

    pub fn init() Prof {
        const v = std.c.getenv("TENSORFOLD_FN_PROFILE") orelse return .{};
        return .{ .every = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0 };
    }

    /// The call's start (null when off); the time since the previous call ended counts as host time between calls.
    pub fn begin(p: *Prof, io: std.Io) ?std.Io.Timestamp {
        if (p.every == 0) return null;
        const now = std.Io.Timestamp.now(io, .awake);
        if (p.last) |l| {
            const gap = since(l, now);
            if (gap < 200) { // longer: the server was idle
                p.ms[@intFromEnum(Kind.gap)] += gap;
                p.n[@intFromEnum(Kind.gap)] += 1;
            }
        }
        return now;
    }

    pub fn end(p: *Prof, io: std.Io, kind: Kind, t0: ?std.Io.Timestamp, rows: usize, parts: usize) void {
        const t = t0 orelse return;
        const now = std.Io.Timestamp.now(io, .awake);
        p.ms[@intFromEnum(kind)] += since(t, now);
        p.n[@intFromEnum(kind)] += 1;
        p.rows += rows;
        p.parts += parts;
        p.last = now;
        const rounds = p.n[@intFromEnum(Kind.verify)] + p.n[@intFromEnum(Kind.shared)];
        if ((kind == .verify or kind == .shared) and rounds >= p.every) p.flush();
    }

    fn flush(p: *Prof) void {
        const rounds: f64 = @floatFromInt(p.n[@intFromEnum(Kind.verify)] + p.n[@intFromEnum(Kind.shared)]);
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("flash next profile: {d} rounds, {d:.1} rows and {d:.2} streams a shared round; ms a round:", .{
            rounds, @as(f64, @floatFromInt(p.rows)) / rounds,
            if (p.n[@intFromEnum(Kind.shared)] > 0) @as(f64, @floatFromInt(p.parts)) / @as(f64, @floatFromInt(p.n[@intFromEnum(Kind.shared)])) else 0,
        }) catch {};
        for (0..N) |i| if (p.n[i] > 0) w.print(" {s} {d:.2} (x{d:.2})", .{
            @tagName(@as(Kind, @enumFromInt(i))), p.ms[i] / rounds, @as(f64, @floatFromInt(p.n[i])) / rounds,
        }) catch {};
        if (p.enq) |enq| if (p.wait) |wt| {
            w.print("; verify launch {d:.2} wait {d:.2}", .{ enq.* / rounds, wt.* / rounds }) catch {};
            enq.* = 0;
            wt.* = 0;
        };
        std.log.info("{s}", .{w.buffered()});
        p.* = .{ .every = p.every, .enq = p.enq, .wait = p.wait };
    }
};

fn since(a: std.Io.Timestamp, b: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(a.durationTo(b).toNanoseconds())) / 1e6;
}
