//! Flash Next (qwen4_exp) on two GB10s behind the native family interface: rank 0's lane core drives the engine
//! through a lane backend that first sends each call to rank 1, which replays it on its own half of the model
//! (``follow``). Every token is drawn inside the forward from both ranks' vocabulary shards (an NCCL exchange), so
//! both ranks reach the same tokens and rank 1 needs nothing back.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("api.zig");
const weights = @import("weights.zig");
const forward = @import("forward.zig");
const be = lanes.backend;

pub const model_type = "qwen4_exp";
pub const formats: []const []const u8 = &.{"affine-experts-int8"};
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 2048;
pub const two_ranks = true;

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    tp: u8 = 2,
    rank: u8 = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29600,
};

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
};

/// Device bytes this rank's half of the weights takes (the n-gram tables stay on the host).
pub fn deviceWeightBytes(io: std.Io, dir: []const u8) u64 {
    return weights.deviceBytes(io, dir) catch 40 << 30;
}

const max_rows = 16;            // a verify window's rows at most (pending + 15 drafts)
const max_depth = max_rows - 1;
/// The head stops at a draft it gives less than this (the Python engine's --mtp-confidence, the recipe's 0.70);
/// the slots past it are held with chance 0, so the lane core's allocator leaves them out of the window.
const draft_confidence: f64 = 0.7;

// -- the protocol rank 0 sends rank 1 ---------------------------------------------------------------------------

const Op = enum(u32) { prefill = 1, verify = 2, keep = 3, draft = 4, release = 5, stop = 6, ready = 7 };

const Writer = struct {
    buf: std.ArrayList(u8) = .empty,
    gpa: std.mem.Allocator,
    fn int(w: *Writer, v: anytype) !void {
        const x: u64 = @intCast(v);
        try w.buf.appendSlice(w.gpa, std.mem.asBytes(&x));
    }
    fn tokens(w: *Writer, t: []const u32) !void {
        try w.int(t.len);
        try w.buf.appendSlice(w.gpa, std.mem.sliceAsBytes(t));
    }
};

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,
    fn int(r: *Reader) !u64 {
        if (r.at + 8 > r.bytes.len) return error.ShortFrame;
        const v = std.mem.readInt(u64, r.bytes[r.at..][0..8], .little);
        r.at += 8;
        return v;
    }
    fn tokens(r: *Reader) ![]const u32 {
        const n: usize = @intCast(try r.int());
        if (r.at + 4 * n > r.bytes.len) return error.ShortFrame;
        const t: []align(1) const u32 = std.mem.bytesAsSlice(u32, r.bytes[r.at..][0 .. 4 * n]);
        r.at += 4 * n;
        return @alignCast(t);
    }
};

// -- the family's state -------------------------------------------------------------------------------------------

const Lane = struct {
    seq: *forward.Seq,
    id: u64,
    held: [max_depth]u32 = undefined,   // drafts held for the next window
    probs: [max_depth]f64 = undefined,  // the head's chance for each (0 past the confidence cut)
    nheld: u32 = 0,
    rows: u32 = 0,                      // the last verify's rows, not yet committed (0: nothing pending)
};

/// Commits the lane's last window: ``kept`` of its rows (the lane core calls keep only to cut a window short, so a
/// window it accepted whole is committed here, before the lane's next call). Rank 1 commits the same.
fn settle(self: *Owned, l: *Lane, kept: u32) !void {
    if (l.rows == 0) return;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(l.id);
    try w.int(l.rows);
    try w.int(kept);
    try self.send(.keep, &w);
    try forward.keep(self.e, l.seq, l.rows, kept);
    l.rows = 0;
}

const Owned = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    rank: u8,
    link: cuda.tp_link.Link,
    nccl: cuda.nccl.Library,
    ctx: api.Ctx,
    kernels: api.Kernels,
    store: api.Store,
    e: *forward.Engine,
    lanes_by: std.AutoHashMap(*lanes.Stream, Lane),
    by_id: std.AutoHashMap(u64, *forward.Seq),    // rank 1's sequences, by rank 0's stream id
    next_id: u64 = 1,
    drawn: [64]u32 = undefined,
    next: u64 = 0,
    costs: [max_rows]lanes.config.Cost = undefined, // a window's ms by width (rank 0, timed at open)
    cost_count: usize = 0,
    mtp_ms: f64 = 0, // one chained head step
    spare: ?*forward.Seq = null, // a released sequence, reset and reused by the next request (its memory, its graphs)

    fn obtain(self: *Owned) !*forward.Seq {
        if (self.spare) |s| {
            self.spare = null;
            try forward.resetSeq(self.e, s);
            return s;
        }
        return forward.newSeq(self.e);
    }

    fn retire(self: *Owned, s: *forward.Seq) void {
        if (self.spare == null) self.spare = s else forward.freeSeq(self.e, s);
    }

    fn take(self: *Owned, t: u32) u64 {
        self.drawn[self.next % self.drawn.len] = t;
        self.next += 1;
        return self.next - 1;
    }

    fn send(self: *Owned, op: Op, w: *Writer) !void {
        try self.link.send(@intFromEnum(op), w.buf.items);
        w.buf.clearRetainingCapacity();
    }
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels_dir: []const u8, o: Options) !Loaded {
    if (o.tp != 2) return error.TwoRanksOnly;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.io = io;
    own.rank = o.rank;
    // the link and the communicator first, while memory is fresh (NCCL registers its buffers at its first use)
    own.link = if (o.rank == 0) try cuda.tp_link.Link.lead(o.master_port, 600_000)
        else try cuda.tp_link.Link.follow(io, try std.Io.net.IpAddress.parse(o.master orelse return error.NoMaster, o.master_port), 600_000);
    errdefer own.link.close();
    own.nccl = try cuda.nccl.Library.open();
    errdefer own.nccl.close();
    const comm = try cuda.tp_link.communicator(&own.nccl, own.link);
    own.ctx = .{ .d = ctx.d, .ctx = ctx, .stream = try cuda.Stream.init(ctx.d, true), .nccl = &own.nccl, .comm = comm, .rank = o.rank };
    { // NCCL connects (and registers its buffers) at its first collective: here, before weights and tables take the
      // memory (a GB10 worker refuses the registration once they are resident)
        var warm = try cuda.DeviceBuffer.alloc(ctx.d, 4096);
        defer warm.free();
        try own.nccl.check(own.nccl.api.ncclAllGather(warm.ptr, warm.ptr + 2048, 256, .i32, comm, own.ctx.stream.handle), "warm gather");
        try own.ctx.stream.synchronize();
    }
    own.kernels = try api.Kernels.load(gpa, io, ctx.d, ctx.device, kernels_dir);
    errdefer own.kernels.deinit();
    own.store = try weights.load(gpa, io, &own.ctx, &own.kernels, dir, o.rank);
    errdefer own.store.deinit();
    own.e = try forward.init(gpa, io, &own.ctx, &own.kernels, &own.store, .{ .context = o.context, .max_rows = max_rows, .depth = if (o.drafts) max_depth else 0 });
    errdefer forward.deinit(own.e);
    try forward.prefetchTables(own.e); // the n-gram tables paged in (and locked) before the first request
    // both halves loaded before rank 0 serves: rank 1 says so once its engine is up (it loads more slowly)
    if (o.rank == 1) try own.link.send(@intFromEnum(Op.ready), "") else {
        const m = try own.link.recv(gpa);
        gpa.free(m.bytes);
        if (m.tag != @intFromEnum(Op.ready)) return error.RankOneNotReady;
    }
    own.lanes_by = .init(gpa);
    own.by_id = .init(gpa);
    own.next_id = 1;
    own.next = 0;
    own.cost_count = 0;
    own.spare = null;
    own.mtp_ms = 0;
    if (o.rank == 0) try calibrate(own, o.drafts); // rank 1 replays it in its follow loop
    return .{
        .backend = .{ .ptr = own, .vtable = &vtable },
        .facts = .{ .exact_width = max_rows, .mtp = o.drafts, .speculate = o.drafts, .speculate_early = false, .drafts = if (o.drafts) max_depth else 1,
                    .hidden_rows = false, .max_streams = 1, .batch_rows = max_rows,
                    .window_costs = own.costs[0..own.cost_count], .mtp_step_ms = own.mtp_ms,
                    .draft_probabilities = o.drafts },
        .rows = if (o.drafts) max_rows else 1,
        .stream_bytes = forward.seqBytes(own.e),
        .ctx = own,
        .deinit = release,
        .follow = if (o.rank == 1) followLoop else null,
    };
}

fn msSince(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
}

/// The lane core's costs on a throwaway sequence: a verify window of each width (best of three after a warm-up,
/// which also captures its graph) and a chained head step. Sent to rank 1 like any request.
fn calibrate(self: *Owned, drafts: bool) !void {
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    const id = self.next_id;
    self.next_id += 1;
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 5000);
    try w.int(id);
    try w.tokens(&prompt);
    try self.send(.prefill, &w);
    const seq = try self.obtain();
    defer {
        w.int(id) catch {};
        self.send(.release, &w) catch {};
        self.retire(seq);
    }
    var tok = try forward.prefill(self.e, seq, &prompt);
    var ids: [max_rows]u32 = undefined;
    var out: [max_rows]u32 = undefined;
    for (1..max_rows + 1) |width| {
        var best: f64 = std.math.inf(f64);
        for (0..4) |rep| {
            @memset(ids[0..width], tok);
            try w.int(id);
            try w.tokens(ids[0..width]);
            try self.send(.verify, &w);
            const t0 = std.Io.Timestamp.now(self.io, .awake);
            try forward.verify(self.e, seq, ids[0..width], out[0..width]);
            if (rep > 0) best = @min(best, msSince(self.io, t0));
            try w.int(id);
            try w.int(width);
            try w.int(1);
            try self.send(.keep, &w);
            try forward.keep(self.e, seq, @intCast(width), 1);
            tok = out[0];
        }
        self.costs[width - 1] = .{ .width = @intCast(width), .ms = best };
    }
    self.cost_count = max_rows;
    if (!drafts) return;
    var held: [max_depth]u32 = undefined;
    var at: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
    for ([_]u32{ 1, 8 }, 0..) |depth, k| {
        for (0..3) |rep| {
            const follow = [_]u32{tok};
            try w.int(id);
            try w.tokens(&follow);
            try w.int(depth);
            try w.int(0); // no confidence cut: every level timed
            try self.send(.draft, &w);
            const t0 = std.Io.Timestamp.now(self.io, .awake);
            try forward.draft(self.e, seq, &follow, depth, held[0..depth]);
            if (rep > 0) at[k] = @min(at[k], msSince(self.io, t0));
        }
    }
    self.mtp_ms = @max(0, (at[1] - at[0]) / 7);
    std.log.info("flash next costs: window 1/4/8/16 rows {d:.1}/{d:.1}/{d:.1}/{d:.1} ms, head step {d:.2} ms", .{
        self.costs[0].ms, self.costs[3].ms, self.costs[7].ms, self.costs[15].ms, self.mtp_ms });
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.Sampled => "the native Flash Next engine draws greedily for now: send temperature 0",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.rank == 0) own.link.send(@intFromEnum(Op.stop), "") catch {};
    if (own.spare) |s| forward.freeSeq(own.e, s);
    forward.deinit(own.e);
    own.store.deinit();
    own.kernels.deinit();
    own.link.close();
    own.nccl.close();
    own.lanes_by.deinit();
    own.by_id.deinit();
    own.gpa.destroy(own);
}

fn of(p: *anyopaque) *Owned {
    return @ptrCast(@alignCast(p));
}

const vtable: be.Backend.VTable = .{
    .prefill = prefillFn,
    .first = firstFn,
    .queue = queueFn,
    .read = readFn,
    .verify = verifyFn,
    .keep = keepFn,
    .draft = draftFn,
    .release = releaseFn,
    .probabilities = probabilitiesFn,
};

/// A new sequence for the stream and its prompt (the head absorbs it); the first token is drawn here.
fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
    const self = of(p);
    if (s.sampling != null) return error.Sampled;
    const ids = s.prompt();
    if (ids.len == 0 or ids.len + s.max_new + max_rows > forward.maxLen(self.e)) return error.PromptTooLong;
    const gop = try self.lanes_by.getOrPut(s);
    if (gop.found_existing) self.retire(gop.value_ptr.seq);
    const id = self.next_id;
    self.next_id += 1;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(id);
    try w.tokens(ids);
    try self.send(.prefill, &w);
    const seq = try self.obtain();
    gop.value_ptr.* = .{ .seq = seq, .id = id };
    _ = self.take(try forward.prefill(self.e, seq, ids));
}

fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
    const self = of(p);
    if (position != s.prompt_len) return error.PositionMismatch;
    return self.next - 1;
}

fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
    return error.NotPipelined;
}

fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
    const self = of(p);
    if (handle >= self.next or self.next - handle > self.drawn.len) return error.NoSuchToken;
    return self.drawn[handle % self.drawn.len];
}

fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
    const self = of(p);
    if (windows.len != 1) return error.SharedRoundsNotBuilt;
    const win = windows[0];
    if (win.parents != null) return error.TreesNotBuilt;
    const l = self.lanes_by.getPtr(win.stream) orelse return error.NoLane;
    const rows = win.rows();
    if (rows > max_rows) return error.WindowTooWide;
    try settle(self, l, l.rows); // the previous window, accepted whole
    var ids: [max_rows]u32 = undefined;
    ids[0] = win.pending;
    if (win.held > 0) @memcpy(ids[1..][0..win.held], l.held[0..win.held]) else @memcpy(ids[1..][0..win.tokens.len], win.tokens);
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(l.id);
    try w.tokens(ids[0..rows]);
    try self.send(.verify, &w);
    try forward.verify(self.e, l.seq, ids[0..rows], out[0].sampled[0..rows]);
    @memcpy(out[0].drafts[0 .. rows - 1], ids[1..rows]);
    l.rows = @intCast(rows);
    l.nheld = 0;
}

fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const self = of(p);
    for (windows, paths) |win, path| {
        if (path.len == 0) return error.EmptyPath;
        for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        const l = self.lanes_by.getPtr(win.stream) orelse return error.NoLane;
        try settle(self, l, @intCast(path.len));
    }
}

fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const self = of(p);
    for (requests) |r| {
        const l = self.lanes_by.getPtr(r.stream) orelse return error.NoLane;
        if (r.lanes != null) return error.TreesNotBuilt;
        try settle(self, l, l.rows); // a window the core did not cut is kept whole
        var follow: [max_rows]u32 = undefined;
        const n: usize = if (r.rows) |rows| rows.len else 1;
        if (r.rows) |rows| {
            for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            @memcpy(follow[0..n], r.follow[0..n]);
        } else follow[0] = switch (r.first orelse return error.NoFirstToken) {
            .handle => |h| try readFn(p, h),
            .value => |v| v,
        };
        const depth: u32 = @min(r.depth, max_depth);
        var w: Writer = .{ .gpa = self.gpa };
        defer w.buf.deinit(self.gpa);
        try w.int(l.id);
        try w.tokens(follow[0..n]);
        try w.int(depth);
        try w.int(1); // the confidence cut (rank 1 must stop where rank 0 does: the head's steps gather)
        try self.send(.draft, &w);
        const got = try forward.draftUpTo(self.e, l.seq, follow[0..n], depth, draft_confidence, l.held[0..depth]);
        for (0..depth) |i| {
            l.probs[i] = if (i < got) self.e.draft_p[i] else 0;
            if (i >= got) l.held[i] = l.held[got - 1];
        }
        l.nheld = depth;
    }
}

/// Each held draft's chance of landing: the head's probability, 0 past the confidence cut.
fn probabilitiesFn(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
    const self = of(p);
    const l = self.lanes_by.getPtr(s) orelse return false;
    if (out.len > l.nheld) return false;
    @memcpy(out, l.probs[0..out.len]);
    return true;
}

fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
    const self = of(p);
    const kv = self.lanes_by.fetchRemove(s) orelse return;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    w.int(kv.value.id) catch {};
    self.send(.release, &w) catch {};
    self.retire(kv.value.seq);
}

/// Rank 1: rank 0's calls, replayed in order on this rank's half of the model, until rank 0 stops.
fn followLoop(p: *anyopaque) anyerror!void {
    const self = of(p);
    var held: [max_depth]u32 = undefined;
    var sampled: [max_rows]u32 = undefined;
    while (true) {
        const m = self.link.recv(self.gpa) catch |e| switch (e) {
            error.PeerClosed => return,
            else => return e,
        };
        defer self.gpa.free(m.bytes);
        var r: Reader = .{ .bytes = m.bytes };
        switch (@as(Op, @enumFromInt(m.tag))) {
            .prefill => {
                const id = try r.int();
                const seq = try self.obtain();
                try self.by_id.put(id, seq);
                _ = try forward.prefill(self.e, seq, try r.tokens());
            },
            .verify => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const ids = try r.tokens();
                try forward.verify(self.e, seq, ids, sampled[0..ids.len]);
            },
            .keep => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const rows: u32 = @intCast(try r.int());
                try forward.keep(self.e, seq, rows, @intCast(try r.int()));
            },
            .draft => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const follow = try r.tokens();
                const depth: u32 = @intCast(try r.int());
                const cut: f64 = if (try r.int() != 0) draft_confidence else 0;
                _ = try forward.draftUpTo(self.e, seq, follow, depth, cut, held[0..depth]);
            },
            .release => {
                const kv = self.by_id.fetchRemove(try r.int()) orelse continue;
                self.retire(kv.value);
            },
            .stop => return,
            .ready => {},
        }
    }
}
