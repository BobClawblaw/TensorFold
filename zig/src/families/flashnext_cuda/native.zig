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

// -- the protocol rank 0 sends rank 1 ---------------------------------------------------------------------------

const Op = enum(u32) { prefill = 1, verify = 2, keep = 3, draft = 4, release = 5, stop = 6 };

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
    nheld: u32 = 0,
    rows: u32 = 0,                      // the last verify's rows, committed by keep
};

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
    own.kernels = try api.Kernels.load(gpa, io, ctx.d, ctx.device, kernels_dir);
    errdefer own.kernels.deinit();
    own.store = try weights.load(gpa, io, &own.ctx, &own.kernels, dir, o.rank);
    errdefer own.store.deinit();
    own.e = try forward.Engine.init(gpa, io, &own.ctx, &own.kernels, &own.store, .{ .context = o.context, .max_rows = max_rows, .depth = if (o.drafts) max_depth else 0 });
    own.lanes_by = .init(gpa);
    own.by_id = .init(gpa);
    own.next_id = 1;
    own.next = 0;
    return .{
        .backend = .{ .ptr = own, .vtable = &vtable },
        .facts = .{ .exact_width = max_rows, .mtp = o.drafts, .speculate = false, .speculate_early = false, .drafts = if (o.drafts) max_depth else 1,
                    .hidden_rows = false, .max_streams = 1, .batch_rows = max_rows },
        .rows = if (o.drafts) max_rows else 1,
        .stream_bytes = own.e.seqBytes(),
        .ctx = own,
        .deinit = release,
        .follow = if (o.rank == 1) followLoop else null,
    };
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
    own.e.deinit();
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
};

/// A new sequence for the stream and its prompt (the head absorbs it); the first token is drawn here.
fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
    const self = of(p);
    if (s.sampling != null) return error.Sampled;
    const ids = s.prompt();
    if (ids.len == 0 or ids.len + s.max_new + max_rows > self.e.maxLen()) return error.PromptTooLong;
    const gop = try self.lanes_by.getOrPut(s);
    if (gop.found_existing) self.e.freeSeq(gop.value_ptr.seq);
    const id = self.next_id;
    self.next_id += 1;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(id);
    try w.tokens(ids);
    try self.send(.prefill, &w);
    const seq = try self.e.newSeq();
    gop.value_ptr.* = .{ .seq = seq, .id = id };
    _ = self.take(try self.e.prefill(seq, ids));
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
    var ids: [max_rows]u32 = undefined;
    ids[0] = win.pending;
    if (win.held > 0) @memcpy(ids[1..][0..win.held], l.held[0..win.held]) else @memcpy(ids[1..][0..win.tokens.len], win.tokens);
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(l.id);
    try w.tokens(ids[0..rows]);
    try self.send(.verify, &w);
    try self.e.verify(l.seq, ids[0..rows], out[0].sampled[0..rows]);
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
        var w: Writer = .{ .gpa = self.gpa };
        defer w.buf.deinit(self.gpa);
        try w.int(l.id);
        try w.int(l.rows);
        try w.int(path.len);
        try self.send(.keep, &w);
        try self.e.keep(l.seq, l.rows, @intCast(path.len));
    }
}

fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const self = of(p);
    for (requests) |r| {
        const l = self.lanes_by.getPtr(r.stream) orelse return error.NoLane;
        if (r.lanes != null) return error.TreesNotBuilt;
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
        try self.send(.draft, &w);
        try self.e.draft(l.seq, follow[0..n], depth, l.held[0..depth]);
        l.nheld = depth;
    }
}

fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
    const self = of(p);
    const kv = self.lanes_by.fetchRemove(s) orelse return;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    w.int(kv.value.id) catch {};
    self.send(.release, &w) catch {};
    self.e.freeSeq(kv.value.seq);
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
                const seq = try self.e.newSeq();
                try self.by_id.put(id, seq);
                _ = try self.e.prefill(seq, try r.tokens());
            },
            .verify => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const ids = try r.tokens();
                try self.e.verify(seq, ids, sampled[0..ids.len]);
            },
            .keep => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const rows: u32 = @intCast(try r.int());
                try self.e.keep(seq, rows, @intCast(try r.int()));
            },
            .draft => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const follow = try r.tokens();
                const depth: u32 = @intCast(try r.int());
                try self.e.draft(seq, follow, depth, held[0..depth]);
            },
            .release => {
                const kv = self.by_id.fetchRemove(try r.int()) orelse continue;
                self.e.freeSeq(kv.value);
            },
            .stop => return,
        }
    }
}
