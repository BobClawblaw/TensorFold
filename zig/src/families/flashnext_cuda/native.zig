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
const vision = @import("vision.zig");
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
    vision: bool = false, // load the vision tower (rank 0) and take image prompts
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
const batch_rows = 64;          // a shared round's rows at most, every stream's window together
const max_streams = 16;         // streams a shared round packs at most
const max_depth = max_rows - 1;
/// The head stops at a draft it gives less than this (the Python engine's --mtp-confidence, the recipe's 0.70);
/// the slots past it are held with chance 0, so the lane core's allocator leaves them out of the window.
const draft_confidence: f64 = 0.7;

// -- the protocol rank 0 sends rank 1 ---------------------------------------------------------------------------

const Op = enum(u32) { prefill = 1, verify = 2, keep = 3, draft = 4, release = 5, stop = 6, ready = 7, shared = 8, drafts = 9, image = 10 };

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

/// A stream's draw rule in a frame: present, then seed, temperature, top_k, top_p, min_p (floats as their bits).
fn writeSampling(w: *Writer, s: ?lanes.Sampling) !void {
    const r = s orelse return w.int(0);
    try w.int(1);
    try w.int(r.seed);
    try w.int(@as(u64, @bitCast(r.temperature)));
    try w.int(r.top_k);
    try w.int(@as(u64, @bitCast(r.top_p)));
    try w.int(@as(u64, @bitCast(r.min_p)));
}

fn readSampling(r: *Reader) !?lanes.Sampling {
    if (try r.int() == 0) return null;
    return .{ .seed = try r.int(), .temperature = @bitCast(try r.int()), .top_k = @intCast(try r.int()), .top_p = @bitCast(try r.int()), .min_p = @bitCast(try r.int()) };
}

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
    tower: ?vision.Tower = null,                  // rank 0, with vision: the image encoder
    images: std.AutoHashMap(u64, []u8),           // rank 1: an image frame waiting for its prefill, by stream id
    next_id: u64 = 1,
    drawn: [64]u32 = undefined,
    next: u64 = 0,
    costs: [max_rows]lanes.config.Cost = undefined, // a window's ms by width (rank 0, timed at open)
    cost_count: usize = 0,
    shared: [8]lanes.config.Cost = undefined, // a shared round's ms by its total rows
    shared_count: usize = 0,
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
    own.e = try forward.init(gpa, io, &own.ctx, &own.kernels, &own.store, .{ .context = o.context, .max_rows = batch_rows, .depth = if (o.drafts) max_depth else 0 });
    errdefer forward.deinit(own.e);
    try forward.prefetchTables(own.e); // the n-gram tables paged in (and locked) before the first request
    { // the sequences' caches grow within what is free now, less the server's reserve and a margin (a GB10 shares
      // its memory with the host: past this a growth is refused instead of starving the system)
        var free: usize = 0;
        var total: usize = 0;
        try ctx.d.check(ctx.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        const keep_free: u64 = 14 << 30;
        forward.setGrowthBudget(own.e, if (free > keep_free) free - keep_free else 0);
    }
    // both halves loaded before rank 0 serves: rank 1 says so once its engine is up (it loads more slowly)
    if (o.rank == 1) try own.link.send(@intFromEnum(Op.ready), "") else {
        const m = try own.link.recv(gpa);
        gpa.free(m.bytes);
        if (m.tag != @intFromEnum(Op.ready)) return error.RankOneNotReady;
    }
    own.lanes_by = .init(gpa);
    own.by_id = .init(gpa);
    own.images = .init(gpa);
    own.tower = null;
    if (o.vision and o.rank == 0) own.tower = try vision.Tower.load(gpa, io, ctx.d, dir);
    own.next_id = 1;
    own.next = 0;
    own.cost_count = 0;
    own.shared_count = 0;
    own.spare = null;
    own.mtp_ms = 0;
    if (o.rank == 0) try calibrate(own, o.drafts); // rank 1 replays it in its follow loop
    // the calibration's sequence is not kept: the server budgets streams from the memory left after open
    if (own.spare) |sp| {
        forward.freeSeq(own.e, sp);
        own.spare = null;
    }
    return .{
        .backend = .{ .ptr = own, .vtable = &vtable },
        .facts = .{ .exact_width = max_rows, .mtp = o.drafts, .speculate = o.drafts, .speculate_early = false, .drafts = if (o.drafts) max_depth else 1,
                    .hidden_rows = o.drafts, .max_streams = if (o.drafts) max_streams else 1, .batch_rows = batch_rows,
                    .window_costs = own.costs[0..own.cost_count], .mtp_step_ms = own.mtp_ms,
                    .shared_costs = own.shared[0..own.shared_count],
                    .draft_probabilities = o.drafts, .draft_streams = o.drafts },
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
    try writeSampling(&w, null);
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
    try calibrateShared(self);
    std.log.info("flash next costs: window 1/4/8/16 rows {d:.1}/{d:.1}/{d:.1}/{d:.1} ms, head step {d:.2} ms", .{
        self.costs[0].ms, self.costs[3].ms, self.costs[7].ms, self.costs[15].ms, self.mtp_ms });
}

/// Shared rounds timed on throwaway sequences: 2-row windows over 2, 4, 8 and 16 streams and 4-row windows over 16
/// (best of two after a warm-up), the lane core's prices for a round by its total rows.
fn calibrateShared(self: *Owned) !void {
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    var ids: [max_streams]u64 = undefined;
    var seqs: [max_streams]*forward.Seq = undefined;
    var made: usize = 0;
    defer for (ids[0..made], seqs[0..made]) |id, sq| {
        w.int(id) catch {};
        self.send(.release, &w) catch {};
        forward.freeSeq(self.e, sq);
    };
    var prompt: [32]u32 = undefined;
    var first: [max_streams]u32 = undefined;
    for (0..max_streams) |k| {
        for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + ((i + k * 7) * 37) % 5000);
        ids[k] = self.next_id;
        self.next_id += 1;
        try w.int(ids[k]);
        try w.tokens(&prompt);
        try writeSampling(&w, null);
        try self.send(.prefill, &w);
        seqs[k] = try forward.newSeq(self.e);
        made += 1;
        first[k] = try forward.prefill(self.e, seqs[k], &prompt);
    }
    const shapes = [_][2]usize{ .{ 2, 2 }, .{ 4, 2 }, .{ 8, 2 }, .{ 16, 2 }, .{ 16, 4 } };
    for (shapes) |sh| {
        const n = sh[0];
        const rows = sh[1];
        var best: f64 = std.math.inf(f64);
        for (0..3) |rep| {
            var toks: [max_streams][4]u32 = undefined;
            var parts: [max_streams]forward.Part = undefined;
            try w.int(n);
            for (0..n) |k| {
                @memset(toks[k][0..rows], first[k]);
                parts[k] = .{ .s = seqs[k], .ids = toks[k][0..rows] };
                try w.int(ids[k]);
                try w.tokens(toks[k][0..rows]);
            }
            try self.send(.shared, &w);
            var flat: [batch_rows]u32 = undefined;
            const t0 = std.Io.Timestamp.now(self.io, .awake);
            try forward.verifyShared(self.e, parts[0..n], flat[0 .. n * rows]);
            if (rep > 0) best = @min(best, msSince(self.io, t0));
            for (0..n) |k| { // keep one row: the caches advance a little each pass
                try w.int(ids[k]);
                try w.int(rows);
                try w.int(1);
                try self.send(.keep, &w);
                try forward.keep(self.e, seqs[k], @intCast(rows), 1);
            }
        }
        self.shared[self.shared_count] = .{ .width = @intCast(n * rows), .ms = best };
        self.shared_count += 1;
    }
    std.log.info("flash next shared rounds: 4/8/16/32/64 rows {d:.1}/{d:.1}/{d:.1}/{d:.1}/{d:.1} ms", .{
        self.shared[0].ms, self.shared[1].ms, self.shared[2].ms, self.shared[3].ms, self.shared[4].ms });
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.NoVision => "image inputs need the server started with --vision",
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
    if (own.tower) |*t| t.deinit();
    var it = own.images.valueIterator();
    while (it.next()) |v| own.gpa.free(v.*);
    own.images.deinit();
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
    const ids = s.prompt();
    if (ids.len == 0 or ids.len + s.max_new + max_rows > forward.maxLen(self.e)) return error.PromptTooLong;
    // images: encoded and their positions computed before rank 1 hears of the request (a refusal stays on rank 0)
    var frame: ?[]u8 = null;
    defer if (frame) |f| self.gpa.free(f);
    var pos: ?vision.Positions = null;
    defer if (pos) |*q| q.deinit(self.gpa);
    var feats: ?cuda.DeviceBuffer = null;
    defer if (feats) |*f| f.free();
    if (s.images.len > 0) {
        const tower = &(self.tower orelse return error.NoVision);
        var grids = try self.gpa.alloc([3]i64, s.images.len);
        defer self.gpa.free(grids);
        var npatch: usize = 0;
        for (s.images, 0..) |im, k| {
            grids[k] = im.grid;
            npatch += im.patches.len;
        }
        const patches = try self.gpa.alloc(f32, npatch);
        defer self.gpa.free(patches);
        var at: usize = 0;
        for (s.images) |im| {
            @memcpy(patches[at..][0..im.patches.len], im.patches);
            at += im.patches.len;
        }
        pos = try vision.mediaPositions(self.gpa, ids, grids, .{});
        const rows = pos.?.rows.len;
        feats = try cuda.DeviceBuffer.alloc(self.ctx.d, rows * 2560 * 2);
        try tower.encode(&self.e.k, patches, grids, feats.?.ptr);
        // rank 1's frame: rows, delta, the positions [3, L] and the features (bf16), after the stream id
        const L = ids.len;
        var w0: Writer = .{ .gpa = self.gpa };
        try w0.int(self.next_id);
        try w0.int(rows);
        try w0.int(@as(u64, @bitCast(pos.?.delta)));
        try w0.int(L);
        try w0.buf.appendSlice(self.gpa, std.mem.sliceAsBytes(pos.?.pos));
        try w0.buf.appendSlice(self.gpa, std.mem.sliceAsBytes(pos.?.rows));
        const fat = w0.buf.items.len;
        try w0.buf.resize(self.gpa, fat + rows * 2560 * 2);
        try self.e.k.stream.synchronize();
        try self.ctx.d.check(self.ctx.d.api.cuMemcpyDtoH_v2(w0.buf.items[fat..].ptr, feats.?.ptr, rows * 2560 * 2), "image features");
        frame = try w0.buf.toOwnedSlice(self.gpa);
    }
    const gop = try self.lanes_by.getOrPut(s);
    if (gop.found_existing) self.retire(gop.value_ptr.seq);
    const id = self.next_id;
    self.next_id += 1;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    if (frame) |f| try self.link.send(@intFromEnum(Op.image), f);
    try w.int(id);
    try w.tokens(ids);
    try writeSampling(&w, s.sampling);
    try self.send(.prefill, &w);
    const seq = try self.obtain();
    seq.sampling = s.sampling;
    gop.value_ptr.* = .{ .seq = seq, .id = id };
    if (pos) |q| try forward.attach(self.e, seq, q.rows, feats.?.ptr, q.pos, q.delta);
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
    if (windows.len > 1) return verifyShared(self, windows, out);
    const win = windows[0];
    if (win.parents != null) return error.TreesNotBuilt;
    const l = self.lanes_by.getPtr(win.stream) orelse return error.NoLane;
    const rows = win.rows();
    if (rows > max_rows) return error.WindowTooWide;
    try settle(self, l, l.rows); // the previous window, accepted whole
    // the core keys row r's draw at position pos + 1 + r, as the forward draws it
    if (win.positions.len > 0 and win.positions[0] != @as(u64, @intCast(l.seq.pos + 1))) return error.PositionMismatch;
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

/// A shared round: every window checked and every stream's last window committed before rank 1 hears of it, then
/// one forward over all of them on both ranks (rank 1 replays the same parts in the same order).
fn verifyShared(self: *Owned, windows: []const be.Window, out: []be.Verified) anyerror!void {
    if (windows.len > max_streams) return error.TooManyStreams;
    var lanes_of: [max_streams]*Lane = undefined;
    var ids: [max_streams][max_rows]u32 = undefined;
    var total: usize = 0;
    for (windows, 0..) |win, k| {
        if (win.parents != null) return error.TreesNotBuilt;
        lanes_of[k] = self.lanes_by.getPtr(win.stream) orelse return error.NoLane;
        const rows = win.rows();
        if (rows > max_rows) return error.WindowTooWide;
        total += rows;
    }
    if (total > batch_rows) return error.WindowTooWide;
    for (windows, 0..) |win, k| {
        const l = lanes_of[k];
        try settle(self, l, l.rows);
        ids[k][0] = win.pending;
        if (win.held > 0) @memcpy(ids[k][1..][0..win.held], l.held[0..win.held]) else @memcpy(ids[k][1..][0..win.tokens.len], win.tokens);
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    var parts: [max_streams]forward.Part = undefined;
    try w.int(windows.len);
    for (windows, 0..) |win, k| {
        const rows = win.rows();
        try w.int(lanes_of[k].id);
        try w.tokens(ids[k][0..rows]);
        parts[k] = .{ .s = lanes_of[k].seq, .ids = ids[k][0..rows] };
    }
    try self.send(.shared, &w);
    var flat: [batch_rows]u32 = undefined;
    try forward.verifyShared(self.e, parts[0..windows.len], flat[0..total]);
    var r0: usize = 0;
    for (windows, 0..) |win, k| {
        const rows = win.rows();
        @memcpy(out[k].sampled[0..rows], flat[r0..][0..rows]);
        @memcpy(out[k].drafts[0 .. rows - 1], ids[k][1..rows]);
        lanes_of[k].rows = @intCast(rows);
        lanes_of[k].nheld = 0;
        r0 += rows;
    }
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
    if (requests.len > 1) return draftMany(self, requests);
    for (requests) |r| {
        const l = self.lanes_by.getPtr(r.stream) orelse return error.NoLane;
        if (r.lanes != null) return error.TreesNotBuilt;
        // the kept rows (a shared round drafts before it keeps; a lone one has kept already and settles nothing)
        try settle(self, l, if (r.rows) |rows| @intCast(rows.len) else l.rows);
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

/// Several streams' drafts in one batch (each level one forward over every stream still chaining), sent to rank 1
/// as one op after every request is checked and every stream's kept rows committed.
fn draftMany(self: *Owned, requests: []const be.DraftRequest) anyerror!void {
    if (requests.len > max_streams) return error.TooManyStreams;
    var lanes_of: [max_streams]*Lane = undefined;
    var follow: [max_streams][max_rows]u32 = undefined;
    var nf: [max_streams]usize = undefined;
    var depth: [max_streams]u32 = undefined;
    for (requests, 0..) |r, k| {
        lanes_of[k] = self.lanes_by.getPtr(r.stream) orelse return error.NoLane;
        if (r.lanes != null) return error.TreesNotBuilt;
        if (r.rows) |rows| {
            if (rows.len == 0 or rows.len > max_rows) return error.WindowTooWide;
            for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
        } else if (r.first == null) return error.NoFirstToken;
        depth[k] = @min(r.depth, max_depth);
    }
    for (requests, 0..) |r, k| {
        const l = lanes_of[k];
        try settle(self, l, if (r.rows) |rows| @intCast(rows.len) else l.rows);
        if (r.rows) |rows| {
            nf[k] = rows.len;
            @memcpy(follow[k][0..rows.len], r.follow[0..rows.len]);
        } else {
            nf[k] = 1;
            follow[k][0] = switch (r.first.?) {
                .handle => |h| try readFn(self, h),
                .value => |v| v,
            };
        }
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(requests.len);
    var reqs: [max_streams]forward.DraftReq = undefined;
    for (0..requests.len) |k| {
        const l = lanes_of[k];
        try w.int(l.id);
        try w.tokens(follow[k][0..nf[k]]);
        try w.int(depth[k]);
        reqs[k] = .{ .s = l.seq, .follow = follow[k][0..nf[k]], .depth = depth[k], .out = l.held[0..depth[k]], .probs = l.probs[0..depth[k]] };
    }
    try self.send(.drafts, &w);
    try forward.draftBatch(self.e, reqs[0..requests.len], draft_confidence);
    for (reqs[0..requests.len], 0..) |r, k| {
        const l = lanes_of[k];
        for (r.got..depth[k]) |i| { // past the confidence cut: held with chance 0 (the allocator leaves them out)
            l.held[i] = l.held[r.got - 1];
            l.probs[i] = 0;
        }
        l.nheld = depth[k];
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

/// Rank 1: an image frame (see prefillFn) attached to its sequence.
fn attachFrame(self: *Owned, seq: *forward.Seq, f: []const u8) !void {
    var r: Reader = .{ .bytes = f };
    _ = try r.int(); // the stream id
    const rows: usize = @intCast(try r.int());
    const delta: i64 = @bitCast(try r.int());
    const L: usize = @intCast(try r.int());
    if (r.at + L * 12 + rows * 4 + rows * 5120 > f.len) return error.ShortFrame;
    const pos = try self.gpa.alloc(i32, 3 * L);
    defer self.gpa.free(pos);
    @memcpy(std.mem.sliceAsBytes(pos), f[r.at..][0 .. L * 12]);
    r.at += L * 12;
    const idx = try self.gpa.alloc(u32, rows);
    defer self.gpa.free(idx);
    @memcpy(std.mem.sliceAsBytes(idx), f[r.at..][0 .. rows * 4]);
    r.at += rows * 4;
    var fd = try cuda.DeviceBuffer.alloc(self.ctx.d, rows * 5120);
    defer fd.free();
    try self.ctx.d.check(self.ctx.d.api.cuMemcpyHtoD_v2(fd.ptr, f[r.at..].ptr, rows * 5120), "image features");
    try forward.attach(self.e, seq, idx, fd.ptr, pos, delta);
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
            .image => {
                const id = try r.int();
                try self.images.put(id, try self.gpa.dupe(u8, m.bytes));
            },
            .prefill => {
                const id = try r.int();
                const seq = try self.obtain();
                try self.by_id.put(id, seq);
                if (self.images.fetchRemove(id)) |kv| {
                    defer self.gpa.free(kv.value);
                    try attachFrame(self, seq, kv.value);
                }
                const toks = try r.tokens();
                seq.sampling = try readSampling(&r);
                _ = try forward.prefill(self.e, seq, toks);
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
                if (kv.key == 1) { // the calibration's sequence (rank 0 frees its own at open, see there)
                    forward.freeSeq(self.e, kv.value);
                    continue;
                }
                self.retire(kv.value);
            },
            .stop => return,
            .ready => {},
            .drafts => {
                const n: usize = @intCast(try r.int());
                if (n == 0 or n > max_streams) return error.TooManyStreams;
                var reqs: [max_streams]forward.DraftReq = undefined;
                var outs: [max_streams][max_depth]u32 = undefined;
                var probs: [max_streams][max_depth]f64 = undefined;
                for (reqs[0..n], 0..) |*q, k| {
                    const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                    const follow = try r.tokens();
                    const depth: u32 = @intCast(try r.int());
                    q.* = .{ .s = seq, .follow = follow, .depth = depth, .out = outs[k][0..depth], .probs = probs[k][0..depth] };
                }
                try forward.draftBatch(self.e, reqs[0..n], draft_confidence);
            },
            .shared => {
                const n: usize = @intCast(try r.int());
                if (n == 0 or n > max_streams) return error.TooManyStreams;
                var parts: [max_streams]forward.Part = undefined;
                var total: usize = 0;
                for (parts[0..n]) |*pt| {
                    const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                    pt.* = .{ .s = seq, .ids = try r.tokens() };
                    total += pt.ids.len;
                }
                var flat: [batch_rows]u32 = undefined;
                if (total > flat.len) return error.WindowTooWide;
                try forward.verifyShared(self.e, parts[0..n], flat[0..total]);
            },
        }
    }
}
