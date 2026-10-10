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
const Prof = @import("prof.zig").Prof;
const be = lanes.backend;

pub const model_type = "qwen4_exp";
pub const formats: []const []const u8 = &.{"affine-experts-int8"};
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 2048;
pub const two_ranks = true;

/// Structured output: rank 0 masks each window's grammar rows (gramRows), rank 1 the same rows from its frames.
pub const structures = true;

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    tp: u8 = 2,
    rank: u8 = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29600,
    vision: bool = false, // load the vision tower (rank 0) and take image prompts
    kv_bits: u8 = 8, // the attention caches' codes: 8 (int8) or 4 (int4, two a byte), as Python's --kv-dtype
};

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 0, // every admitted stream takes its own sequence
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
const batch_rows = 128;         // a shared round's rows at most, every stream's window together (16 streams x 8)
const max_streams = 16;         // streams a shared round packs at most
const max_depth = max_rows - 1;
/// Rows a stream's caches are backed past what its next forward writes before rank 1 hears of it: a verify window,
/// the kept rows and the head's drafts after it all fall within (so the forwards' own growth finds them mapped).
const grow_margin = 2 * max_rows + 2;
/// The head stops at a draft it gives less than this (the Python engine's --mtp-confidence, the recipe's 0.70);
/// the slots past it are held with chance 0, so the lane core's allocator leaves them out of the window.
/// TENSORFOLD_FN_CONFIDENCE overrides it (a measurement's knob; both ranks must see the same value).
var draft_confidence: f64 = 0.7;
/// What the lane core's allocator gets for a held draft (TENSORFOLD_FN_CHANCES, a measurement's knob): ``step`` the
/// head's probability for it given the drafts before it, ``chain`` that times theirs (its chance of landing), ``off``
/// none (the depth rule sizes chains from each stream's acceptance by depth, as the Python engine's Flash Next does).
var chances: enum { step, chain, off } = .step;

// -- the protocol rank 0 sends rank 1 ---------------------------------------------------------------------------

const Op = enum(u32) { prefill = 1, verify = 2, keep = 3, draft = 4, release = 5, stop = 6, ready = 7, shared = 8, drafts = 9, image = 10, keeps = 11, prefills = 12, grow = 13 };

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
    g: ?Gram = null,                    // the reply's grammar (rank 0)
    first: u64 = 0,                     // the prompt's first token (a handle: readFn)
};

/// A reply's grammar on rank 0 (engine/grammar.py's Constraint): its matcher at the reply's committed tokens.
const Gram = struct {
    m: lanes.grammar.Matcher,
    after: ?u32,          // with thinking on: the grammar starts after this token
    active: bool,
    fed: usize = 0,       // reply tokens the matcher has followed
};

/// A released sequence kept to resume from (multi.py's kept): its prompt to the kept point, rank 1's id for it.
const Kept = struct { ids: []u32, seq: *forward.Seq, id: u64 };

/// Kept prompt states (engine.py KEEP).
const keep_max = 8;

/// Rank 0: drop kept state ``i`` on both ranks.
fn dropKept(self: *Owned, i: usize) void {
    const k = self.kept.orderedRemove(i);
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    w.int(k.id) catch {};
    w.int(0) catch {};
    self.send(.release, &w) catch {};
    self.gpa.free(k.ids);
    self.retire(k.seq);
}

/// A window's grammar rows: which rows and their bits, both ranks' halves.
const Rows = struct {
    n: usize = 0,
    rows: [max_rows]u32 = undefined,
    bits: [2][]u32 = undefined, // rank r's MASK_WORDS words a row (views into the owner's scratch)
};

const MW = forward.MASK_WORDS;

/// Follow the reply's committed tokens, then the rows of a window over ``ids`` (row 0 the pending token, drafts
/// after) the grammar constrains, as grammar.py's window(): each row the matcher reaches (a draft it rejects ends the
/// path: no later row can be accepted), its allowed bits. The matcher is rolled back to the committed tokens.
fn gramRows(g: *Gram, s: *const lanes.Stream, ids: []const u32, full: []u32, out: *Rows) !void {
    out.n = 0;
    const emitted = s.context.items[s.prompt_len..];
    for (emitted[g.fed..]) |t| {
        if (!g.active) {
            g.active = g.after != null and t == g.after.?;
            continue;
        }
        if (g.m.terminated()) break;
        if (!try g.m.accept(t)) return error.GrammarRejected;
    }
    g.fed = emitted.len;
    var active = g.active;
    var taken: usize = 0;
    defer g.m.rollback(taken) catch {};
    for (0..ids.len) |r| {
        if (active and !g.m.terminated()) {
            const words = full[0 .. 2 * MW];
            try g.m.fill(words);
            @memcpy(out.bits[0][out.n * MW ..][0..MW], words[0..MW]);
            @memcpy(out.bits[1][out.n * MW ..][0..MW], words[MW..]);
            out.rows[out.n] = @intCast(r);
            out.n += 1;
        }
        if (r + 1 == ids.len) break;
        const next = ids[r + 1];
        if (!active) {
            active = g.after != null and next == g.after.?;
            continue;
        }
        if (g.m.terminated() or !try g.m.accept(next)) break;
        taken += 1;
    }
}

/// A frame's grammar rows for rank 1: the count, the rows, rank 1's bits.
fn writeRows(w: *Writer, rows: *const Rows) !void {
    try w.int(rows.n);
    for (rows.rows[0..rows.n]) |r| try w.int(r);
    try w.buf.appendSlice(w.gpa, std.mem.sliceAsBytes(rows.bits[1][0 .. rows.n * MW]));
}

/// Rank 1: a frame's grammar rows into ``rows``/``bits`` from ``at`` on (null: none).
fn readRows(r: *Reader, rows: []u32, bits: []u32) !?forward.Mask {
    const n: usize = @intCast(try r.int());
    if (n == 0) return null;
    if (n > rows.len) return error.WindowTooWide;
    for (rows[0..n]) |*x| x.* = @intCast(try r.int());
    const bytes = n * MW * 4;
    if (r.at + bytes > r.bytes.len) return error.ShortFrame;
    @memcpy(std.mem.sliceAsBytes(bits[0 .. n * MW]), r.bytes[r.at..][0..bytes]);
    r.at += bytes;
    return .{ .rows = rows[0..n], .bits = bits[0 .. n * MW] };
}

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

/// ``settle`` for several lanes: one frame and one keep for all of them (forward.keepMany).
fn settleMany(self: *Owned, ls: []const *Lane, kept: []const u32) !void {
    var items: [max_streams]forward.Keep = undefined;
    var n: usize = 0;
    for (ls, kept) |l, kp| {
        if (l.rows == 0) continue;
        items[n] = .{ .s = l.seq, .rows = l.rows, .kept = kp };
        n += 1;
    }
    if (n == 0) return;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(n);
    for (ls, kept) |l, kp| if (l.rows != 0) {
        try w.int(l.id);
        try w.int(l.rows);
        try w.int(kp);
    };
    try self.send(.keeps, &w);
    try forward.keepMany(self.e, items[0..n]);
    for (ls) |l| l.rows = 0;
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
    spares: std.ArrayList(*forward.Seq) = .empty, // released sequences, reset and reused by the next requests (their memory, their graphs)
    kept: std.ArrayList(Kept) = .empty, // released sequences kept at their prompts' kept points (oldest first)
    refused: std.ArrayList(*lanes.Stream) = .empty, // streams a round's cache growth was refused for (refusedFn)
    streams: usize = max_streams, // the streams admission budgeted: active, kept and spare sequences stay within it
    gfull: []u32 = &.{}, // a grammar row's bits over the whole vocabulary
    gbits: [2][]u32 = .{ &.{}, &.{} }, // a round's grammar rows, each rank's half (rank 1: what the frames carry)
    growsbuf: [batch_rows]u32 = undefined,
    prof: Prof = .{}, // TENSORFOLD_FN_PROFILE (prof.zig)

    /// The scratch grammar rows need, made on first use.
    fn gramScratch(self: *Owned) !void {
        if (self.gfull.len > 0) return;
        self.gfull = try self.gpa.alloc(u32, 2 * MW);
        for (&self.gbits) |*b| b.* = try self.gpa.alloc(u32, batch_rows * MW);
    }

    fn obtain(self: *Owned) !*forward.Seq {
        return self.spares.pop() orelse forward.newSeq(self.e);
    }

    /// A released sequence, reset now (its grown caches back to the budget, its state zeroed on the stream) and kept
    /// for the next request while every sequence stays within the budgeted streams: the Python engine holds a state
    /// per slot, where allocating and zeroing one per request held up each new stream's first token.
    fn retire(self: *Owned, s: *forward.Seq) void {
        const live = if (self.rank == 0) self.lanes_by.count() + self.kept.items.len else self.by_id.count();
        if (live + self.spares.items.len >= self.streams) return forward.freeSeq(self.e, s);
        forward.resetSeq(self.e, s) catch return forward.freeSeq(self.e, s);
        self.spares.append(self.gpa, s) catch forward.freeSeq(self.e, s);
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

    /// Room in the budget for ``bytes`` more, made before rank 1 hears of the growth: memory no live stream holds goes
    /// first (spare sequences, then rank 0's oldest kept prompt states, dropped on both ranks); false when it stays short.
    fn make(self: *Owned, bytes: u64) bool {
        while (!forward.fits(self.e, bytes)) if (!self.relieve(true)) return false;
        return true;
    }

    /// ``seq``'s caches backed through ``target``, spare sequences freed first when the budget is short (or the GPU:
    /// a growth ``make`` let in may still fail to map). Kept states are dropped by ``make`` alone, before any frame
    /// whose answer rank 1 waits on.
    fn grow(self: *Owned, seq: *forward.Seq, target: i64) !void {
        while (true) {
            forward.reserve(self.e, seq, target) catch |err| {
                if (err == error.OutOfDeviceMemory and self.relieve(false)) continue;
                return err;
            };
            return;
        }
    }

    fn relieve(self: *Owned, kept: bool) bool {
        if (self.spares.pop()) |s| {
            forward.freeSeq(self.e, s);
            return true;
        }
        if (!kept or self.rank != 0 or self.kept.items.len == 0) return false;
        const k = self.kept.orderedRemove(0);
        var w: Writer = .{ .gpa = self.gpa };
        defer w.buf.deinit(self.gpa);
        w.int(k.id) catch {};
        w.int(0) catch {};
        self.send(.release, &w) catch {};
        self.gpa.free(k.ids);
        forward.freeSeq(self.e, k.seq);
        return true;
    }

    fn refuse(self: *Owned, s: *lanes.Stream) !void {
        for (self.refused.items) |x| if (x == s) return;
        try self.refused.append(self.gpa, s);
    }

    /// Rank 1: a prompt frame's growth answered (``grown``: null when rank 0 asked for none) and rank 0's verdict
    /// awaited (a kept state to drop and try again, while rank 1 is short); whether the pass runs.
    fn settleGrowth(self: *Owned, grown: ?bool, prompts: []const forward.Prompt) !bool {
        var ok = grown orelse return true;
        while (true) {
            try self.link.send(@intFromEnum(Op.grow), &.{@intFromBool(ok)});
            const v = try self.link.recv(self.gpa);
            defer self.gpa.free(v.bytes);
            if (v.tag != @intFromEnum(Op.grow) or v.bytes.len == 0) return error.RankZeroOutOfStep;
            if (v.bytes[0] != 2) return v.bytes[0] == 1;
            if (v.bytes.len != 9) return error.RankZeroOutOfStep;
            if (self.by_id.fetchRemove(std.mem.readInt(u64, v.bytes[1..9], .little))) |kv| self.retire(kv.value);
            ok = true;
            for (prompts) |pr| self.grow(pr.s, @intCast(pr.ids.len + grow_margin)) catch |err| switch (err) {
                error.OutOfDeviceMemory => ok = false,
                else => return err,
            };
        }
    }

    /// Rank 0's word on a prompt frame's growth (rank 1 waits for it before the pass): 1 go, 0 refused, 2 the kept
    /// state ``drop`` dropped on both ranks and rank 1 to try again.
    fn verdict(self: *Owned, code: u8, drop: ?u64) !void {
        var b: [9]u8 = undefined;
        b[0] = code;
        if (drop) |id| std.mem.writeInt(u64, b[1..9], id, .little);
        try self.link.send(@intFromEnum(Op.grow), b[0..if (drop != null) 9 else 1]);
    }

    /// Rank 1's answer to a growth it was asked for (a .grow frame, or a prompt frame's): 1 a byte where it grew.
    fn answer(self: *Owned, n: usize) ![]u8 {
        const m = try self.link.recv(self.gpa);
        if (m.tag != @intFromEnum(Op.grow) or m.bytes.len != n) {
            self.gpa.free(m.bytes);
            return error.RankOneOutOfStep;
        }
        return m.bytes;
    }
};

/// TENSORFOLD_GROWTH_LIMIT_MIB: this rank's sequences' caches grow by this much at most past what is mapped when the
/// calibration ends (a test's budget: growths refused with short prompts, on either rank alone).
fn growthLimit(e: *forward.Engine) void {
    const v = std.c.getenv("TENSORFOLD_GROWTH_LIMIT_MIB") orelse return;
    const mib = std.fmt.parseInt(u64, std.mem.span(v), 10) catch return;
    forward.setGrowthBudget(e, @min(e.budget.cap, e.budget.used + (mib << 20)));
    std.log.info("flash next: cache growth within {d} MiB ({d} MiB mapped)", .{ e.budget.cap >> 20, e.budget.used >> 20 });
}

/// Every lane's caches backed through its target before rank 1 hears of the round: rank 0 grows first, then rank 1
/// (a .grow frame it answers), so a growth refused on either rank refuses the round on both before any forward (one
/// refused on one rank alone left the other in a forward whose collectives never met). The streams refused wait in
/// ``refused`` for the lane host (refusedFn), which ends them alone; the round's other streams go on.
fn reserveRound(self: *Owned, ls: []const *Lane, ss: []const *lanes.Stream, targets: []const i64) !void {
    var ask: [max_streams]usize = undefined;
    var n: usize = 0;
    var refused = false;
    var pending: u64 = 0;
    for (ls, targets, 0..) |l, t, k| { // the growths the budget takes (after spares and kept states), in order
        const bytes = forward.wants(l.seq, t);
        if (bytes == 0) continue;
        if (!self.make(pending + bytes)) {
            try self.refuse(ss[k]);
            refused = true;
            continue;
        }
        pending += bytes;
        ask[n] = k;
        n += 1;
    }
    if (n > 0) { // both ranks grow at once: rank 1 from the frame, rank 0 here, then rank 1's answer
        var w: Writer = .{ .gpa = self.gpa };
        defer w.buf.deinit(self.gpa);
        try w.int(n);
        for (ask[0..n]) |k| {
            try w.int(ls[k].id);
            try w.int(targets[k]);
        }
        try self.send(.grow, &w);
        var mine: [max_streams]bool = @splat(true);
        for (ask[0..n], 0..) |k, i| self.grow(ls[k].seq, targets[k]) catch |err| switch (err) {
            error.OutOfDeviceMemory => mine[i] = false,
            else => return err,
        };
        var theirs: [max_streams]bool = undefined;
        {
            const oks = try self.answer(n);
            defer self.gpa.free(oks);
            for (oks, 0..) |ok, i| theirs[i] = ok == 1;
        }
        // rank 1 short where rank 0 is not (kept prompt states hold its memory too): the oldest go on both ranks and
        // rank 1 tries those again, while there are some
        while (self.kept.items.len > 0) {
            var again: [max_streams]usize = undefined;
            var m: usize = 0;
            for (0..n) |i| if (mine[i] and !theirs[i]) {
                again[m] = i;
                m += 1;
            };
            if (m == 0) break;
            dropKept(self, 0);
            var w2: Writer = .{ .gpa = self.gpa };
            defer w2.buf.deinit(self.gpa);
            try w2.int(m);
            for (again[0..m]) |i| {
                try w2.int(ls[ask[i]].id);
                try w2.int(targets[ask[i]]);
            }
            try self.send(.grow, &w2);
            const oks = try self.answer(m);
            defer self.gpa.free(oks);
            for (again[0..m], oks) |i, ok| theirs[i] = ok == 1;
        }
        for (ask[0..n], 0..) |k, i| if (!theirs[i] or !mine[i]) {
            try self.refuse(ss[k]);
            refused = true;
        };
    }
    if (refused) return error.OutOfDeviceMemory;
}

/// The streams the last rounds' cache growth was refused for (the lane host ends them; the others go on).
fn refusedFn(p: *anyopaque, out: []*lanes.Stream) usize {
    const self = of(p);
    const n = @min(out.len, self.refused.items.len);
    @memcpy(out[0..n], self.refused.items[0..n]);
    self.refused.clearRetainingCapacity();
    return n;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (o.tp != 2) return error.TwoRanksOnly;
    const kernels_dir = kernels orelse return error.NoCapturedKernels; // the Triton set captured from Python
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.io = io;
    own.rank = o.rank;
    own.prof = if (o.rank == 0) .init() else .{};
    if (std.c.getenv("TENSORFOLD_FN_CHANCES")) |v| chances = std.meta.stringToEnum(@TypeOf(chances), std.mem.span(v)) orelse chances;
    if (std.c.getenv("TENSORFOLD_FN_CONFIDENCE")) |v| draft_confidence = std.fmt.parseFloat(f64, std.mem.span(v)) catch draft_confidence;
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
    own.e = try forward.init(gpa, io, &own.ctx, &own.kernels, &own.store, .{ .context = o.context, .max_rows = batch_rows, .depth = if (o.drafts) max_depth else 0, .kv_bits = o.kv_bits });
    if (std.c.getenv("TENSORFOLD_FN_SPLIT_GRAPHS")) |v| own.e.split_gathers = std.mem.eql(u8, std.mem.span(v), "1");
    if (own.prof.every > 0) {
        own.prof.enq = &own.e.enq_ms;
        own.prof.wait = &own.e.wait_ms;
    }
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
    // and with its growth budget: rank 0 grows within the smaller of the two, so rank 1's growth (the same as rank 0's)
    // fits its own budget too
    if (o.rank == 1) try own.link.send(@intFromEnum(Op.ready), std.mem.asBytes(&own.e.budget.cap)) else {
        const m = try own.link.recv(gpa);
        defer gpa.free(m.bytes);
        if (m.tag != @intFromEnum(Op.ready)) return error.RankOneNotReady;
        if (m.bytes.len == 8) forward.setGrowthBudget(own.e, @min(own.e.budget.cap, std.mem.readInt(u64, m.bytes[0..8], .little)));
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
    own.spares = .empty;
    own.kept = .empty;
    own.refused = .empty;
    own.streams = max_streams;
    own.gfull = &.{};
    own.gbits = .{ &.{}, &.{} };
    own.mtp_ms = 0;
    if (o.rank == 0) { // rank 1 replays it in its follow loop, then hears it ended
        try calibrate(own, o.drafts);
        try own.link.send(@intFromEnum(Op.ready), "");
        growthLimit(own.e);
    }
    // the calibration's sequence is not kept: the server budgets streams from the memory left after open
    while (own.spares.pop()) |sp| forward.freeSeq(own.e, sp);
    return .{
        .backend = .{ .ptr = own, .vtable = &vtable },
        .facts = .{ .exact_width = max_rows, .mtp = o.drafts, .speculate = o.drafts, .speculate_early = false, .drafts = if (o.drafts) max_depth else 1,
                    .hidden_rows = o.drafts, .max_streams = if (o.drafts) max_streams else 1, .batch_rows = batch_rows,
                    .window_costs = own.costs[0..own.cost_count], .mtp_step_ms = own.mtp_ms,
                    .shared_costs = own.shared[0..own.shared_count],
                    .draft_probabilities = o.drafts and chances != .off, .draft_streams = o.drafts },
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
    try w.int(0); // no grammar rows
    try w.int(0); // nor a kept state, nor a kept point, nor a reach to agree on
    try w.int(0);
    try w.int(0);
    try w.int(0);
    try self.send(.prefill, &w);
    const seq = try self.obtain();
    defer {
        w.int(id) catch {};
        w.int(0) catch {};
        self.send(.release, &w) catch {};
        self.retire(seq);
    }
    var tok = try forward.prefill(self.e, seq, &prompt, 0);
    var ids: [max_rows]u32 = undefined;
    var out: [max_rows]u32 = undefined;
    for (1..max_rows + 1) |width| {
        var best: f64 = std.math.inf(f64);
        for (0..4) |rep| {
            @memset(ids[0..width], tok);
            try w.int(id);
            try w.tokens(ids[0..width]);
            try w.int(0);
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

/// Shared rounds timed on throwaway sequences: 2-row windows over 2, 4, 8 and 16 streams and 4-, 6- and 8-row windows
/// over 16 (best of two after a warm-up), the lane core's prices for a round by its total rows.
fn calibrateShared(self: *Owned) !void {
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    var ids: [max_streams]u64 = undefined;
    var seqs: [max_streams]*forward.Seq = undefined;
    var made: usize = 0;
    defer for (ids[0..made], seqs[0..made]) |id, sq| {
        w.int(id) catch {};
        w.int(0) catch {};
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
        for (0..5) |_| try w.int(0);
        try self.send(.prefill, &w);
        seqs[k] = try forward.newSeq(self.e);
        made += 1;
        first[k] = try forward.prefill(self.e, seqs[k], &prompt, 0);
    }
    const shapes = [_][2]usize{ .{ 2, 2 }, .{ 4, 2 }, .{ 8, 2 }, .{ 16, 2 }, .{ 16, 4 }, .{ 16, 6 }, .{ 16, 8 } };
    for (shapes) |sh| {
        const n = sh[0];
        const rows = sh[1];
        var best: f64 = std.math.inf(f64);
        for (0..3) |rep| {
            var toks: [max_streams][8]u32 = undefined;
            var parts: [max_streams]forward.Part = undefined;
            try w.int(n);
            for (0..n) |k| {
                @memset(toks[k][0..rows], first[k]);
                parts[k] = .{ .s = seqs[k], .ids = toks[k][0..rows] };
                try w.int(ids[k]);
                try w.tokens(toks[k][0..rows]);
                try w.int(0);
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
    std.log.info("flash next shared rounds: 4/8/16/32/64/96/128 rows {d:.1}/{d:.1}/{d:.1}/{d:.1}/{d:.1}/{d:.1}/{d:.1} ms", .{
        self.shared[0].ms, self.shared[1].ms, self.shared[2].ms, self.shared[3].ms, self.shared[4].ms, self.shared[5].ms, self.shared[6].ms });
}

/// The streams admission budgeted (native/cuda.zig after open): kept prompt states stay within them.
pub fn setStreams(p: *anyopaque, n: usize) void {
    of(p).streams = n;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    if (err == error.GrammarRejected) return "the reply's grammar rejected a token it was made to take";
    if (err == error.Grammar) return "the reply's grammar failed (xgrammar)";
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
    while (own.spares.pop()) |s| forward.freeSeq(own.e, s);
    own.spares.deinit(own.gpa);
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
    .prefill = timedPrefill,
    .prefill_many = timedPrefillMany,
    .first = firstFn,
    .queue = queueFn,
    .read = readFn,
    .verify = timedVerify,
    .keep = timedKeep,
    .draft = timedDraft,
    .release = releaseFn,
    .probabilities = probabilitiesFn,
    .refused = refusedFn,
};

fn timedPrefill(p: *anyopaque, s: *lanes.Stream) anyerror!void {
    const t = of(p).prof.begin(of(p).io);
    defer of(p).prof.end(of(p).io, .prefill, t, 0, 0);
    return prefillFn(p, s);
}

fn timedPrefillMany(p: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
    const t = of(p).prof.begin(of(p).io);
    defer of(p).prof.end(of(p).io, .prefills, t, 0, ss.len);
    return prefillManyFn(p, ss);
}

fn timedVerify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
    const t = of(p).prof.begin(of(p).io);
    var rows: usize = 0;
    for (windows) |w| rows += w.rows();
    defer of(p).prof.end(of(p).io, if (windows.len > 1) .shared else .verify, t, rows, if (windows.len > 1) windows.len else 0);
    return verifyFn(p, windows, out);
}

fn timedKeep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const t = of(p).prof.begin(of(p).io);
    defer of(p).prof.end(of(p).io, .keep, t, 0, 0);
    return keepFn(p, windows, paths);
}

fn timedDraft(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const t = of(p).prof.begin(of(p).io);
    defer of(p).prof.end(of(p).io, if (requests.len > 1) .drafts else .draft, t, 0, 0);
    return draftFn(p, requests);
}

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
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    var resumed: u64 = 0;
    const pr = try setupPrefill(self, s, &w, 0, &resumed);
    const target: i64 = @intCast(pr.ids.len + grow_margin);
    const bytes = forward.wants(pr.s, target);
    if (bytes > 0 and !self.make(bytes)) {
        // refused before rank 1 hears of the prompt: a kept state it resumed is dropped there (the lane is released
        // as any failed one, its id unknown to rank 1)
        if (resumed != 0) {
            var wr: Writer = .{ .gpa = self.gpa };
            defer wr.buf.deinit(self.gpa);
            try wr.int(resumed);
            try wr.int(0);
            try self.send(.release, &wr);
        }
        return error.OutOfDeviceMemory;
    }
    if (frame) |f| try self.link.send(@intFromEnum(Op.image), f);
    try self.send(.prefill, &w);
    try agree(self, &.{pr}); // the lane is released as any failed one when either rank is short
    if (pos) |q| try forward.attach(self.e, pr.s, q.rows, feats.?.ptr, q.pos, q.delta);
    self.lanes_by.getPtr(s).?.first = self.take(try forward.prefill(self.e, pr.s, pr.ids, pr.start));
}

/// A stream's lane for its prompt pass (prefillFn, prefillManyFn): the kept state it resumes from, its grammar (its
/// first row's bits at grammar row ``at`` of the scratch), its sequence; rank 1's frame fields into ``w``.
fn setupPrefill(self: *Owned, s: *lanes.Stream, w: *Writer, at: usize, resumed: *u64) !forward.Prompt {
    const ids = s.prompt();
    resumed.* = 0;
    // a kept prompt state this prompt extends (the longest), and the point this prompt keeps (drafting, no images)
    const cutting = s.drafts and s.images.len == 0;
    var reuse: ?Kept = null;
    if (cutting) {
        var best: ?usize = null;
        for (self.kept.items, 0..) |k, i| {
            if (k.ids.len < ids.len and std.mem.eql(u32, k.ids, ids[0..k.ids.len]) and (best == null or k.ids.len > self.kept.items[best.?].ids.len)) best = i;
        }
        if (best) |i| reuse = self.kept.orderedRemove(i);
    }
    errdefer if (reuse) |k| { // refused before rank 1 heard of the prompt: the state is dropped on both ranks
        self.kept.insert(self.gpa, 0, k) catch {};
        if (self.kept.items.len > 0 and self.kept.items[0].seq == k.seq) dropKept(self, 0);
    };
    const gop = try self.lanes_by.getOrPut(s);
    if (gop.found_existing) self.retire(gop.value_ptr.seq);
    // every sequence within the budgeted streams: the oldest kept states go first
    while (true) { // spares first, then the oldest kept states (a new stream takes a spare when it resumes none)
        const spares = self.spares.items.len - @intFromBool(reuse == null and self.spares.items.len > 0);
        if (self.lanes_by.count() + self.kept.items.len + spares <= self.streams) break;
        if (spares > 0) {
            forward.freeSeq(self.e, self.spares.orderedRemove(0));
        } else if (self.kept.items.len > 0) dropKept(self, 0) else break;
    }
    const id = self.next_id;
    self.next_id += 1;
    // a grammar: compiled (the server's compile is cached), its first row when it starts with the reply
    var gram: ?Gram = null;
    errdefer if (gram) |g| g.m.free();
    var grows: Rows = .{};
    if (s.structure) |st| {
        try self.gramScratch();
        if (st.compiler.words != 2 * MW) return error.GrammarVocabulary;
        var err: [512]u8 = undefined;
        const compiled = try st.compiler.compile(self.io, st.kind, st.text, &err);
        defer compiled.free();
        gram = .{ .m = try compiled.matcher(), .after = st.after, .active = st.after == null };
        grows.bits = .{ self.gbits[0][at * MW ..], self.gbits[1][at * MW ..] };
        try gramRows(&gram.?, s, &.{0}, self.gfull, &grows);
    }
    try w.int(id);
    try w.tokens(ids);
    try writeSampling(w, s.sampling);
    try writeRows(w, &grows);
    const start: usize = if (reuse) |k| k.ids.len else 0;
    const cut: ?i64 = if (cutting) @max(1, @as(i64, @intCast(ids.len)) - 1) else null;
    try w.int(if (reuse) |k| k.id else 0);
    try w.int(start);
    try w.int(if (cut) |c| @as(u64, @intCast(c)) else 0);
    const seq = if (reuse) |k| k.seq else try self.obtain();
    // the positions the prompt and its first drafts reach: each rank backs its caches through them (a sequence comes
    // new on one rank and reused on the other, so only then do both map the same) and rank 1 answers whether it could
    try w.int(ids.len + grow_margin);
    if (reuse) |k| {
        resumed.* = k.id;
        self.gpa.free(k.ids);
    }
    reuse = null;
    seq.sampling = s.sampling;
    seq.cut_at = cut;
    s.cached = @intCast(start);
    if (grows.n > 0) { // the rows outlive this call (the pass reads them): kept in the scratch beside the bits
        @memcpy(self.growsbuf[at..][0..grows.n], grows.rows[0..grows.n]);
        seq.mask = .{ .rows = self.growsbuf[at..][0..grows.n], .bits = grows.bits[0][0 .. grows.n * MW] };
    }
    if (gop.found_existing) if (gop.value_ptr.g) |g| g.m.free();
    gop.value_ptr.* = .{ .seq = seq, .id = id, .g = gram };
    gram = null;
    return .{ .s = seq, .ids = ids, .start = start };
}

/// Several streams' prompts in one pass (forward.prefillMany): text prompts that fill one prompt chunk together,
/// each set up as prefillFn sets one up, one frame for rank 1; false (nothing done) when they do not go together.
fn prefillManyFn(p: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
    const self = of(p);
    if (ss.len < 2 or ss.len > max_streams) return false;
    var rows: usize = 0;
    for (ss) |s| {
        const ids = s.prompt();
        if (s.images.len > 0 or ids.len == 0 or ids.len + s.max_new + max_rows > forward.maxLen(self.e)) return false;
        rows += ids.len;
    }
    if (rows > forward.PREFILL_ROWS) return false;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(ss.len);
    for (ss) |s| if (s.structure != null) return false; // grammars one by one (a refused one stays alone)
    var prompts: [max_streams]forward.Prompt = undefined;
    var resumed: [max_streams]u64 = undefined;
    for (ss, 0..) |s, k| prompts[k] = setupPrefill(self, s, &w, k, &resumed[k]) catch |err| {
        undoSetups(self, ss[0..k], resumed[0..k]);
        return err;
    };
    var bytes: u64 = 0;
    for (prompts[0..ss.len]) |pr| bytes += forward.wants(pr.s, @intCast(pr.ids.len + grow_margin));
    if (bytes > 0 and !self.make(bytes)) { // each alone then: the one the budget refuses ends on its own
        undoSetups(self, ss, resumed[0..ss.len]);
        return false;
    }
    try self.send(.prefills, &w);
    try agree(self, prompts[0..ss.len]);
    var firsts: [max_streams]u32 = undefined;
    const t0 = std.Io.Timestamp.now(self.io, .awake);
    try forward.prefillMany(self.e, prompts[0..ss.len], firsts[0..ss.len]);
    std.log.debug("flash next: {d} prompts in one pass ({d} rows) in {d:.1} ms", .{ ss.len, rows, msSince(self.io, t0) });
    for (ss, 0..) |s, k| self.lanes_by.getPtr(s).?.first = self.take(firsts[k]);
    return true;
}


/// A prompt frame's growth, both ranks at once: rank 0 backs its prompts' caches while rank 1 backs its own from the
/// frame, then rank 1's answer and rank 0's verdict (rank 1 runs the pass only on a go); error.OutOfDeviceMemory,
/// neither rank having run anything, when either is short. Every prompt frame asks: a sequence new on one rank may be
/// reused on the other (its first step mapped), and only once both back the prompt do they map the same.
fn agree(self: *Owned, prompts: []const forward.Prompt) !void {
    var ok = true;
    for (prompts) |pr| {
        const target: i64 = @intCast(pr.ids.len + grow_margin);
        if (forward.wants(pr.s, target) == 0) continue;
        self.grow(pr.s, target) catch |err| switch (err) {
            error.OutOfDeviceMemory => ok = false,
            else => {
                self.verdict(0, null) catch {};
                return err;
            },
        };
    }
    while (true) {
        const theirs = try self.answer(1);
        const short = theirs[0] == 0;
        self.gpa.free(theirs);
        if (!short or !ok) {
            try self.verdict(if (ok and !short) 1 else 0, null);
            if (ok and !short) return;
            return error.OutOfDeviceMemory;
        }
        // rank 1 is short where rank 0 is not (kept prompt states hold rank 1's memory as much as rank 0's): the
        // oldest goes on both ranks and rank 1 tries again, while there is one
        if (self.kept.items.len == 0) {
            try self.verdict(0, null);
            return error.OutOfDeviceMemory;
        }
        const k = self.kept.orderedRemove(0);
        self.gpa.free(k.ids);
        self.retire(k.seq);
        try self.verdict(2, k.id);
    }
}

/// Rank 1 has heard of none of these prompts: their lanes undone, a kept state each resumed dropped there.
fn undoSetups(self: *Owned, ss: []const *lanes.Stream, resumed: []const u64) void {
    for (ss, resumed) |done, rid| {
        const kv = self.lanes_by.fetchRemove(done) orelse continue;
        if (kv.value.g) |g| g.m.free();
        if (rid != 0) {
            var wr: Writer = .{ .gpa = self.gpa };
            defer wr.buf.deinit(self.gpa);
            wr.int(rid) catch {};
            wr.int(0) catch {};
            self.send(.release, &wr) catch {};
        }
        self.retire(kv.value.seq);
    }
}

fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
    const self = of(p);
    if (position != s.prompt_len) return error.PositionMismatch;
    return (self.lanes_by.get(s) orelse return error.NoLane).first;
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
    try reserveRound(self, &.{l}, &.{win.stream}, &.{forward.reach(l.seq, @as(i64, @intCast(rows)) + grow_margin)});
    // the core keys row r's draw at position pos + 1 + r, as the forward draws it
    if (win.positions.len > 0 and win.positions[0] != @as(u64, @intCast(l.seq.pos + 1))) return error.PositionMismatch;
    var ids: [max_rows]u32 = undefined;
    ids[0] = win.pending;
    if (win.held > 0) @memcpy(ids[1..][0..win.held], l.held[0..win.held]) else @memcpy(ids[1..][0..win.tokens.len], win.tokens);
    var grows: Rows = .{};
    if (l.g) |*g| {
        grows.bits = .{ self.gbits[0], self.gbits[1] };
        try gramRows(g, win.stream, ids[0..rows], self.gfull, &grows);
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(l.id);
    try w.tokens(ids[0..rows]);
    try writeRows(&w, &grows);
    try self.send(.verify, &w);
    if (grows.n > 0) l.seq.mask = .{ .rows = grows.rows[0..grows.n], .bits = grows.bits[0][0 .. grows.n * MW] };
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
    var whole: [max_streams]u32 = undefined; // each lane's previous window, accepted whole
    for (lanes_of[0..windows.len], 0..) |l, k| whole[k] = l.rows;
    try settleMany(self, lanes_of[0..windows.len], whole[0..windows.len]);
    {
        var ss: [max_streams]*lanes.Stream = undefined;
        var targets: [max_streams]i64 = undefined;
        for (windows, 0..) |win, k| {
            ss[k] = win.stream;
            targets[k] = forward.reach(lanes_of[k].seq, @as(i64, @intCast(win.rows())) + grow_margin);
        }
        try reserveRound(self, lanes_of[0..windows.len], ss[0..windows.len], targets[0..windows.len]);
    }
    for (windows, 0..) |win, k| {
        const l = lanes_of[k];
        ids[k][0] = win.pending;
        if (win.held > 0) @memcpy(ids[k][1..][0..win.held], l.held[0..win.held]) else @memcpy(ids[k][1..][0..win.tokens.len], win.tokens);
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    var parts: [max_streams]forward.Part = undefined;
    var grows: [max_streams]Rows = undefined;
    var used: usize = 0; // grammar rows so far: each part's bits after the previous parts'
    try w.int(windows.len);
    for (windows, 0..) |win, k| {
        const rows = win.rows();
        grows[k] = .{};
        if (lanes_of[k].g) |*g| {
            grows[k].bits = .{ self.gbits[0][used * MW ..], self.gbits[1][used * MW ..] };
            try gramRows(g, win.stream, ids[k][0..rows], self.gfull, &grows[k]);
            used += grows[k].n;
        }
        try w.int(lanes_of[k].id);
        try w.tokens(ids[k][0..rows]);
        try writeRows(&w, &grows[k]);
        parts[k] = .{ .s = lanes_of[k].seq, .ids = ids[k][0..rows] };
        if (grows[k].n > 0) lanes_of[k].seq.mask = .{ .rows = grows[k].rows[0..grows[k].n], .bits = grows[k].bits[0][0 .. grows[k].n * MW] };
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
    if (windows.len > max_streams) return error.TooManyStreams;
    var ls: [max_streams]*Lane = undefined;
    var kept: [max_streams]u32 = undefined;
    for (windows, paths, 0..) |win, path, k| {
        if (path.len == 0) return error.EmptyPath;
        for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        ls[k] = self.lanes_by.getPtr(win.stream) orelse return error.NoLane;
        kept[k] = @intCast(path.len);
    }
    if (windows.len == 1) return settle(self, ls[0], kept[0]);
    try settleMany(self, ls[0..windows.len], kept[0..windows.len]);
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
        try reserveRound(self, &.{l}, &.{r.stream}, &.{forward.reach(l.seq, @as(i64, @intCast(n + depth)) + 2)});
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
    var kept: [max_streams]u32 = undefined;
    for (requests, 0..) |r, k| kept[k] = if (r.rows) |rows| @intCast(rows.len) else lanes_of[k].rows;
    try settleMany(self, lanes_of[0..requests.len], kept[0..requests.len]);
    for (requests, 0..) |r, k| {
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
    {
        var ss: [max_streams]*lanes.Stream = undefined;
        var targets: [max_streams]i64 = undefined;
        for (requests, 0..) |r, k| {
            ss[k] = r.stream;
            targets[k] = forward.reach(lanes_of[k].seq, @as(i64, @intCast(nf[k] + depth[k])) + 2);
        }
        try reserveRound(self, lanes_of[0..requests.len], ss[0..requests.len], targets[0..requests.len]);
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
    if (chances == .chain) for (1..out.len) |i| {
        out[i] *= out[i - 1];
    };
    return true;
}

fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
    const self = of(p);
    const kv = self.lanes_by.fetchRemove(s) orelse return;
    if (kv.value.g) |g| g.m.free();
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    const seq = kv.value.seq;
    const at: usize = @intCast(seq.snap.at);
    const kept_ids: ?[]u32 = if (at > 0 and at <= s.prompt_len) self.gpa.dupe(u32, s.context.items[0..at]) catch null else null;
    w.int(kv.value.id) catch {};
    w.int(@intFromBool(kept_ids != null)) catch {}; // 1: rank 1 keeps it by id
    self.send(.release, &w) catch {};
    if (kept_ids) |k| {
        self.kept.append(self.gpa, .{ .ids = k, .seq = seq, .id = kv.value.id }) catch {
            self.gpa.free(k);
            self.retire(seq);
            return;
        };
        while (self.kept.items.len > keep_max) dropKept(self, 0);
    } else self.retire(seq);
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

/// Rank 1: a prompt's frame fields (prefillFn's): its sequence (a kept one it resumes, else a new one) set up with
/// its draw, grammar row (at row ``at`` of the scratch), kept point and image.
fn followPrefill(self: *Owned, r: *Reader, at: usize, grown: *?bool) !forward.Prompt {
    const id = try r.int();
    const toks = try r.tokens();
    const sampling = try readSampling(r);
    const mask = try readRows(r, self.growsbuf[at..], self.gbits[1][at * MW ..]);
    const reuse = try r.int();
    const start: usize = @intCast(try r.int());
    const cut = try r.int();
    const reach: i64 = @intCast(try r.int());
    const seq = if (reuse != 0) (self.by_id.fetchRemove(reuse) orelse return error.NoSequence).value else try self.obtain();
    try self.by_id.put(id, seq);
    if (reach > 0) { // backed through the prompt's reach, as rank 0 backs its own (0: the calibration's, unasked)
        var ok = true;
        self.grow(seq, reach) catch |err| switch (err) {
            error.OutOfDeviceMemory => ok = false,
            else => return err,
        };
        grown.* = (grown.* orelse true) and ok;
    }
    if (self.images.fetchRemove(id)) |kv| {
        defer self.gpa.free(kv.value);
        try attachFrame(self, seq, kv.value);
    }
    seq.sampling = sampling;
    seq.mask = mask;
    seq.cut_at = if (cut != 0) @intCast(cut) else null;
    return .{ .s = seq, .ids = toks, .start = start };
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
                try self.gramScratch();
                var grown: ?bool = null;
                const pr = try followPrefill(self, &r, 0, &grown);
                if (try self.settleGrowth(grown, &.{pr})) _ = try forward.prefill(self.e, pr.s, pr.ids, pr.start);
            },
            .prefills => {
                const n: usize = @intCast(try r.int());
                if (n < 2 or n > max_streams) return error.TooManyStreams;
                try self.gramScratch();
                var prompts: [max_streams]forward.Prompt = undefined;
                var grown: ?bool = null;
                for (prompts[0..n], 0..) |*pr, k| pr.* = try followPrefill(self, &r, k, &grown);
                var firsts: [max_streams]u32 = undefined;
                if (try self.settleGrowth(grown, prompts[0..n])) try forward.prefillMany(self.e, prompts[0..n], firsts[0..n]);
            },
            .verify => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const ids = try r.tokens();
                try self.gramScratch();
                seq.mask = try readRows(&r, self.growsbuf[0..], self.gbits[1]);
                try forward.verify(self.e, seq, ids, sampled[0..ids.len]);
            },
            .keep => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const rows: u32 = @intCast(try r.int());
                try forward.keep(self.e, seq, rows, @intCast(try r.int()));
            },
            .keeps => {
                const n: usize = @intCast(try r.int());
                if (n == 0 or n > max_streams) return error.TooManyStreams;
                var items: [max_streams]forward.Keep = undefined;
                for (items[0..n]) |*it| {
                    const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                    const rows: u32 = @intCast(try r.int());
                    it.* = .{ .s = seq, .rows = rows, .kept = @intCast(try r.int()) };
                }
                try forward.keepMany(self.e, items[0..n]);
            },
            .draft => {
                const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                const follow = try r.tokens();
                const depth: u32 = @intCast(try r.int());
                const cut: f64 = if (try r.int() != 0) draft_confidence else 0;
                _ = try forward.draftUpTo(self.e, seq, follow, depth, cut, held[0..depth]);
            },
            .release => {
                const rid = try r.int();
                if (try r.int() == 1) continue; // kept: it stays under its id until a prefill resumes it or it is dropped
                const kv = self.by_id.fetchRemove(rid) orelse continue;
                if (kv.key == 1) { // the calibration's sequence (rank 0 frees its own at open, see there)
                    forward.freeSeq(self.e, kv.value);
                    continue;
                }
                self.retire(kv.value);
            },
            .stop => return,
            .ready => growthLimit(self.e), // the calibration ended
            .grow => { // rank 0 grew these sequences' caches for its next frame: rank 1 grows them too, and answers
                const n: usize = @intCast(try r.int());
                if (n == 0 or n > max_streams) return error.TooManyStreams;
                var oks: [max_streams]u8 = undefined;
                for (oks[0..n]) |*ok| {
                    const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                    const target: i64 = @intCast(try r.int());
                    ok.* = 1;
                    self.grow(seq, target) catch |err| switch (err) {
                        error.OutOfDeviceMemory => ok.* = 0,
                        else => return err,
                    };
                }
                try self.link.send(@intFromEnum(Op.grow), oks[0..n]);
            },
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
                var used: usize = 0;
                try self.gramScratch();
                for (parts[0..n]) |*pt| {
                    const seq = self.by_id.get(try r.int()) orelse return error.NoSequence;
                    pt.* = .{ .s = seq, .ids = try r.tokens() };
                    seq.mask = try readRows(&r, self.growsbuf[used..], self.gbits[1][used * MW ..]);
                    if (seq.mask) |mk| used += mk.rows.len;
                    total += pt.ids.len;
                }
                var flat: [batch_rows]u32 = undefined;
                if (total > flat.len) return error.WindowTooWide;
                try forward.verifyShared(self.e, parts[0..n], flat[0..total]);
            },
        }
    }
}
