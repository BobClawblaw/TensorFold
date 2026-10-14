//! Flash Next's isolated sessions behind the shared lane scheduler; target rounds batch, MTP drafts are serial.
const std = @import("std");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;
const Session = @import("session.zig").Session;
const meta = @import("batch_meta.zig");
const forward = @import("batch_forward.zig");
const snap = @import("snapshot.zig");
const be = lanes.backend;
const Stream = lanes.Stream;
const Buf = fz.Buf;
const mtl = @import("metal");
const segments = @import("../../core/segments.zig");
const slots = @import("slots.zig");
const sample = @import("sample.zig");
const batch_costs = @import("batch_costs.zig");

/// A stream's sampling when it samples (a temperature above 0); null: the GPU's argmax stands.
fn sampling(stream: *const Stream) ?lanes.Sampling {
    const p = stream.sampling orelse return null;
    return if (p.temperature > 0) p else null;
}

/// Row `row` of a [rows, VOCAB] bf16 logits buffer, drawn with `p` keyed at `position`.
fn drawRow(gpa: std.mem.Allocator, logits: Buf, row: usize, p: lanes.Sampling, position: u64) !u32 {
    const bytes = logits.b.contents()[logits.off + row * fz.VOCAB * 2 ..][0 .. fz.VOCAB * 2];
    return sample.draw(gpa, @as([*]const u16, @ptrCast(@alignCast(bytes.ptr)))[0..fz.VOCAB], p, position);
}

pub const Backend = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    primary: *fz.Model,
    sessions: []Session,
    by: std.AutoHashMapUnmanaged(*Stream, usize) = .empty,
    hidden: Buf,
    window_count: u64 = 0,
    shared_count: u64 = 0,
    chain_drafts: bool = false,
    draft_calls: u64 = 0,
    draft_waits: u64 = 0,
    draft_seconds: f64 = 0,
    sized: bool = false, // FLASHNEXT_SLOT_CONTEXTS set: sessions of several sizes; unset, every stream takes the first free one and none grows
    grow: usize = std.math.maxInt(usize), // reply tokens a session must hold at admission past the prompt (the rest: move)
    paused: std.ArrayList(*Stream) = .empty, // streams that outgrew their session and wait for a larger one to free
    moves: u64 = 0,
    serial_drafts: bool = false, // FZ_BATCH_MTP_SERIAL=1: each stream's drafts in their own command buffers
    costs: ?batch_costs.Costs = null, // timeCosts: the lane allocator's prices

    /// Sessions of the given capacities (the first, the engine's own, holds its whole window whatever is given).
    pub fn init(gpa: std.mem.Allocator, e: *Engine, capacities: []const usize) !Backend {
        if (capacities.len < 2 or capacities.len > fz.MAXR or e.r.tp != null) return error.BadBatchSlots;
        const sessions = try gpa.alloc(Session, capacities.len);
        errdefer gpa.free(sessions);
        var made: usize = 0;
        errdefer for (sessions[0..made]) |*s| s.deinit(gpa);
        for (sessions, capacities, 0..) |*s, capacity, i| {
            s.* = try Session.init(gpa, e, capacity, i == 0);
            made += 1;
        }
        const hidden = try e.r.device.buffer(fz.MAXR * fz.WIDE * 2, fz.opts);
        const env = struct {
            fn on(name: [*:0]const u8) bool {
                return if (std.c.getenv(name)) |v| v[0] == '1' else false;
            }
        };
        return .{ .gpa = gpa, .e = e, .primary = e.m, .sessions = sessions, .hidden = .{ .b = hidden }, .chain_drafts = env.on("FZ_BATCH_MTP_CHAIN"), .serial_drafts = env.on("FZ_BATCH_MTP_SERIAL") };
    }

    /// The lane allocator's prices, timed on the last session before any stream holds it; none (unpriced) when it can't be.
    pub fn timeCosts(b: *Backend) !void {
        if (std.c.getenv("FZ_BATCH_UNPRICED") != null) {
            std.log.info("flash next: FZ_BATCH_UNPRICED=1: shared rounds unpriced, nothing timed", .{});
            return;
        }
        const s = &b.sessions[b.sessions.len - 1];
        if (!batch_costs.fits(s.m.cap)) {
            std.log.info("flash next: a session holds {d} rows, under the {d} that timing shared rounds takes: shared rounds unpriced", .{ s.m.cap, batch_costs.ROWS });
            return;
        }
        const c = try batch_costs.time(b, s);
        b.costs = c;
        std.log.info("flash next: shared forwards {d:.2} {d:.2} {d:.2} {d:.2} {d:.2} ms at 1, 2, 4, 8, 16 rows; a chained draft {d:.2} ms", .{ c.shared[0].ms, c.shared[1].ms, c.shared[3].ms, c.shared[7].ms, c.shared[15].ms, c.step_ms });
    }

    pub fn deinit(b: *Backend) void {
        b.e.m = b.primary;
        b.paused.deinit(b.gpa);
        b.by.deinit(b.gpa);
        for (b.sessions) |*s| s.deinit(b.gpa);
        b.gpa.free(b.sessions);
        b.hidden.b.deinit();
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    pub fn facts(b: *Backend) lanes.Model {
        const c: ?*const batch_costs.Costs = if (b.costs) |*x| x else null;
        // shared rounds priced from the timed forwards, each draft past a stream's first at a chained step's time; a
        // lone stream's depth from the same (FZ_LONE_UNPRICED=1: the heuristic)
        const lone = c != null and std.c.getenv("FZ_LONE_UNPRICED") == null;
        return .{ .exact_width = fz.MAXR, .first_copy_rows = 4, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 8, .streams_exact = true, .hidden_rows = true, .batch_rows = fz.MAXR, .max_streams = @intCast(b.sessions.len), .draft_streams = !b.serial_drafts, .shared_costs = if (c) |x| &x.shared else &.{}, .draft_step_ms = if (c != null and !b.serial_drafts) c.?.step_ms else 0, .window_costs = if (lone) &c.?.shared else &.{}, .mtp_step_ms = if (lone) c.?.step_ms else 0 };
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }
    fn session(b: *Backend, s: *Stream) !*Session {
        return &b.sessions[b.by.get(s) orelse return error.UnknownStream];
    }
    fn settle(b: *Backend) !void {
        var needed = false;
        for (b.sessions) |*s| if (s.used) {
            if (s.round.pending) try s.keep(s.round.rows);
            needed = needed or s.m.state != 0 or s.m.state_row != 0;
        };
        if (!needed) return;
        const cb = b.e.r.queue.commandBuffer();
        const blit = cb.blit();
        for (b.sessions) |*s| if (s.used) s.normalizeGpu(blit);
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("session state copy failed: {s}", .{message});
            return error.GpuFailed;
        }
        for (b.sessions) |*s| if (s.used) {
            s.m.state = 0;
            s.m.state_row = 0;
        };
    }

    const Binding = struct {
        model: *fz.Model,
        host: @FieldType(Engine, "host"),
        cins: [2]Buf,
        select: ?fz.Select,
        shape: @TypeOf(@as(fz.Select, undefined).q_shape),
        passed: @FieldType(Engine, "passed"),
        pub fn restore(old: Binding, b: *Backend) void {
            b.e.m = old.model;
            b.e.host = old.host;
            b.e.cins = old.cins;
            b.e.r.sel = old.select;
            b.e.r.shapes.put(b.e.r.arena, "Kc_shape", old.shape) catch unreachable;
            b.e.passed = old.passed;
        }
    };

    pub fn bind(b: *Backend, s: *Session) !Binding {
        const e = b.e;
        const old: Binding = .{ .model = e.m, .host = e.host, .cins = e.cins, .select = e.r.sel, .shape = e.r.shapes.get("Kc_shape").?, .passed = e.passed };
        try e.r.shapes.put(e.r.arena, "Kc_shape", s.shape);
        e.m = &s.m;
        e.host = .{ .rows = s.m.t.rows, .mdims = s.m.t.mdims, .pos8 = s.m.t.pos8, .nk8 = s.m.t.nk8, .kvmeta = s.m.t.kvmeta, .slots = s.m.mtp.slots };
        e.cins[0] = s.m.ple.cin;
        e.r.sel = s.select[fz.CATCH - 1];
        e.passed = null;
        e.hostMode();
        return old;
    }

    /// A new stream's session (slots.pick); LaneMemory while only a busy one holds it (it waits), ContextFull when none can.
    fn slotFor(b: *Backend, prompt: usize, max_new: usize) !usize {
        var caps: [fz.MAXR]usize = undefined;
        var used: [fz.MAXR]bool = undefined;
        var most: usize = 0;
        for (b.sessions, 0..) |s, j| {
            caps[j] = s.m.cap;
            used[j] = s.used;
            most = @max(most, s.m.cap);
        }
        const n = b.sessions.len;
        const whole = prompt + max_new + slots.WINDOW;
        const i = slots.pick(caps[0..n], used[0..n], b.sized, prompt, max_new, b.grow) orelse
            return if (!b.sized) error.NoFreeSlot else if (whole > most) error.ContextFull else error.LaneMemory;
        if (whole > (if (b.sized) most else caps[i])) return error.ContextFull; // sized, a smaller one grows into the largest
        return i;
    }

    /// Before a round, sized only: a stream its session cannot hold to its reply's end moves to a free larger one, or pauses (slots.room).
    fn makeRoom(b: *Backend, stream: *Stream, rows: usize) !void {
        if (!b.sized) return;
        const i = b.by.get(stream) orelse return error.UnknownStream;
        const s = &b.sessions[i];
        var most: usize = 0;
        var larger = false;
        for (b.sessions) |o| {
            most = @max(most, o.m.cap);
            larger = larger or (!o.used and o.m.cap > s.m.cap);
        }
        switch (slots.room(s.m.cap, s.m.pos, rows, stream.prompt().len + stream.max_new + slots.WINDOW, most, larger)) {
            .ok => {},
            .move => try b.move(stream, i),
            .pause => {
                if (!stream.paused) {
                    try b.paused.append(b.gpa, stream);
                    std.log.info("flash next: a stream at {d} tokens waits for a session larger than {d} tokens to free", .{ s.m.pos, s.m.cap });
                }
                stream.paused = true;
            },
            .full => return error.ContextFull,
        }
    }

    /// The stream in session `i` into the smallest free larger session holding its next `grow` tokens, else the largest free larger one.
    fn move(b: *Backend, stream: *Stream, i: usize) !void {
        const from = &b.sessions[i];
        const left: usize = @intCast(@max(stream.budgetLeft(), 0));
        const need = from.m.pos + @min(left, b.grow) + 2 * slots.WINDOW;
        var best: ?usize = null; // the smallest free larger session that holds `need`
        var most: ?usize = null; // else the largest free larger one
        for (b.sessions, 0..) |o, j| if (!o.used and o.m.cap > from.m.cap) {
            if (o.m.cap >= need and (best == null or o.m.cap < b.sessions[best.?].m.cap)) best = j;
            if (most == null or o.m.cap > b.sessions[most.?].m.cap) most = j;
        };
        const j = best orelse most orelse return error.LaneMemory;
        try b.settle();
        try from.moveTo(&b.sessions[j], b.e);
        try b.by.put(b.gpa, stream, j);
        from.used = false;
        from.round = .{};
        from.held_n = 0;
        b.moves += 1;
        std.log.info("flash next: a stream at {d} tokens moved from a {d}-token session to a {d}-token one ({d} moves)", .{ from.m.pos, from.m.cap, b.sessions[j].m.cap, b.moves });
    }

    /// Paused streams into sessions that freed (slots.resumePaused): a move that fails but for LaneMemory ends the stream.
    fn resumePaused(b: *Backend) void {
        slots.resumePaused(&b.paused, b);
    }

    /// slots.resumePaused's move: the stream into a free larger session, LaneMemory while none is.
    pub fn moveOut(b: *Backend, stream: *Stream) !void {
        try b.move(stream, b.by.get(stream) orelse return error.UnknownStream);
    }

    /// slots.resumePaused's failure: the stream ends with an error the host reports, and its session frees now.
    pub fn failed(b: *Backend, stream: *Stream, err: anyerror) void {
        std.log.err("flash next: a paused stream at {d} tokens could not move to a larger session ({s}); it ends with an error", .{ stream.context.items.len, @errorName(err) });
        slots.endPaused(stream);
        const kv = b.by.fetchRemove(stream) orelse return;
        b.sessions[kv.value].used = false;
        b.sessions[kv.value].round = .{};
        b.sessions[kv.value].held_n = 0;
    }

    /// LaneHost's admission check: a free session holds the request now.
    pub fn fits(ptr: *anyopaque, prompt: u32, max_new: u32) bool {
        _ = self(ptr).slotFor(prompt, max_new) catch return false;
        return true;
    }

    fn prefill(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const b = self(ptr);
        const prompt = stream.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        try b.settle();
        const i = try b.slotFor(prompt.len, stream.max_new);
        const s = &b.sessions[i];
        try b.by.put(b.gpa, stream, i);
        errdefer {
            _ = b.by.remove(stream);
            s.used = false;
        }
        s.used = true;
        s.held_n = 0;
        s.round = .{};
        s.row0 = 0;
        outputs(b, s, 0);
        const old = try b.bind(s);
        defer old.restore(b);
        s.m.reset();
        stream.cached = 0;
        if (stream.reuse.saved) |saved| {
            const state: *snap.State = @ptrCast(@alignCast(saved));
            if (state.at < prompt.len and state.at < s.m.cap) {
                try snap.restore(b.e, state);
                stream.cached = @intCast(state.at);
            } else stream.reuse_failed = true;
        }
        var at: usize = s.m.pos;
        while (at < prompt.len) {
            if (stream.isCancelled()) return error.Cancelled;
            var end = prompt.len;
            for (stream.chunks) |mark| if (mark > at) {
                end = @min(end, mark);
                break;
            };
            for (stream.reuse.marks) |mark| if (mark > at) {
                end = @min(end, mark);
                break;
            };
            const call = if (b.e.segments) segments.next(end - at, b.e.pr.step, @import("engine.zig").SEG_MIN) else segments.Call{ .rows = @min(end - at, b.e.pr.step), .parts = 1 };
            end = at + call.rows;
            const ps = [2]*fz.Prompt{ b.e.pr, b.e.pr2 };
            s.first = try fz.Prompt.chunkN(ps[0..call.parts], &s.m, b.gpa, prompt[at..end]);
            // the pass's last row's logits are in the model's head buffer until the next forward: a sampled first token now
            if (end == prompt.len) if (sampling(stream)) |p| {
                s.first = try drawRow(b.gpa, s.m.t.logits, 0, p, prompt.len);
            };
            for (ps[0..call.parts], 0..) |p, k| {
                const start = at + segments.start(call.rows, call.parts, k);
                const rows = segments.rows(call.rows, call.parts, k);
                const count = if (start + rows < prompt.len) rows else rows - 1;
                try p.mtpKeys(&s.m, start, prompt[start + 1 .. start + 1 + count], p.last);
                if (start + rows == prompt.len) Session.copy(p.last.at((rows - 1) * fz.WIDE * 2), s.last, fz.WIDE * 2);
            }
            s.normalize();
            at = end;
            if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(at)) != null)
                if (stream.reuse.hook) |hook| hook.at(hook.ptr, stream, @intCast(at));
        }
        s.m.mtp.pos = prompt.len - 1;
        s.m.mtp.drafted = 0;
    }

    fn first(ptr: *anyopaque, stream: *Stream, position: u64) anyerror!u64 {
        const s = try self(ptr).session(stream);
        if (position != s.m.pos) return error.PositionMismatch;
        return s.first;
    }
    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.Unsupported;
    }
    fn read(_: *anyopaque, handle: u64) anyerror!u32 {
        return @intCast(handle);
    }

    pub fn outputs(b: *Backend, s: *Session, row: usize) void {
        var li: usize = 0;
        for (&s.m.layers) |*l| if (l.linear) {
            l.cs[1] = .{ .b = b.e.o_cs, .off = (li * fz.MAXR + row) * fz.CS_ROW };
            l.so[1] = .{ .b = b.e.o_so, .off = (li * fz.MAXR + row) * fz.SO_ROW };
            li += 1;
        };
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const b = self(ptr);
        try b.settle();
        if (windows.len == 0 or windows.len > fz.MAXR or out.len != windows.len) return error.BadBatch;
        for (windows) |w| try b.makeRoom(w.stream, w.rows());
        var desc: [fz.MAXR]meta.Window = undefined;
        var segs: [fz.MAXR]forward.Seg = undefined;
        for (windows, out, 0..) |w, o, i| {
            const s = try b.session(w.stream);
            if (w.parents != null or w.early or w.held > s.held_n) return error.WindowOutOfStep;
            if (w.rows() > fz.MAXR or w.rows() > s.m.cap - s.m.pos or
                w.positions.len != w.rows() or o.sampled.len != w.rows() or o.drafts.len != w.rows() - 1)
                return error.WindowOutOfStep;
            for (w.positions, 0..) |p, j| if (p != s.m.pos + j + 1) return error.PositionMismatch;
            desc[i] = .{ .slot = @intCast(b.by.get(w.stream).?), .pos = @intCast(s.m.pos), .rows = @intCast(w.rows()) };
        }
        const plan = try meta.Plan.init(desc[0..windows.len], fz.CAP);
        const t = &b.primary.t;
        for (windows, out, 0..) |w, o, i| {
            const s = try b.session(w.stream);
            const row = plan.starts[i];
            s.row0 = row;
            try s.round.begin(s.m.pos, w.rows());
            s.tokens[0] = w.pending;
            @memcpy(s.tokens[1..][0..w.held], s.held[0..w.held]);
            @memcpy(s.tokens[1 + w.held ..][0..w.tokens.len], w.tokens);
            s.m.windowMeta(s.round.rows);
            s.m.pleIds(s.tokens[0..s.round.rows]);
            @memcpy(t.ids8.b.slice(u32, fz.MAXR)[row..][0..s.round.rows], s.tokens[0..s.round.rows]);
            @memcpy(t.ple_ids.b.slice(u32, fz.MAXR * 16)[row * 16 ..][0 .. s.round.rows * 16], s.m.t.ple_ids.b.slice(u32, s.round.rows * 16));
            outputs(b, s, row);
            segs[i] = .{ .s = s, .row0 = row, .rows = s.round.rows };
            @memcpy(o.drafts, s.tokens[1..s.round.rows]);
        }
        t.rows.b.slice(i32, 1)[0] = @intCast(plan.total);
        t.mdims.b.slice(i32, 2)[0] = @intCast(plan.total);
        @memcpy(t.pos8.b.slice(u32, fz.MAXR)[0..plan.total], plan.positions[0..plan.total]);
        const cb = b.e.r.queue.commandBuffer();
        b.e.r.enc = cb.compute(.serial);
        try forward.encode(b.e, segs[0..windows.len], plan.total, t.ids8);
        try b.primary.finish(cb);
        Session.copy(b.primary.last, b.hidden, plan.total * fz.WIDE * 2);
        for (out, 0..) |o, i| @memcpy(o.sampled, t.picks.b.slice(u32, plan.total)[plan.starts[i]..][0..o.sampled.len]);
        for (windows, out, 0..) |w, o, i| if (sampling(w.stream)) |p| {
            for (o.sampled, w.positions, 0..) |*tok, pos, j| tok.* = try drawRow(b.gpa, t.logits, plan.starts[i] + j, p, pos);
        };
        b.window_count += 1;
        if (windows.len > 1) b.shared_count += 1;
        if (std.c.getenv("FZ_BATCH_LOG") != null)
            std.log.info("flashnext batch: streams {d}, rows {d}, shared rounds {d}", .{ windows.len, plan.total, b.shared_count });
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |r, j| if (r != j) return error.TreeWindowsUnsupported;
            try (try b.session(w.stream)).keep(path.len);
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        if (!b.serial_drafts) return b.draftTogether(requests);
        for (requests) |r| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            if (r.early or r.lanes != null or r.ranks or r.depth >= fz.MAXR) return error.DraftOutOfStep;
            const s = try b.session(r.stream);
            var one: [1]u32 = undefined;
            const follow = if (r.first) |f| blk: {
                one[0] = switch (f) {
                    .value => |v| v,
                    .handle => |h| @intCast(h),
                };
                break :blk one[0..];
            } else r.follow;
            const hidden = if (r.rows) |path| blk: {
                try s.round.draft(r.start, path, follow.len);
                break :blk b.hidden.at(s.row0 * fz.WIDE * 2);
            } else s.last;
            const old = try b.bind(s);
            defer old.restore(b);
            s.held_n = r.depth;
            if (follow.len == 0) return error.DraftOutOfStep;
            const start = mtl.clock.seconds();
            if (b.chain_drafts and r.depth > 0 and follow.len <= fz.MAXR) {
                try s.draftChain(follow, hidden, r.depth);
                b.draft_waits += 1;
            } else {
                const token = try s.m.mtpAbsorb(follow, hidden);
                if (r.depth > 0) s.held[0] = token;
                if (r.depth > 1) for (1..r.depth) |i| {
                    s.held[i] = try s.m.mtpChain(s.held[i - 1]);
                };
                b.draft_waits += (follow.len + fz.MAXR - 1) / fz.MAXR + (r.depth -| 1);
            }
            b.draft_calls += 1;
            b.draft_seconds += mtl.clock.seconds() - start;
        }
    }

    /// Every request's absorb and chained drafts in one command buffer, one wait (a long follow absorbs alone first).
    fn draftTogether(b: *Backend, requests: []const be.DraftRequest) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const start = mtl.clock.seconds();
        var follows: [fz.MAXR][]const u32 = undefined;
        var hiddens: [fz.MAXR]Buf = undefined;
        var ones: [fz.MAXR][1]u32 = undefined;
        if (requests.len > fz.MAXR) return error.DraftOutOfStep;
        for (requests, 0..) |r, k| {
            if (r.early or r.lanes != null or r.ranks or r.depth >= fz.MAXR) return error.DraftOutOfStep;
            const s = try b.session(r.stream);
            follows[k] = if (r.first) |f| blk: {
                ones[k][0] = switch (f) {
                    .value => |v| v,
                    .handle => |h| @intCast(h),
                };
                break :blk ones[k][0..];
            } else r.follow;
            if (follows[k].len == 0) return error.DraftOutOfStep;
            hiddens[k] = if (r.rows) |path| blk: {
                try s.round.draft(r.start, path, follows[k].len);
                break :blk b.hidden.at(s.row0 * fz.WIDE * 2);
            } else s.last;
            if (follows[k].len > fz.MAXR) { // all but the last MAXR rows absorbed first, on their own
                const cut = follows[k].len - fz.MAXR;
                const old = try b.bind(s);
                defer old.restore(b);
                _ = try s.m.mtpAbsorb(follows[k][0..cut], hiddens[k]);
                hiddens[k] = hiddens[k].at(cut * fz.WIDE * 2);
                follows[k] = follows[k][cut..];
                b.draft_waits += (cut + fz.MAXR - 1) / fz.MAXR;
            }
        }
        const cb = b.e.r.queue.commandBuffer();
        b.e.r.enc = cb.compute(if (b.e.r.serial) .serial else .concurrent);
        for (requests, 0..) |r, k| {
            const s = try b.session(r.stream);
            const old = try b.bind(s);
            defer old.restore(b);
            s.encodeDrafts(follows[k], hiddens[k], r.depth) catch |err| {
                b.e.r.enc.end();
                return err;
            };
        }
        try b.primary.finish(cb);
        for (requests) |r| (try b.session(r.stream)).heldDrafts(r.depth);
        b.draft_waits += 1;
        b.draft_calls += requests.len;
        b.draft_seconds += mtl.clock.seconds() - start;
    }

    fn release(ptr: *anyopaque, stream: *Stream) void {
        const b = self(ptr);
        if (std.mem.indexOfScalar(*Stream, b.paused.items, stream)) |k| _ = b.paused.orderedRemove(k);
        const kv = b.by.fetchRemove(stream) orelse return;
        b.sessions[kv.value].used = false;
        b.sessions[kv.value].round = .{};
        b.sessions[kv.value].held_n = 0;
        if (b.paused.items.len > 0) b.resumePaused();
    }

    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return self(ptr).e.snap_pool.size(snap.bytes(at));
    }
    pub fn snapCharged(_: *anyopaque, saved: *anyopaque) u64 {
        const state: *snap.State = @ptrCast(@alignCast(saved));
        return state.cap;
    }
    pub fn snapSpare(ptr: *anyopaque) u64 {
        return self(ptr).e.snap_pool.spare();
    }
    pub fn snapTrim(ptr: *anyopaque, room: u64) void {
        self(ptr).e.snap_pool.trim(room);
    }
    pub fn snapReuses(ptr: *anyopaque, at: u32) bool {
        return self(ptr).e.snap_pool.fits(snap.bytes(at));
    }
    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const b = self(ptr);
        const s = try b.session(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        const old = try b.bind(s);
        defer old.restore(b);
        return try snap.save(b.e, b.gpa, at);
    }
    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores;
    }
    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        snap.drop(self(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};
