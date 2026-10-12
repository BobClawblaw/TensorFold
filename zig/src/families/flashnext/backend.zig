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
const sample = @import("sample.zig");
const think = @import("think.zig");

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
    lone_handoffs: u64 = 0, // lone streams the engine's rounds handed to the shared ones

    pub fn init(gpa: std.mem.Allocator, e: *Engine, n: u32, capacity: usize) !Backend {
        if (n < 2 or n > fz.MAXR or e.r.tp != null) return error.BadBatchSlots;
        const sessions = try gpa.alloc(Session, n);
        errdefer gpa.free(sessions);
        var made: usize = 0;
        errdefer for (sessions[0..made]) |*s| s.deinit(gpa);
        for (sessions, 0..) |*s, i| {
            s.* = try Session.init(gpa, e, capacity, i == 0);
            made += 1;
        }
        const hidden = try e.r.device.buffer(fz.MAXR * fz.WIDE * 2, fz.opts);
        return .{ .gpa = gpa, .e = e, .primary = e.m, .sessions = sessions, .hidden = .{ .b = hidden }, .chain_drafts = if (std.c.getenv("FZ_BATCH_MTP_CHAIN")) |v| v[0] == '1' else false };
    }

    pub fn deinit(b: *Backend) void {
        b.e.m = b.primary;
        b.by.deinit(b.gpa);
        for (b.sessions) |*s| s.deinit(b.gpa);
        b.gpa.free(b.sessions);
        b.hidden.b.deinit();
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    pub fn facts(b: *Backend) lanes.Model {
        return .{ .exact_width = fz.MAXR, .first_copy_rows = 4, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 8, .streams_exact = true, .hidden_rows = true, .batch_rows = fz.MAXR, .max_streams = @intCast(b.sessions.len), .draft_streams = false };
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
        fn restore(old: Binding, b: *Backend) void {
            b.e.m = old.model;
            b.e.host = old.host;
            b.e.cins = old.cins;
            b.e.r.sel = old.select;
            b.e.r.shapes.put(b.e.r.arena, "Kc_shape", old.shape) catch unreachable;
            b.e.passed = old.passed;
        }
    };

    fn bind(b: *Backend, s: *Session) !Binding {
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

    fn prefill(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try self(ptr).fill(stream, null);
    }

    /// A stream's whole prompt pass, in session `slot` when given (free), else the first free one.
    fn fill(b: *Backend, stream: *Stream, slot: ?usize) !void {
        const prompt = stream.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        try b.settle();
        const i = slot orelse for (b.sessions, 0..) |s, j| {
            if (!s.used) break j;
        } else return error.NoFreeSlot;
        if (b.sessions[i].used) return error.NoFreeSlot;
        const s = &b.sessions[i];
        if (prompt.len + stream.max_new + fz.MAXR + 1 > s.m.cap) return error.ContextFull;
        try b.by.put(b.gpa, stream, i);
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

    /// LaneHost's driver for a lone stream: the shared sessions' own prompt pass into the engine's session (the states
    /// the prompt cache keeps are theirs), then the engine's GPU-side rounds, two deep, until the stream finishes
    /// (false) or `hooks.yield` asks for it: its state then stands as a shared round leaves it and the lane core takes
    /// it on (true). `hooks`: LaneHost's LoneHooks.
    pub fn lone(b: *Backend, stream: *Stream, hooks: anytype) !bool {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const prompt = stream.prompt();
        const s = &b.sessions[0]; // the GPU-side rounds run on the engine's own buffers
        if (s.used) return error.NoFreeSlot;
        if (prompt.len + stream.max_new + @import("engine.zig").MARGIN > s.m.cap) return error.ContextFull;
        var handed = false;
        defer if (!handed) release(b, stream);
        try b.fill(stream, 0);
        _ = try stream.commit(b.gpa, &.{s.first}, &.{});
        hooks.committed(hooks.ctx);
        if (stream.finished) return false;
        const Hooks = @TypeOf(hooks);
        const Ctx = struct {
            gpa: std.mem.Allocator,
            stream: *Stream,
            hooks: Hooks,
            failed: ?anyerror = null,
            fn prefilled(_: *anyopaque) void {}
            fn tokens(p: *anyopaque, toks: []const u32) bool {
                const c: *@This() = @ptrCast(@alignCast(p));
                _ = c.stream.commit(c.gpa, toks, &.{}) catch |err| {
                    c.failed = err;
                    return true;
                };
                c.hooks.committed(c.hooks.ctx);
                return c.stream.finished;
            }
            fn cancelled(p: *anyopaque) bool { // a request waits, or one is cancelled: the lane core takes over
                const c: *@This() = @ptrCast(@alignCast(p));
                return c.hooks.yield(c.hooks.ctx);
            }
        };
        var c: Ctx = .{ .gpa = b.gpa, .stream = stream, .hooks = hooks };
        var ho: Engine.Handoff = .{};
        {
            const old = try b.bind(s);
            defer old.restore(b);
            outputs(b, s, 0);
            var watch = try think.Watch.init(.{}); // no thinking budget: lone_host_lone.fits keeps those in the lane core
            const res = try b.e.decode(prompt, s.first, s.last, stream.max_new, stream.eos, null, &watch, .{ .ctx = &c, .prefilled = Ctx.prefilled, .tokens = Ctx.tokens, .cancelled = Ctx.cancelled }, &ho);
            if (c.failed) |err| return err;
            stream.rounds += res.rounds;
            stream.drafted += res.drafted;
            stream.accepted += res.accepted;
            if (res.rounds > 0) stream.min_rows = if (stream.min_rows == 0) res.min_rows else @min(stream.min_rows, res.min_rows);
        }
        if (!ho.yielded) {
            if (!stream.finished) { // the rounds ended where the stream did not (eos or length by the engine's count)
                stream.finished = true;
                stream.reason = .stop;
            }
            return false;
        }
        const ctx = stream.context.items;
        const at = ho.cache; // the cache's rows: every committed token but the last, the next window's first
        if (at + 1 != ctx.len or at < 2) return error.HandoffOutOfStep;
        s.m.pos = at;
        s.m.state = 0;
        s.m.state_row = 0;
        s.m.ple.hist = .{ ctx[at - 2], ctx[at - 1] };
        s.m.mtp.pos = at;
        s.m.mtp.drafted = 0;
        s.m.mtp.pooled_n = @min(s.m.mtp.pooled_n, at / 4); // blocks the rounds pooled are pooled again here
        for (&s.m.layers) |*l| if (!l.linear) {
            l.pooled_n = @min(l.pooled_n, at / 4);
        };
        s.held_n = 0;
        s.round = .{};
        stream.cache_len = at;
        stream.pending = ctx[at];
        stream.next = null;
        handed = true;
        b.lone_handoffs += 1;
        std.log.info("flash next: a lone stream at {d} tokens handed to the shared rounds ({d} handoffs)", .{ at, b.lone_handoffs });
        return true;
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

    fn outputs(b: *Backend, s: *Session, row: usize) void {
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

    fn release(ptr: *anyopaque, stream: *Stream) void {
        const b = self(ptr);
        const kv = b.by.fetchRemove(stream) orelse return;
        b.sessions[kv.value].used = false;
        b.sessions[kv.value].round = .{};
        b.sessions[kv.value].held_n = 0;
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
