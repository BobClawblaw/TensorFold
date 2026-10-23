//! DeepSeek-V4.1-Flash on the lane engine: chunked prompts, DSpark drafts in windows, several streams a forward.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const vision = @import("vision.zig");
const image = @import("image.zig");
const Model = @import("model.zig").Model;
const draw = @import("../lfm2_moe/backend.zig").draw;
const be = lanes.backend;

/// `chunk`: a lone stream's prompt rows a call (the scratch); `shared_chunk`: its rows a call beside other streams.
pub const Options = struct { capacity: usize, chunk: usize = 1024, shared_chunk: usize = 1024 };

/// The widest window: the pending row and the head's five drafts.
pub const window_rows = 1 + c.draft_block;
/// Streams a shared forward takes at most.
pub const max_streams = 4;
/// Shared forwards timed at load: their total rows, split evenly over `max_streams` streams.
const shared_rows = [_]usize{ 8, 12, 16, 24 };

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    /// The draft head's own scratch (a draft round runs between a window and the next).
    drafts: ?st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    /// Forward ms by window width and one draft round's, timed at load: the lane core weighs depths against them.
    costs: [window_rows]lanes.config.Cost = undefined,
    shared: [shared_rows.len]lanes.config.Cost = undefined,
    draft_ms: f64 = 0,
    /// Prompt passes under way, a chunk a `prefillStepFn` call.
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, Fill) = .empty,

    /// A prompt pass's next row and its images' rows.
    const Fill = struct { at: usize, vis: ?fwd.Vis };

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, model.device, @min(options.chunk, st.max_rows)), .drafts = null };
        errdefer out.scratch.deinit();
        if (model.weights.stages != null) {
            out.drafts = try st.Scratch.initDraft(gpa, model.device, 8);
            try out.time();
        }
        return out;
    }

    pub fn deinit(self: *Metal) void {
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        var fills = self.fills.valueIterator();
        while (fills.next()) |f| if (f.vis) |v| self.freeVis(v);
        self.fills.deinit(self.gpa);
        if (self.drafts) |*d| d.deinit();
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Metal) lanes.Model {
        if (self.drafts == null) return .{ .exact_width = 1, .batch_rows = 1, .max_streams = 1, .streams_exact = true, .plain_guard = true };
        // shared forwards: every row on the row kernels, so a stream's rows give its own bits (forward.many)
        return .{
            .exact_width = window_rows,
            .mtp = true,
            .speculate = true, // drafts from a window's kept rows (draftFn), after its keep
            .speculate_early = false,
            .drafts = c.draft_block,
            .window_costs = &self.costs,
            // the head drafts its five in one call: priced a step at a fifth of it
            .mtp_step_ms = self.draft_ms / c.draft_block,
            .plain_guard = true,
            .hidden_rows = true,
            .batch_rows = max_streams * window_rows,
            .max_streams = max_streams,
            .shared_costs = &self.shared,
            .draft_step_ms = self.draft_ms, // the head drafts a shared round's streams one after another
            .streams_exact = true,
        };
    }

    /// Each window width's forward ms and a draft round's on a 256-token history (the fastest of three).
    fn time(self: *Metal) !void {
        var cache = try st.Cache.initWith(self.gpa, self.model.device, 512, true);
        defer cache.deinit();
        var ids: [256]u32 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % c.vocab);
        try fwd.run(self.model, &self.scratch, &cache, &ids, .last, null);
        const base = cache.len;
        var held: [c.draft_block]u32 = undefined;
        for (&self.costs, 1..) |*cost, w| {
            var best: f64 = std.math.floatMax(f64);
            for (0..3) |_| {
                const t0 = now();
                try fwd.run(self.model, &self.scratch, &cache, ids[0..w], .all, null);
                best = @min(best, now() - t0);
                cache.len = base;
            }
            cost.* = .{ .width = @intCast(w), .ms = best };
        }
        var best: f64 = std.math.floatMax(f64);
        for (0..3) |_| {
            const t0 = now();
            try fwd.draft(self.model, &self.drafts.?, &cache, ids[0], base - 1, &held);
            best = @min(best, now() - t0);
        }
        self.draft_ms = best;
        var more: [max_streams - 1]st.Cache = undefined;
        var made: usize = 0;
        defer for (more[0..made]) |*m| m.deinit();
        for (&more) |*m| {
            m.* = try st.Cache.initWith(self.gpa, self.model.device, 512, true);
            made += 1;
            try fwd.run(self.model, &self.scratch, m, &ids, .last, null);
        }
        const caches = [max_streams]*st.Cache{ &cache, &more[0], &more[1], &more[2] };
        for (&self.shared, shared_rows) |*cost, total| {
            const each = total / max_streams;
            var segs: [max_streams]fwd.Seg = undefined;
            for (&segs, caches, 0..) |*g, cc, k| g.* = .{ .cache = cc, .tokens = ids[0..each], .r0 = k * each };
            best = std.math.floatMax(f64);
            for (0..3) |_| {
                const t0 = now();
                try fwd.many(self.model, &self.scratch, &segs, .all);
                best = @min(best, now() - t0);
                for (caches) |cc| cc.len = base;
            }
            cost.* = .{ .width = @intCast(total), .ms = best };
        }
        std.log.info("deepseek_v41: window ms by width 1 {d:.1}, 2 {d:.1}, 4 {d:.1}, 6 {d:.1}; shared over 4 streams 8 rows {d:.1}, 16 {d:.1}, 24 {d:.1}; a draft round {d:.1} ms", .{ self.costs[0].ms, self.costs[1].ms, self.costs[3].ms, self.costs[5].ms, self.shared[0].ms, self.shared[2].ms, self.shared[3].ms, self.draft_ms });
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .prefill_step = prefillStepFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .held = heldFn, .release = releaseFn, .reports_fill = true } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownDeepseekStream;
    }

    /// The whole prompt pass: the chunks `prefillStepFn` takes one a call, run back to back.
    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        while (!try cast(ptr).chunk(stream)) {}
    }

    /// One chunk of the prompt pass, the other streams' rounds running between calls; true once the prompt is in.
    fn prefillStepFn(ptr: *anyopaque, stream: *lanes.Stream) !bool {
        if (stream.isCancelled()) return error.Cancelled;
        return cast(ptr).chunk(stream);
    }

    /// The next chunk of the stream's prompt, its caches made (and a kept prefix restored) before the first.
    fn chunk(self: *Metal, stream: *lanes.Stream) !bool {
        errdefer releaseFn(self, stream);
        const ids = stream.prompt();
        const fill = self.fills.getPtr(stream) orelse try self.begin(stream);
        // a lone stream takes the widest chunk; beside other streams (they decode between chunks) the shared one
        const step = if (self.caches.count() == 1) self.scratch.rows else @min(self.scratch.rows, self.options.shared_chunk);
        const end = cut(stream, fill.at, step, step > self.options.shared_chunk);
        const after = cut(stream, end, step, step > self.options.shared_chunk);
        const vis: ?*const fwd.Vis = if (fill.vis) |*v| v else null;
        const next: ?fwd.Next = if (after > end) .{ .tokens = ids[end..after], .vis = vis } else null;
        try fwd.runChunk(self.model, &self.scratch, try self.cacheOf(stream), ids[fill.at..end], if (end == ids.len) .last else .none, vis, next);
        fill.at = end;
        stream.fill_at.store(@intCast(end), .monotonic);
        if (stream.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null)
            hk.at(hk.ptr, stream, @intCast(end));
        if (end < ids.len) return false;
        self.endFill(stream);
        return true;
    }

    /// Where a chunk from `at` ends: `step` rows on, at a chunk boundary (`wide`: the last it reaches) or a save mark.
    fn cut(stream: *lanes.Stream, at: usize, step: usize, wide: bool) usize {
        const ids = stream.prompt();
        var end: usize = @min(ids.len, at + step);
        var reached: ?usize = null;
        for (stream.chunks) |boundary| if (boundary > at) {
            if (boundary > end) break;
            reached = boundary;
            if (!wide) break;
        };
        if (reached) |r| if (!wide or end < ids.len) {
            end = r;
        };
        for (stream.reuse.marks) |m| if (m > at) {
            end = @min(end, m);
            break;
        };
        return end;
    }

    /// A prompt pass starts: the stream's caches, a kept prefix of its prompt restored, its images through the tower.
    fn begin(self: *Metal, stream: *lanes.Stream) !*Fill {
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidDeepseekPrompt;
        const rows = @min(self.options.capacity, ids.len + stream.max_new + window_rows);
        var held: usize = 0;
        var them = self.caches.valueIterator();
        while (them.next()) |other| held += other.*.capacity;
        if (held + rows > self.options.capacity) return error.LaneMemory; // the streams share --context: it waits for room
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.initWith(self.gpa, self.model.device, rows, self.drafts != null);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        var at: usize = 0;
        stream.cached = 0;
        if (stream.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
            if (snap.at < ids.len) if (snapshot.restore(cache, snap)) |_| {
                at = snap.at;
            } else |_| {};
            stream.cached = @intCast(at);
            stream.reuse_failed = at == 0;
        }
        const vis: ?fwd.Vis = if (stream.images.len > 0) try self.images(stream, ids.len) else null;
        const got = self.fills.getOrPut(self.gpa, stream) catch |e| {
            if (vis) |v| self.freeVis(v);
            return e;
        };
        got.value_ptr.* = .{ .at = at, .vis = vis };
        return got.value_ptr;
    }

    /// The stream's prompt pass is over: its images' rows are freed.
    fn endFill(self: *Metal, stream: *lanes.Stream) void {
        const entry = self.fills.fetchRemove(stream) orelse return;
        if (entry.value.vis) |v| self.freeVis(v);
    }

    fn freeVis(self: *Metal, v: fwd.Vis) void {
        v.rows.deinit();
        self.gpa.free(v.codes);
    }

    /// The stream's images through the tower: their aligner rows and each prompt position's code (-1 for text).
    fn images(self: *Metal, stream: *lanes.Stream, prompt_len: usize) !fwd.Vis {
        const v = &(self.model.vision orelse return error.DeepseekCheckpointHasNoVisionTower);
        var total: usize = 0;
        for (stream.images) |img| {
            const g = vision.tokenGrid(img.gh, img.gw);
            if (img.tokens != vision.spanTokens(img.gh, img.gw) or img.at + img.tokens > prompt_len) return error.DeepseekImageOutOfStep;
            total += @as(usize, g[0]) * g[1];
        }
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const rows = try self.model.device.buffer(total * c.hidden * 2, opts);
        errdefer rows.deinit();
        const codes = try self.gpa.alloc(i32, prompt_len);
        errdefer self.gpa.free(codes);
        @memset(codes, -1);
        var row: usize = 0;
        for (stream.images) |img| {
            const g = vision.tokenGrid(img.gh, img.gw);
            const pixels = try self.model.device.buffer(img.pixels.len * 2, opts);
            defer pixels.deinit();
            for (pixels.slice(u16, img.pixels.len), img.pixels) |*d, x| d.* = image.bf16(x);
            const out = try self.model.device.buffer(@as(usize, g[0]) * g[1] * c.hidden * 2, opts);
            defer out.deinit();
            try vision.run(self.model, v, pixels, img.gh, img.gw, out);
            @memcpy((rows.contents() + row * c.hidden * 2)[0 .. @as(usize, g[0]) * g[1] * c.hidden * 2], out.contents()[0 .. @as(usize, g[0]) * g[1] * c.hidden * 2]);
            var p: usize = img.at;
            codes[p] = -2;
            p += 1;
            for (0..g[0]) |y| {
                for (0..g[1]) |x| {
                    codes[p] = @intCast(row + y * g[1] + x);
                    p += 1;
                }
                codes[p] = -4;
                p += 1;
            }
            codes[p] = -3;
            row += @as(usize, g[0]) * g[1];
        }
        return .{ .codes = codes, .rows = rows };
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidDeepseekPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidDeepseekPosition;
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidDeepseekPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try fwd.run(self.model, &self.scratch, cache, &.{token}, .last, null);
        return try draw(self.gpa, cache.logits.slice(u16, c.vocab), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidDeepseekToken;
        return @intCast(handle);
    }

    /// The stream's held drafts as tokens (structured output masks them before a window takes them).
    fn heldFn(ptr: *anyopaque, stream: *lanes.Stream, out: []u32) !void {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (out.len > cache.held_n) return error.DeepseekWindowOutOfStep;
        @memcpy(out, cache.held[0..out.len]);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        if (windows.len != out.len or windows.len == 0 or windows.len > max_streams) return error.InvalidDeepseekWindows;
        const mask_words = (c.vocab + 31) / 32;
        var ids: [max_streams][window_rows]u32 = undefined;
        var segs: [max_streams]fwd.Seg = undefined;
        var r0: usize = 0;
        for (windows, out, ids[0..windows.len], segs[0..windows.len]) |w, o, *row_ids, *g| {
            const rows = w.rows();
            if (rows > window_rows or w.parents != null or w.early) return error.UnsupportedDeepseekWindow;
            const cache = try self.cacheOf(w.stream);
            if (w.held > cache.held_n) return error.DeepseekWindowOutOfStep;
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidDeepseekPosition;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidDeepseekMasks;
            if (o.sampled.len != rows or o.drafts.len != rows - 1) return error.InvalidDeepseekOutputs;
            row_ids[0] = w.pending;
            @memcpy(row_ids[1..][0..w.held], cache.held[0..w.held]);
            @memcpy(row_ids[1 + w.held ..][0..w.tokens.len], w.tokens);
            g.* = .{ .cache = cache, .tokens = row_ids[0..rows], .r0 = r0 };
            r0 += rows;
        }
        try fwd.many(self.model, &self.scratch, segs[0..windows.len], .all);
        for (windows, out, segs[0..windows.len]) |w, o, g| {
            const rows = g.tokens.len;
            const logits = self.scratch.logits.slice(u16, (g.r0 + rows) * c.vocab)[g.r0 * c.vocab ..];
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, logits[r * c.vocab ..][0..c.vocab], w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, g.tokens[1..rows]);
            g.cache.held_n = 0;
        }
    }

    /// A window's kept rows are a prefix of it: the cache goes back to them (everything it holds is by position).
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |r, k| if (r != k) return error.UnsupportedDeepseekTree;
            const cache = try self.cacheOf(w.stream);
            if (path.len == 0 or path.len > cache.last_rows) return error.DeepseekKeepOutOfStep;
            cache.len = cache.last_start + path.len;
            const row = self.scratch.logits.contents() + (cache.last_r0 + path.len - 1) * c.vocab * 2;
            @memcpy(cache.logits.contents()[0 .. c.vocab * 2], row[0 .. c.vocab * 2]);
        }
    }

    /// The head drafts from the stream's last kept row with its pending token; `depth` of the five are held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const self = cast(ptr);
        for (requests) |r| {
            if (r.early or r.lanes != null or r.ranks) return error.UnsupportedDeepseekDraft;
            const cache = try self.cacheOf(r.stream);
            cache.held_n = 0;
            if (r.rows) |kept| { // a shared round drafts before its keep: the cache goes back to the kept rows first
                if (r.start != cache.last_start or kept.len == 0 or kept.len > cache.last_rows) return error.DeepseekDraftOutOfStep;
                for (kept, 0..) |row, k| if (row != k) return error.UnsupportedDeepseekTree;
                cache.len = cache.last_start + kept.len;
            }
            if (r.depth == 0) continue;
            if (self.drafts == null) return error.DeepseekCheckpointHasNoDraftHead;
            const ds = &self.drafts.?;
            const token: u32 = if (r.first) |f| switch (f) {
                .value => |v| v,
                .handle => |h| @intCast(h),
            } else if (r.follow.len > 0) r.follow[r.follow.len - 1] else return error.DeepseekDraftOutOfStep;
            if (r.position != cache.len + 1) return error.DeepseekDraftOutOfStep;
            try fwd.draft(self.model, ds, cache, token, cache.len - 1, &cache.held);
            cache.held_n = @min(r.depth, c.draft_block);
        }
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        self.endFill(stream);
        if (self.caches.fetchRemove(stream)) |entry| {
            if (self.scratch.prepared) |p| if (p.cache == entry.value) { // a chunk gathered ahead for this stream
                self.scratch.prepared = null;
            };
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

/// The prompt cache's view of the streams: kept states copied at chunk ends, restored by the prompt pass.
pub const Snaps = struct {
    fn metal(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }
    pub fn bytes(ptr: *anyopaque, at: u32) u64 {
        return snapshot.bytes(at, metal(ptr).drafts != null);
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = metal(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        return @ptrCast(try snapshot.save(self.gpa, cache, at));
    }
    pub fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.RestoredInPrefill; // the prompt pass restores, before its first chunk
    }
    pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        snapshot.drop(metal(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};

fn now() f64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) * 1e3 + @as(f64, @floatFromInt(t.nsec)) / 1e6;
}
