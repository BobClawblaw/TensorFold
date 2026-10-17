//! DeepSeek-V4.1-Flash behind the lane engine's backend interface: one stream, the prompt in chunks, windows of up
//! to six rows verifying the DSpark head's drafts (drafted replies equal one-row ones: tf-dsv41-forward DS_DRAFT),
//! draws on the host from each row's bf16 logits.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const draw = @import("../lfm2_moe/backend.zig").draw;
const be = lanes.backend;

pub const Options = struct { capacity: usize, chunk: usize = 256 };

/// The widest window: the pending row and the head's five drafts.
pub const window_rows = 1 + c.draft_block;

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
    draft_ms: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, model.device, @min(options.chunk, st.max_rows)), .drafts = null };
        errdefer out.scratch.deinit();
        if (model.weights.stages != null) {
            out.drafts = try st.Scratch.init(gpa, model.device, 8);
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
        if (self.drafts) |*d| d.deinit();
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Metal) lanes.Model {
        if (self.drafts == null) return .{ .exact_width = 1, .batch_rows = 1, .max_streams = 1, .streams_exact = true, .plain_guard = true };
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
            .batch_rows = window_rows,
            .max_streams = 1,
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
        std.log.info("deepseek_v41: window ms by width 1 {d:.1}, 2 {d:.1}, 4 {d:.1}, 6 {d:.1}; a draft round {d:.1} ms", .{ self.costs[0].ms, self.costs[1].ms, self.costs[3].ms, self.costs[5].ms, self.draft_ms });
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownDeepseekStream;
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidDeepseekPrompt;
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.initWith(self.gpa, self.model.device, @min(self.options.capacity, ids.len + stream.max_new + window_rows), self.drafts != null);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        stream.cached = 0;
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + self.scratch.rows);
            try fwd.run(self.model, &self.scratch, cache, ids[at..end], if (end == ids.len) .last else .none, null);
            at = end;
        }
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

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        if (windows.len != out.len) return error.InvalidDeepseekWindows;
        const mask_words = (c.vocab + 31) / 32;
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (rows > window_rows or w.parents != null or w.early) return error.UnsupportedDeepseekWindow;
            const cache = try self.cacheOf(w.stream);
            if (w.held > cache.held_n) return error.DeepseekWindowOutOfStep;
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidDeepseekPosition;
            if (w.masks.len != 0 and w.masks.len != rows * mask_words) return error.InvalidDeepseekMasks;
            var ids: [window_rows]u32 = undefined;
            ids[0] = w.pending;
            @memcpy(ids[1..][0..w.held], cache.held[0..w.held]);
            @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
            try fwd.run(self.model, &self.scratch, cache, ids[0..rows], .all, null);
            const logits = self.scratch.logits.slice(u16, rows * c.vocab);
            if (o.sampled.len != rows or o.drafts.len != rows - 1) return error.InvalidDeepseekOutputs;
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, logits[r * c.vocab ..][0..c.vocab], w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, ids[1..rows]);
            cache.held_n = 0;
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
            const row = self.scratch.logits.contents() + (path.len - 1) * c.vocab * 2;
            @memcpy(cache.logits.contents()[0 .. c.vocab * 2], row[0 .. c.vocab * 2]);
        }
    }

    /// The head drafts from the stream's last committed row with its pending token; `depth` of the five are held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const self = cast(ptr);
        for (requests) |r| {
            if (r.early or r.lanes != null or r.ranks) return error.UnsupportedDeepseekDraft;
            const cache = try self.cacheOf(r.stream);
            cache.held_n = 0;
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
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};

fn now() f64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) * 1e3 + @as(f64, @floatFromInt(t.nsec)) / 1e6;
}
