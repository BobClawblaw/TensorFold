//! DeepSeek-V4.1-Flash behind the lane engine's backend interface: one stream, one row a round (no drafts yet), the
//! prompt in chunks, draws on the host from the row's bf16 logits.
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

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = try st.Scratch.init(gpa, model.device, @min(options.chunk, st.max_rows)) };
        return out;
    }

    pub fn deinit(self: *Metal) void {
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.destroy(self);
    }

    pub fn facts(_: *const Metal) lanes.Model {
        return .{ .exact_width = 1, .hidden_rows = false, .batch_rows = 1, .max_streams = 1, .streams_exact = true, .plain_guard = true };
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
        cache.* = try st.Cache.init(self.gpa, self.model.device, @min(self.options.capacity, ids.len + stream.max_new + 1));
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
        for (windows, out) |w, o| {
            if (w.rows() != 1 or w.parents != null or w.held != 0) return error.UnsupportedDeepseekWindow;
            const cache = try self.cacheOf(w.stream);
            if (w.positions[0] != cache.len + 1) return error.InvalidDeepseekPosition;
            try fwd.run(self.model, &self.scratch, cache, &.{w.pending}, .last, null);
            const mask_words = (c.vocab + 31) / 32;
            const mask = if (w.masks.len > 0) w.masks[0..mask_words] else &.{};
            o.sampled[0] = try draw(self.gpa, cache.logits.slice(u16, c.vocab), w.stream.sampling, w.positions[0], mask);
        }
    }

    fn keepFn(_: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        for (windows, paths) |w, path| if (w.rows() != 1 or path.len != 1) return error.UnsupportedDeepseekWindow;
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.DeepseekDraftsNotServedYet;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            entry.value.deinit();
            self.gpa.destroy(entry.value);
        }
    }
};
