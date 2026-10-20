//! DeepSeek-V4.1-Flash served by the lane host and scheduler.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.deepseek_v41;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");

const Host = struct {
    gpa: std.mem.Allocator,
    model: *q.Model,
    metal: *q.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the model's queue, for the lane host's idle ticker
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: a zero budget
    io: std.Io = undefined,
    markers: [1]u32 = .{image_token}, // the template writes the image token alone (encoding.py's placeholder)

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

const cannot = "the native DeepSeek-V4.1 engine cannot load this checkpoint ({s})";

/// <｜deepseek_image｜>: the config's image_token_id, each image's place in the rendered prompt.
const image_token: u32 = 129264;

/// Info.vision's prepare: an image's bytes to the tower's patches (deepseek_v41/image.zig), HTTP thread.
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try q.image.prepare(h.io, a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const config = q.config.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    const window: usize = if (o.context) |w| @intCast(w) else @min(config.context, 65536);
    if (window < 1 or window > config.context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside DeepSeek-V4.1-Flash's 1 to {d} tokens", .{ window, config.context });
        return null;
    }
    const model = q.Model.load(gpa, io, o.dir, true) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    const chunk = 1024; // a prompt pass's rows a call beside other streams (the lane host's step and the prompt cache's save points)
    const lone_chunk = q.state.max_rows; // a lone stream's rows a call (the scratch): more rows share each expert's weights
    h.metal = try q.backend.Metal.init(gpa, model, .{ .capacity = window, .chunk = lone_chunk, .shared_chunk = chunk });
    errdefer h.metal.deinit();
    const rows: u32 = if (h.metal.drafts != null) q.backend.window_rows else 1;
    h.config = try lanes.Config.init(gpa, h.metal.facts(), rows, rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.io = io;
    h.markers = .{image_token};
    var offer: ?api.Vision = null;
    if (o.vision) |v| {
        if (model.vision == null) {
            problem.* = try std.fmt.allocPrint(a, "--vision: this DeepSeek-V4.1 checkpoint has no image tower", .{});
            return null;
        }
        offer = .{ .ctx = h, .prepare = prepareImage, .image_token = image_token, .markers = &h.markers, .image_tokens = v.image_tokens, .max_images = v.max_images, .explain = q.image.decoderProblem };
    }
    // up to four streams: their windows share a forward, and a new prompt goes in a chunk between their rounds
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = @max(1, @min(o.lanes, 4)), .context_window = @intCast(window), .prefill_step = chunk, .decode_share = 0.5, .structures = true, .answer_limit = true, .vision = offer });
    h.cache = null;
    var why: []const u8 = "";
    const dev = model.device;
    const plan = cache_fit.budget(o.prompt_cache_gib, o.prompt_cache_over_cap, dev.maxWorkingSet() -| dev.allocated() -| (8 << 30), a, &why, "") catch |e| {
        if (e == error.CacheOverCap) problem.* = why;
        return e;
    };
    h.host.info_.prompt_cache_plan = plan;
    if (plan.budget_bytes > 0) {
        const S = q.backend.Snaps;
        h.cache = api.prompt_cache.Store.init(gpa, .{
            .ptr = h.metal,
            .vtable = &.{ .bytes = S.bytes, .save = S.save, .restore = S.restore, .drop = S.drop },
        }, .{ .chunk = chunk }, plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.warm = .{ .queue = h.metal.model.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
