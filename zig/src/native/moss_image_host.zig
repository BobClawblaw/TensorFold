//! MOSS-Image served: the encoder, transformer and decoder resident, one picture at a time in arrival order.
const std = @import("std");
const api = @import("engine_api");
const mtl = @import("metal");
const tf = @import("tensorfold");
const tokenizer = @import("tokenizer");
const mi = tf.moss_image;

pub const Options = struct {
    /// The largest picture served: 1536 x 1024's pixels, the largest checked.
    max_pixels: u64 = 1536 * 1024,
    /// The longest side served: 1536, as 1536 x 1024 and 1024 x 1536 were checked.
    max_side: u32 = 1536,
    max_steps: u32 = 200,
    max_prompt_bytes: u32 = 16384,
    /// Requests that may wait behind the one running.
    queue: u32 = 8,
};

const Job = struct { id: api.Id, request: *const api.ImageRequest, sink: api.Sink };

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    encoder: *mi.Encoder,
    tok: tokenizer.Tokenizer,
    model: *mi.Model,
    decoder: *mi.Decoder,
    schedule: mi.schedule.Config,
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    waiting: std.ArrayList(Job) = .empty,
    running: ?api.Id = null,
    /// The running request was cancelled: it stops before its next step.
    stop: std.atomic.Value(bool) = .init(false),
    worker: ?std.Thread = null,
    closing: bool = false,

    const vtable: api.Engine.VTable = .{
        .info = info,
        .submit = submitText,
        .submit_image = submit,
        .cancel = cancel,
        .status = status,
        .memory = memory,
        .drain = drain,
    };

    fn self(p: *anyopaque) *Host {
        return @ptrCast(@alignCast(p));
    }
    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }
    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }
    fn info(p: *anyopaque) api.Info {
        return self(p).info_;
    }
    fn submitText(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }

    fn valid(h: *const Host, r: *const api.ImageRequest) bool {
        const im = h.info_.images.?;
        _ = mi.config.grid(r.width, r.height) catch return false;
        if (r.width > im.max_side or r.height > im.max_side or @as(u64, r.width) * r.height > im.max_pixels) return false;
        if (r.steps == 0 or r.steps > im.max_steps) return false;
        if (!std.math.isFinite(r.guidance) or r.guidance < 0) return false;
        if (r.prompt.len == 0 or r.prompt.len > im.max_prompt_bytes or r.negative.len > im.max_prompt_bytes) return false;
        if (!std.unicode.utf8ValidateSlice(r.prompt) or !std.unicode.utf8ValidateSlice(r.negative)) return false;
        return true;
    }

    fn submit(p: *anyopaque, id: api.Id, r: *const api.ImageRequest, sink: api.Sink) api.ImageSubmitError!void {
        const h = self(p);
        if (!h.valid(r)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing) return error.Closed;
        if (h.running == id) return error.InvalidRequest;
        for (h.waiting.items) |j| if (j.id == id) return error.InvalidRequest;
        if (h.waiting.items.len >= h.info_.images.?.queue) return error.Busy;
        h.waiting.append(h.gpa, .{ .id = id, .request = r, .sink = sink }) catch return error.Busy;
        h.wake.signal(h.io);
    }

    /// A waiting request leaves the queue at once; the running one stops before its next step.
    fn cancel(p: *anyopaque, id: api.Id) void {
        const h = self(p);
        h.lock();
        var dropped: ?Job = null;
        for (h.waiting.items, 0..) |j, i| if (j.id == id) {
            dropped = h.waiting.orderedRemove(i);
            break;
        };
        if (h.running == id) h.stop.store(true, .release);
        h.unlock();
        if (dropped) |j| finish(j, .cancelled, "");
    }

    fn status(p: *anyopaque, out: *api.Status, _: []u32) void {
        const h = self(p);
        h.lock();
        defer h.unlock();
        out.* = .{ .running = @intFromBool(h.running != null), .waiting = @intCast(h.waiting.items.len) };
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn finish(j: Job, reason: api.Reason, message: []const u8) void {
        const e: api.Event = .{ .finished = .{ .reason = reason, .message = message } };
        j.sink.event(j.sink.ctx, j.id, &e);
    }

    fn run(h: *Host) void {
        while (true) {
            h.lock();
            while (h.waiting.items.len == 0 and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
            if (h.waiting.items.len == 0) {
                h.unlock();
                return;
            }
            const j = h.waiting.orderedRemove(0);
            h.running = j.id;
            h.stop.store(false, .release);
            h.unlock();
            var arena = std.heap.ArenaAllocator.init(h.gpa);
            const done = h.picture(arena.allocator(), j.request);
            arena.deinit();
            h.lock();
            h.running = null;
            h.unlock();
            if (done) |finished| {
                finish(j, if (finished) .stop else .cancelled, "");
            } else |e| {
                std.log.err("moss image: a picture failed ({s})", .{@errorName(e)});
                finish(j, .failed, "the picture failed on the engine");
            }
        }
    }

    /// One request's picture as a PNG, as tf-moss-image-generate makes it; false when cancelled between steps.
    fn picture(h: *Host, a: std.mem.Allocator, r: *const api.ImageRequest) !bool {
        const guided = r.guidance != 1;
        var rows: [2][]u16 = undefined;
        const texts = [2][]const u8{ r.prompt, r.negative };
        for (texts[0..if (guided) 2 else 1], rows[0..if (guided) 2 else 1]) |t, *m| {
            const ids = try mi.encoder.ids(&h.tok, a, t);
            m.* = try a.alloc(u16, ids.len * mi.encoder_config.hidden);
            try h.encoder.rows(ids, m.*, null);
        }
        const plan = try mi.schedule.make(a, h.schedule, r.steps);
        const g = try mi.config.grid(r.width, r.height);
        const x = try mi.noise.latents(a, r.seed, g.rows, g.columns, mi.config.channels);
        try h.model.resize(r.width, r.height);
        var prompt = try h.model.prompt(rows[0]);
        defer prompt.deinit();
        var negative: ?mi.Prompt = if (guided) try h.model.prompt(rows[1]) else null;
        defer if (negative) |*n| n.deinit();
        const cond = try a.alloc(f32, x.len);
        const uncond = try a.alloc(f32, x.len);
        for (plan.timesteps, 0..) |t, s| {
            if (h.stop.load(.acquire)) return false;
            _ = try mi.pipeline.step(h.model, &prompt, if (negative) |*n| n else null, x, t, plan.sigmas[s], plan.sigmas[s + 1], r.guidance, cond, uncond);
        }
        try h.decoder.resize(r.width, r.height);
        const rgb = try a.alloc(u8, @as(usize, r.width) * r.height * 3);
        _ = try h.decoder.decode(x, rgb);
        try r.png.appendSlice(r.gpa, try mi.png.encode(a, rgb, r.width, r.height));
        return true;
    }

    fn drain(p: *anyopaque) void {
        const h = self(p);
        h.lock();
        h.closing = true;
        h.stop.store(true, .release);
        h.wake.signal(h.io);
        h.unlock();
        if (h.worker) |t| t.join();
        h.worker = null;
    }

    fn close(p: *anyopaque) void {
        const h = self(p);
        drain(h);
        h.waiting.deinit(h.gpa);
        h.decoder.deinit();
        h.model.deinit();
        h.tok.deinit();
        h.encoder.deinit();
        h.gpa.destroy(h);
    }
};

/// The checkpoint folder `dir` (its model_index.json names MOSSImagePipeline) loaded for pictures up to the options'.
pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !api.Opened {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const device = try mtl.Device.init();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "scheduler", "scheduler_config.json" }), a, .limited(1 << 20));
    const schedule = try mi.schedule.parse(a, text);
    const encoder = try mi.Encoder.load(gpa, io, device, dir);
    errdefer encoder.deinit();
    var tok = try tokenizer.loadTokenizer(io, gpa, try std.fs.path.join(a, &.{ dir, "tokenizer", "tokenizer.json" }));
    errdefer tok.deinit();
    // scratch for the card's 1024 x 1024; a larger picture grows it (Model.resize, Decoder.resize)
    const model = try mi.Model.load(gpa, io, device, try std.fs.path.join(a, &.{ dir, "transformer" }), mi.config.max_text, 1024, 1024);
    errdefer model.deinit();
    const decoder = try mi.Decoder.load(gpa, io, device, try std.fs.path.join(a, &.{ dir, "vae" }), 1024, 1024);
    errdefer decoder.deinit();
    h.* = .{
        .gpa = gpa,
        .io = io,
        .device = device,
        .encoder = encoder,
        .tok = tok,
        .model = model,
        .decoder = decoder,
        .schedule = schedule,
        .info_ = .{
            .name = "moss_image",
            .text_generation = false,
            .lanes = 1,
            .images = .{ .max_pixels = o.max_pixels, .max_side = o.max_side, .max_steps = o.max_steps, .max_prompt_bytes = o.max_prompt_bytes, .queue = o.queue },
        },
    };
    h.worker = try std.Thread.spawn(.{}, Host.run, .{h});
    std.log.info("moss image: serving pictures up to {d} pixels ({d} on a side), {d} steps, {d} waiting", .{ o.max_pixels, o.max_side, o.max_steps, o.queue });
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
}

/// The image engine for `o`, or null with `problem` set: every text-engine option is refused, as none applies.
pub fn openServe(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.context != null or o.max_frames != null or o.cfg_enabled or o.cfg_batched or o.lanes_fixed or o.tp != 1 or
        o.rank != 0 or o.vision != null or o.precision != null or o.record != null or o.recall or o.learn != null or
        o.learn_conversations or o.speed_up != null or o.slide or o.drafter != null or o.mtp_drafts != null or
        o.mtp_confidence != null or o.policy.len != 0 or (o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0) or
        (o.keep != null and o.keep.? != 0) or (o.device != null and o.device.? != 0) or
        (o.segments != null and o.segments.? != 1))
    {
        problem.* = "MOSS-Image serves pictures alone: it takes no context, lanes, drafts, caches, vision, learning or TP options";
        return null;
    }
    return open(gpa, io, o.dir, .{}) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the MOSS-Image checkpoint could not be loaded ({s})", .{@errorName(e)});
        return null;
    };
}
