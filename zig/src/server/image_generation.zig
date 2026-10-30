//! POST /v1/images/generations: one picture from a prompt, as OpenAI's images API returns it (b64_json, a PNG).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const openai = @import("openai.zig");
const pyrepr = @import("pyrepr.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const path = "/v1/images/generations";

/// The pipeline's defaults: 1024 x 1024, 50 steps, seed 42, guidance 3, the empty negative prompt.
const defaults = .{ .width = 1024, .height = 1024, .steps = 50, .seed = 42, .guidance = 3.0 };

/// OpenAI's fields this route honors, then the pipeline's own; any other field is refused.
const fields = [_][]const u8{ "model", "prompt", "n", "size", "response_format", "negative_prompt", "seed", "steps", "guidance" };

pub fn post(srv: *Server, c: *Conn, a: Allocator) !void {
    const im = srv.info.images orelse {
        // dropped, so it cannot reach the next request on this connection
        _ = try http_body.read(c, a, http_body.limit);
        return routes.unknown(c, a);
    };
    var cx: Cx = .{ .a = a };
    const id = try ids.make(a, "img-", 32);
    const raw = http_body.readJson(c, &cx) catch |e| return refuse(c, &cx, id, e);
    const request = prepare(srv, &cx, im, raw) catch |e| return refuse(c, &cx, id, e);
    generate(srv, c, &cx, id, request);
}

const Plan = struct { prompt: []const u8, negative: []const u8, width: u32, height: u32, steps: u32, seed: u64, guidance: f32 };

/// Every check that can refuse the request, before anything reaches the engine.
fn prepare(srv: *Server, cx: *Cx, im: api.ImageInfo, raw: Value) errors.Refused!Plan {
    if (raw != .object) return cx.refuse("image request body must be a JSON object");
    for (raw.object.keys()) |key| {
        var known = false;
        for (fields) |f| known = known or std.mem.eql(u8, f, key);
        if (!known) return cx.fail(.request, "{s} is not supported by this model", .{key});
    }
    const model = raw.strField("model") orelse return cx.fail(.request, "model is required: send {s}", .{srv.config.served_name});
    var served = false;
    for (srv.config.model_ids) |m| served = served or std.mem.eql(u8, m, model);
    if (!served) return cx.fail(.request, "model {s} is not served here: send {s}", .{ try pyrepr.repr(cx.a, .{ .string = model }), srv.config.served_name });
    const prompt = raw.strField("prompt") orelse return cx.refuse("prompt must be a non-empty string");
    if (prompt.len == 0 or !std.unicode.utf8ValidateSlice(prompt)) return cx.refuse("prompt must be a non-empty string of UTF-8 text");
    if (prompt.len > im.max_prompt_bytes) return cx.fail(.request, "prompt must be at most {d} bytes", .{im.max_prompt_bytes});
    const negative = if (raw.field("negative_prompt")) |v| (if (v == .string) v.string else return cx.refuse("negative_prompt must be a string")) else "";
    if (!std.unicode.utf8ValidateSlice(negative) or negative.len > im.max_prompt_bytes) return cx.fail(.request, "negative_prompt must be UTF-8 text of at most {d} bytes", .{im.max_prompt_bytes});
    if (raw.field("n")) |n| if ((try integer(cx, n, "n")) != 1) return cx.refuse("n must be 1: this model makes one picture a request");
    if (raw.field("response_format")) |f| if (f != .string or !std.mem.eql(u8, f.string, "b64_json")) return cx.refuse("response_format must be b64_json");
    var p: Plan = .{ .prompt = prompt, .negative = negative, .width = defaults.width, .height = defaults.height, .steps = defaults.steps, .seed = defaults.seed, .guidance = defaults.guidance };
    if (raw.field("size")) |s| {
        if (s != .string) return cx.refuse("size must be a string such as \"1024x1024\"");
        const x = std.mem.indexOfScalar(u8, s.string, 'x') orelse return cx.refuse("size must be WIDTHxHEIGHT, such as \"1024x1024\"");
        p.width = std.fmt.parseInt(u32, s.string[0..x], 10) catch return cx.refuse("size must be WIDTHxHEIGHT, such as \"1024x1024\"");
        p.height = std.fmt.parseInt(u32, s.string[x + 1 ..], 10) catch return cx.refuse("size must be WIDTHxHEIGHT, such as \"1024x1024\"");
    }
    if (p.width == 0 or p.height == 0 or p.width % 16 != 0 or p.height % 16 != 0)
        return cx.refuse("size's width and height must be positive multiples of 16");
    if (p.width > im.max_side or p.height > im.max_side or @as(u64, p.width) * p.height > im.max_pixels)
        return cx.fail(.request, "size must be at most {d} pixels a side and {d} pixels in all", .{ im.max_side, im.max_pixels });
    if (raw.field("steps")) |v| {
        const n = try integer(cx, v, "steps");
        if (n < 1 or n > im.max_steps) return cx.fail(.request, "steps must be 1 through {d}", .{im.max_steps});
        p.steps = @intCast(n);
    }
    if (raw.field("seed")) |v| {
        const n = try integer(cx, v, "seed");
        if (n < 0) return cx.refuse("seed must be 0 or more");
        p.seed = @intCast(n);
    }
    if (raw.field("guidance")) |v| {
        const g: f64 = switch (v) {
            .float => |f| f,
            .int => @floatFromInt(try integer(cx, v, "guidance")),
            else => return cx.refuse("guidance must be a number"),
        };
        if (!std.math.isFinite(g) or g < 0 or g > 1e6) return cx.refuse("guidance must be a finite number, 0 or more");
        p.guidance = @floatCast(g);
    }
    return p;
}

fn integer(cx: *Cx, v: Value, name: []const u8) errors.Refused!i64 {
    if (v != .int) return cx.fail(.request, "{s} must be an integer", .{name});
    return std.fmt.parseInt(i64, v.int, 10) catch cx.fail(.request, "{s} is out of range", .{name});
}

/// One picture: queued on the engine, then sent whole once it is made; a client that leaves cancels it.
fn generate(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, p: Plan) void {
    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(srv.gpa);
    const request: api.ImageRequest = .{ .prompt = p.prompt, .negative = p.negative, .width = p.width, .height = p.height, .steps = p.steps, .seed = p.seed, .guidance = p.guidance, .png = &png, .gpa = srv.gpa };
    var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    const engine_id = srv.next_id.fetchAdd(1, .monotonic);
    const started = std.Io.Clock.awake.now(srv.io).toNanoseconds();
    srv.engine.submitImage(engine_id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return fail(c, cx, id, switch (e) {
        error.Busy => .capacity,
        error.InvalidRequest, error.Unsupported => .request,
        error.Closed => .other,
    }, switch (e) {
        error.Busy => "the image queue is full; retry shortly",
        error.Closed => "the scheduler is closed",
        error.InvalidRequest => "the engine refused this image request",
        error.Unsupported => "this model does not make pictures",
    });
    // the engine holds the request and its png until finished; nothing below returns before that
    while (true) {
        box.mutex.lockUncancelable(box.io);
        if (!box.finished) box.cond.waitTimeout(box.io, &box.mutex, .{ .duration = .{ .raw = .fromMilliseconds(200), .clock = .awake } }) catch {};
        const finished = box.finished;
        box.mutex.unlock(box.io);
        if (finished) break;
        if (c.peerGone()) {
            srv.engine.cancel(engine_id);
            box.mutex.lockUncancelable(box.io);
            while (!box.finished) box.cond.waitUncancelable(box.io, &box.mutex);
            box.mutex.unlock(box.io);
            log.line("{s}: the client left; the picture was cancelled", .{id});
            return;
        }
    }
    if (box.reason != .stop) return fail(c, cx, id, .server, if (box.message.len > 0) box.message else "the picture was not made");
    const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(srv.io).toNanoseconds() - started)) / std.time.ns_per_s;
    reply(srv, c, cx, id, png.items, seconds) catch {
        c.close = true;
    };
}

fn reply(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, png: []const u8, seconds: f64) !void {
    const encoder = std.base64.standard.Encoder;
    const b64 = try cx.a.alloc(u8, encoder.calcSize(png.len));
    _ = encoder.encode(b64, png);
    const item = try json.newObject(cx.a);
    try item.put(cx.a, "b64_json", .{ .string = b64 });
    const data = try cx.a.alloc(Value, 1);
    data[0] = .{ .object = item };
    const body = try json.newObject(cx.a);
    try body.put(cx.a, "created", try json.intValue(cx.a, std.Io.Clock.real.now(srv.io).toSeconds()));
    try body.put(cx.a, "data", .{ .array = data });
    const text = try json.stringify(cx.a, .{ .object = body }, .{});
    log.line("{s}: a {d}-byte picture in {d:.1} s", .{ id, png.len, seconds });
    c.sendJson(200, text);
}

fn refuse(c: *Conn, cx: *Cx, id: []const u8, e: errors.Refused) void {
    if (e == error.OutOfMemory) {
        cx.kind = .server;
        cx.message = "out of memory";
    }
    send(c, cx, id);
}

fn fail(c: *Conn, cx: *Cx, id: []const u8, kind: errors.Kind, message: []const u8) void {
    cx.kind = kind;
    cx.message = message;
    send(c, cx, id);
}

/// OpenAI's error object, logged first: the access line shows only the status.
fn send(c: *Conn, cx: *Cx, id: []const u8) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, cx.message)});
    const body = openai.errorBody(cx.a, cx, null) catch return;
    const text = json.stringify(cx.a, openai.wrapError(cx.a, body) catch return, .{}) catch return;
    c.sendJson(cx.status(), text);
}
