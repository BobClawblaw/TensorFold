//! POST /v1/images/generations on the wire, against a scripted image engine over a socketpair.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const routes = @import("routes.zig");
const harness = @import("speech_stream_test.zig");
const Conn = @import("http_conn.zig").Conn;

const gpa = std.testing.allocator;

/// Appends `png` and finishes, inside submit unless `hold`, when it waits for a cancel on its own thread.
const Images = struct {
    png: []const u8 = "picture bytes",
    submit_error: ?api.ImageSubmitError = null,
    reason: api.Reason = .stop,
    hold: bool = false,
    images: bool = true,
    seen: ?api.ImageRequest = null,
    /// The request's text, copied: the route's own storage is gone once it returns.
    prompt: [64]u8 = undefined,
    negative: [64]u8 = undefined,
    cancelled: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn engine(e: *Images) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .submit_image = image, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const e: *Images = @ptrCast(@alignCast(ctx));
        return .{ .text_generation = false, .images = if (e.images) .{ .max_pixels = 1200 * 880, .max_side = 1344, .max_steps = 200, .max_prompt_bytes = 64, .queue = 2 } else null };
    }
    fn image(ctx: *anyopaque, id: api.Id, request: *const api.ImageRequest, sink: api.Sink) api.ImageSubmitError!void {
        const e: *Images = @ptrCast(@alignCast(ctx));
        if (e.submit_error) |err| return err;
        e.seen = request.*;
        @memcpy(e.prompt[0..request.prompt.len], request.prompt);
        @memcpy(e.negative[0..request.negative.len], request.negative);
        e.seen.?.prompt = e.prompt[0..request.prompt.len];
        e.seen.?.negative = e.negative[0..request.negative.len];
        if (e.hold) {
            e.thread = std.Thread.spawn(.{}, wait, .{ e, id, sink }) catch return error.Busy;
            return;
        }
        if (e.reason == .stop) request.png.appendSlice(request.gpa, e.png) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = e.reason } });
    }
    fn wait(e: *Images, id: api.Id, sink: api.Sink) void {
        while (!e.cancelled.load(.acquire)) std.Thread.yield() catch {};
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const e: *Images = @ptrCast(@alignCast(ctx));
        e.cancelled.store(true, .release);
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// One request through the route on its own thread; the reply read from the client end.
const Served = struct {
    srv: *server.Server,
    text: harness.Bytes = .{},
    client: std.c.fd_t,
    route_fd: std.c.fd_t,
    route: std.Thread,
    got: std.ArrayList(u8) = .empty,

    fn start(s: *Served, e: *Images, body: []const u8) !void {
        var pair: [2]std.c.fd_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        s.* = .{ .srv = undefined, .client = pair[0], .route_fd = pair[1], .route = undefined };
        s.srv = try server.Server.init(gpa, std.testing.io, e.engine(), s.text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        var head: [160]u8 = undefined;
        try write(s.client, try std.fmt.bufPrint(&head, "POST /v1/images/generations HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{body.len}));
        try write(s.client, body);
        s.route = try std.Thread.spawn(.{}, serve, .{s});
    }
    fn serve(s: *Served) void {
        var conn = Conn.init(gpa, s.route_fd, "test") catch return;
        defer conn.deinit();
        conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const outcome = conn.readRequest(arena.allocator(), false) catch return;
        if (outcome == .ready) routes.dispatch(s.srv, &conn, arena.allocator());
        _ = std.c.shutdown(s.route_fd, 1);
    }
    /// The whole reply, once the route has finished it.
    fn reply(s: *Served) ![]const u8 {
        s.route.join();
        var tmp: [4096]u8 = undefined;
        while (true) {
            const n = std.c.read(s.client, &tmp, tmp.len);
            if (n <= 0) break;
            try s.got.appendSlice(gpa, tmp[0..@intCast(n)]);
        }
        return s.got.items;
    }
    fn deinit(s: *Served) void {
        _ = std.c.close(s.client);
        _ = std.c.close(s.route_fd);
        s.got.deinit(gpa);
        s.srv.deinit();
    }
};

fn write(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

fn has(text: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, text, part) != null;
}

test "a picture comes back as b64_json, and every option reaches the engine" {
    var e: Images = .{};
    var s: Served = undefined;
    try s.start(&e, "{\"model\":\"m\",\"prompt\":\"a fox\",\"negative_prompt\":\"blur\",\"size\":\"1344x768\",\"steps\":30,\"seed\":5,\"guidance\":1.5,\"n\":1,\"response_format\":\"b64_json\"}");
    defer s.deinit();
    const got = try s.reply();
    try std.testing.expect(has(got, "200 OK"));
    try std.testing.expect(has(got, "\"b64_json\": \"cGljdHVyZSBieXRlcw==\""));
    const r = e.seen.?;
    try std.testing.expectEqualStrings("a fox", r.prompt);
    try std.testing.expectEqualStrings("blur", r.negative);
    try std.testing.expectEqual(@as(u32, 1344), r.width);
    try std.testing.expectEqual(@as(u32, 768), r.height);
    try std.testing.expectEqual(@as(u32, 30), r.steps);
    try std.testing.expectEqual(@as(u64, 5), r.seed);
    try std.testing.expectEqual(@as(f32, 1.5), r.guidance);
}

test "the pipeline's defaults fill what a request leaves out" {
    var e: Images = .{};
    var s: Served = undefined;
    try s.start(&e, "{\"model\":\"m\",\"prompt\":\"a fox\"}");
    defer s.deinit();
    try std.testing.expect(has(try s.reply(), "200 OK"));
    const r = e.seen.?;
    try std.testing.expectEqualStrings("", r.negative);
    try std.testing.expectEqual(@as(u32, 1024), r.width);
    try std.testing.expectEqual(@as(u32, 1024), r.height);
    try std.testing.expectEqual(@as(u32, 50), r.steps);
    try std.testing.expectEqual(@as(u64, 42), r.seed);
    try std.testing.expectEqual(@as(f32, 3), r.guidance);
}

test "what this model cannot do is refused before the engine sees it" {
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "{\"model\":\"m\",\"prompt\":\"x\",\"n\":2}", "n must be 1" },
        .{ "{\"model\":\"m\",\"prompt\":\"x\",\"quality\":\"hd\"}", "quality is not supported" },
        .{ "{\"model\":\"m\",\"prompt\":\"x\",\"size\":\"1000x1024\"}", "multiples of 16" },
        .{ "{\"model\":\"m\",\"prompt\":\"x\",\"size\":\"2048x2048\"}", "at most 1344 pixels a side" },
        .{ "{\"model\":\"m\",\"prompt\":\"x\",\"steps\":0}", "steps must be 1 through 200" },
        .{ "{\"model\":\"m\",\"prompt\":\"x\",\"response_format\":\"url\"}", "response_format must be b64_json" },
        .{ "{\"model\":\"m\",\"prompt\":\"x\",\"guidance\":-1}", "guidance must be a finite number" },
        .{ "{\"model\":\"m\",\"prompt\":\"\"}", "prompt must be a non-empty string" },
        .{ "{\"model\":\"m\",\"prompt\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}", "prompt must be at most 64 bytes" },
        .{ "{\"model\":\"other\",\"prompt\":\"x\"}", "is not served here" },
    };
    for (cases) |c| {
        var e: Images = .{};
        var s: Served = undefined;
        try s.start(&e, c[0]);
        defer s.deinit();
        const got = try s.reply();
        try std.testing.expect(has(got, "400 Bad Request"));
        try std.testing.expect(has(got, c[1]));
        try std.testing.expect(e.seen == null);
    }
}

test "a full queue is 503, a failed picture 500" {
    var busy: Images = .{ .submit_error = error.Busy };
    var s: Served = undefined;
    try s.start(&busy, "{\"model\":\"m\",\"prompt\":\"x\"}");
    try std.testing.expect(has(try s.reply(), "503 Service Unavailable"));
    s.deinit();
    var failed: Images = .{ .reason = .failed };
    try s.start(&failed, "{\"model\":\"m\",\"prompt\":\"x\"}");
    try std.testing.expect(has(try s.reply(), "500 Internal Server Error"));
    s.deinit();
}

test "a client that leaves cancels the picture, which is waited out" {
    var e: Images = .{ .hold = true };
    var s: Served = undefined;
    try s.start(&e, "{\"model\":\"m\",\"prompt\":\"x\"}");
    defer s.deinit();
    while (e.seen == null) std.Thread.yield() catch {};
    _ = std.c.shutdown(s.client, 1);
    _ = std.c.close(s.client);
    s.client = -1;
    s.route.join(); // returns only after the finished event: the request outlived the picture
    if (e.thread) |t| t.join();
    try std.testing.expect(e.cancelled.load(.acquire));
}

test "a model that makes no pictures has no image route" {
    var e: Images = .{ .images = false };
    var s: Served = undefined;
    try s.start(&e, "{\"model\":\"m\",\"prompt\":\"x\"}");
    defer s.deinit();
    const got = try s.reply();
    try std.testing.expect(has(got, "404 Not Found"));
    try std.testing.expect(e.seen == null);
}
