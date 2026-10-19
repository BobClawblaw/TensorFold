//! A prompt going in a chunk at a time while another stream decodes: `Info.decode_share` paces its chunks.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

/// The live stream's rounds between the long prompt's chunks, each chunk taking 10 ms, under `share`.
fn roundsBetweenChunks(share: f32) ![3]u64 {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .decode_share = share });
    try host.start();
    defer host.stop();
    const Seen = struct {
        target: *lanes.fake.Fake,
        at: [4]u64 = .{ 0, 0, 0, 0 }, // the rounds run when each of the long prompt's chunks began
        tokens: std.atomic.Value(u32) = .init(0),
        done: std.atomic.Value(u32) = .init(0),

        fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len < 8) return;
            seen.at[chunk] = seen.target.rounds;
            std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
        }
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .tokens => if (id == 1) {
                    _ = seen.tokens.fetchAdd(1, .release);
                },
                .finished => _ = seen.done.fetchAdd(1, .release),
                else => {},
            }
        }
    };
    var seen: Seen = .{ .target = &target };
    target.prefill_hook = Seen.hook;
    target.prefill_hook_ctx = &seen;
    const e = host.engine();
    const short = [_]u32{ 2, 7, 1, 8 };
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const decoding: Request = .{ .prompt = &short, .max_tokens = 1 << 30 };
    const filling: Request = .{ .prompt = &long, .max_tokens = 4 };
    try e.submit(1, &decoding, .{ .ctx = &seen, .event = Seen.event });
    while (seen.tokens.load(.acquire) == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try e.submit(2, &filling, .{ .ctx = &seen, .event = Seen.event });
    while (seen.done.load(.acquire) == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    e.cancel(1);
    while (seen.done.load(.acquire) < 2) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    return .{ seen.at[1] - seen.at[0], seen.at[2] - seen.at[1], seen.at[3] - seen.at[2] };
}

test "without a decode share, the live stream gets one round between a prompt's chunks" {
    for (try roundsBetweenChunks(0)) |n| try std.testing.expectEqual(@as(u64, 1), n);
}

test "with a decode share, the live stream's rounds fill the time it is owed between a prompt's chunks" {
    // each 10 ms chunk owes 10 ms of rounds at a half share; a fake round takes microseconds
    for (try roundsBetweenChunks(0.5)) |n| try std.testing.expect(n >= 2);
}
