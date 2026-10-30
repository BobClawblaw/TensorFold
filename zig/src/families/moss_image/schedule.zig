//! MOSS-Image's denoising schedule, computed as the publisher's MOSSImageScheduler computes it in float32.
const std = @import("std");

/// The settings MOSSImageScheduler reads from scheduler/scheduler_config.json, at its own defaults.
pub const Config = struct { train: usize = 1000, shift: f64 = 5.0, training_shift: f64 = 1.0 };

/// The scheduler config this family runs (MOSSImageScheduler), else an error; the keys its class ignores stay ignored.
pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const o = doc.value.object;
    const name = o.get("_class_name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "MOSSImageScheduler")) return error.NotMossImage;
    var c: Config = .{};
    if (o.get("num_train_timesteps")) |v| {
        if (v != .integer or v.integer < 2) return error.BadConfig;
        c.train = @intCast(v.integer);
    }
    inline for (.{ .{ "shift", "shift" }, .{ "training_shift", "training_shift" } }) |kv| if (o.get(kv[0])) |v| {
        @field(c, kv[1]) = switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => return error.BadConfig,
        };
        if (!(@field(c, kv[1]) > 0) or !std.math.isFinite(@field(c, kv[1]))) return error.BadConfig;
    };
    return c;
}

pub const Schedule = struct {
    /// Each step's timestep, sigma times the training steps.
    timesteps: []f32,
    /// Each step's noise level, then the final 0.
    sigmas: []f32,

    pub fn deinit(s: Schedule, gpa: std.mem.Allocator) void {
        gpa.free(s.timesteps);
        gpa.free(s.sigmas);
    }
};

/// The schedule for `steps` steps: the shifted grid snapped to the nearest training sigma, repeats dropped.
pub fn make(gpa: std.mem.Allocator, c: Config, steps: usize) !Schedule {
    if (steps == 0) return error.BadSteps;
    const wanted = try grid(gpa, steps, c.shift, c.train);
    defer gpa.free(wanted);
    const training = try grid(gpa, c.train, c.training_shift, c.train);
    defer gpa.free(training);
    std.mem.reverse(f32, training);
    const snapped = try gpa.alloc(f32, steps);
    defer gpa.free(snapped);
    var kept: usize = 0;
    for (wanted) |x| {
        const up = @min(std.sort.lowerBound(f32, training, x, order), c.train - 1);
        const lo = up -| 1;
        const v = training[if (@abs(x - training[lo]) <= @abs(training[up] - x)) lo else up];
        if (kept == 0 or snapped[kept - 1] != v) {
            snapped[kept] = v;
            kept += 1;
        }
    }
    const timesteps = try gpa.alloc(f32, kept);
    errdefer gpa.free(timesteps);
    const sigmas = try gpa.alloc(f32, kept + 1);
    for (timesteps, sigmas[0..kept], snapped[0..kept]) |*t, *s, v| {
        t.* = v * @as(f32, @floatFromInt(c.train));
        s.* = v;
    }
    sigmas[kept] = 0;
    return .{ .timesteps = timesteps, .sigmas = sigmas };
}

fn order(x: f32, y: f32) std.math.Order {
    return std.math.order(x, y);
}

/// `count` sigmas from 1 down to the lowest training one: torch's float32 linspace, then the shift.
fn grid(gpa: std.mem.Allocator, count: usize, shift: f64, train: usize) ![]f32 {
    const out = try gpa.alloc(f32, count);
    const total: f64 = @floatFromInt(train);
    linspace(out, 1, @floatCast(shift / (total + shift - 1)));
    const s: f32 = @floatCast(shift);
    const s1: f32 = @floatCast(shift - 1);
    for (out) |*v| v.* = s * v.* / (1 + s1 * v.*);
    return out;
}

/// torch.linspace on the CPU in float32: each half fused from its own end.
fn linspace(out: []f32, start: f32, end: f32) void {
    if (out.len == 1) {
        out[0] = start;
        return;
    }
    const step = (end - start) / @as(f32, @floatFromInt(out.len - 1));
    const half = out.len / 2;
    for (out, 0..) |*v, i| v.* = if (i < half)
        @mulAdd(f32, step, @floatFromInt(i), start)
    else
        @mulAdd(f32, -step, @floatFromInt(out.len - i - 1), end);
}

const released =
    \\{"_class_name": "MOSSImageScheduler", "_diffusers_version": "0.38.0", "base_image_seq_len": 256, "base_shift": 0.5,
    \\ "invert_sigmas": false, "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 5.0,
    \\ "shift_terminal": null, "stochastic_sampling": false, "time_shift_type": "exponential", "training_shift": 1.0,
    \\ "use_beta_sigmas": false, "use_dynamic_shifting": false, "use_exponential_sigmas": false, "use_karras_sigmas": false}
;

fn bits(values: []const f32) []const u32 {
    return std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(values));
}

test "the released scheduler config parses to the class's settings; another class does not" {
    const a = std.testing.allocator;
    const c = try parse(a, released);
    try std.testing.expectEqual(Config{}, c);
    const other = try std.mem.replaceOwned(u8, a, released, "MOSSImageScheduler", "FlowMatchEulerDiscreteScheduler");
    defer a.free(other);
    try std.testing.expectError(error.NotMossImage, parse(a, other));
    const bad = try std.mem.replaceOwned(u8, a, released, "\"shift\": 5.0", "\"shift\": -1.0");
    defer a.free(bad);
    try std.testing.expectError(error.BadConfig, parse(a, bad));
}

test "short schedules equal the publisher's scheduler's bit for bit" {
    const a = std.testing.allocator;
    const want = [_]struct { t: []const u32, s: []const u32 }{
        .{ .t = &.{0x447a0000}, .s = &.{ 0x3f800000, 0 } },
        .{ .t = &.{ 0x447a0000, 0x41c00000 }, .s = &.{ 0x3f800000, 0x3cc49ba6, 0 } },
        .{ .t = &.{ 0x447a0000, 0x4450c000, 0x41c00000 }, .s = &.{ 0x3f800000, 0x3f55c28f, 0x3cc49ba6, 0 } },
    };
    for (want, 1..) |w, steps| {
        const s = try make(a, .{}, steps);
        defer s.deinit(a);
        try std.testing.expectEqualSlices(u32, w.t, bits(s.timesteps));
        try std.testing.expectEqualSlices(u32, w.s, bits(s.sigmas));
    }
}

test "past 218 steps, steps that snap to the same training sigma are dropped" {
    const a = std.testing.allocator;
    const full = try make(a, .{}, 218);
    defer full.deinit(a);
    try std.testing.expectEqual(@as(usize, 218), full.timesteps.len);
    const fewer = try make(a, .{}, 219);
    defer fewer.deinit(a);
    try std.testing.expect(fewer.timesteps.len < 219);
    try std.testing.expectEqual(fewer.timesteps.len + 1, fewer.sigmas.len);
    try std.testing.expectError(error.BadSteps, make(a, .{}, 0));
}
