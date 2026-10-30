//! MOSS-Image's starting noise: torch.randn on the CPU in float32 from a seeded generator, as torch on macOS draws it.
const std = @import("std");

/// The Mersenne Twister torch's CPU generator runs, seeded as manual_seed seeds it (the seed's low 32 bits).
pub const Mt19937 = struct {
    state: [624]u32,
    at: usize,

    pub fn init(seed: u64) Mt19937 {
        var m: Mt19937 = .{ .state = undefined, .at = 624 };
        m.state[0] = @truncate(seed);
        for (1..624) |i| m.state[i] = 1812433253 *% (m.state[i - 1] ^ (m.state[i - 1] >> 30)) +% @as(u32, @intCast(i));
        return m;
    }

    pub fn next(m: *Mt19937) u32 {
        if (m.at == 624) {
            for (0..624) |k| {
                const y = (m.state[k] & 0x80000000) | (m.state[(k + 1) % 624] & 0x7fffffff);
                m.state[k] = m.state[(k + 397) % 624] ^ (y >> 1) ^ (if (y & 1 != 0) @as(u32, 0x9908b0df) else 0);
            }
            m.at = 0;
        }
        var y = m.state[m.at];
        m.at += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// torch's float uniform in [0, 1): the output's low 24 bits.
    fn uniform(m: *Mt19937) f32 {
        return @as(f32, @floatFromInt(m.next() & 0xffffff)) * 0x1p-24;
    }
};

const SinCos = extern struct { sin: f32, cos: f32 };
/// Apple libm's logf and __sincosf_stret, as torch's macOS build calls them.
const Math = struct { log: *const fn (f32) callconv(.c) f32, sincos: *const fn (f32) callconv(.c) SinCos };

/// `out` as torch.randn(out.len) on the CPU in float32 under manual_seed(`seed`): uniforms, then Box-Muller by 16.
pub fn draw(out: []f32, seed: u64) !void {
    // torch takes another path below 16 values
    if (out.len < 16) return error.TooFewValues;
    // looked up in libSystem itself: Zig's runtime defines its own logf, which differs in the last bit
    const lib = std.c.dlopen("/usr/lib/libSystem.B.dylib", .{ .LAZY = true }) orelse return error.NoSystemMath;
    defer _ = std.c.dlclose(lib);
    const math: Math = .{
        .log = @ptrCast(@alignCast(std.c.dlsym(lib, "logf") orelse return error.NoSystemMath)),
        .sincos = @ptrCast(@alignCast(std.c.dlsym(lib, "__sincosf_stret") orelse return error.NoSystemMath)),
    };
    var m = Mt19937.init(seed);
    for (out) |*v| v.* = m.uniform();
    var i: usize = 0;
    while (i + 16 <= out.len) : (i += 16) fill(out[i..][0..16], math);
    if (out.len % 16 != 0) {
        // a ragged end: the last 16 drawn again from fresh uniforms
        const last = out[out.len - 16 ..][0..16];
        for (last) |*v| v.* = m.uniform();
        fill(last, math);
    }
}

/// Box-Muller over 16 uniforms: radius from the first 8 (Apple libm's logf), angle from the last 8 (its sincosf).
fn fill(d: *[16]f32, math: Math) void {
    for (0..8) |j| {
        const radius = @sqrt(-2 * math.log(1 - d[j]));
        const theta: f32 = @floatCast(2.0 * std.math.pi * @as(f64, d[j + 8]));
        const sc = math.sincos(theta);
        d[j] = radius * sc.cos;
        d[j + 8] = radius * sc.sin;
    }
}

/// The pipeline's starting latents for a `rows` x `columns` grid: torch.randn((1, 128, rows, columns)) as rows of 128.
pub fn latents(gpa: std.mem.Allocator, seed: u64, rows: usize, columns: usize, channels: usize) ![]f32 {
    const n = try std.math.mul(usize, try std.math.mul(usize, rows, columns), channels);
    const planes = try gpa.alloc(f32, n);
    defer gpa.free(planes);
    try draw(planes, seed);
    const out = try gpa.alloc(f32, n);
    for (0..channels) |c| for (0..rows * columns) |p| {
        out[p * channels + c] = planes[c * rows * columns + p];
    };
    return out;
}

fn bits(values: []const f32) []const u32 {
    return std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(values));
}

test "the Mersenne Twister's first output for seed 5489" {
    var m = Mt19937.init(5489);
    try std.testing.expectEqual(@as(u32, 3499211612), m.next());
}

test "16 and a ragged 20 values equal torch.randn's bit for bit" {
    var sixteen: [16]f32 = undefined;
    try draw(&sixteen, 42);
    try std.testing.expectEqualSlices(u32, &.{ 0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f2dacd5, 0xbf9e0592, 0xbd306787, 0xbfcd65ba, 0xbf408bff, 0x3fd3095a, 0xbec8f2f6, 0xbfb3a963, 0xbf3a566d, 0xbf0f36cc, 0xbf44d2a1, 0x3f432f9f }, bits(&sixteen));
    var twenty: [20]f32 = undefined;
    try draw(&twenty, 7);
    try std.testing.expectEqualSlices(u32, &.{ 0xbf51f456, 0x3eca9023, 0x3f661ed5, 0xbfb1b739, 0x3e4b34eb, 0x3f4835ea, 0x3f85017b, 0xbf397993, 0xbf595213, 0xbf9bc459, 0xbda5eea5, 0x3fae4525, 0xc003ef41, 0x3f2c91dd, 0xbfa96365, 0xbfae0ce5, 0xbf5de028, 0xbf1062b9, 0xc0039d15, 0xbeb6a456 }, bits(&twenty));
    var few: [8]f32 = undefined;
    try std.testing.expectError(error.TooFewValues, draw(&few, 1));
}
