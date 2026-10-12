//! Qwen3.8 Flash Next's image tower against a reference dump (tools/zig/flashnext_vision_fixtures.py: transformers).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const vision = tf.flashnext_vision;

const usage =
    \\tf-flashnext-vision MODEL_DIR REF_DIR [STAGE]
    \\REF_DIR holds meta.json (grid_thw), patches.f32 and the stage's reference: embed.f32 (STAGE embed, the patch
    \\projection plus the position table), block0.f32 (STAGE block0), blocks.f32 (STAGE blocks, after the last block)
    \\or features.f32 (STAGE features, the default: the tower's output rows). Writes native_STAGE.f32 there and prints
    \\the largest and mean absolute differences, the reference's mean magnitude and the rows' cosine similarity; for
    \\features also against features_fp32.f32 (the fp32 tower), beside the bf16 reference's own cosine to it.
;

fn readFile(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    return a.dupe(u8, f.bytes[0..f.size]);
}

fn floats(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const raw = try readFile(a, dir, name);
    const out = try a.alloc(f32, raw.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), raw[0 .. out.len * 4]);
    return out;
}

const Cmp = struct { max_d: f64, mean_d: f64, mean_ref: f64, cos_mean: f64, cos_worst: f64 };

fn compare(got: []const f32, want: []const f32, width: usize) Cmp {
    var c: Cmp = .{ .max_d = 0, .mean_d = 0, .mean_ref = 0, .cos_mean = 0, .cos_worst = 1 };
    const rows = got.len / width;
    for (0..rows) |r| {
        var dot: f64 = 0;
        var ng: f64 = 0;
        var nw: f64 = 0;
        for (0..width) |j| {
            const g: f64 = got[r * width + j];
            const w: f64 = want[r * width + j];
            c.max_d = @max(c.max_d, @abs(g - w));
            c.mean_d += @abs(g - w);
            c.mean_ref += @abs(w);
            dot += g * w;
            ng += g * g;
            nw += w * w;
        }
        const cs = dot / (@sqrt(ng * nw) + 1e-30);
        c.cos_worst = @min(c.cos_worst, cs);
        c.cos_mean += cs;
    }
    const n: f64 = @floatFromInt(got.len);
    c.mean_d /= n;
    c.mean_ref /= n;
    c.cos_mean /= @floatFromInt(rows);
    return c;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const dir = args[1];
    const ref = args[2];
    const stage = if (args.len > 3) args[3] else "features";
    const stop: ?u32 = if (std.mem.eql(u8, stage, "embed")) 0 else if (std.mem.eql(u8, stage, "block0")) 1 else if (std.mem.eql(u8, stage, "blocks")) vision.depth else if (std.mem.eql(u8, stage, "features")) null else return error.UnknownStage;

    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, try readFile(a, ref, "meta.json"), .{});
    const grid = meta.object.get("grid_thw").?.array.items;
    const gh: u32 = @intCast(grid[1].integer);
    const gw: u32 = @intCast(grid[2].integer);
    const pixels = try floats(a, ref, "patches.f32");
    const P = gh * gw;
    const T = P / vision.patches_per_token;

    try vision.check(try readFile(a, dir, "config.json"), a);
    const device = try mtl.Device.init();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    var t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const v = try vision.Vision.load(std.heap.page_allocator, device, dir, @max(T, 1024));
    defer v.deinit();
    var t1 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    std.debug.print("tower loaded in {d:.2} s ({d:.2} GiB with scratch for {d} patches, attention chunks of {d} rows)\n", .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @as(f64, @floatFromInt(v.bytes)) / (1 << 30), v.max_patches, v.chunk });
    const out_buf = try device.buffer(@as(usize, T) * vision.out_hidden * 2, mtl.ResourceOptions.shared);
    defer out_buf.deinit();

    for (0..2) |run| { // the second run is the timed one
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
        try v.encode(enc, pixels, gh, gw, .{ .buf = out_buf }, stop);
        enc.end();
        cb.commit();
        cb.wait();
        t1 = std.Io.Clock.awake.now(init.io).toNanoseconds();
        if (cb.failure()) |msg| {
            std.debug.print("command buffer failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
        if (run == 1) std.debug.print("encode: {d} patches -> {d} tokens in {d:.1} ms\n", .{ P, T, @as(f64, @floatFromInt(t1 - t0)) / 1e6 });
    }
    const width: usize = if (stop != null) vision.hidden else vision.out_hidden;
    const rows: usize = if (stop != null) P else T;
    const src: [*]const u16 = if (stop != null) @ptrCast(@alignCast(v.x.addr())) else @ptrCast(@alignCast(out_buf.contents()));
    const got = try a.alloc(f32, rows * width);
    for (got, 0..) |*g, i| g.* = @bitCast(@as(u32, src[i]) << 16);
    {
        const path = try std.fmt.allocPrint(a, "{s}/native_{s}.f32", .{ ref, stage });
        const f = try std.Io.Dir.cwd().createFile(init.io, path, .{});
        defer f.close(init.io);
        try f.writeStreamingAll(init.io, std.mem.sliceAsBytes(got));
    }
    const want = try floats(a, ref, try std.fmt.allocPrint(a, "{s}.f32", .{stage}));
    if (want.len != got.len) {
        std.debug.print("size: native {d} values, reference {d}\n", .{ got.len, want.len });
        return error.SizeMismatch;
    }
    const c = compare(got, want, width);
    std.debug.print("{s}: max |d| {d:.4}, mean |d| {d:.5}, mean |ref| {d:.4}, cosine mean {d:.6} worst {d:.6} ({d} rows of {d})\n", .{ stage, c.max_d, c.mean_d, c.mean_ref, c.cos_mean, c.cos_worst, rows, width });
    if (stop == null) {
        const exact = floats(a, ref, "features_fp32.f32") catch return;
        const n = compare(got, exact, width);
        const r = compare(want, exact, width);
        std.debug.print("against fp32: native cosine mean {d:.6} worst {d:.6}; the bf16 reference's mean {d:.6} worst {d:.6}\n", .{ n.cos_mean, n.cos_worst, r.cos_mean, r.cos_worst });
    }
}
