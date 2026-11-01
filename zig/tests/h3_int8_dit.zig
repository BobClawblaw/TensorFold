//! FastH3's native transformer against a case from tools/zig/h3_case.py: the first forward's two velocities, then
//! every denoising step, timed; the final video and audio rows are written raw for the Python decoders.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;
extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;
extern "c" fn usleep(microseconds: c_uint) c_int;
extern "c" fn getppid() c_int;

/// Until `path` exists: the exporter writes its files under another name and renames them when complete. Gives
/// up when the process that started this one is gone, or after `limit` seconds.
fn waitFor(path: [*:0]const u8, parent: c_int, limit: f64) !void {
    const started = mtl.clock.seconds();
    while (access(path, 0) != 0) {
        if (getppid() != parent) return error.ParentGone;
        if (mtl.clock.seconds() - started > limit) return error.CaseTimedOut;
        _ = usleep(10_000);
    }
}

const Match = struct { cosine: f64, relative: f64, worst_row: f64 };

/// Cosine and relative error over everything, and the largest relative error of one row of `width`.
fn match(ours: []const f32, theirs: []const f32, width: usize) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var dd: f64 = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |a, b| {
            dot += @as(f64, a) * b;
            aa += @as(f64, a) * a;
            bb += @as(f64, b) * b;
            rd += (@as(f64, a) - b) * (@as(f64, a) - b);
            rb += @as(f64, b) * b;
        }
        dd += rd;
        worst = @max(worst, @sqrt(rd / @max(rb, 1e-30)));
    }
    return .{ .cosine = dot / @sqrt(aa * bb), .relative = @sqrt(dd / bb), .worst_row = worst };
}

fn write(gpa: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, values: []const f32) !void {
    const path = try std.fmt.allocPrint(gpa, "{s}" ++ suffix ++ "\x00", .{prefix});
    defer gpa.free(path);
    const file = fopen(@ptrCast(path.ptr), "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(values.ptr, 4, values.len, file) != values.len) return error.CannotWrite;
}

/// One Euler step as the reference takes it: x0 from the sigma the model saw, then the grid's ratio.
fn advance(x: []f32, velocity: []const f32, seen: f32, ratio: f32) void {
    for (x, velocity) |*o, v| o.* = ratio * o.* + (1.0 - ratio) * (o.* + seen * v);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5 and args.len != 6) {
        std.debug.print("usage: tf-h3-dit transformer_dir shards case.safetensors out_prefix [dense | swiglu64 | row-scales | half | w7 | profile | wait | lora=PATH | DEFINE, ...]\n", .{});
        return error.BadArguments;
    }
    const gpa = init.gpa;
    const parent = getppid();
    const device = try mtl.Device.init();
    // the optional last argument names kernel shapes to try: "dense", "swiglu64", "row-scales", or defines such as
    // "H3_ATT_FIXED"; "wait" quantizes the checkpoint while the case is still being written (it appears under its
    // own name when complete); "lora=PATH" merges an adapter's factors file (tools/zig/h3_case.py --lora) first
    var options = h3.Options{};
    var defines: [256]u8 = undefined;
    var used: usize = 0;
    var dense = false;
    var profile = false;
    var wait = false;
    var lora: ?[:0]u8 = null;
    if (args.len == 6) {
        var it = std.mem.splitScalar(u8, args[5], ',');
        while (it.next()) |word| {
            if (std.mem.eql(u8, word, "dense")) dense = true else if (std.mem.eql(u8, word, "swiglu64")) options.swiglu_rows = 64 else if (std.mem.eql(u8, word, "row-scales")) options.tile_scales = false else if (std.mem.eql(u8, word, "half")) options.weights = .half else if (std.mem.eql(u8, word, "w7")) options.weights = .w7 else if (std.mem.eql(u8, word, "profile")) profile = true else if (std.mem.eql(u8, word, "wait")) wait = true else if (std.mem.startsWith(u8, word, "lora=")) lora = try gpa.dupeSentinel(u8, word[5..], 0) else {
                const text = try std.fmt.bufPrint(defines[used..], "#define {s}\n", .{word});
                used += text.len;
            }
        }
    }
    options.defines = defines[0..used];
    const loading = mtl.clock.seconds();
    var adapter: ?h3.Tensors = null;
    if (lora) |path| {
        if (wait) try waitFor(path.ptr, parent, 900);
        adapter = try h3.Tensors.open(gpa, device, path.ptr);
    }
    const model = try h3.Model.prepare(gpa, args[1], try std.fmt.parseInt(usize, args[2], 10), if (adapter) |*a| a else null);
    if (adapter) |*a| a.deinit();
    std.debug.print("weights in {d:.1} s ({d} projections with an adapter merged)\n", .{ mtl.clock.seconds() - loading, model.adapted });
    if (wait) try waitFor(args[3].ptr, parent, 900);
    var case = try h3.Tensors.open(gpa, device, args[3]);
    defer case.deinit();
    try model.attach(&case, options);
    model.dense = dense;
    const steps = (case.names.get("video_step") orelse return error.MissingTensor).shape[0];
    const nv = model.video * model.video_in;
    const na = model.audio * model.audio_in;
    const video = try case.copy(f32, gpa, try case.at("video", .f32, nv), nv);
    const audio = try case.copy(f32, gpa, try case.at("audio", .f32, na), na);
    const video_start = try case.copy(f32, gpa, try case.at("video", .f32, nv), nv);
    const audio_start = try case.copy(f32, gpa, try case.at("audio", .f32, na), na);
    const expect_video = try case.copy(f32, gpa, try case.at("video_velocity", .f32, nv), nv);
    const expect_audio = try case.copy(f32, gpa, try case.at("audio_velocity", .f32, na), na);
    const video_step = try case.copy(f32, gpa, try case.at("video_step", .f32, 2 * steps), 2 * steps);
    const audio_step = try case.copy(f32, gpa, try case.at("audio_step", .f32, 2 * steps), 2 * steps);
    std.debug.print("{s}: {d} rows (text {d}, keyframe {d}, audio {d}, video {d}), {d} tiles ({d} prefix, {d} video kept a query), {d} blocks of {d}/{d}/{d}, {d} steps, loaded in {d:.1} s\n", .{ device.name(), model.rows, model.text, model.condition, model.audio, model.video, model.tiles, model.prefix_tiles, model.keep, model.blocks.len, model.hidden, model.inner, model.mlp, steps, mtl.clock.seconds() - loading });

    const vv = try gpa.alloc(f32, nv);
    const av = try gpa.alloc(f32, na);
    // a knock-out or a probe only wants the profile
    const quick = args.len == 6 and (std.mem.indexOf(u8, args[5], "H3_KO") != null or std.mem.indexOf(u8, args[5], "H3_PROBE") != null);
    for (0..if (quick) 0 else steps) |step| {
        const took = try model.forward(video, audio, step, vv, av);
        if (step == 0) {
            const m = match(vv, expect_video, model.video_in);
            const n = match(av, expect_audio, model.audio_in);
            std.debug.print("first video velocity against the reference: cosine {d:.6}, relative error {d:.5}, worst row {d:.5}\n", .{ m.cosine, m.relative, m.worst_row });
            std.debug.print("first audio velocity against the reference: cosine {d:.6}, relative error {d:.5}, worst row {d:.5}\n", .{ n.cosine, n.relative, n.worst_row });
        }
        advance(video, vv, video_step[2 * step], video_step[2 * step + 1]);
        advance(audio, av, audio_step[2 * step], audio_step[2 * step + 1]);
        std.debug.print("step {d}/{d}: {d:.3} s wall, {d:.3} s on the GPU\n", .{ step + 1, steps, took.wall, took.gpu });
        if (step == 0 and !dense) {
            const adj = model.adjacency();
            std.debug.print("last block's chosen key tiles: {d}, {d} of them directly after the one before\n", .{ adj.chosen, adj.following });
        }
    }
    try write(gpa, args[4], ".video.f32", video);
    try write(gpa, args[4], ".audio.f32", audio);
    if (!profile and !quick) return;
    model.profile = true;
    _ = try model.forward(video_start, audio_start, 0, vv, av);
    std.debug.print("one forward by stage, each in its own command buffer:\n", .{});
    for (h3.stages, model.spent) |stage, seconds| std.debug.print("  {s}: {d:.3} s\n", .{ stage, seconds });
}
