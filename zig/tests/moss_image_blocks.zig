//! tf-moss-image-blocks: one step of MOSS-Image's transformer, each block alone, gated against float32.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;

const usage =
    \\tf-moss-image-blocks TRANSFORMER_DIR BLOCKS
    \\BLOCKS (moss_image_case.py blocks) holds one step's branch: its input latents, timestep and prompt rows, a float32
    \\run's stream into each block and after the last, its velocity, and torch bf16's embedding, each block and the final
    \\layer alone on that stream rounded to bf16. Ours runs the same steps (Model.steps); each must be within twice torch
    \\bf16's distance from float32 (or bf16 rounding's, when larger) in cosine error and in its worst row, and a second
    \\run must repeat the first byte for byte; else it fails.
;

const check = mi.reference;
const H = mi.config.hidden;

fn tensor(f: *const tf.safetensors.File, name: []const u8, dtype: tf.safetensors.DType) ![]const u8 {
    const t = f.get(name) orelse {
        std.debug.print("BLOCKS has no {s}\n", .{name});
        return error.MissingTensor;
    };
    if (t.dtype != dtype) return error.BadBlocksTensor;
    return t.bytes;
}

fn floats(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]f32 {
    const bytes = try tensor(f, name, .f32);
    const out = try a.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn halves(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]u16 {
    const bytes = try tensor(f, name, .bf16);
    const out = try a.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn scalar(f: *const tf.safetensors.File, name: []const u8) !i64 {
    const bytes = try tensor(f, name, .i64);
    if (bytes.len != 8) return error.BadBlocksTensor;
    return std.mem.readInt(i64, bytes[0..8], .little);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var f = try tf.safetensors.File.open(a, io, args[2]);
    defer f.close(io);
    const step = try scalar(&f, "step");
    const skip_at = try scalar(&f, "skip");
    const skip: ?usize = if (skip_at < 0) null else std.math.cast(usize, skip_at) orelse return error.BadBlocksTensor;
    const timestep = (try floats(a, &f, "timestep"))[0];
    const latents = try floats(a, &f, "latents");
    const prompt = try halves(a, &f, "prompt");
    const stream = try floats(a, &f, "stream");
    const alone = try halves(a, &f, "alone");
    const velocity_fp32 = try floats(a, &f, "velocity_fp32");
    const velocity_alone = try floats(a, &f, "velocity_alone");
    const grid = (f.get("latents") orelse return error.MissingTensor).shape;
    const image = grid[0] * grid[1];
    const text = prompt.len / mi.config.context;
    const N = image + text;
    const kept: usize = mi.config.layers - @as(usize, @intFromBool(skip != null));
    if (stream.len != (kept + 1) * N * H or alone.len != stream.len or latents.len != image * mi.config.channels or velocity_fp32.len != latents.len or velocity_alone.len != latents.len) {
        std.debug.print("BLOCKS' tensors do not fit {d} image rows, {d} prompt rows and {d} blocks\n", .{ image, text, kept });
        return 2;
    }

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const model = try mi.Model.load(std.heap.page_allocator, io, device, args[1], text, grid[1] * 16, grid[0] * 16);
    defer model.deinit();
    var p = try model.prompt(prompt);
    defer p.deinit();
    const inputs = try a.alloc(u16, stream.len);
    for (inputs, stream) |*o, x| o.* = mi.host.bf16(x);
    const rows = try a.alloc(u16, stream.len);
    const velocity = try a.alloc(f32, latents.len);
    try model.steps(&p, latents, timestep, skip, inputs, rows, velocity);
    std.debug.print("{s}: step {d}, {s} branch, {d} image and {d} prompt rows, {d} blocks\n", .{ device.name(), step, if (skip == null) "prompt" else "negative", image, text, kept });

    // block k's float32 output is the stream into block k + 1 (the embedding's, into block 0)
    const verdicts = try a.alloc(check.Verdict, kept + 2);
    var total: f64 = 0;
    var most: f64 = 0;
    var most_at: usize = 0;
    var ratio: f64 = 0;
    for (0..kept + 2) |k| {
        const last = k == kept + 1;
        const exact = if (last) velocity_fp32 else stream[k * N * H ..][0 .. N * H];
        const theirs = if (last) velocity_alone else try widen(a, alone[k * N * H ..][0 .. N * H]);
        const width: usize = if (last) mi.config.channels else H;
        const ours = if (last) check.match(f32, velocity, exact, width) else check.match(u16, rows[k * N * H ..][0 .. N * H], exact, width);
        const reference = check.match(f32, theirs, exact, width);
        const v = check.gate(ours, reference, check.rounding(exact, width));
        verdicts[k] = v;
        const used = @max(v.cosine_err / v.cosine_limit, v.row / v.row_limit);
        total += used;
        if (used > most) {
            most = used;
            most_at = k;
        }
        ratio += ours.worst_row / @max(reference.worst_row, 1e-30);
        const name = if (k == 0) "the embedding" else if (last) "the final layer" else try std.fmt.allocPrint(a, "block {d}", .{if (skip) |s| (if (k - 1 < s) k - 1 else k) else k - 1});
        std.debug.print("step {d}, {s}: 1-cosine {e:.2} of {e:.2}, worst row {d:.5} (row {d}) of {d:.5}; torch bf16's worst row {d:.5}: {s}\n", .{ k, name, v.cosine_err, v.cosine_limit, v.row, v.row_at, v.row_limit, reference.worst_row, if (v.pass) "pass" else "FAIL" });
    }
    const steps_n: f64 = @floatFromInt(kept + 2);
    std.debug.print("{d} steps: a mean of {d:.2} of the limit, at most {d:.2} (step {d}); our worst row {d:.2} times torch bf16's on average\n", .{ kept + 2, total / steps_n, most, most_at, ratio / steps_n });

    const rows_again = try a.alloc(u16, stream.len);
    const velocity_again = try a.alloc(f32, latents.len);
    try model.steps(&p, latents, timestep, skip, inputs, rows_again, velocity_again);
    const same = std.mem.eql(u16, rows, rows_again) and std.mem.eql(u8, std.mem.sliceAsBytes(velocity), std.mem.sliceAsBytes(velocity_again));
    std.debug.print("run again, the same bytes: {}\n", .{same});
    const pass = check.accept(verdicts, same);
    std.debug.print("{s}\n", .{if (pass) "PASS every step within its gate, repeated byte for byte" else "FAIL"});
    return if (pass) 0 else 1;
}

fn widen(a: std.mem.Allocator, x: []const u16) ![]f32 {
    const out = try a.alloc(f32, x.len);
    for (out, x) |*o, v| o.* = mi.host.widen(v);
    return out;
}
