//! tf-moss-image: MOSS-Image on a moss_image_case.py case, the transformer compared and timed, then decoded.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;
const prompting = @import("moss_image_prompts.zig");

const usage =
    \\tf-moss-image TRANSFORMER_DIR CASE OUT_DIR [--prompt FILE [--negative FILE]]
    \\CASE (tools/zig/moss_image_case.py) holds the prompt's and the empty negative prompt's encoder rows, the timesteps
    \\and noise levels and the starting latents. The schedule is made natively from the checkpoint's scheduler config and
    \\must equal the case's bit for bit. When the case records its seed, the starting latents are drawn natively too
    \\(torch.randn on the CPU) and must equal the case's bit for bit. A `case` also holds a reference run (each step's
    \\input and both branches' velocities, in bf16 and for its first steps in float32): each step then runs both branches
    \\on the reference's input and compares them, beside torch bf16's own distance from float32 where the case has it
    \\(tf-moss-image-blocks gates each block alone); a step run twice must repeat byte for byte, or the run fails. Then
    \\the whole schedule runs from the starting latents with the card's guidance (scale 3, the negative branch skipping
    \\block 22), timed. The transformer is then freed and the checkpoint's image decoder (the vae/ folder beside
    \\TRANSFORMER_DIR) turns the latents into the picture. OUT_DIR gets latents.f32 (rows x columns x 128), picture.png
    \\and run.json. With --prompt, the encoder in TRANSFORMER_DIR's checkpoint folder makes both prompts' rows (the
    \\negative from --negative's file, else empty) in place of the case's.
;

const negative_skip = mi.pipeline.negative_skip;

const check = mi.reference;

/// The case tensor `name` of `dtype`, copied out as T values.
fn take(comptime T: type, a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8, dtype: tf.safetensors.DType) ![]T {
    const t = f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
    if (t.dtype != dtype) return error.BadCaseTensor;
    const out = try a.alloc(T, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn dims(f: *const tf.safetensors.File, name: []const u8) ![]const usize {
    const e = f.names.getPtr(name) orelse return error.MissingTensor;
    return e.shape[0..e.rank];
}

fn print(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    try list.appendSlice(a, try std.fmt.allocPrint(a, fmt, args));
}

fn json(list: *std.ArrayList(u8), a: std.mem.Allocator, key: []const u8, m: check.Match) !void {
    try print(list, a, ", \"{s}\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.8}, \"worst_row\": {d:.6}}}", .{ key, m.max_abs, m.cosine, m.worst_row });
}

/// Each step's two branches on the reference's input against its velocities; true when a repeated forward repeats.
fn teacher(a: std.mem.Allocator, case: *const tf.safetensors.File, model: *mi.Model, branches: *[2]mi.Prompt, timesteps: []const f32, count: usize, out: *std.ArrayList(u8)) !bool {
    const steps = timesteps.len;
    const inputs = try take(f32, a, case, "inputs", .f32);
    const reference = [2][]f32{ try take(f32, a, case, "velocity", .f32), try take(f32, a, case, "velocity_negative", .f32) };
    const exact = [2]?[]f32{
        if (case.get("velocity_f32") != null) try take(f32, a, case, "velocity_f32", .f32) else null,
        if (case.get("velocity_negative_f32") != null) try take(f32, a, case, "velocity_negative_f32", .f32) else null,
    };
    if (inputs.len != steps * count) return error.BadCaseTensor;
    for (reference) |r| if (r.len != steps * count) return error.BadCaseTensor;
    try print(out, a, ", \"teacher_forced\": [", .{});
    const skips = [2]?usize{ null, negative_skip };
    const labels = [2][]const u8{ "prompt", "negative" };
    const velocity = try a.alloc(f32, count);
    const again = try a.alloc(f32, count);
    for (0..steps) |s| {
        try print(out, a, "{s}{{\"step\": {d}, \"timestep\": {d:.4}", .{ if (s == 0) "" else ", ", s + 1, timesteps[s] });
        for (0..2) |b| {
            _ = try model.forward(&branches[b], inputs[s * count ..][0..count], timesteps[s], skips[b], velocity);
            const theirs = reference[b][s * count ..][0..count];
            const m = check.match(f32, velocity, theirs, mi.config.channels);
            std.debug.print("step {d} {s}: against the bf16 reference max |diff| {d:.5}, cosine {d:.7}, worst row {d:.5}\n", .{ s + 1, labels[b], m.max_abs, m.cosine, m.worst_row });
            try json(out, a, try std.fmt.allocPrint(a, "{s}_bf16", .{labels[b]}), m);
            if (exact[b]) |f| if (f.len >= (s + 1) * count) {
                const want = f[s * count ..][0..count];
                const e = check.match(f32, velocity, want, mi.config.channels);
                const r = check.match(f32, theirs, want, mi.config.channels);
                std.debug.print("        against the fp32 reference max |diff| {d:.5}, cosine {d:.7}, worst row {d:.5}; the bf16 reference's own: {d:.5}, {d:.7}, {d:.5}\n", .{ e.max_abs, e.cosine, e.worst_row, r.max_abs, r.cosine, r.worst_row });
                try json(out, a, try std.fmt.allocPrint(a, "{s}_fp32", .{labels[b]}), e);
                try json(out, a, try std.fmt.allocPrint(a, "{s}_bf16_ref_vs_fp32", .{labels[b]}), r);
            };
        }
        try out.append(a, '}');
    }
    _ = try model.forward(&branches[0], inputs[0..count], timesteps[0], null, again);
    _ = try model.forward(&branches[1], inputs[0..count], timesteps[1], negative_skip, velocity);
    _ = try model.forward(&branches[0], inputs[0..count], timesteps[0], null, velocity);
    const repeat = std.mem.eql(u8, std.mem.sliceAsBytes(velocity), std.mem.sliceAsBytes(again));
    std.debug.print("the same step twice, another between: {s}\n", .{if (repeat) "identical bytes" else "DIFFERENT bytes"});
    try print(out, a, "], \"repeat_identical\": {}", .{repeat});
    return repeat;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var prompt_file: ?[]const u8 = null;
    var negative_file: ?[]const u8 = null;
    var i: usize = 4;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--prompt")) prompt_file = args[i + 1] else if (std.mem.eql(u8, args[i], "--negative")) negative_file = args[i + 1] else break;
    }
    if (args.len < 4 or i != args.len or (negative_file != null and prompt_file == null)) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var case = try tf.safetensors.File.open(a, io, args[2]);
    defer case.close(io);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const checkpoint = std.fs.path.dirname(args[1]) orelse ".";
    const prompts = if (prompt_file) |file| try encode(a, io, device, checkpoint, file, negative_file) else Prompts{ .rows = .{ try take(u16, a, &case, "prompt", .bf16), try take(u16, a, &case, "negative", .bf16) }, .encode_s = null };
    var out: std.ArrayList(u8) = .empty;
    const run = try denoise(a, io, device, args[1], &case, prompts, &out);

    const width = run.columns * 16;
    const height = run.rows * 16;
    const loading = mtl.clock.seconds();
    const vae = try std.fs.path.join(a, &.{ checkpoint, "vae" });
    const decoder = try mi.Decoder.load(std.heap.page_allocator, io, device, vae, width, height);
    defer decoder.deinit();
    const decoder_load_s = mtl.clock.seconds() - loading;
    const rgb = try a.alloc(u8, width * height * 3);
    const t = try decoder.decode(run.latents, rgb);
    std.debug.print("decoded {d}x{d}: {d:.3} s ({d:.3} s on the GPU), the decoder loaded in {d:.2} s\n", .{ width, height, t.wall, t.gpu, decoder_load_s });
    try print(&out, a, ", \"decoder_load_s\": {d:.3}, \"decode_s\": {d:.4}}}\n", .{ decoder_load_s, t.wall });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/latents.f32", .{args[3]}), .data = std.mem.sliceAsBytes(run.latents) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/picture.png", .{args[3]}), .data = try mi.png.encode(a, rgb, width, height) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/run.json", .{args[3]}), .data = out.items });
    std.debug.print("{s}\n", .{if (run.pass) "PASS" else "FAIL"});
    return if (run.pass) 0 else 1;
}

const Prompts = struct { rows: [2][]u16, encode_s: ?f64 };

/// Both prompts' rows from the checkpoint's own encoder, which is freed before this returns.
fn encode(a: std.mem.Allocator, io: std.Io, device: mtl.Device, checkpoint: []const u8, prompt_file: []const u8, negative_file: ?[]const u8) !Prompts {
    const texts = [2][]const u8{ try prompting.read(a, io, prompt_file), if (negative_file) |n| try prompting.read(a, io, n) else "" };
    var made: [2][]u16 = undefined;
    const encode_s = try prompting.encode(a, io, device, checkpoint, &texts, &made);
    std.debug.print("both prompts encoded natively: {d:.3} s\n", .{encode_s});
    return .{ .rows = made, .encode_s = encode_s };
}

const Run = struct { latents: []f32, rows: usize, columns: usize, pass: bool };

/// The case's schedule on `prompts` through the transformer of `dir`, freed before this returns; timings into `out`.
fn denoise(a: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, case: *const tf.safetensors.File, prompts: Prompts, out: *std.ArrayList(u8)) !Run {
    const grid = try dims(case, "latents");
    if (grid.len != 3 or grid[2] != mi.config.channels) return error.BadCaseTensor;
    const rows = grid[0] * grid[1];
    const count = rows * mi.config.channels;
    const prompt = prompts.rows[0];
    const negative = prompts.rows[1];
    const text = prompt.len / mi.config.context;
    const negative_rows = negative.len / mi.config.context;
    const theirs = try take(f32, a, case, "latents", .f32);
    if (theirs.len != count) return error.BadCaseTensor;
    var start = theirs;
    var noise_equal = true;
    if (case.get("seed") != null) {
        const seed = std.math.cast(u64, (try take(i64, a, case, "seed", .i64))[0]) orelse return error.BadCaseTensor;
        start = try mi.noise.latents(a, seed, grid[0], grid[1], mi.config.channels);
        noise_equal = std.mem.eql(u8, std.mem.sliceAsBytes(start), std.mem.sliceAsBytes(theirs));
        std.debug.print("starting latents: drawn natively from seed {d}, the same bits as the case's: {}\n", .{ seed, noise_equal });
    } else std.debug.print("starting latents: the case's (it records no seed)\n", .{});
    const plan = try schedule(a, io, std.fs.path.dirname(dir) orelse ".", case);
    const timesteps = plan.timesteps;
    const sigmas = plan.sigmas;
    const steps = timesteps.len;

    const loading = mtl.clock.seconds();
    const model = try mi.Model.load(std.heap.page_allocator, io, device, dir, @max(text, negative_rows), grid[1] * 16, grid[0] * 16);
    defer model.deinit();
    const load_s = mtl.clock.seconds() - loading;
    std.debug.print("{s}: {d} image rows, prompts of {d} and {d} rows, {d} steps; loaded in {d:.1} s\n", .{ device.name(), rows, text, negative_rows, steps, load_s });

    const refining = mtl.clock.seconds();
    var branches: [2]mi.Prompt = undefined;
    branches[0] = try model.prompt(prompt);
    defer branches[0].deinit();
    branches[1] = try model.prompt(negative);
    defer branches[1].deinit();
    const refine_s = mtl.clock.seconds() - refining;
    std.debug.print("both prompts refined: {d:.3} s\n", .{refine_s});
    try print(out, a, "{{\"device\": \"{s}\", \"image_rows\": {d}, \"prompt_rows\": {d}, \"negative_rows\": {d}, \"load_s\": {d:.2}, \"refine_s\": {d:.4}", .{ device.name(), rows, text, negative_rows, load_s, refine_s });
    if (prompts.encode_s) |s| try print(out, a, ", \"native_encoder\": true, \"encode_s\": {d:.4}", .{s});
    try print(out, a, ", \"schedule_equal\": {}, \"noise_equal\": {}", .{ plan.equal, noise_equal });
    var pass = plan.equal and noise_equal;
    if (case.get("inputs") != null) pass = try teacher(a, case, model, &branches, timesteps, count, out) and pass;
    try print(out, a, ", \"run\": [", .{});

    const x = try a.dupe(f32, start);
    const cond = try a.alloc(f32, count);
    const uncond = try a.alloc(f32, count);
    var total: f64 = 0;
    for (0..steps) |s| {
        const t = try mi.pipeline.step(model, &branches[0], &branches[1], x, timesteps[s], sigmas[s], sigmas[s + 1], mi.pipeline.guidance, cond, uncond);
        total += t.wall;
        std.debug.print("step {d}/{d}: {d:.3} s ({d:.3} s on the GPU, {d:.3} s modulation on the host)\n", .{ s + 1, steps, t.wall, t.gpu, t.host });
        try print(out, a, "{s}{{\"wall_s\": {d:.4}, \"gpu_s\": {d:.4}, \"host_s\": {d:.4}}}", .{ if (s == 0) "" else ", ", t.wall, t.gpu, t.host });
    }
    try print(out, a, "], \"denoise_s\": {d:.3}", .{total});
    std.debug.print("denoise: {d:.2} s for {d} steps ({d} forwards)\n", .{ total, steps, 2 * steps });
    return .{ .latents = x, .rows = grid[0], .columns = grid[1], .pass = pass };
}

/// The native schedule for the case's step count, and whether it equals the case's own bit for bit.
fn schedule(a: std.mem.Allocator, io: std.Io, checkpoint: []const u8, case: *const tf.safetensors.File) !struct { timesteps: []f32, sigmas: []f32, equal: bool } {
    const theirs = .{ try take(f32, a, case, "timesteps", .f32), try take(f32, a, case, "sigmas", .f32) };
    const steps: usize = if (case.get("steps") != null) std.math.cast(usize, (try take(i64, a, case, "steps", .i64))[0]) orelse return error.BadCaseTensor else theirs[0].len;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ checkpoint, "scheduler", "scheduler_config.json" }), a, .limited(1 << 20));
    const ours = try mi.schedule.make(a, try mi.schedule.parse(a, text), steps);
    const equal = std.mem.eql(u8, std.mem.sliceAsBytes(ours.timesteps), std.mem.sliceAsBytes(theirs[0])) and std.mem.eql(u8, std.mem.sliceAsBytes(ours.sigmas), std.mem.sliceAsBytes(theirs[1]));
    std.debug.print("schedule: {d} steps from scheduler_config.json, the same bits as the case's: {}\n", .{ ours.timesteps.len, equal });
    return .{ .timesteps = ours.timesteps, .sigmas = ours.sigmas, .equal = equal };
}
