//! tf-moss-image-generate: a MOSS-Image picture from a prompt and options alone, every stage on this machine.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;
const prompting = @import("moss_image_prompts.zig");

const usage =
    \\tf-moss-image-generate MODEL_DIR OUT.png (--prompt TEXT | --prompt-file FILE) [options]
    \\MODEL_DIR is the OpenMOSS-Team/MOSS-Image-Base folder. As the publisher's pipeline does, with its defaults:
    \\  --negative TEXT | --negative-file FILE   the negative prompt (empty)
    \\  --width W --height H                     positive multiples of 16 (1024 x 1024)
    \\  --steps N                                Euler steps asked for (50)
    \\  --seed S                                 the starting noise's seed, as torch.manual_seed takes it (42)
    \\  --guidance G                             the guidance scale, 0 or more (3); at 1 the negative branch is not run
    \\The negative branch skips block 22, as the card's pipeline does. Skip-layer guidance (the pipeline's slg_layers,
    \\slg_scale, slg_start, slg_stop) is not supported: an --slg-* option is refused.
;

const Options = struct {
    prompt: ?[]const u8 = null,
    negative: []const u8 = "",
    width: usize = 1024,
    height: usize = 1024,
    steps: u32 = 50,
    seed: u64 = 42,
    guidance: f32 = mi.pipeline.guidance,
};

fn parse(a: std.mem.Allocator, io: std.Io, args: []const []const u8) !Options {
    var o: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const k = args[i];
        if (std.mem.startsWith(u8, k, "--slg")) return error.SkipLayerGuidance;
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--prompt")) {
            o.prompt = v;
        } else if (std.mem.eql(u8, k, "--prompt-file")) {
            o.prompt = try prompting.read(a, io, v);
        } else if (std.mem.eql(u8, k, "--negative")) {
            o.negative = v;
        } else if (std.mem.eql(u8, k, "--negative-file")) {
            o.negative = try prompting.read(a, io, v);
        } else if (std.mem.eql(u8, k, "--width")) {
            o.width = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--height")) {
            o.height = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--steps")) {
            o.steps = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--seed")) {
            o.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, k, "--guidance")) {
            o.guidance = try std.fmt.parseFloat(f32, v);
        } else return error.Usage;
    }
    if (o.prompt == null or o.steps == 0 or !std.math.isFinite(o.guidance) or o.guidance < 0) return error.Usage;
    return o;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const o = (if (args.len < 3) error.Usage else parse(a, io, args[3..])) catch |err| {
        if (err == error.SkipLayerGuidance) {
            std.debug.print("skip-layer guidance (--slg-*) is not supported: it has not been checked against the publisher's pipeline\n", .{});
        } else std.debug.print("{s}\n", .{usage});
        return 2;
    };
    const g = mi.config.grid(o.width, o.height) catch {
        std.debug.print("{d}x{d}: the width and height must be positive multiples of 16\n", .{ o.width, o.height });
        return 2;
    };
    const dir = args[1];
    const started = mtl.clock.seconds();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();

    // both prompts' rows, the encoder freed before the transformer loads; the prompt's alone at guidance 1
    const guided = o.guidance != 1;
    var rows: [2][]u16 = undefined;
    const texts = [2][]const u8{ o.prompt.?, o.negative };
    const encode_s = try prompting.encode(a, io, device, dir, texts[0..if (guided) 2 else 1], rows[0..if (guided) 2 else 1]);

    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "scheduler", "scheduler_config.json" }), a, .limited(1 << 20));
    const plan = try mi.schedule.timesteps(a, try mi.schedule.parse(a, text), o.steps);
    const x = try a.alloc(f32, g.rows * g.columns * mi.config.channels);
    try mi.noise.latents(a, o.seed, g.rows, g.columns, x);

    // the guided steps; the transformer is freed before the decoder loads
    const denoise_s = denoise: {
        const longest = @max(rows[0].len, if (guided) rows[1].len else 0) / mi.config.context;
        const model = try mi.Model.load(std.heap.page_allocator, io, device, try std.fs.path.join(a, &.{ dir, "transformer" }), longest, o.width, o.height);
        defer model.deinit();
        var prompt = try model.prompt(rows[0]);
        defer prompt.deinit();
        var negative: ?mi.Prompt = if (guided) try model.prompt(rows[1]) else null;
        defer if (negative) |*n| n.deinit();
        const cond = try a.alloc(f32, x.len);
        const uncond = try a.alloc(f32, x.len);
        var total: f64 = 0;
        for (plan.timesteps, 0..) |t, s| {
            const spent = try mi.pipeline.step(model, &prompt, if (negative) |*n| n else null, x, t, plan.sigmas[s], plan.sigmas[s + 1], o.guidance, cond, uncond);
            total += spent.wall;
        }
        break :denoise total;
    };

    const decoder = try mi.Decoder.load(std.heap.page_allocator, io, device, try std.fs.path.join(a, &.{ dir, "vae" }), o.width, o.height);
    defer decoder.deinit();
    const rgb = try a.alloc(u8, o.width * o.height * 3);
    const decoded = try decoder.decode(x, rgb);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = try mi.png.encode(a, rgb, o.width, o.height) });
    std.debug.print("{s}: {d}x{d}, seed {d}, {d} steps at guidance {d}: prompts {d:.2} s, denoise {d:.1} s, decode {d:.2} s; {d:.1} s in all\n", .{ args[2], o.width, o.height, o.seed, plan.timesteps.len, o.guidance, encode_s, denoise_s, decoded.wall, mtl.clock.seconds() - started });
    return 0;
}
