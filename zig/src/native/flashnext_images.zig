//! Flash Next's images (--vision, one Mac): the shared Qwen-VL tower's rows for a request's prompt pass, its frame.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const vl = tf.qwen_vl;
const vp = tf.flashnext_replay.vis_prompt;
const Tower = tf.qwen_vl_tower.Tower;
const Allocator = std.mem.Allocator;

/// The interleaved M-RoPE sections fn_attn_prep_mrope is written for, and the width it rotates.
const sections = [3]usize{ 11, 11, 10 };
const rotary = 64;

pub const Vision = struct {
    gpa: Allocator,
    c: vl.Config,
    tower: *Tower,
    prep: vp.Prep, // fn_attn_prep_mrope
    capacity: u32, // image tokens one request holds (--vision-image-tokens)
    max_images: u32, // and images (--vision-max-images)
    markers: [3]u32, // vision_start, image_pad, vision_end: text that spells one stays text

    /// Buffers to keep resident with the engine's.
    pub fn buffers(v: *const Vision) [2]mtl.Buffer {
        return v.tower.buffers();
    }

    pub fn deinit(v: *Vision) void {
        v.prep.deinit();
        v.tower.deinit();
        v.gpa.destroy(v);
    }
};

/// The checkpoint's tower for the replay engine's width under its rope, for `o.image_tokens` image tokens a request.
pub fn open(gpa: Allocator, io: std.Io, eng: *fx.Engine, dir: []const u8, o: api.VisionOpen, a: Allocator, why: *[]const u8) !*Vision {
    if (eng.r.tp != null) {
        why.* = "--vision: images are not read across a Flash Next pair yet; serve on one Mac, or leave --vision out";
        return error.VisionOnPair;
    }
    const read = struct {
        fn f(al: Allocator, i: std.Io, d: []const u8, name: []const u8) ?[]const u8 {
            const path = std.fs.path.join(al, &.{ d, name }) catch return null;
            return std.Io.Dir.cwd().readFileAlloc(i, path, al, .limited(16 << 20)) catch null;
        }
    }.f;
    const config = read(a, io, dir, "config.json") orelse return error.NoConfig;
    const c = (vl.Config.parse(a, config, read(a, io, dir, "preprocessor_config.json"), read(a, io, dir, "processor_config.json")) catch |err| {
        why.* = "--vision: this checkpoint's vision_config or image processor names a tower the native engine does not read";
        return err;
    }) orelse {
        why.* = try std.fmt.allocPrint(a, "--vision: {s} has no image tower (its config.json names no vision_config)", .{dir});
        return error.NoVisionTower;
    };
    const mc = try tf.flashnext_config.parse(a, config);
    if (c.out_hidden != tf.flashnext_replay.D or !mc.mrope_interleaved or !std.mem.eql(usize, &mc.mrope_section, &sections) or mc.rotary_dim != rotary) {
        why.* = "--vision: the image tower's output width or the text's M-RoPE sections do not match Flash Next's";
        return error.VisionShape;
    }
    const v = try gpa.create(Vision);
    errdefer gpa.destroy(v);
    v.* = .{ .gpa = gpa, .c = c, .tower = undefined, .prep = undefined, .capacity = o.image_tokens, .max_images = o.max_images, .markers = .{ c.vision_start, c.image_token, c.vision_end } };
    // one image a pass: the tower's scratch holds the largest image the processor makes under the request's budget
    const largest: u32 = @intCast(@min(@as(u64, o.image_tokens), c.max_pixels / (@as(u64, c.factor()) * c.factor())));
    v.tower = try Tower.load(gpa, io, eng.r.device, dir, c, largest);
    errdefer v.tower.deinit();
    v.prep = try vp.Prep.init(eng.r.device);
    std.log.info("flash next: image tower loaded ({d:.2} GiB with scratch for {d} image tokens a request)", .{ @as(f64, @floatFromInt(v.tower.bytes)) / (1 << 30), o.image_tokens });
    return v;
}

/// The request's images through the tower: the prompt pass's rows (four streams each) and frame; `release` frees them.
pub fn build(v: *Vision, eng: *fx.Engine, a: Allocator, images: []const api.Image, prompt_len: usize) !*vp.VisPrompt {
    const merge = v.c.merge;
    var total: usize = 0;
    const spans = try a.alloc(vp.Span, images.len);
    for (images, spans, 0..) |img, *s, i| {
        if (img.gh % merge != 0 or img.gw % merge != 0 or img.tokens * merge * merge != img.gh * img.gw) return error.ImageGrid;
        if (@as(usize, img.at) + img.tokens > prompt_len or (i > 0 and img.at < spans[i - 1].at + spans[i - 1].n)) return error.ImageOutOfPrompt;
        s.* = .{ .at = img.at, .n = img.tokens, .mh = img.gh / merge, .mw = img.gw / merge, .row = @intCast(total) };
        total += img.tokens;
    }
    if (total > v.capacity) return error.ImageTooLarge;
    const row = tf.flashnext_replay.D * 2;
    const rows = try eng.r.device.buffer(total * 4 * row, mtl.ResourceOptions.shared);
    errdefer rows.deinit();
    var most: usize = 0;
    for (spans) |s| most = @max(most, s.n);
    const out = try eng.r.device.buffer(most * row, mtl.ResourceOptions.shared);
    defer out.deinit();
    const t0 = std.c.mach_absolute_time();
    for (images, spans) |img, s| { // one image a pass: the tower's host-written tables are one image's
        try v.tower.image(eng.r.queue, img.pixels, img.gh, img.gw, .{ .buf = out });
        const src = out.contents();
        const dst = rows.contents();
        for (0..s.n) |k| for (0..4) |stream| @memcpy(dst[((s.row + k) * 4 + stream) * row ..][0..row], src[k * row ..][0..row]);
    }
    std.log.info("flash next vision: {d} images, {d} tokens through the tower in {d:.1} ms", .{ images.len, total, @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e3 });
    const p = try a.create(vp.VisPrompt);
    const last = spans[spans.len - 1];
    p.* = .{ .rows = rows, .spans = spans, .pos3 = try vp.frame(a, spans), .frame_end = last.at + last.n, .prep = v.prep.prep };
    return p;
}

/// The engine's prompt pass is over: its image rows go.
pub fn release(eng: *fx.Engine, p: ?*vp.VisPrompt) void {
    eng.m.vis = null;
    if (p) |q| q.rows.deinit();
}

/// Info.vision's prepare: an image's bytes to the tower's patches (core/qwen_vl geometry), on the HTTP thread.
pub fn prepare(v: *const Vision, io: std.Io, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const p = try vl.geometry.prepare(io, a, bytes, v.c, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}
