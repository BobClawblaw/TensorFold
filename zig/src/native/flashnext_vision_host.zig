//! Image input for Flash Next behind the native server (--vision, one Mac): the tower beside the replay engine, the
//! server's prepare hook (flashnext/image.zig), and each image request's prompt for the engine's prompt pass
//! (flashnext/vis_prompt.zig): the tower's rows over the image tokens and the multimodal positions.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const fnv = tf.flashnext_vision;
const vp = tf.flashnext_replay.vis_prompt;

/// <|vision_start|>, <|image_pad|> (the token the template writes once per image), <|vision_end|>.
pub const markers = [3]u32{ 248053, 248056, 248054 };
/// <|vision_start|>, <|video_pad|> (once per video, and each group's rows), <|vision_end|>.
pub const video_markers = [3]u32{ 248053, 248057, 248054 };

pub const Vis = struct {
    gpa: std.mem.Allocator,
    tower: *fnv.Vision,
    markers: [3]u32 = markers,
    video_markers: [3]u32 = video_markers,
    lim: api.VisionOpen,

    /// One image's tokens at most while a request's images fit --vision-image-tokens together (the Python lane's
    /// TOKENS_PER_IMAGE): the tower's scratch holds one such image.
    pub const image_most: u32 = 4096;

    /// The server's image hook: this tower's offer, `ctx` this Vis.
    pub fn offer(v: *Vis) api.Vision {
        // a video: frames in pairs, each "<X.Y seconds>", <|vision_start|>, its <|video_pad|> rows, <|vision_end|>
        const video: api.VideoOffer = .{ .prepare = prepareVideo, .video_token = video_markers[1], .group_open = video_markers[0], .group_close = video_markers[2], .group_token = video_markers[1], .time_first = true, .markers = &v.video_markers, .video_tokens = v.lim.video_tokens, .max_videos = v.lim.max_videos, .max_frames = v.lim.max_frames };
        // video only with libtfvideo beside the server: without it a video part is refused (400), not failed (500)
        return .{ .ctx = v, .prepare = prepare, .image_token = markers[1], .markers = &v.markers, .image_tokens = v.lim.image_tokens, .max_images = v.lim.max_images, .image_most = image_most, .video = if (api.tfvideo.version() != null) video else null };
    }

    pub fn deinit(v: *Vis) void {
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    /// The request's images through the tower: the prompt pass's image rows and positions (freed by `release`).
    pub fn build(v: *Vis, eng: *fx.Engine, a: std.mem.Allocator, images: []const api.Image, prompt_len: usize) !*vp.VisPrompt {
        return vp.build(v.tower, eng.r.device, eng.r.queue, a, images, prompt_len, .{});
    }
};

/// The tower for `dir` with scratch for one image (Vis.image_most tokens, fewer under a smaller budget); refused on a
/// speed-up pair.
pub fn attach(gpa: std.mem.Allocator, eng: *fx.Engine, dir: []const u8, lim: api.VisionOpen) !*Vis {
    if (eng.r.tp != null) return error.VisionOnPair;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const path = try std.fmt.allocPrintSentinel(arena.allocator(), "{s}/config.json", .{dir}, 0);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    try fnv.check(f.bytes[0..f.size], arena.allocator());
    const v = try gpa.create(Vis);
    errdefer gpa.destroy(v);
    const one = @min(lim.image_tokens, Vis.image_most);
    v.* = .{ .gpa = gpa, .tower = try fnv.Vision.load(gpa, eng.r.device, dir, one), .lim = lim };
    std.log.info("flash next: image tower loaded ({d:.2} GiB with scratch for an image of {d} tokens; {d} image tokens a request)", .{ @as(f64, @floatFromInt(v.tower.bytes)) / (1 << 30), one, lim.image_tokens });
    return v;
}

/// The engine's prompt pass is over: its image rows go.
pub fn release(eng: *fx.Engine, p: ?*vp.VisPrompt) void {
    eng.m.vis = null;
    if (p) |q| q.rows.deinit();
}

fn prepare(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const p = try tf.flashnext_image.prepare(a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}

fn prepareVideo(_: *anyopaque, a: std.mem.Allocator, src: api.VideoSource, max_tokens: u32, most: u32) anyerror!api.PreparedVideo {
    const fv = tf.flashnext_video;
    const rate: f64 = if (src.rate_den > 0) @as(f64, @floatFromInt(src.rate_num)) / @as(f64, @floatFromInt(src.rate_den)) else 0;
    const idx = try fv.sample(a, src.frames, rate, most);
    const size = try fv.smartResize(@intCast(idx.len), src.height, src.width);
    const gh = size[0] / fnv.patch;
    const gw = size[1] / fnv.patch;
    const per_group: u32 = gh * gw / fnv.patches_per_token;
    const n: usize = (idx.len + 1) / 2;
    if (per_group * n > max_tokens) return error.VideoTooLarge;
    var frames: fv.Frames = .{ .a = a, .w = size[1], .h = size[0], .px = try a.alloc([]u8, idx.len) };
    try src.decode(src.ctx, idx, .{ .ctx = &frames, .take = fv.Frames.take });
    const groups = try a.alloc(api.PreparedImage, n);
    for (groups, 0..) |*g, k| {
        const f0 = frames.px[2 * k];
        const f1 = if (2 * k + 1 < idx.len) frames.px[2 * k + 1] else frames.px[idx.len - 1]; // the last frame again
        const pixels = try fv.group(a, f0, f1, size[1], size[0]);
        g.* = .{ .pixels = pixels, .gh = gh, .gw = gw, .tokens = per_group, .hash = std.hash.Wyhash.hash(0x71de, std.mem.sliceAsBytes(pixels)) };
    }
    return .{ .groups = groups, .times = try fv.times(a, idx, rate) };
}
