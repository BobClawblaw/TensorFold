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

pub const Vis = struct {
    gpa: std.mem.Allocator,
    tower: *fnv.Vision,
    markers: [3]u32 = markers,
    lim: api.VisionOpen,

    /// The server's image hook: this tower's offer, `ctx` this Vis.
    pub fn offer(v: *Vis) api.Vision {
        return .{ .ctx = v, .prepare = prepare, .image_token = markers[1], .markers = &v.markers, .image_tokens = v.lim.image_tokens, .max_images = v.lim.max_images };
    }

    pub fn deinit(v: *Vis) void {
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    /// The request's images through the tower: the prompt pass's image rows and positions (freed by `release`).
    pub fn build(v: *Vis, eng: *fx.Engine, a: std.mem.Allocator, images: []const api.Image, prompt_len: usize) !*vp.VisPrompt {
        return vp.build(v.tower, eng.r.device, eng.r.queue, a, images, prompt_len);
    }
};

/// The tower for `dir` with scratch for `lim.image_tokens` tokens at once; refused on a speed-up pair.
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
    v.* = .{ .gpa = gpa, .tower = try fnv.Vision.load(gpa, eng.r.device, dir, lim.image_tokens), .lim = lim };
    std.log.info("flash next: image tower loaded ({d:.2} GiB with scratch for {d} image tokens a request)", .{ @as(f64, @floatFromInt(v.tower.bytes)) / (1 << 30), lim.image_tokens });
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
