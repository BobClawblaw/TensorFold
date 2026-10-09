//! Image inputs of a chat request, as the Python server's image frontend takes them (tensorfold/vision/images.py,
//! qwen_processing.py): each ``image_url`` part (a data: URL; JPEG or PNG) decoded and prepared, and replaced in its
//! message by the text the chat template writes for an image (``<|vision_start|><|image_pad|><|vision_end|>``); the
//! rendered prompt's image pads are then expanded to each image's visual tokens (``expand``).

const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const qi = api.qwen_image;

pub const placeholder = "<|vision_start|><|image_pad|><|vision_end|>";
const max_bytes_one: usize = 10 * 1024 * 1024;
const max_bytes_all: usize = 20 * 1024 * 1024;
const max_pixels_all: u64 = 32 * 1024 * 1024;
const max_tokens_one: u64 = 4096;

pub const Extracted = struct { messages: Value, images: []api.Image };

/// The image URL of a content part (an OpenAI ``image_url`` part, or ``image`` / ``input_image`` spellings) and its
/// ``detail``; null for a part that is not an image.
fn imagePart(part: Value) ?struct { url: ?[]const u8, detail: []const u8 } {
    if (part != .object) return null;
    const ty = part.get("type");
    const is_image = (ty != null and ty.? == .string and (std.mem.eql(u8, ty.?.string, "image_url") or std.mem.eql(u8, ty.?.string, "image") or
        std.mem.eql(u8, ty.?.string, "input_image"))) or part.get("image_url") != null or part.get("image") != null;
    if (!is_image) return null;
    var url: ?[]const u8 = null;
    var detail: []const u8 = "auto";
    if (part.get("image_url")) |iu| switch (iu) {
        .string => |s| url = s,
        .object => {
            if (iu.get("url")) |u| if (u == .string) {
                url = u.string;
            };
            if (iu.get("detail")) |d| if (d == .string) {
                detail = d.string;
            };
        },
        else => {},
    };
    if (url == null) if (part.get("image")) |im| if (im == .string) {
        url = im.string;
    };
    if (part.get("detail")) |d| if (d == .string) {
        detail = d.string;
    };
    return .{ .url = url, .detail = detail };
}

fn hasImages(messages: Value) bool {
    if (messages != .array) return false;
    for (messages.array) |m| {
        if (m != .object) continue;
        const c = m.get("content") orelse continue;
        if (c != .array) continue;
        for (c.array) |p| if (imagePart(p) != null) return true;
    }
    return false;
}

/// The bytes of a ``data:<type>;base64,<data>`` URL.
fn dataBytes(cx: *Cx, url: []const u8) errors.Refused![]u8 {
    if (std.mem.startsWith(u8, url, "http://") or std.mem.startsWith(u8, url, "https://"))
        return cx.refuse("image URLs are not fetched by this server; send the image as a data: URL (base64)");
    if (!std.mem.startsWith(u8, url, "data:")) return cx.refuse("an image must be a data: URL with base64 bytes");
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return cx.refuse("an image data: URL needs a comma before its bytes");
    if (!std.mem.endsWith(u8, url[0..comma], ";base64")) return cx.refuse("an image data: URL must be base64-encoded");
    const b64 = url[comma + 1 ..];
    const dec = std.base64.standard.Decoder;
    const n = dec.calcSizeForSlice(b64) catch return cx.refuse("an image's base64 bytes are invalid");
    if (n > max_bytes_one) return cx.refuse("an image exceeds the 10 MiB limit");
    const out = try cx.a.alloc(u8, n);
    dec.decode(out, b64) catch return cx.refuse("an image's base64 bytes are invalid");
    return out;
}

/// A request's images extracted and prepared, its messages with their image parts as the template's image text;
/// null when it has none. A server without --vision refuses images.
pub fn extract(srv: *Server, cx: *Cx, messages: ?Value) errors.Refused!?Extracted {
    const list = messages orelse return null;
    if (!hasImages(list)) return null;
    if (!srv.info.vision) return cx.refuse("image inputs need a server started with --vision");
    var urls: std.ArrayList(struct { url: ?[]const u8, detail: []const u8 }) = .empty;
    const out = try cx.a.alloc(Value, list.array.len);
    for (list.array, 0..) |m, i| {
        out[i] = m;
        if (m != .object) continue;
        const c = m.get("content") orelse continue;
        if (c != .array) continue;
        const role = m.get("role");
        var parts = try cx.a.alloc(Value, c.array.len);
        var any = false;
        for (c.array, 0..) |p, j| {
            parts[j] = p;
            const im = imagePart(p) orelse continue;
            if (role != null and role.? == .string and (std.mem.eql(u8, role.?.string, "system") or std.mem.eql(u8, role.?.string, "developer")))
                return cx.refuse("a system message cannot contain images");
            try urls.append(cx.a, .{ .url = im.url, .detail = im.detail });
            const t = try json.newObject(cx.a);
            try t.put(cx.a, "type", .{ .string = "text" });
            try t.put(cx.a, "text", .{ .string = placeholder });
            parts[j] = .{ .object = t };
            any = true;
        }
        if (any) {
            const copy = try json.copyObject(cx.a, m.object);
            try copy.put(cx.a, "content", .{ .array = parts });
            out[i] = .{ .object = copy };
        }
    }
    const n = urls.items.len;
    if (n > srv.config.vision_max_images) return cx.fail(.request, "a request may carry at most {d} images; this one has {d}", .{ srv.config.vision_max_images, n });
    const budget: u64 = srv.config.vision_image_tokens;
    if (budget < n) return cx.refuse("the image count exceeds the visual-token budget");
    const per: u64 = @min(budget / n, max_tokens_one);
    const images = try cx.a.alloc(api.Image, n);
    var total_bytes: usize = 0;
    var total_pixels: u64 = 0;
    for (urls.items, 0..) |u, k| {
        const bytes = try dataBytes(cx, u.url orelse return cx.refuse("an image part needs an image_url with a url"));
        total_bytes += bytes.len;
        if (total_bytes > max_bytes_all) return cx.refuse("a request's images exceed the 20 MiB limit");
        var img = qi.decode(cx.a, bytes, .{}) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ImageTooLarge => return cx.refuse("image dimensions exceed the decoded pixel limit"),
            else => return cx.refuse("image bytes are invalid or unsupported; use JPEG or PNG"),
        };
        total_pixels += @as(u64, img.w) * img.h;
        if (total_pixels > max_pixels_all) return cx.refuse("a request's images exceed the decoded pixel limit");
        const cap = if (std.mem.eql(u8, u.detail, "low")) @min(per, 256) else per;
        const p = qi.prepare(cx.a, img, cap) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return cx.refuse("absolute aspect ratio must be smaller than 200"),
        };
        img.deinit(cx.a);
        images[k] = .{ .patches = p.patches, .grid = p.grid };
    }
    return .{ .messages = .{ .array = out }, .images = images };
}

/// The rendered prompt with each image pad repeated to its image's visual tokens (the pads in prompt order).
pub fn expand(srv: *Server, cx: *Cx, ids: []const u32, images: []const api.Image) errors.Refused![]const u32 {
    const pad_ids = srv.text.encode(cx.a, "<|image_pad|>", false) catch return cx.fail(.server, "the tokenizer has no <|image_pad|> token", .{});
    if (pad_ids.len != 1) return cx.fail(.server, "the tokenizer has no single <|image_pad|> token", .{});
    const pad = pad_ids[0];
    var out: std.ArrayList(u32) = .empty;
    var k: usize = 0;
    for (ids) |t| {
        if (t != pad) {
            try out.append(cx.a, t);
            continue;
        }
        if (k >= images.len) return cx.fail(.server, "the rendered prompt has more image pads than images", .{});
        const g = images[k].grid;
        const n: usize = @intCast(@divExact(g[1] * g[2], 4));
        try out.appendNTimes(cx.a, pad, n);
        k += 1;
    }
    if (k != images.len) return cx.fail(.server, "the rendered prompt has {d} image pads for {d} images", .{ k, images.len });
    return out.items;
}
