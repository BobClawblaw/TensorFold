//! MOSS-Image's prompts through the checkpoint's own encoder, for tf-moss-image and tf-moss-image-generate.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;
const tokenizer = @import("tokenizer");

/// Each text's rows (rows, 4096) bf16 from the encoder in `checkpoint`, which is freed before this returns.
pub fn encode(a: std.mem.Allocator, io: std.Io, device: mtl.Device, checkpoint: []const u8, texts: []const []const u8, rows: [][]u16) !f64 {
    const enc = try mi.Encoder.load(std.heap.page_allocator, io, device, checkpoint);
    defer enc.deinit();
    var tok = try tokenizer.loadTokenizer(io, a, try std.fs.path.join(a, &.{ checkpoint, "tokenizer", "tokenizer.json" }));
    defer tok.deinit();
    const encoding = mtl.clock.seconds();
    for (texts, rows) |t, *m| {
        const ids = try mi.encoder.ids(&tok, a, t);
        m.* = try a.alloc(u16, ids.len * mi.encoder_config.hidden);
        try enc.rows(ids, m.*, null);
    }
    return mtl.clock.seconds() - encoding;
}

/// A prompt file's text, trimmed as the case tool trims it.
pub fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return std.mem.trim(u8, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)), " \t\r\n");
}
