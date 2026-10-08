//! Development weights: a Python pack (the engine's prepared tensors, raw, with their index) loaded by name into an
//! api.Store, so the forward can run before the checkpoint loader exists and be checked against the same bytes.

const std = @import("std");
const cuda = @import("cuda");
const api = @import("api.zig");

/// ``bin``/``index``: the pack; ``ngram_json``/``ngram_root``: the n-gram geometry and where its shards live.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, bin: []const u8, index: []const u8, ngram_json: []const u8, ngram_root: []const u8) !api.Store {
    var store = api.Store.init(gpa);
    errdefer store.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const tensors = parsed.value.object.get("tensors").?.array.items;
    var total: u64 = 0;
    for (tensors) |tv| {
        const t = tv.object;
        if (t.get("host") != null) continue;
        total += std.mem.alignForward(u64, @intCast(t.get("bytes").?.integer), 256);
    }
    var all = try cuda.DeviceBuffer.alloc(d, total);
    errdefer all.free();
    try store.owned.append(gpa, all);
    const file = try std.Io.Dir.cwd().openFile(io, bin, .{});
    defer file.close(io);
    const host = try gpa.alloc(u8, 1 << 28);
    defer gpa.free(host);
    var at: u64 = 0;
    for (tensors) |tv| {
        const t = tv.object;
        if (t.get("host") != null) continue;
        const bytes: u64 = @intCast(t.get("bytes").?.integer);
        var src: u64 = @intCast(t.get("offset").?.integer);
        var left = bytes;
        var dst = all.ptr + at;
        while (left > 0) {
            const n = @min(left, host.len);
            if (try file.readPositionalAll(io, host[0..n], src) != n) return error.ShortRead;
            try d.check(d.api.cuMemcpyHtoD_v2(dst, host.ptr, n), "pack upload");
            left -= n;
            src += n;
            dst += n;
        }
        var tensor: api.Tensor = .{ .ptr = all.ptr + at, .dtype = api.parseDType(t.get("dtype").?.string) orelse return error.UnknownDType };
        const shape = t.get("shape").?.array.items;
        tensor.nd = @intCast(shape.len);
        for (shape, 0..) |s, i| tensor.shape[i] = s.integer;
        try store.put(t.get("name").?.string, tensor);
        at += std.mem.alignForward(u64, bytes, 256);
    }
    try store.host.put(try gpa.dupe(u8, "ngram.json"), try std.Io.Dir.cwd().readFileAlloc(io, ngram_json, gpa, .limited(1 << 24)));
    try store.host.put(try gpa.dupe(u8, "ngram_root"), try gpa.dupe(u8, ngram_root));
    return store;
}
