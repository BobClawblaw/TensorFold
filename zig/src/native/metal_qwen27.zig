//! The 27B engine on Metal: the qwen3_5 checkpoints it opens (metal.zig adds every qwen3_5_moe), and its open.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const Allocator = std.mem.Allocator;
const qwen27 = @import("qwen27_served.zig");
const metal = @import("metal.zig");
const metal_spill = @import("metal_spill.zig");

/// A qwen3_5 checkpoint opens the 27B engine unless it has the pinned 2B's hidden size or a tied head (the 2B's).
pub fn engine(a: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return true;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return true;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return true;
    if (doc != .object) return true;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    for ([_]std.json.Value{ doc, text }) |o| {
        if (o.object.get("tie_word_embeddings")) |v| if (v == .bool and v.bool) return false;
        if (o.object.get("hidden_size")) |v| if (v == .integer and v.integer == tf.qwen35.config.hidden) return false;
    }
    return true;
}

/// Qwen3.8-27B on the lane host: `--parallel` streams in shared rounds, DFlash2 drafts from --drafter for one of them.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const context = o.context orelse 32768;
    if (context <= 0 or context > 262144) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside this Qwen checkpoint's 1 to 262,144 tokens", .{context});
        return null;
    }
    if (o.drafts and o.drafter != null and o.drafter_bits != 0 and o.drafter_bits != 4) {
        problem.* = "--drafter-bits takes 0 (bf16) or 4";
        return null;
    }
    var why: []const u8 = "";
    const drafter = if (o.drafts) o.drafter else null;
    const learn: ?@import("qwen27_disk.zig").Learn = if (o.learn_conversations) if (o.learn) |root| .{ .root = root, .dir = o.dir, .drafter = drafter, .cap = metal.gibBytes(o.learn_gib), .floor = metal.gibBytes(o.learn_min_free_gib), .daily = metal.gibBytes(o.learn_daily_gib), .spill = o.spill, .slots = metal_spill.slots(o) } else null else null;
    const h = qwen27.open(gpa, io, o.dir, .{ .context = @intCast(context), .streams = o.lanes, .streams_fixed = o.lanes_fixed, .drafter = drafter, .drafter_bits = o.drafter_bits, .cache_gib = o.prompt_cache_gib, .cache_over_cap = o.prompt_cache_over_cap, .learn = learn, .vision = o.vision }, a, &why) catch |err| {
        problem.* = if (why.len > 0) why else try std.fmt.allocPrint(a, "the native Qwen engine (27B class) cannot load {s} ({s})", .{ o.dir, @errorName(err) });
        return null;
    };
    return .{ .engine = h.engine(), .close = qwen27.close, .ctx = h };
}

test "a qwen3_5 checkpoint opens the 27B engine unless it has the 2B's geometry or a tied head" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expect(engine(a, io, dir)); // no config.json: the 27B's loader explains the refusal
    const cases = [_]struct { config: []const u8, big: bool }{
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":true}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"text_config\":{}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{\"tie_word_embeddings\":false}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":false,\"hidden_size\":2048}}", .big = false },
    };
    for (cases) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.config });
        try std.testing.expectEqual(c.big, engine(a, io, dir));
    }
}
