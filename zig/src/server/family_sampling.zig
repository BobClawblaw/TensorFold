//! Served sampling: the family's defaults, generation_config.json, the serve flags; and the instruct set for thinking off.
const std = @import("std");
const json = @import("json");

/// The serve flags' sampling values; each set one replaces what the layers below gave.
pub const Flags = struct {
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    top_k: ?i64 = null,
    min_p: ?f64 = null,
};

/// The request fields --instruct-sampling may name.
pub const instruct_fields = [_][]const u8{ "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty" };

/// DeepSeek-V4.x: temperature 1.0, top_p 1.0, no top_k (the V4 and V4.1 model cards' recommended sampling).
pub fn put(a: std.mem.Allocator, out: *json.Object, model_type: []const u8) !void {
    if (!std.mem.eql(u8, model_type, "deepseek_v41") and !std.mem.eql(u8, model_type, "deepseek_v4")) return;
    try out.put(a, "temperature", .{ .float = 1.0 });
    try out.put(a, "top_p", .{ .float = 1.0 });
}

/// Qwen 3.5, Qwen3.8-27B and Flash-Next: the card gives a thinking set (generation_config.json) and an instruct set.
pub fn hasInstruct(model_type: []const u8) bool {
    for ([_][]const u8{ "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text" }) |t| {
        if (std.mem.eql(u8, model_type, t)) return true;
    }
    return false;
}

/// The Qwen cards' instruct set: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 where penalties apply.
pub fn putInstruct(a: std.mem.Allocator, out: *json.Object, model_type: []const u8, penalties: bool) !bool {
    if (!hasInstruct(model_type)) return false;
    try out.put(a, "temperature", .{ .float = 0.7 });
    try out.put(a, "top_p", .{ .float = 0.8 });
    try out.put(a, "top_k", try json.intValue(a, 20));
    if (penalties) try out.put(a, "presence_penalty", .{ .float = 1.5 });
    return true;
}

/// The three layers over each other (``do_sample`` false is greedy); null when none sets a value.
pub fn resolve(a: std.mem.Allocator, model_type: []const u8, generation_config: ?[]const u8, flags: Flags) !?json.Value {
    const out = try json.newObject(a);
    _ = try layers(a, out, model_type, generation_config, flags, null);
    return if (out.count() > 0) json.Value{ .object = out } else null;
}

/// Thinking-off defaults: `resolve`'s layers with the instruct set under the flags, then ``pairs``; null: as `resolve`.
pub fn resolveInstruct(a: std.mem.Allocator, model_type: []const u8, generation_config: ?[]const u8, flags: Flags, penalties: bool, pairs: ?[]const u8) !?json.Value {
    if (pairs) |p| if (std.mem.eql(u8, std.mem.trim(u8, p, " "), "none")) return null;
    const out = try json.newObject(a);
    var apart = try layers(a, out, model_type, generation_config, flags, penalties);
    if (pairs) |p| {
        try parsePairs(a, out, p);
        apart = true;
    }
    return if (apart) json.Value{ .object = out } else null;
}

/// ``--instruct-sampling``'s text, checked: ``none``, or name=value pairs `parsePairs` reads.
pub fn checkPairs(a: std.mem.Allocator, text: []const u8) error{ OutOfMemory, InvalidInstructSampling }!void {
    if (std.mem.eql(u8, std.mem.trim(u8, text, " "), "none")) return;
    const out = try json.newObject(a);
    try parsePairs(a, out, text);
}

/// Comma-separated ``name=value`` pairs of instruct fields (top_k an integer, the rest finite numbers) into ``out``.
pub fn parsePairs(a: std.mem.Allocator, out: *json.Object, text: []const u8) error{ OutOfMemory, InvalidInstructSampling }!void {
    var any = false;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |raw| {
        const pair = std.mem.trim(u8, raw, " ");
        if (pair.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse return error.InvalidInstructSampling;
        const name = std.mem.trim(u8, pair[0..eq], " ");
        const value = std.mem.trim(u8, pair[eq + 1 ..], " ");
        const known = for (instruct_fields) |f| {
            if (std.mem.eql(u8, f, name)) break true;
        } else false;
        if (!known) return error.InvalidInstructSampling;
        if (std.mem.eql(u8, name, "top_k")) {
            const k = std.fmt.parseInt(i64, value, 10) catch return error.InvalidInstructSampling;
            try out.put(a, name, try json.intValue(a, k));
        } else {
            const x = std.fmt.parseFloat(f64, value) catch return error.InvalidInstructSampling;
            if (!std.math.isFinite(x)) return error.InvalidInstructSampling;
            try out.put(a, name, .{ .float = x });
        }
        any = true;
    }
    if (!any) return error.InvalidInstructSampling;
}

/// Family defaults, generation_config.json, the instruct set when ``instruct`` (its penalties flag), then the flags.
fn layers(a: std.mem.Allocator, out: *json.Object, model_type: []const u8, generation_config: ?[]const u8, flags: Flags, instruct: ?bool) !bool {
    try put(a, out, model_type);
    if (generation_config) |bytes| if ((try json.parse(a, bytes)) == .ok) {
        const cfg = (try json.parse(a, bytes)).ok;
        for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
        if (cfg.get("do_sample")) |d| if (d == .bool) {
            if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
        };
    };
    const put_instruct = if (instruct) |penalties| try putInstruct(a, out, model_type, penalties) else false;
    if (flags.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (flags.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (flags.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (flags.min_p) |t| try out.put(a, "min_p", .{ .float = t });
    return put_instruct;
}

test "DeepSeek-V4.x serves its cards' sampling defaults; other families keep none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "deepseek_v41", "deepseek_v4" }) |kind| {
        const out = (try resolve(a, kind, null, .{})).?;
        try std.testing.expectEqual(@as(f64, 1.0), out.get("temperature").?.float);
        try std.testing.expectEqual(@as(f64, 1.0), out.get("top_p").?.float);
        try std.testing.expect(out.get("top_k") == null);
    }
    try std.testing.expect((try resolve(a, "glm5_next", null, .{})) == null);
}

test "family sampling defaults sit under generation_config.json and the serve flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try resolve(a, "deepseek_v41", "{\"temperature\": 0.6, \"top_k\": 40}", .{});
    try std.testing.expectEqual(@as(f64, 0.6), cfg.?.get("temperature").?.float);
    try std.testing.expectEqual(@as(?i64, 40), cfg.?.get("top_k").?.int64());
    try std.testing.expectEqual(@as(f64, 1.0), cfg.?.get("top_p").?.float);
    const flag = try resolve(a, "deepseek_v41", "{\"temperature\": 0.6}", .{ .temperature = 0.3, .top_p = 0.9 });
    try std.testing.expectEqual(@as(f64, 0.3), flag.?.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 0.9), flag.?.get("top_p").?.float);
    const greedy = try resolve(a, "deepseek_v41", "{\"do_sample\": false}", .{});
    try std.testing.expectEqual(@as(f64, 0), greedy.?.get("temperature").?.float);
}

test "Qwen 3.5 and 3.8 serve their cards' instruct set with thinking off, under the flags and --instruct-sampling" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"do_sample\": true, \"temperature\": 1.0, \"top_k\": 20, \"top_p\": 0.95}";
    for ([_][]const u8{ "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text" }) |kind| {
        const thinking = (try resolve(a, kind, cfg, .{})).?;
        try std.testing.expectEqual(@as(f64, 1.0), thinking.get("temperature").?.float);
        try std.testing.expect(thinking.get("presence_penalty") == null);
        const out = (try resolveInstruct(a, kind, cfg, .{}, true, null)).?;
        try std.testing.expectEqual(@as(f64, 0.7), out.get("temperature").?.float);
        try std.testing.expectEqual(@as(f64, 0.8), out.get("top_p").?.float);
        try std.testing.expectEqual(@as(?i64, 20), out.get("top_k").?.int64());
        try std.testing.expectEqual(@as(f64, 1.5), out.get("presence_penalty").?.float);
        try std.testing.expect(out.get("frequency_penalty") == null);
    }
    // an engine without penalties gets the set without presence_penalty
    const plain = (try resolveInstruct(a, "qwen3_5", cfg, .{}, false, null)).?;
    try std.testing.expectEqual(@as(f64, 0.7), plain.get("temperature").?.float);
    try std.testing.expect(plain.get("presence_penalty") == null);
    // the serve flags apply to both modes
    const flagged = (try resolveInstruct(a, "qwen3_5", cfg, .{ .temperature = 0, .top_k = 40 }, true, null)).?;
    try std.testing.expectEqual(@as(f64, 0), flagged.get("temperature").?.float);
    try std.testing.expectEqual(@as(?i64, 40), flagged.get("top_k").?.int64());
    try std.testing.expectEqual(@as(f64, 1.5), flagged.get("presence_penalty").?.float);
    // --instruct-sampling pairs win for thinking off only; none keeps one set for every request
    const pairs = (try resolveInstruct(a, "qwen3_5", cfg, .{ .temperature = 0 }, true, "temperature=0.6, presence_penalty=0,top_k=10")).?;
    try std.testing.expectEqual(@as(f64, 0.6), pairs.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 0), pairs.get("presence_penalty").?.float);
    try std.testing.expectEqual(@as(?i64, 10), pairs.get("top_k").?.int64());
    try std.testing.expectEqual(@as(f64, 0.8), pairs.get("top_p").?.float);
    try std.testing.expect((try resolveInstruct(a, "qwen3_5", cfg, .{}, true, "none")) == null);
    // other families have no instruct set unless the flag gives one
    try std.testing.expect((try resolveInstruct(a, "deepseek_v41", null, .{}, true, null)) == null);
    try std.testing.expect((try resolveInstruct(a, "glm5_next", cfg, .{}, true, null)) == null);
    const given = (try resolveInstruct(a, "glm5_next", cfg, .{}, true, "presence_penalty=1")).?;
    try std.testing.expectEqual(@as(f64, 1.0), given.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 1.0), given.get("presence_penalty").?.float);
    // malformed pairs
    for ([_][]const u8{ "", " ", "temperature", "temperature=x", "foo=1", "top_k=1.5", "temperature=inf", "=0.5" }) |bad| {
        try std.testing.expectError(error.InvalidInstructSampling, resolveInstruct(a, "qwen3_5", cfg, .{}, true, bad));
        try std.testing.expectError(error.InvalidInstructSampling, checkPairs(a, bad));
    }
    try checkPairs(a, "none");
    try checkPairs(a, "temperature=0.7,top_p=0.8,top_k=20,min_p=0,presence_penalty=1.5,frequency_penalty=0");
}
