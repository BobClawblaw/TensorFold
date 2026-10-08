//! Admission for the Qwen3.5-family MLX affine checkpoints the native engine is qualified on: the tied 2B and the 27B.
const std = @import("std");

pub const vocab = 248320;
pub const linear_dim = 128;
pub const conv_taps = 4;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const group = 64;
pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000000;

/// A qualified checkpoint shape: everything the kernels and buffers size by, read from config.json and matched here.
pub const Geometry = struct {
    name: []const u8,
    hidden: usize,
    intermediate: usize,
    layers: usize,
    linear_k_heads: usize, // DeltaNet key (and query) heads
    linear_v_heads: usize, // DeltaNet value heads: a multiple of the key heads, each key head serving a run of them
    query_heads: usize,
    kv_heads: usize,
    tied: bool, // the output projection is the packed input embedding

    /// The DeltaNet input projection's row: queries, keys, then values.
    pub fn convDim(g: Geometry) usize {
        return (2 * g.linear_k_heads + g.linear_v_heads) * linear_dim;
    }
    pub fn kInner(g: Geometry) usize {
        return g.linear_k_heads * linear_dim;
    }
    pub fn vInner(g: Geometry) usize {
        return g.linear_v_heads * linear_dim;
    }
    pub fn qInner(g: Geometry) usize {
        return g.query_heads * head_dim;
    }
    pub fn kvInner(g: Geometry) usize {
        return g.kv_heads * head_dim;
    }
    /// The widest row an activation buffer holds: the recurrence's values or the attention's heads.
    pub fn inner(g: Geometry) usize {
        return @max(g.vInner(), g.qInner());
    }
    /// One layer's recurrent state (fp32) and its conv window (bf16).
    pub fn deltaBytes(g: Geometry) usize {
        return g.linear_v_heads * linear_dim * linear_dim * 4;
    }
    pub fn convBytes(g: Geometry) usize {
        return (conv_taps - 1) * g.convDim() * 2;
    }
};

pub const geometries = [_]Geometry{
    .{ .name = "Qwen3.5-2B", .hidden = 2048, .intermediate = 6144, .layers = 24, .linear_k_heads = 16, .linear_v_heads = 16, .query_heads = 8, .kv_heads = 2, .tied = true },
    .{ .name = "Qwen3.8-27B", .hidden = 5120, .intermediate = 17408, .layers = 64, .linear_k_heads = 16, .linear_v_heads = 48, .query_heads = 24, .kv_heads = 4, .tied = false },
};

pub const Config = struct {
    context: usize,
    g: Geometry,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

pub fn linear(index: usize) bool {
    return index % 4 != 3;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadQwenConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadQwenConfig;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadQwenConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 1) return error.BadQwenConfig;
    return @intCast(v.integer);
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedQwenConfig;
}

fn boolean(o: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const v = try field(o, key);
    if (v != .bool or v.bool != expected) return error.UnsupportedQwenConfig;
}

fn quantization(v: std.json.Value) !void {
    const o = try object(v);
    if (try integer(o, "bits") != 4 or try integer(o, "group_size") != group) return error.UnsupportedQwenQuantization;
    if (o.get("mode")) |mode| {
        if (mode != .string or !std.mem.eql(u8, mode.string, "affine")) return error.UnsupportedQwenQuantization;
    }
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size") and !std.mem.eql(u8, k, "mode"))
            return error.UnsupportedQwenQuantization;
    }
}

/// The qualified geometry whose dimensions the config names, else UnsupportedQwenGeometry.
fn geometry(text: std.json.ObjectMap) !Geometry {
    const fixed = .{
        .{ "vocab_size", vocab },                 .{ "linear_key_head_dim", linear_dim }, .{ "linear_value_head_dim", linear_dim },
        .{ "linear_conv_kernel_dim", conv_taps }, .{ "head_dim", head_dim },              .{ "full_attention_interval", 4 },
    };
    inline for (fixed) |d| if (try integer(text, d[0]) != d[1]) return error.UnsupportedQwenGeometry;
    const hidden = try integer(text, "hidden_size");
    const intermediate = try integer(text, "intermediate_size");
    const layers = try integer(text, "num_hidden_layers");
    const k_heads = try integer(text, "linear_num_key_heads");
    const v_heads = try integer(text, "linear_num_value_heads");
    const query_heads = try integer(text, "num_attention_heads");
    const kv_heads = try integer(text, "num_key_value_heads");
    for (geometries) |g| {
        if (g.hidden == hidden and g.intermediate == intermediate and g.layers == layers and g.linear_k_heads == k_heads and
            g.linear_v_heads == v_heads and g.query_heads == query_heads and g.kv_heads == kv_heads) return g;
    }
    return error.UnsupportedQwenGeometry;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    try string(root, "model_type", "qwen3_5");
    const text = try object(try field(root, "text_config"));
    try string(text, "model_type", "qwen3_5_text");
    const g = try geometry(text);
    try boolean(text, "tie_word_embeddings", g.tied);
    if (root.get("tie_word_embeddings") != null) try boolean(root, "tie_word_embeddings", g.tied);
    try boolean(text, "attention_bias", false);
    try boolean(text, "attn_output_gate", true);
    try string(text, "hidden_act", "silu");
    try string(text, "mamba_ssm_dtype", "float32");
    if (try number(try field(text, "rms_norm_eps")) != 1e-6) return error.UnsupportedQwenConfig;
    const types = try field(text, "layer_types");
    if (types != .array or types.array.items.len != g.layers) return error.UnsupportedQwenGeometry;
    for (types.array.items, 0..) |kind, i| {
        if (kind != .string or !std.mem.eql(u8, kind.string, if (linear(i)) "linear_attention" else "full_attention"))
            return error.UnsupportedQwenGeometry;
    }
    const rope = try object(try field(text, "rope_parameters"));
    try string(rope, "rope_type", "default");
    try boolean(rope, "mrope_interleaved", true);
    if (try number(try field(rope, "rope_theta")) != theta or
        try number(try field(rope, "partial_rotary_factor")) != 0.25) return error.UnsupportedQwenRotary;
    const sections = try field(rope, "mrope_section");
    if (sections != .array or sections.array.items.len != 3) return error.UnsupportedQwenRotary;
    for (sections.array.items, [_]i64{ 11, 11, 10 }) |s, expected| {
        if (s != .integer or s.integer != expected) return error.UnsupportedQwenRotary;
    }
    try quantization(root.get("quantization") orelse root.get("quantization_config") orelse return error.UnsupportedQwenQuantization);
    if (root.get("quantization_config")) |q| try quantization(q);
    const context = try integer(text, "max_position_embeddings");
    if (context > 262144) return error.UnsupportedQwenConfig;
    return .{ .context = context, .g = g };
}

const fixture_head =
    \\{
    \\  "model_type": "qwen3_5",
    \\  "text_config": {
    \\    "attention_bias": false,
    \\    "attention_dropout": 0.0,
    \\    "attn_output_gate": true,
    \\    "dtype": "bfloat16",
    \\    "eos_token_id": 248044,
    \\    "full_attention_interval": 4,
    \\    "head_dim": 256,
    \\    "hidden_act": "silu",
    \\    "initializer_range": 0.02,
    \\    "linear_conv_kernel_dim": 4,
    \\    "linear_key_head_dim": 128,
    \\    "linear_value_head_dim": 128,
    \\    "max_position_embeddings": 262144,
    \\    "mlp_only_layers": [],
    \\    "model_type": "qwen3_5_text",
    \\    "mtp_num_hidden_layers": 1,
    \\    "mtp_use_dedicated_embeddings": false,
    \\    "rms_norm_eps": 1e-06,
    \\    "use_cache": true,
    \\    "vocab_size": 248320,
    \\    "mamba_ssm_dtype": "float32",
    \\    "rope_parameters": {
    \\      "mrope_interleaved": true,
    \\      "mrope_section": [11, 11, 10],
    \\      "rope_type": "default",
    \\      "rope_theta": 10000000,
    \\      "partial_rotary_factor": 0.25
    \\    },
;

const fixture_tail =
    \\  },
    \\  "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
    \\  "quantization_config": { "group_size": 64, "bits": 4, "mode": "affine" },
;

/// A config.json for geometry `g`, as the two checkpoints' read (minus their vision and generation fields).
fn fixture(gpa: std.mem.Allocator, g: Geometry) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll(fixture_head);
    try w.print("    \"hidden_size\": {d},\n    \"intermediate_size\": {d},\n    \"num_hidden_layers\": {d},\n", .{ g.hidden, g.intermediate, g.layers });
    try w.print("    \"linear_num_key_heads\": {d},\n    \"linear_num_value_heads\": {d},\n", .{ g.linear_k_heads, g.linear_v_heads });
    try w.print("    \"num_attention_heads\": {d},\n    \"num_key_value_heads\": {d},\n    \"tie_word_embeddings\": {},\n", .{ g.query_heads, g.kv_heads, g.tied });
    try w.writeAll("    \"layer_types\": [");
    for (0..g.layers) |i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ", ", if (linear(i)) "linear_attention" else "full_attention" });
    try w.writeAll("]\n");
    try w.writeAll(fixture_tail);
    try w.print("  \"tie_word_embeddings\": {}\n}}\n", .{g.tied});
    return out.toOwnedSlice();
}

test "admit the two qualified geometries, text rotary layout and affine 4-bit format" {
    const gpa = std.testing.allocator;
    for (geometries) |g| {
        const text = try fixture(gpa, g);
        defer gpa.free(text);
        const config = try parse(gpa, text);
        try std.testing.expectEqual(@as(usize, 262144), config.context);
        try std.testing.expectEqualStrings(g.name, config.g.name);
        try std.testing.expectEqual(g.hidden, config.g.hidden);
        try std.testing.expectEqual(g.tied, config.g.tied);
    }
    const text = try fixture(gpa, geometries[0]);
    defer gpa.free(text);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    const compact = try std.json.Stringify.valueAlloc(gpa, doc.value, .{});
    defer gpa.free(compact);
    const changes = .{
        .{ "\"hidden_size\":2048", "\"hidden_size\":1024", error.UnsupportedQwenGeometry },
        .{ "\"linear_num_value_heads\":16", "\"linear_num_value_heads\":48", error.UnsupportedQwenGeometry },
        .{ "\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false", error.UnsupportedQwenConfig },
        .{ "\"bits\":4", "\"bits\":8", error.UnsupportedQwenQuantization },
        .{ "\"group_size\":64", "\"group_size\":32", error.UnsupportedQwenQuantization },
        .{ "\"mode\":\"affine\"", "\"mode\":\"mxfp4\"", error.UnsupportedQwenQuantization },
        .{ "\"rope_theta\":10000000", "\"rope_theta\":100000", error.UnsupportedQwenRotary },
        .{ "\"mrope_section\":[11,11,10]", "\"mrope_section\":[16,8,8]", error.UnsupportedQwenRotary },
        .{ "\"max_position_embeddings\":262144", "\"max_position_embeddings\":524288", error.UnsupportedQwenConfig },
    };
    inline for (changes) |change| {
        try std.testing.expect(std.mem.indexOf(u8, compact, change[0]) != null);
        const bad = try std.mem.replaceOwned(u8, gpa, compact, change[0], change[1]);
        defer gpa.free(bad);
        try std.testing.expectError(change[2], parse(gpa, bad));
    }
}

test "the geometries' derived rows match the checkpoints' tensors" {
    const two = geometries[0];
    try std.testing.expectEqual(@as(usize, 6144), two.convDim());
    try std.testing.expectEqual(@as(usize, 2048), two.vInner());
    try std.testing.expectEqual(@as(usize, 2048), two.qInner());
    const big = geometries[1];
    try std.testing.expectEqual(@as(usize, 10240), big.convDim());
    try std.testing.expectEqual(@as(usize, 6144), big.vInner());
    try std.testing.expectEqual(@as(usize, 6144), big.qInner());
    try std.testing.expectEqual(@as(usize, 1024), big.kvInner());
    try std.testing.expectEqual(@as(usize, 48 * 128 * 128 * 4), big.deltaBytes());
}
