//! A TP rank's MiniMax-M3.1-Flash weights: MXFP8 or NVFP4 lanes, NVFP4 or 3-bit experts, a bf16 or MXFP8 head.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const expert_load = @import("cuda_expert_load.zig");

const Tensor = core.checkpoint.Tensor;
pub const Linear = cuda.qmmf.Weight;

/// The checkpoint's text tensors sit under this prefix; the vision tower beside it.
pub const prefix = "language_model.";

/// q | k | v | index q | index k of this rank's heads stacked (index k whole), o by columns, the head norms.
pub const Attention = struct { qkv: Linear, o: Linear, q_norm: u64, k_norm: u64, iq_norm: u64, ik_norm: u64 };
/// gate | up of the rank's intermediate rows stacked, down by its columns.
pub const Mlp = struct { gu: Linear, down: Linear };
/// The fp32 router [experts][hidden] and its score bias, the rank's share of the routed experts, the shared expert.
pub const Moe = struct { router: u64, bias: u64, routed: cuda.experts.Layer, shared: Mlp };
pub const Layer = struct { in_norm: u64, post_norm: u64, attn: Attention, ffn: union(enum) { dense: Mlp, moe: Moe } };

/// config.json's nvfp4_mix: the MoE layers with NVFP4 experts (or TF_MINIMAX_NVFP4_LAYERS), the converted formats.
pub const Mix = struct {
    nvfp4: [cfg.max_layers]bool = @splat(false),
    fp4_qkv: bool = false,
    fp4_o: bool = false,
    fp4_mlp: bool = false,
    fp8_head: bool = false,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const cfg.Config, override: ?[]const u8) !Mix {
        var m: Mix = .{};
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        const mix = doc.value.object.get("nvfp4_mix") orelse return if (override) |text| m.parse(text, c) else error.NoMix;
        if (mix.object.get("dense_fp4")) |kinds| {
            if (kinds != .array) return error.BadMix;
            for (kinds.array.items) |v| {
                if (v != .string) return error.BadMix;
                if (std.mem.eql(u8, v.string, "qkv")) m.fp4_qkv = true else if (std.mem.eql(u8, v.string, "o")) m.fp4_o = true else if (std.mem.eql(u8, v.string, "mlp")) m.fp4_mlp = true else return error.BadMix;
            }
        }
        if (mix.object.get("head_format")) |v| {
            if (v != .string) return error.BadMix;
            if (std.mem.eql(u8, v.string, "mxfp8")) m.fp8_head = true else if (!std.mem.eql(u8, v.string, "bf16")) return error.BadMix;
        }
        if (override) |text| return m.parse(text, c);
        const list = (mix.object.get("nvfp4_layers") orelse return error.NoMix).array;
        for (list.items) |v| {
            if (v != .integer or v.integer < 0 or v.integer >= c.layers) return error.BadMix;
            m.nvfp4[@intCast(v.integer)] = true;
        }
        return m;
    }

    fn parse(m: *Mix, text: []const u8, c: *const cfg.Config) !Mix {
        if (std.mem.eql(u8, text, "all")) {
            for (0..c.layers) |i| m.nvfp4[i] = c.layer[i].moe;
            return m.*;
        }
        if (std.mem.eql(u8, text, "none")) return m.*;
        var it = std.mem.splitScalar(u8, text, ',');
        while (it.next()) |part| {
            if (part.len == 0) continue;
            const dash = std.mem.indexOfScalar(u8, part, '-');
            const lo = try std.fmt.parseInt(usize, if (dash) |d| part[0..d] else part, 10);
            const hi = if (dash) |d| try std.fmt.parseInt(usize, part[d + 1 ..], 10) else lo;
            if (hi < lo or hi >= c.layers) return error.BadMix;
            for (lo..hi + 1) |i| m.nvfp4[i] = true;
        }
        return m.*;
    }

    pub fn count(m: *const Mix, c: *const cfg.Config) usize {
        var n: usize = 0;
        for (0..c.layers) |i| n += @intFromBool(m.nvfp4[i]);
        return n;
    }
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    share: shard.Share,
    mix: Mix,
    embed: u64 = 0, // bf16 [vocab][hidden], whole on every rank
    norm: u64 = 0, // the final norm, bf16 [hidden]
    head: Linear = undefined, // bf16 or MXFP8 as stored: this rank's vocabulary rows
    fp4_projections: u32 = 0, // decoder projections stored NVFP4
    fp8_projections: u32 = 0, // and MXFP8
    fp8_head: bool = false,
    layers: []Layer = &.{},
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0, // device bytes the weights hold

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    /// Whether the decoder's projections are all NVFP4 and the head MXFP8 (the dense FP4 conversion).
    pub fn denseFp4(w: *const Weights) bool {
        return w.fp4_projections > 0 and w.fp8_projections == 0 and w.fp8_head;
    }
};

/// What a load runs on the device: the stream its uploads and packs go on, the expert packers.
pub const Tools = struct { d: *const cuda.Driver, stream: cuda.Stream, a3: *const cuda.experts3.Affine3Experts, a4: *const cuda.experts.Affine4Experts };

/// Rows [lo, hi) and inputs [klo, khi) of a stored MXFP8 projection, one part of a stacked lane weight.
pub const MxPart = struct { name: []const u8, rows: ?[2]usize = null, cols: ?[2]usize = null };

pub const Loader = struct {
    gpa: std.mem.Allocator,
    t: Tools,
    w: *Weights,
    ck: *core.Checkpoint,
    host: std.ArrayList(u8) = .empty,
    name_buf: [256]u8 = undefined,

    /// A device copy of `bytes`, owned by the weights.
    pub fn put(L: *Loader, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(L.t.d, bytes);
        errdefer {
            var owned = b;
            owned.free();
        }
        try L.w.buffers.append(L.gpa, b);
        L.w.bytes += b.len;
        return b.ptr;
    }

    /// The host bytes gathered so far uploaded as one buffer.
    pub fn flush(L: *Loader) !u64 {
        const ptr = try L.put(L.host.items);
        L.host.clearRetainingCapacity();
        return ptr;
    }

    pub fn name(L: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&L.name_buf, fmt, args);
    }

    pub fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        return L.ck.get(try L.name(fmt, args));
    }

    /// A tensor of `dtype` with exactly `dims`.
    pub fn expect(L: *Loader, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, dims: []const usize) !Tensor {
        const t = try L.get(fmt, args);
        if (t.dtype != dtype or t.rank != dims.len or !std.mem.eql(usize, t.shape[0..t.rank], dims)) {
            std.log.err("minimax_m3: {s} is not {s}{any}", .{ L.name_buf[0..], @tagName(dtype), dims });
            return error.UnexpectedMiniMaxTensor;
        }
        return t;
    }

    /// A bf16 vector of `n` as stored, on the device.
    fn vector(L: *Loader, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
        const t = try L.expect(fmt, args, .bf16, &.{n});
        return L.put(t.bytes);
    }

    /// A decoder projection's parts in their stored format: NVFP4 (`.weight_packed`) or MXFP8.
    fn proj(L: *Loader, parts: []const MxPart) !Linear {
        if (L.ck.has(try L.name("{s}.weight_packed", .{parts[0].name}))) {
            L.w.fp4_projections += 1;
            return L.fp4(parts);
        }
        L.w.fp8_projections += 1;
        return L.mx(parts);
    }

    /// NVFP4 parts (`.weight_packed`, `.weight_scale`, the dividing `.weight_global_scale`) stacked, one global.
    fn fp4(L: *Loader, parts: []const MxPart) !Linear {
        var total: usize = 0;
        var k: usize = 0;
        var global: ?u32 = null;
        for (parts, 0..) |p, i| {
            const w = try L.get("{s}.weight_packed", .{p.name});
            if (w.dtype != .u8 or w.rank != 2) return error.UnexpectedMiniMaxTensor;
            const g = try L.get("{s}.weight_global_scale", .{p.name});
            if (g.dtype != .f32 or g.numel() != 1) return error.UnexpectedMiniMaxTensor;
            const bits = std.mem.readInt(u32, g.bytes[0..4], .little);
            if (global != null and global.? != bits) return error.UnexpectedMiniMaxTensor;
            global = bits;
            const kk = w.dim(1) * 2;
            const rows = p.rows orelse [2]usize{ 0, w.dim(0) };
            const cols = p.cols orelse [2]usize{ 0, kk };
            if (rows[1] > w.dim(0) or cols[1] > kk or rows[0] >= rows[1] or cols[0] >= cols[1]) return error.UnexpectedMiniMaxTensor;
            if (cols[0] % 64 != 0 or (cols[1] - cols[0]) % 64 != 0 or (rows[1] - rows[0]) % 64 != 0) return error.UnexpectedMiniMaxTensor;
            if (i == 0) k = cols[1] - cols[0] else if (k != cols[1] - cols[0]) return error.UnexpectedMiniMaxTensor;
            total += rows[1] - rows[0];
        }
        const gs: f32 = @bitCast(global.?);
        if (!(gs > 0)) return error.UnexpectedMiniMaxTensor;
        const codes = try L.gpa.alloc(u8, total * k / 2);
        defer L.gpa.free(codes);
        const scales = try L.gpa.alloc(u8, total * k / 16);
        defer L.gpa.free(scales);
        var at: usize = 0;
        for (parts) |p| {
            const w = try L.get("{s}.weight_packed", .{p.name});
            const s = try L.get("{s}.weight_scale", .{p.name});
            const kk = w.dim(1) * 2;
            if (s.dtype != .f8_e4m3 or s.rank != 2 or s.dim(0) != w.dim(0) or s.dim(1) != kk / 16) return error.UnexpectedMiniMaxTensor;
            const rows = p.rows orelse [2]usize{ 0, w.dim(0) };
            const cols = p.cols orelse [2]usize{ 0, kk };
            for (rows[0]..rows[1]) |r| {
                @memcpy(codes[at * (k / 2) ..][0 .. k / 2], w.bytes[r * (kk / 2) + cols[0] / 2 ..][0 .. k / 2]);
                @memcpy(scales[at * (k / 16) ..][0 .. k / 16], s.bytes[r * (kk / 16) + cols[0] / 16 ..][0 .. k / 16]);
                at += 1;
            }
        }
        const words = try L.gpa.alloc(u32, cuda.nvfp4.padded(total) / 64 * (k / 64) * 512);
        defer L.gpa.free(words);
        const tiles = try L.gpa.alloc(u8, cuda.nvfp4.padded(total) * (k / 16));
        defer L.gpa.free(tiles);
        cuda.nvfp4.packWords(words, codes, total, k);
        cuda.nvfp4.packScales(tiles, scales, total, k);
        return cuda.nvfp4.weight(try L.put(std.mem.sliceAsBytes(words)), try L.put(tiles), 1.0 / gs, total, k);
    }

    /// MXFP8 parts (e4m3 `.weight` [n, k], e8m0 `.weight_scale_inv` [n, k/32]) cut and stacked as one lane weight.
    pub fn mx(L: *Loader, parts: []const MxPart) !Linear {
        var total: usize = 0;
        var k: usize = 0;
        for (parts, 0..) |p, i| {
            const w = try L.get("{s}.weight", .{p.name});
            if (w.dtype != .f8_e4m3 or w.rank != 2) return error.UnexpectedMiniMaxTensor;
            const rows = p.rows orelse [2]usize{ 0, w.dim(0) };
            const cols = p.cols orelse [2]usize{ 0, w.dim(1) };
            if (rows[1] > w.dim(0) or cols[1] > w.dim(1) or rows[0] >= rows[1] or cols[0] >= cols[1]) return error.UnexpectedMiniMaxTensor;
            if (cols[0] % 64 != 0 or (cols[1] - cols[0]) % 64 != 0 or (rows[1] - rows[0]) % 64 != 0) return error.UnexpectedMiniMaxTensor;
            if (i == 0) k = cols[1] - cols[0] else if (k != cols[1] - cols[0]) return error.UnexpectedMiniMaxTensor;
            total += rows[1] - rows[0];
        }
        const codes = try L.gpa.alloc(u8, total * k);
        defer L.gpa.free(codes);
        const scales = try L.gpa.alloc(u8, total * k / 32);
        defer L.gpa.free(scales);
        var at: usize = 0;
        for (parts) |p| {
            const w = try L.get("{s}.weight", .{p.name});
            const s = try L.get("{s}.weight_scale_inv", .{p.name});
            if (s.dtype != .u8 or s.rank != 2 or s.dim(0) != w.dim(0) or s.dim(1) != w.dim(1) / 32) return error.UnexpectedMiniMaxTensor;
            const kk = w.dim(1);
            const rows = p.rows orelse [2]usize{ 0, w.dim(0) };
            const cols = p.cols orelse [2]usize{ 0, kk };
            for (rows[0]..rows[1]) |r| {
                @memcpy(codes[at * k ..][0..k], w.bytes[r * kk + cols[0] ..][0..k]);
                @memcpy(scales[at * (k / 32) ..][0 .. k / 32], s.bytes[r * (kk / 32) + cols[0] / 32 ..][0 .. k / 32]);
                at += 1;
            }
        }
        const npad = cuda.mxfp8.padded(total);
        const pc = try L.gpa.alloc(u8, npad * k);
        defer L.gpa.free(pc);
        cuda.mxfp8.packCodes(pc, codes, total, k);
        const ps = try L.gpa.alloc(u8, npad * k / 32);
        defer L.gpa.free(ps);
        cuda.mxfp8.packScales(ps, scales, total, k);
        const dc = try L.put(pc);
        const ds = try L.put(ps);
        return cuda.mxfp8.weight(dc, ds, total, k);
    }

    fn attention(L: *Loader, i: u32) !Attention {
        const c = &L.w.config;
        const s = L.w.share;
        const D: usize = c.head_dim;
        const q = s.queries(c);
        var names: [5][96]u8 = undefined;
        const qn = try std.fmt.bufPrint(&names[0], prefix ++ "model.layers.{d}.self_attn.q_proj", .{i});
        const kn = try std.fmt.bufPrint(&names[1], prefix ++ "model.layers.{d}.self_attn.k_proj", .{i});
        const vn = try std.fmt.bufPrint(&names[2], prefix ++ "model.layers.{d}.self_attn.v_proj", .{i});
        const iqn = try std.fmt.bufPrint(&names[3], prefix ++ "model.layers.{d}.self_attn.index_q_proj", .{i});
        const ikn = try std.fmt.bufPrint(&names[4], prefix ++ "model.layers.{d}.self_attn.index_k_proj", .{i});
        const parts = [_]MxPart{
            .{ .name = qn, .rows = .{ q.lo * D, q.hi * D } },
            .{ .name = kn, .rows = .{ s.kv.lo * D, s.kv.hi * D } },
            .{ .name = vn, .rows = .{ s.kv.lo * D, s.kv.hi * D } },
            .{ .name = iqn, .rows = .{ s.kv.lo * c.index_dim, s.kv.hi * c.index_dim } },
            .{ .name = ikn },
        };
        const qkv = try L.proj(&parts);
        var on: [96]u8 = undefined;
        const o = try L.proj(&.{.{ .name = try std.fmt.bufPrint(&on, prefix ++ "model.layers.{d}.self_attn.o_proj", .{i}), .cols = .{ q.lo * D, q.hi * D } }});
        return .{
            .qkv = qkv,
            .o = o,
            .q_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.q_norm.weight", .{i}, D),
            .k_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.k_norm.weight", .{i}, D),
            .iq_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.index_q_norm.weight", .{i}, c.index_dim),
            .ik_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.index_k_norm.weight", .{i}, c.index_dim),
        };
    }

    /// A gate/up/down MLP under `module` cut to `span` of its intermediate: gate | up rows, down columns.
    fn mlp(L: *Loader, module: []const u8, span: shard.Span) !Mlp {
        var names: [3][160]u8 = undefined;
        const g = try std.fmt.bufPrint(&names[0], "{s}.gate_proj", .{module});
        const u = try std.fmt.bufPrint(&names[1], "{s}.up_proj", .{module});
        const d = try std.fmt.bufPrint(&names[2], "{s}.down_proj", .{module});
        return .{
            .gu = try L.proj(&.{ .{ .name = g, .rows = .{ span.lo, span.hi } }, .{ .name = u, .rows = .{ span.lo, span.hi } } }),
            .down = try L.proj(&.{.{ .name = d, .cols = .{ span.lo, span.hi } }}),
        };
    }

    fn moe(L: *Loader, i: u32) !Moe {
        const c = &L.w.config;
        const router = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.gate.weight", .{i}, .f32, &.{ c.experts, c.hidden });
        const rp = try L.put(router.bytes);
        const bias = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.e_score_correction_bias", .{i}, .f32, &.{c.experts});
        const bp = try L.put(bias.bytes);
        var mb: [128]u8 = undefined;
        const shared = try L.mlp(try std.fmt.bufPrint(&mb, prefix ++ "model.layers.{d}.block_sparse_moe.shared_experts", .{i}), L.w.share.shared);
        const routed = if (L.w.mix.nvfp4[i]) try expert_load.nvfp4(L, i) else try expert_load.affine3(L, i);
        return .{ .router = rp, .bias = bp, .routed = routed, .shared = shared };
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = &L.w.config;
        var mb: [96]u8 = undefined;
        return .{
            .in_norm = try L.vector(prefix ++ "model.layers.{d}.input_layernorm.weight", .{i}, c.hidden),
            .post_norm = try L.vector(prefix ++ "model.layers.{d}.post_attention_layernorm.weight", .{i}, c.hidden),
            .attn = try L.attention(i),
            .ffn = if (c.layer[i].moe) .{ .moe = try L.moe(i) } else .{ .dense = try L.mlp(try std.fmt.bufPrint(&mb, prefix ++ "model.layers.{d}.mlp", .{i}), L.w.share.dense) },
        };
    }
};

/// The rank's weights of `layers` (a prefix for a thin check), the embeddings whole, its vocabulary's head rows.
pub fn load(gpa: std.mem.Allocator, io: std.Io, tools: Tools, dir: []const u8, c: cfg.Config, share: shard.Share, mix: Mix, layers: []const u32) !Weights {
    var w: Weights = .{ .gpa = gpa, .config = c, .share = share, .mix = mix };
    errdefer w.deinit();
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var L: Loader = .{ .gpa = gpa, .t = tools, .w = &w, .ck = &ck };
    defer L.host.deinit(gpa);
    w.layers = try gpa.alloc(Layer, layers.len);
    for (layers, w.layers) |i, *l| {
        if (i >= c.layers) return error.BadLayer;
        l.* = try L.layer(i);
        ck.release(); // the pages read so far, before they crowd the device's allocations
    }
    const embed = try L.expect(prefix ++ "model.embed_tokens.weight", .{}, .bf16, &.{ c.vocab, c.hidden });
    w.embed = try L.put(embed.bytes);
    w.norm = try L.vector(prefix ++ "model.norm.weight", .{}, c.hidden);
    if (ck.has(prefix ++ "lm_head.weight_scale_inv")) { // MXFP8, from the dense FP4 conversion
        w.head = try L.mx(&.{.{ .name = prefix ++ "lm_head", .rows = .{ share.vocab.lo, share.vocab.hi } }});
        w.fp8_head = true;
    } else {
        const head = try L.expect(prefix ++ "lm_head.weight", .{}, .bf16, &.{ c.vocab, c.hidden });
        const n = share.vocab.len();
        const h: usize = c.hidden;
        const rows: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, head.bytes[share.vocab.lo * h * 2 .. share.vocab.hi * h * 2]));
        const tiles = try gpa.alloc(u16, cuda.mxfp8.padded(n) * h);
        defer gpa.free(tiles);
        cuda.qmmf.packBf16(tiles, rows, n, h);
        w.head = cuda.modelopt.bf16Weight(try L.put(std.mem.sliceAsBytes(tiles)), n, h);
    }
    if ((w.fp4_projections > 0) != (mix.fp4_qkv or mix.fp4_o or mix.fp4_mlp) or w.fp8_head != mix.fp8_head) {
        std.log.err("minimax_m3: config.json's nvfp4_mix (dense_fp4, head_format) does not name the formats stored", .{});
        return error.BadMix;
    }
    if (w.fp4_projections > 0) std.log.info("minimax_m3: {d} projections NVFP4, {d} MXFP8; head {s}", .{ w.fp4_projections, w.fp8_projections, if (w.fp8_head) "MXFP8" else "bf16" });
    try tools.stream.synchronize();
    return w;
}

test "the mix: config.json's layers, or an override of ranges, all or none" {
    var c = cfg.Config{};
    for (0..c.layers) |i| c.layer[i] = .{ .sparse = true, .moe = i >= 3 };
    var m: Mix = .{};
    _ = try m.parse("20-29,31,59", &c);
    try std.testing.expect(m.nvfp4[20] and m.nvfp4[29] and m.nvfp4[31] and m.nvfp4[59] and !m.nvfp4[30] and !m.nvfp4[3]);
    try std.testing.expectEqual(@as(usize, 12), m.count(&c));
    var all: Mix = .{};
    _ = try all.parse("all", &c);
    try std.testing.expectEqual(@as(usize, 57), all.count(&c));
    var none: Mix = .{};
    _ = try none.parse("none", &c);
    try std.testing.expectEqual(@as(usize, 0), none.count(&c));
    var bad: Mix = .{};
    try std.testing.expectError(error.BadMix, bad.parse("58-60", &c));
}
