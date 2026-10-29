//! A native MiniMax-M3.1-Flash model: config, the checkpoint in Metal buffers, its matrices and every pipeline.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const rows = @import("../../core/row_projection.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const frags = @import("../../core/frags.zig");
const st = @import("state.zig");
const weights = @import("weights.zig");
const vision = @import("vision.zig");
const exl3 = @import("exl3.zig");
const exl3_rows = @import("../../core/exl3_rows.zig");

/// Nemotron's generated keyed sampler (its vocabulary compiled in; this model's replaces it).
fn sampler() ?ks.nemotron.Kernel {
    for (ks.nemotron.all) |k| if (std.mem.eql(u8, k.key, "sample")) return k;
    return null;
}

pub const DrafterKind = enum { dflash2, dspark };

/// MINIMAX_DRAFTER: dspark (the release's head) or dflash2 (incoai's, the default); null otherwise, refused at load.
pub fn drafterKind() ?DrafterKind {
    const v = std.c.getenv("MINIMAX_DRAFTER") orelse return .dflash2;
    return std.meta.stringToEnum(DrafterKind, std.mem.span(v));
}

/// A prompt chunk's tensor-unit matmul for one (bits, group) the checkpoint uses.
pub const Mm = struct { bits: u8, group: usize, pipes: affine_mm.Pipes };

pub const glue_names = [_][:0]const u8{ "mm_embed", "mm_norm", "mm_add_norm", "mm_add", "mm_swiglu", "mm_router", "mm_route_pick", "mm_moe_add_norm", "mm_moe_add", "mm_sort", "mm_rows_by_order", "mm_prep", "mm_index_scores", "mm_index_select", "mm_decode_vec", "mm_merge_vec", "mm_argmax", "mm_route_groups", "mm_widen", "mm_narrow" };
pub const Glue = enum { embed, norm, add_norm, add, swiglu, router, route_pick, moe_add_norm, moe_add, sort, rows_by_order, prep, index_scores, index_select, decode_vec, merge_vec, argmax, route_groups, widen, narrow };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: weights.Weights,
    glue: [glue_names.len]mtl.Pipeline,
    rows: rows.Pipelines,
    /// Prompt attention on the tensor units over each row's kept blocks; decode attention is glue's mm_decode_vec.
    prompt_attn: mtl.Pipeline,
    /// A prompt chunk's indexer block scores on the tensor units (index_nax.metal).
    index_nax: mtl.Pipeline,
    /// A pipelined step's keyed draws (host twins: lanes' gpu_rule, gpu_full): top-k candidates, whole vocabulary.
    sample: mtl.Pipeline,
    sample_full: mtl.Pipeline,
    mm: [4]Mm = undefined,
    mms: usize = 0,
    /// The residual's copy into a drafter's taps after a tapped layer (dspark.metal's mm_tap).
    tap: mtl.Pipeline,
    /// A drafter block's attention over its context ring and itself (dspark.metal's mm_draft_vec, either drafter's).
    draft_vec: mtl.Pipeline,
    /// A decode window's rows of each expert at 3 and 4 bits (experts.metal), grouped by mm_route_groups.
    grouped: [2]mtl.Pipeline,
    /// Each layer's place in the drafter's taps, -1 when untapped (no drafter: all -1).
    tap_at: [cfg.max_layers]i8 = @splat(-1),
    taps: u32 = 0,
    /// The image tower (--vision), its shard added to the checkpoint by loadVision.
    vision: ?*vision.Vision = null,
    /// An EXL3 pack's projections, bf16 embedding rows and fp16 index pair (exl3_glue.metal).
    xr: ?exl3_rows.Pipelines = null,
    xm: ?exl3_rows.MmPipelines = null,
    embed_bf16: ?mtl.Pipeline = null,
    half_rows: ?mtl.Pipeline = null,
    /// The pack's DSpark head's widened bf16 projections (exl3_glue.metal's mm_bf16_rows).
    bf16_rows: ?mtl.Pipeline = null,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.mms = 0;
        m.vision = null;
        m.xr = null;
        m.xm = null;
        m.embed_bf16 = null;
        m.half_rows = null;
        m.bf16_rows = null;
        m.config = try cfg.Config.read(gpa, io, dir);
        const c = &m.config;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        // the vision tower's shard stays on disk until loadVision, and the drafter not in use
        const skip = if (drafterKind() == DrafterKind.dspark) "model-dflash2.safetensors" else "model-dspark.safetensors";
        for (files) |path| if (std.mem.indexOf(u8, std.fs.path.basename(path), vision.shard_stem) == null and !std.mem.endsWith(u8, path, skip)) try m.checkpoint.addFile(m.device, path, "");
        const packed_exl3 = try exl3.isExl3(gpa, io, dir);
        m.weights = if (packed_exl3) try weights.loadExl3(gpa, m.device, &m.checkpoint, c, st.batch_rows) else try weights.load(gpa, &m.checkpoint, c);
        errdefer m.weights.deinit(gpa);
        errdefer m.deinitExl3();
        if (packed_exl3) {
            m.xr = try exl3_rows.Pipelines.load(m.device);
            m.xm = try exl3_rows.MmPipelines.load(m.device, gpa);
            const lib = try mtl.Library.fromSource(m.device, ks.minimax_m3_exl3_glue, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.embed_bf16 = try mtl.Pipeline.init(m.device, lib, "mm_embed_bf16", false);
            m.half_rows = try mtl.Pipeline.init(m.device, lib, "mm_half_rows", false);
            m.bf16_rows = try mtl.Pipeline.init(m.device, lib, "mm_bf16_rows", false);
        }
        {
            const lib = try mtl.Library.fromSource(m.device, ks.minimax_m3_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.glue[0..made]) |p| p.deinit();
            for (glue_names, 0..) |n, i| {
                m.glue[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.glue) |p| p.deinit();
        m.rows = try rows.Pipelines.load(m.device);
        errdefer m.rows.deinit();
        {
            const body = try frags.source(m.device, gpa, ks.minimax_m3_attention_nax);
            defer gpa.free(body);
            const src = try std.fmt.allocPrint(gpa, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WORDS {d}\n#define SCALE {e}f\n{s}", .{ c.head_dim, c.heads, c.heads / c.kv_heads, c.kv_heads, st.selWords(c), 1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))), body });
            defer gpa.free(src);
            const lib = try mtl.Library.fromSource(m.device, src, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.prompt_attn = try mtl.Pipeline.init(m.device, lib, "mm_prompt_attention", false);
        }
        errdefer m.prompt_attn.deinit();
        {
            const body = try frags.source(m.device, gpa, ks.minimax_m3_index_nax);
            defer gpa.free(body);
            const src = try std.fmt.allocPrint(gpa, "#define DH {d}\n#define IH {d}\n{s}", .{ c.index_dim, c.index_heads, body });
            defer gpa.free(src);
            const lib = try mtl.Library.fromSource(m.device, src, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.index_nax = try mtl.Pipeline.init(m.device, lib, "mm_index_scores_nax", false);
        }
        errdefer m.index_nax.deinit();
        {
            const generated = sampler() orelse return error.MiniMaxSamplerMissing;
            const v = try std.fmt.allocPrint(gpa, "constexpr int V = {d};", .{c.vocab});
            defer gpa.free(v);
            const text = try std.mem.replaceOwned(u8, gpa, generated.source, "constexpr int V = 131072;", v);
            defer gpa.free(text);
            if (std.mem.indexOf(u8, text, v) == null) return error.MiniMaxSamplerMissing;
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.sample = try mtl.Pipeline.init(m.device, lib, generated.function, false);
        }
        errdefer m.sample.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, ks.nemotron_sample, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.sample_full = try mtl.Pipeline.init(m.device, lib, "tf_sample_full", false);
        }
        errdefer m.sample_full.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, ks.minimax_m3_dspark, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.tap = try mtl.Pipeline.init(m.device, lib, "mm_tap", false);
            errdefer m.tap.deinit();
            m.draft_vec = try mtl.Pipeline.init(m.device, lib, "mm_draft_vec", false);
        }
        errdefer m.tap.deinit();
        errdefer m.draft_vec.deinit();
        {
            const text = try std.mem.concat(gpa, u8, &.{ ks.core_row_projection, "\n", ks.minimax_m3_experts });
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.grouped[0] = try mtl.Pipeline.init(m.device, lib, "mm_experts_grouped_q3", false);
            errdefer m.grouped[0].deinit();
            m.grouped[1] = try mtl.Pipeline.init(m.device, lib, "mm_experts_grouped_q4", false);
        }
        errdefer for (m.grouped) |p| p.deinit();
        errdefer for (m.mm[0..m.mms]) |f| for (f.pipes) |p| p.deinit();
        if (packed_exl3) return m;
        try m.addMm(m.weights.head.affine);
        for (m.weights.layers) |l| {
            try m.addMm(l.attn.qkv.affine);
            switch (l.ffn) {
                .dense => |d| try m.addMm(d.gu.affine),
                .moe => |e| {
                    try m.addMm(e.gu.affine.one);
                    try m.addMm(e.down.affine.one);
                    try m.addMm(e.shared.gu.affine);
                },
            }
        }
        return m;
    }

    fn deinitExl3(m: *Model) void {
        if (m.xr) |*p| p.deinit();
        if (m.xm) |*p| p.deinit();
        if (m.embed_bf16) |p| p.deinit();
        if (m.half_rows) |p| p.deinit();
        if (m.bf16_rows) |p| p.deinit();
    }

    /// An encoder's indirect reads: every shard an EXL3 job points into (no-op for affine checkpoints).
    pub fn useWeights(m: *const Model, e: mtl.ComputeEncoder) void {
        if (m.weights.jobs == null) return;
        for (m.checkpoint.shards.items) |sh| e.useResource(sh.buffer, mtl.ResourceUsage.read);
    }

    /// Pipelines for `w`'s format, compiled once per format (at most 4).
    fn addMm(m: *Model, w: rows.Weights) !void {
        if (m.mmFor(w) != null) return;
        if (m.mms == m.mm.len) return error.TooManyMiniMaxFormats;
        const text = try affine_mm.source(m.device, m.gpa, .{ .bits = w.bits, .group = @intCast(w.group) });
        defer m.gpa.free(text);
        const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var f = Mm{ .bits = w.bits, .group = w.group, .pipes = undefined };
        var made: usize = 0;
        errdefer for (f.pipes[0..made]) |p| p.deinit();
        for (affine_mm.names, 0..) |n, i| {
            f.pipes[i] = try mtl.Pipeline.init(m.device, lib, n, false);
            made += 1;
        }
        m.mm[m.mms] = f;
        m.mms += 1;
    }

    /// The tensor-unit pipelines for `w`'s format, if loaded.
    pub fn mmFor(m: *const Model, w: rows.Weights) ?affine_mm.Pipes {
        for (m.mm[0..m.mms]) |f| if (f.bits == w.bits and f.group == w.group) return f.pipes;
        return null;
    }

    /// The drafter's layers, in the order its features concatenate them.
    pub fn setTaps(m: *Model, layers: []const u32) !void {
        m.tap_at = @splat(-1);
        for (layers, 0..) |l, i| {
            if (l >= m.config.layers or i > 127) return error.BadMiniMaxTaps;
            m.tap_at[l] = @intCast(i);
        }
        m.taps = @intCast(layers.len);
    }

    /// The checkpoint's image tower (--vision): its shard into the checkpoint, then the tower over it.
    pub fn loadVision(m: *Model, io: std.Io, dir: []const u8) !void {
        if (m.vision != null) return;
        const path = try vision.shardPath(m.gpa, io, dir);
        defer m.gpa.free(path);
        try m.checkpoint.addFile(m.device, path, "");
        m.vision = try vision.Vision.load(m.gpa, m.device, &m.checkpoint, m.pipe(.widen), m.pipe(.narrow));
    }

    /// An EXL3 pack's DSpark head (dspark/, the release's): its shard into the checkpoint, for dspark.Drafter.
    pub fn loadPackDSpark(m: *Model, io: std.Io, dir: []const u8) !void {
        const sub = try std.fs.path.join(m.gpa, &.{ dir, "dspark" });
        defer m.gpa.free(sub);
        const files = try shards.shardFiles(m.gpa, io, sub);
        defer shards.freeShardFiles(m.gpa, files);
        for (files) |path| try m.checkpoint.addFile(m.device, path, "");
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn deinit(m: *Model) void {
        m.deinitExl3();
        for (m.mm[0..m.mms]) |f| for (f.pipes) |p| p.deinit();
        m.prompt_attn.deinit();
        m.index_nax.deinit();
        m.tap.deinit();
        m.draft_vec.deinit();
        for (m.grouped) |p| p.deinit();
        m.sample.deinit();
        m.sample_full.deinit();
        m.rows.deinit();
        for (m.glue) |p| p.deinit();
        if (m.vision) |v| v.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
