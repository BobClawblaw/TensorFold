//! A native DeepSeek-V4.1-Flash model: config, checkpoint views, engram tables, RoPE tables and pipelines.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");
const engram = @import("engram.zig");
const vision = @import("vision.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    engram: engram.Engram,
    /// The image tower and aligner, when the checkpoint has them.
    vision: ?vision.Weights,
    /// RoPE frequencies (f32 [32]): plain for window-only layers and drafts, YaRN for compressed-key layers.
    freq_plain: mtl.Buffer,
    freq_compressed: mtl.Buffer,
    /// The checkpoint's buffers held resident on the queue (none before macOS 15).
    resident: ?mtl.ResidencySet = null,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, drafts: bool) !*Model {
        const config = try cfg.Config.read(gpa, io, dir);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = config;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ckpt.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        // held resident as read: ~190 GiB of untouched buffers invite the compressor, then jetsam kills the load
        m.resident = m.device.residencySet(files.len) catch null;
        errdefer if (m.resident) |set| set.deinit();
        for (files) |path| {
            try m.checkpoint.addFile(m.device, path, "");
            if (m.resident) |set| {
                set.add(m.checkpoint.shards.items[m.checkpoint.shards.items.len - 1].buffer);
                set.commit();
                set.requestResidency();
            }
        }
        // the DSpark head when asked for and the checkpoint has it (mtp.*)
        m.weights = try weights.load(&m.checkpoint, &m.config, drafts and m.checkpoint.has("mtp.0.attn.wq_a.weight"));
        m.vision = try vision.load(&m.checkpoint, m.config.dense);
        const wq_b = m.weights.layers[0].attn.wq_b;
        m.kernels = try kernels.load(m.device, &m.config, .{ .bits = @intCast(wq_b.bits), .group = @intCast(wq_b.group) });
        errdefer m.kernels.deinit();
        m.engram = try engram.Engram.open(gpa, io, dir, &m.config);
        errdefer m.engram.deinit();
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        m.freq_plain = try m.device.buffer(32 * 4, opts);
        errdefer m.freq_plain.deinit();
        m.freq_compressed = try m.device.buffer(32 * 4, opts);
        frequencies(m.config.rope, false, m.freq_plain.slice(f32, 32));
        frequencies(m.config.rope, true, m.freq_compressed.slice(f32, 32));
        if (m.resident) |set| m.queue.addResidencySet(set);
        return m;
    }

    /// The frequency table a layer's queries and keys rotate by (layers 40.. are the draft stages, window-only).
    pub fn freq(m: *const Model, layer: usize) mtl.Buffer {
        return if (layer < cfg.layers and cfg.ratios[layer] != 0) m.freq_compressed else m.freq_plain;
    }

    pub fn deinit(m: *Model) void {
        if (m.resident) |set| {
            m.queue.removeResidencySet(set);
            set.deinit();
        }
        m.freq_plain.deinit();
        m.freq_compressed.deinit();
        m.engram.deinit();
        m.kernels.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

/// precompute_freqs_cis's 64-dim frequencies 1 / theta^(2i/64), with YaRN's ramp when `compressed`.
pub fn frequencies(r: cfg.Rope, compressed: bool, out: []f32) void {
    const dim: f64 = cfg.rope_dim;
    const base: f64 = if (compressed) r.compress_theta else r.theta;
    for (out, 0..) |*f, i| {
        const e: f64 = @as(f64, @floatFromInt(2 * i)) / dim;
        f.* = @floatCast(1.0 / std.math.pow(f64, base, e));
    }
    if (!compressed) return;
    const orig: f64 = @floatFromInt(r.original);
    const corrected = struct {
        fn at(rotations: f64, d: f64, o: f64, b: f64) f64 {
            return d * @log(o / (rotations * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.at;
    const low = @max(@floor(corrected(r.beta_fast, dim, orig, base)), 0);
    const high = @min(@ceil(corrected(r.beta_slow, dim, orig, base)), dim - 1);
    for (out, 0..) |*f, i| {
        const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / @max(high - low, 1e-3), 0, 1);
        const smooth = 1 - ramp;
        const v: f64 = f.*;
        f.* = @floatCast(v / r.factor * (1 - smooth) + v * smooth);
    }
}
