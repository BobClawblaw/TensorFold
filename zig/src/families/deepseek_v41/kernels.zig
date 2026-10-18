//! The family's pipelines (kernels/metal/deepseek/v41*.metal) and the core row projections.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");
const cfg = @import("config.zig");

pub const names = [_][:0]const u8{
    "ds_embed",             "ds_hc_expand",    "ds_hc_boundary", "ds_hc_collapse",      "ds_rms",
    "ds_gemv_bf_bf",        "ds_gemv_bf_f32",  "ds_rope",        "ds_fp8_qdq",          "ds_fp4_qdq",
    "ds_put_rows",          "ds_put_rows_f32", "ds_pool2",       "ds_index_scores_1x8", "ds_index_scores_2x16",
    "ds_index_scores_kept", "ds_kept_keys",    "ds_block_max",   "ds_topk",             "ds_attend",
    "ds_route",             "ds_swiglu",       "ds_moe_combine", "ds_engram",           "ds_scale",
    "ds_hc_mix_part",       "ds_sort_slots",   "ds_gather_rows", "ds_attend_rows",      "ds_stream_mean",
    "ds_markov",            "ds_markov_pick",  "ds_linear_bias", "ds_bias",             "ds_add",
    "ds_vit_rope",          "ds_vit_attend",   "ds_vit_glu",     "ds_unfold",           "ds_put_image",
    "ds_attend_split",      "ds_attend_merge", "ds_kv_finish",   "ds_hc_expand_mix",
};

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    rows: row.Pipelines,
    /// core/affine_mm by code width (group 64) for prompt chunks on the tensor units; null without them.
    nax: [9]?am.Pipes = @splat(null),

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        const i = comptime for (names, 0..) |n, i| {
            if (std.mem.eql(u8, n, key)) break i;
        } else @compileError("no deepseek_v41 kernel " ++ key);
        return self.pipelines[i];
    }

    /// The tensor-unit pipelines for `w`, when this GPU has them and the checkpoint uses the width.
    pub fn naxPipes(self: *const Kernels, w: cfg.Width) ?am.Pipes {
        if (w.group != 64) return null;
        return self.nax[w.bits];
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
        for (self.nax) |set| if (set) |pipes| for (pipes) |p| p.deinit();
        self.rows.deinit();
    }
};

pub fn load(device: mtl.Device, config: *const cfg.Config, block_dense: cfg.Width) !Kernels {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var out: Kernels = undefined;
    var made: usize = 0;
    errdefer for (out.pipelines[0..made]) |p| p.deinit();
    for (names, 0..) |name, i| {
        out.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    out.rows = try row.Pipelines.load(device);
    errdefer out.rows.deinit();
    out.nax = @splat(null);
    if (device.tensorUnits()) {
        errdefer for (out.nax) |set| if (set) |pipes| for (pipes) |p| p.deinit();
        var widths: [cfg.layers + cfg.stages + 2]cfg.Width = undefined;
        widths[0] = config.dense;
        widths[1] = block_dense; // the blocks' projections, when their own entries differ from the global width
        @memcpy(widths[2..][0..cfg.layers], &config.experts);
        @memcpy(widths[2 + cfg.layers ..], &config.draft_experts);
        for (widths) |w| {
            if (w.group != 64 or out.nax[w.bits] != null) continue;
            out.nax[w.bits] = try naxLoad(device, w.bits);
        }
    }
    return out;
}

/// core/affine_mm at `bits`, group 64 (its GLU entry points unused: the experts' SwiGLU is the family's own).
fn naxLoad(device: mtl.Device, bits: u8) !am.Pipes {
    const a = std.heap.c_allocator;
    const src = try am.source(device, a, .{ .bits = bits, .group = 64 });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: am.Pipes = undefined;
    var made: usize = 0;
    errdefer for (pipes[0..made]) |p| p.deinit();
    for (am.names, 0..) |name, j| {
        pipes[j] = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    return pipes;
}
