//! The family's pipelines (kernels/metal/deepseek/v41.metal) and the core row projections.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const row = @import("../../core/row_projection.zig");

pub const names = [_][:0]const u8{
    "ds_embed",          "ds_hc_expand",    "ds_hc_boundary", "ds_hc_collapse", "ds_rms",
    "ds_gemv_bf_bf",     "ds_gemv_bf_f32",  "ds_rope",        "ds_fp8_qdq",     "ds_fp4_qdq",
    "ds_put_rows",       "ds_put_rows_f32", "ds_pool2",       "ds_index_scores", "ds_candidate_mask",
    "ds_block_max",      "ds_topk",         "ds_attend",      "ds_route",       "ds_swiglu",
    "ds_moe_combine",    "ds_engram",       "ds_scale",
};

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    rows: row.Pipelines,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        const i = comptime for (names, 0..) |n, i| {
            if (std.mem.eql(u8, n, key)) break i;
        } else @compileError("no deepseek_v41 kernel " ++ key);
        return self.pipelines[i];
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
        self.rows.deinit();
    }
};

pub fn load(device: mtl.Device) !Kernels {
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
    return out;
}
