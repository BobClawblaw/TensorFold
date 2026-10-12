//! The shared native glue kernels (zig/src/cuda) in place of the captured Triton set, kernel by kernel, to gate
//! each against the captures. TF_QWEN27_GLUE: a comma list of Kernel names, or "all"; unset, every launch is Triton.

const std = @import("std");
const cuda = @import("cuda");

pub const Kernel = enum { embed, add_rmsnorm, gdn_pre, gated_norm, swiglu, attn_prep, gate_mul, group_sums, paths, attention, prep, dconv, block_attention };

pub const Glue = struct {
    on: std.EnumSet(Kernel),
    norms: cuda.norms.Norms,
    act: cuda.act_rows.ActRows,
    embed: cuda.embed_rows.EmbedRows,
    qk: cuda.qk_rows.QkRows,
    gdn: cuda.gdn_prep.GdnPrep,
    tree: cuda.tree_attention.TreeAttention,
    block: cuda.block_attention.BlockAttention,
    /// tree attention's groups a row: fixed by the context (a graph replays at any length), set by the engine.
    ng: usize = 0,

    /// The kernels TF_QWEN27_GLUE names, loaded; null when it names none.
    pub fn load(d: *const cuda.Driver) !?Glue {
        const env = std.mem.span(std.c.getenv("TF_QWEN27_GLUE") orelse return null);
        var on = std.EnumSet(Kernel).empty;
        var it = std.mem.tokenizeAny(u8, env, ", ");
        while (it.next()) |name| {
            if (std.mem.eql(u8, name, "all")) {
                on = std.EnumSet(Kernel).full;
                continue;
            }
            on.insert(std.meta.stringToEnum(Kernel, name) orelse {
                std.log.err("TF_QWEN27_GLUE: no kernel {s}", .{name});
                return error.UnknownGlueKernel;
            });
        }
        if (on.count() == 0) return null;
        return try loadSet(d, on);
    }

    /// The modules, taking the kernels in `on`.
    pub fn loadSet(d: *const cuda.Driver, on: std.EnumSet(Kernel)) !Glue {
        var g: Glue = undefined;
        g.on = on;
        g.ng = 0;
        g.norms = try cuda.norms.Norms.load(d);
        errdefer g.norms.unload();
        g.act = try cuda.act_rows.ActRows.load(d);
        errdefer g.act.unload();
        g.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer g.embed.unload();
        g.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer g.qk.unload();
        g.gdn = try cuda.gdn_prep.GdnPrep.load(d);
        errdefer g.gdn.unload();
        g.tree = try cuda.tree_attention.TreeAttention.load(d);
        errdefer g.tree.unload();
        g.block = try cuda.block_attention.BlockAttention.load(d);
        var names: [256]u8 = undefined;
        var n: usize = 0;
        var keys = on.iterator();
        while (keys.next()) |k| {
            const s = @tagName(k);
            if (n + s.len + 1 > names.len) break;
            @memcpy(names[n..][0..s.len], s);
            names[n + s.len] = ' ';
            n += s.len + 1;
        }
        std.log.info("native glue in place of Triton: {s}", .{names[0..n]});
        return g;
    }

    pub fn unload(g: *Glue) void {
        g.block.unload();
        g.tree.unload();
        g.gdn.unload();
        g.qk.unload();
        g.embed.unload();
        g.act.unload();
        g.norms.unload();
    }
};

/// The loaded glue when it takes kernel `k`.
pub fn uses(g: ?*const Glue, k: Kernel) ?*const Glue {
    const x = g orelse return null;
    return if (x.on.contains(k)) x else null;
}
