//! Unlimited-OCR-2's CUDA move kernels under the contract (patches, windows, gathers, embeds), each with a control.
const std = @import("std");
const g = @import("core").contract.gates;
const u = @import("contract_uocr.zig");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_uocr_moves_kit.zig");
const rel = @import("contract_uocr_moves_rel.zig");
const rows = @import("contract_uocr_moves_rows.zig");

pub const Group = struct {
    im2col: [6]kit.Im2col = undefined,
    windows: [5]kit.Window = undefined,
    rel: [3]rel.RelPos = undefined,
    concat: [2]rel.Concat = undefined,
    clip: [2]rel.ClipEmbed = undefined,
    rows: [4]rows.Rows = undefined,
    embed: [2]rows.Embed = undefined,
    embed_rig: rig_.Rig = undefined,

    /// The group's tests, appended to `list` (each control after the kernel it breaks).
    pub fn init(group: *Group, ctx: u.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        const rig: rig_.Rig = .{ .gpu = ctx.gpu, .s = ctx.s };
        const k = ctx.k;
        const ctl = ctx.ctl;
        const geos = kit.geos;
        // the controls run at the patches' geometry (no padding: the swap) and the neck's (padding on every side)
        group.im2col = .{
            .{ .rig = rig, .k = k, .geo = geos[0], .name = geos[0].name },
            .{ .rig = rig, .k = k.with("im2col", try ctl.function("tf_ctl_uocr_moves_im2col_swap")), .geo = geos[0], .name = "control: uocr moves im2col ky and kx swapped (patch 16x16)", .control = .shapes },
            .{ .rig = rig, .k = k, .geo = geos[1], .name = geos[1].name },
            .{ .rig = rig, .k = k.with("im2col", try ctl.function("tf_ctl_uocr_moves_im2col_pad")), .geo = geos[1], .name = "control: uocr moves im2col padding 1 not 0 (neck 3x3)", .control = .shapes },
            .{ .rig = rig, .k = k, .geo = geos[2], .name = geos[2].name },
            .{ .rig = rig, .k = k, .geo = geos[3], .name = geos[3].name },
        };
        group.windows = .{
            .{ .rig = rig, .k = k, .back = false, .name = "cuda uocr moves window 14 (zero padding)" },
            .{ .rig = rig, .k = k.with("window", try ctl.function("tf_ctl_uocr_moves_window_swap")), .back = false, .name = "control: uocr moves window rows and columns swapped in a window", .control = .shapes },
            .{ .rig = rig, .k = k.with("window", try ctl.function("tf_ctl_uocr_moves_window_pad")), .back = false, .name = "control: uocr moves window padding 1 not 0", .control = .shapes },
            .{ .rig = rig, .k = k, .back = true, .name = "cuda uocr moves unwindow_add 14 (crop, bf16 residual add, in place)" },
            .{ .rig = rig, .k = k.with("unwindow_add", try ctl.function("tf_ctl_uocr_moves_unwindow_swap")), .back = true, .name = "control: uocr moves unwindow_add rows and columns swapped in a window", .control = .shapes },
        };
        group.rel = .{
            .{ .rig = rig, .k = k, .name = "cuda uocr moves rel_pos (12 heads of 64, fp32 sums)" },
            .{ .rig = rig, .k = k.with("rel_pos", try ctl.function("tf_ctl_uocr_moves_rel_pos_swap")), .name = "control: uocr moves rel_pos rel_h and rel_w positions swapped", .control = .shapes },
            .{ .rig = rig, .k = k.with("rel_pos", try ctl.function("tf_ctl_uocr_moves_rel_pos_half")), .name = "control: uocr moves rel_pos summed in half", .control = .sums },
        };
        group.concat = .{
            .{ .rig = rig, .k = k, .name = "cuda uocr moves concat (CLIP | SAM rows)" },
            .{ .rig = rig, .k = k.with("concat", try ctl.function("tf_ctl_uocr_moves_concat_wide")), .name = "control: uocr moves concat left row one column wider", .control = .shapes },
        };
        group.clip = .{
            .{ .rig = rig, .k = k, .name = "cuda uocr moves clip_embed (class row, positions, one bf16 add)" },
            .{ .rig = rig, .k = k.with("clip_embed", try ctl.function("tf_ctl_uocr_moves_clip_embed_pos")), .name = "control: uocr moves clip_embed class row takes position 1", .control = .shapes },
        };
        group.rows = .{
            .{ .rig = rig, .k = k, .scatter = false, .name = "cuda uocr moves gather 1280 (y[i] = x[rows[i]])" },
            .{ .rig = rig, .k = k.with("gather", try ctl.function("tf_ctl_uocr_moves_gather_even")), .scatter = false, .name = "control: uocr moves gather odd rows read the even row's index", .control = .shapes },
            .{ .rig = rig, .k = k, .scatter = true, .name = "cuda uocr moves scatter 1280 (slots[dest[i]] = y[i])" },
            .{ .rig = rig, .k = k.with("scatter", try ctl.function("tf_ctl_uocr_moves_scatter_even")), .scatter = true, .name = "control: uocr moves scatter odd rows written to the even row's slot", .control = .shapes },
        };
        group.embed_rig = rig;
        group.embed = .{
            .{ .rig = &group.embed_rig, .k = k, .name = "cuda uocr moves embed 1280 (token rows from the table, image rows left)" },
            .{ .rig = &group.embed_rig, .k = k.with("embed", try ctl.function("tf_ctl_uocr_moves_embed_image")), .name = "control: uocr moves embed image rows overwritten from the table", .control = .shapes },
        };
        for (&group.im2col) |*x| try list.append(a, x.kernel());
        for (&group.windows) |*x| try list.append(a, x.kernel());
        for (&group.rel) |*x| try list.append(a, x.kernel());
        for (&group.concat) |*x| try list.append(a, x.kernel());
        for (&group.clip) |*x| try list.append(a, x.kernel());
        for (&group.rows) |*x| try list.append(a, x.kernel());
        for (&group.embed) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(group: *Group) void {
        inline for (.{ "im2col", "windows", "rel", "concat", "clip", "rows", "embed" }) |f| {
            for (&@field(group, f)) |*x| x.deinit();
        }
        group.embed_rig.deinit();
    }
};
