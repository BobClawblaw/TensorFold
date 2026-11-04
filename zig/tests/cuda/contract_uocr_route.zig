//! Unlimited-OCR-2's CUDA routing under the contract (block keys, DASH router, MoE gate, combine), with controls.
const std = @import("std");
const g = @import("core").contract.gates;
const u = @import("contract_uocr.zig");
const rig_ = @import("contract_rig.zig");
const base = @import("contract_uocr_route_base.zig");
const Route = @import("contract_uocr_route_pick.zig").Route;
const moe = @import("contract_uocr_route_moe.zig");

const BlockKeys = base.BlockKeys;

/// A control: the kernel it stands in for, its function in the controls image, its name and the gate it must fail.
const Control = struct { sym: [:0]const u8, name: []const u8, gate: g.Gate };

const keys_controls = [_]Control{
    .{ .sym = "tf_ctl_uocr_route_keys_nonfinite", .name = "control: uocr route tf_uocr_block_keys writes a non-finite sum", .gate = .nonfinite },
};
const route_controls = [_]Control{
    .{ .sym = "tf_ctl_uocr_route_tie_high", .name = "control: uocr route tf_uocr_route, a tie goes to the higher block", .gate = .shapes },
    .{ .sym = "tf_ctl_uocr_route_gap_short", .name = "control: uocr route tf_uocr_route, the local gap one block short", .gate = .shapes },
    .{ .sym = "tf_ctl_uocr_route_page_last", .name = "control: uocr route tf_uocr_route, a block's last page left out", .gate = .shapes },
};
const gate_controls = [_]Control{
    .{ .sym = "tf_ctl_uocr_route_gate_tie_high", .name = "control: uocr route tf_uocr_moe_gate, a tie goes to the higher expert", .gate = .shapes },
};
const combine_controls = [_]Control{
    .{ .sym = "tf_ctl_uocr_route_combine_unrounded", .name = "control: uocr route tf_uocr_moe_combine drops bf16(sum)", .gate = .shapes },
};

pub const Group = struct {
    keys: [1 + keys_controls.len]BlockKeys = undefined,
    routes: [1 + route_controls.len]Route = undefined,
    gates: [1 + gate_controls.len]moe.MoeGate = undefined,
    combines: [1 + combine_controls.len]moe.Combine = undefined,

    /// One kernel's test and its controls, each a copy whose launches go to the control's function.
    fn fill(comptime T: type, comptime name: []const u8, ctx: u.Ctx, out: []T, label: []const u8, ctls: []const Control) !void {
        const rig = ctx.rig;
        out[0] = .{ .rig = rig, .k = ctx.k, .name = label };
        for (ctls, out[1..]) |ctl, *x| x.* = .{ .rig = rig, .k = ctx.k.with(name, try ctx.ctl.function(ctl.sym)), .name = ctl.name, .control = ctl.gate };
    }

    /// The group's tests, appended to `list` (each control after the kernel it breaks).
    pub fn init(group: *Group, ctx: u.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        try fill(BlockKeys, "block_keys", ctx, &group.keys, "cuda uocr route tf_uocr_block_keys block 32 router 128", &keys_controls);
        try fill(Route, "route", ctx, &group.routes, "cuda uocr route tf_uocr_route ranked 16 read 12 blocks to 4096", &route_controls);
        try fill(moe.MoeGate, "moe_gate", ctx, &group.gates, "cuda uocr route tf_uocr_moe_gate experts 64 top 6 hidden 1280", &gate_controls);
        try fill(moe.Combine, "moe_combine", ctx, &group.combines, "cuda uocr route tf_uocr_moe_combine top 6 hidden 1280", &combine_controls);
        for (&group.keys) |*x| try list.append(a, x.kernel());
        for (&group.routes) |*x| try list.append(a, x.kernel());
        for (&group.gates) |*x| try list.append(a, x.kernel());
        for (&group.combines) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(group: *Group) void {
        for (&group.keys) |*x| x.deinit();
        for (&group.routes) |*x| x.deinit();
        for (&group.gates) |*x| x.deinit();
        for (&group.combines) |*x| x.deinit();
    }
};
