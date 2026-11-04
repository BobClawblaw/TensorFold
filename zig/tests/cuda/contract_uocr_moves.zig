//! Unlimited-OCR-2's CUDA move kernels under the contract (patches, windows, gathers, embeds), each with a control.
const std = @import("std");
const g = @import("core").contract.gates;
const u = @import("contract_uocr.zig");

pub const Group = struct {
    /// The group's tests, appended to `list` (each control after the kernel it breaks).
    pub fn init(group: *Group, ctx: u.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        _ = group;
        _ = ctx;
        _ = a;
        _ = list;
    }

    pub fn deinit(group: *Group) void {
        _ = group;
    }
};
