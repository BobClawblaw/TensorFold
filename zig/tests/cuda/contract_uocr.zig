//! Unlimited-OCR-2's CUDA kernels under the contract (uocr-contract): the Metal family's five groups, with controls.
const std = @import("std");
const cuda = @import("cuda");
const uocr = @import("uocr_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const g = @import("core").contract.gates;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// What a group's tests launch with: the family's kernels, the controls image, and the one rig every test shares.
pub const Ctx = struct { gpu: check.Gpu, s: cuda.Stream, k: uocr.kernels.Kernels, ctl: cuda.Module, rig: *rig_.Rig };

const Rows = @import("contract_uocr_rows.zig").Group;
const Moves = @import("contract_uocr_moves.zig").Group;
const Attention = @import("contract_uocr_attention.zig").Group;
const Route = @import("contract_uocr_route.zig").Group;
const Matvec = @import("contract_uocr_matvec.zig").Group;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: uocr.kernels.Kernels,
    ctl: cuda.Module,
    /// Every test's buffers: a prepare frees the last test's, so the set holds one test's memory at a time.
    rig: rig_.Rig,
    rows: Rows = .{},
    moves: Moves = .{},
    attention: Attention = .{},
    route: Route = .{},
    matvec: Matvec = .{},

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try uocr.kernels.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl, .rig = .{ .gpu = gpu, .s = s } };
        const ctx: Ctx = .{ .gpu = gpu, .s = s, .k = k, .ctl = ctl, .rig = &set.rig };
        try set.rows.init(ctx, a, list);
        try set.moves.init(ctx, a, list);
        try set.attention.init(ctx, a, list);
        try set.route.init(ctx, a, list);
        try set.matvec.init(ctx, a, list);
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.matvec.deinit();
        set.route.deinit();
        set.attention.deinit();
        set.moves.deinit();
        set.rows.deinit();
        set.rig.deinit();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// The Unlimited-OCR-2 kernels' gates 1-4 and their controls alone; `only` keeps the kernels whose name holds it.
pub fn run(gpu: check.Gpu, only: ?[]const u8) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    if (only) |text| {
        var kept: usize = 0;
        for (list.items) |k| if (std.mem.indexOf(u8, k.spec.name, text) != null) {
            list.items[kept] = k;
            kept += 1;
        };
        list.shrinkRetainingCapacity(kept);
    }
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x756f6372, &t);
    std.debug.print("SUMMARY uocr: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "uocr-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
