//! Unlimited-OCR-2's CUDA attention under the contract: the tower's and the decoder's, each with a control.
const std = @import("std");
const core = @import("core");
const u = @import("contract_uocr.zig");
const rig_ = @import("contract_rig.zig");
const tower = @import("contract_uocr_attention_tower.zig");
const decoder = @import("contract_uocr_attention_decoder.zig");

const c = core.contract.spec;
const g = core.contract.gates;

pub const nan16 = 0x7fc0;
/// Gate 4's values: multiples of 64 in 32-key periods (16 values, then their negations, 2 nudged), exact in fp32.
pub const step: f32 = 64;
pub const period = 32;
/// nvcc's expf under the glue flags (no fast math): 2 ulps.
pub const exp_ulps = [2]f64{ 2, 0 };
/// The glue flags' --prec-div=true: an IEEE division, one rounding.
pub const div_roundings = 1;

/// A value as fp32 then bf16 rounds it (to nearest even, twice): an fp32 step on exact inputs, then the bf16 point.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Random bf16 values in [-r, r).
pub fn fillRandom(rng: std.Random, out: []u16, r: f32) void {
    for (out) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * r);
}

/// Gate 4's value at key `j` and column `col` of `base` [period / 2][width].
pub fn stressValue(base: []const f32, width: usize, j: usize, col: usize) f32 {
    const m = j % period;
    if (m < period / 2) return base[m * width + col];
    const x = -base[(m - period / 2) * width + col];
    return if (m - period / 2 < 2) x + step else x;
}

pub fn stressBase(a: std.mem.Allocator, rng: std.Random, width: usize) ![]f32 {
    const base = try a.alloc(f32, period / 2 * width);
    for (base) |*x| x.* = step * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 255)));
    return base;
}

pub const Group = struct {
    /// One rig for every probe: the gates run one call at a time, and the decoder's long caches are freed on the next.
    rig: rig_.Rig = undefined,
    live: bool = false,
    towers: [tower.entries.len]tower.Probe = undefined,
    decoders: [decoder.cases.len]decoder.Probe = undefined,

    /// The group's tests, appended to `list` (each control after the kernel it breaks).
    pub fn init(group: *Group, ctx: u.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        group.rig = .{ .gpu = ctx.gpu, .s = ctx.s };
        group.live = true;
        for (tower.entries, &group.towers) |e, *p| {
            const k = if (tower.control(e[1])) |name| ctx.k.with("attention", try ctx.ctl.function(name)) else ctx.k;
            p.* = .{ .a = ctx.gpu.gpa, .rig = &group.rig, .k = k, .tower = e[0], .case = e[1] };
        }
        for (decoder.cases, &group.decoders) |e, *p| {
            const k = if (decoder.control(e)) |name| ctx.k.with("attend", try ctx.ctl.function(name)) else ctx.k;
            p.* = .{ .a = ctx.gpu.gpa, .rig = &group.rig, .k = k, .case = e };
        }
        for (&group.towers) |*p| try list.append(a, p.kernel());
        for (&group.decoders) |*p| try list.append(a, p.kernel());
    }

    pub fn deinit(group: *Group) void {
        if (!group.live) return;
        for (&group.towers) |*p| p.free();
        for (&group.decoders) |*p| p.free();
        group.rig.deinit();
        group.live = false;
    }
};
