//! A captured graph that serves any sequence laid out alike: the pointers into its regions are rewritten in place.

const std = @import("std");
const abi = @import("abi.zig");
const graph = @import("graph.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const max_regions = 24;

/// One sequence's regions: where each starts, and (for the capturing sequence) how far it reaches.
pub const Regions = struct {
    base: [max_regions]u64 = @splat(0),
    len: [max_regions]u64 = @splat(0),
    n: usize = 0,

    pub fn add(r: *Regions, base: u64, len: u64) void {
        r.base[r.n] = base;
        r.len[r.n] = len;
        r.n += 1;
    }

    /// The region `v` points into, if any.
    pub fn find(r: *const Regions, v: u64) ?u8 {
        for (r.base[0..r.n], r.len[0..r.n], 0..) |b, l, i| if (l > 0 and v >= b and v < b + l) return @intCast(i);
        return null;
    }

    /// `b` starts each of `a`'s regions where `a` does (it may have more regions after them).
    pub fn same(a: *const Regions, b: *const Regions) bool {
        return b.n >= a.n and std.mem.eql(u64, a.base[0..a.n], b.base[0..a.n]);
    }
};

const Word = struct { at: u32, region: u8, off: u64 }; // a pointer at bytes[at], `off` past its region's base

/// A kernel node whose arguments point into the sequence: its launch, its argument bytes and the pointer words.
const KernelPatch = struct {
    node: graph.Node,
    p: abi.KernelNodeParams,
    bytes: []align(16) u8,
    offsets: []u32,
    ptrs: []?*anyopaque,
    words: []Word,

    fn deinit(x: *KernelPatch, gpa: std.mem.Allocator) void {
        gpa.free(x.bytes);
        gpa.free(x.offsets);
        gpa.free(x.ptrs);
        gpa.free(x.words);
    }
};

const End = struct { region: u8, off: u64 };
const CopyPatch = struct { node: graph.Node, p: abi.Memcpy3D, src: ?End, dst: ?End };
const SetPatch = struct { node: graph.Node, p: abi.MemsetParams, dst: End };

/// What moves: pointers into the capturing sequence as offsets from their regions (no plan: a node could not be read).
pub const Plan = struct {
    kernels: []KernelPatch,
    copies: []CopyPatch,
    sets: []SetPatch,
    bound: Regions, // the sequence the executable graph points at now

    pub fn deinit(pl: *Plan, gpa: std.mem.Allocator) void {
        for (pl.kernels) |*x| x.deinit(gpa);
        gpa.free(pl.kernels);
        gpa.free(pl.copies);
        gpa.free(pl.sets);
        pl.* = undefined;
    }

    /// The patched nodes' pointers moved to `to`'s regions (true), or nothing when the graph points there already.
    pub fn bind(pl: *Plan, exec: graph.Exec, ctx: abi.Context, to: *const Regions) Error!bool {
        if (pl.bound.same(to)) return false;
        if (to.n < pl.bound.n) return error.Invalid;
        for (pl.kernels) |*x| {
            for (x.words) |w| std.mem.writeInt(u64, x.bytes[w.at..][0..8], to.base[w.region] + w.off, .little);
            for (x.offsets, 0..) |o, i| x.ptrs[i] = &x.bytes[o];
            x.p.params = x.ptrs.ptr;
            try exec.d.check(exec.d.api.cuGraphExecKernelNodeSetParams_v2(exec.handle, x.node, &x.p), "cuGraphExecKernelNodeSetParams");
        }
        for (pl.copies) |*x| {
            if (x.src) |s| x.p.src_device = to.base[s.region] + s.off;
            if (x.dst) |s| x.p.dst_device = to.base[s.region] + s.off;
            try exec.d.check(exec.d.api.cuGraphExecMemcpyNodeSetParams(exec.handle, x.node, &x.p, ctx), "cuGraphExecMemcpyNodeSetParams");
        }
        for (pl.sets) |*x| {
            x.p.dst = to.base[x.dst.region] + x.dst.off;
            try exec.d.check(exec.d.api.cuGraphExecMemsetNodeSetParams(exec.handle, x.node, &x.p, ctx), "cuGraphExecMemsetNodeSetParams");
        }
        @memcpy(pl.bound.base[0..pl.bound.n], to.base[0..pl.bound.n]);
        return true;
    }
};

/// Argument `i`'s offset and size: true, false past the last argument, null when the driver cannot say.
fn paramInfo(d: *const Driver, p: abi.KernelNodeParams, i: usize, off: *usize, size: *usize) ?bool {
    if (p.func != null) {
        const f = d.api.cuFuncGetParamInfo orelse return null;
        return f(p.func, i, off, size) == abi.success;
    }
    const f = d.api.cuKernelGetParamInfo orelse return null;
    if (p.kern == null) return null;
    return f(p.kern, i, off, size) == abi.success;
}

fn end(r: *const Regions, kind: c_uint, v: u64) ?End {
    if (kind != abi.memory_device and kind != abi.memory_unified) return null;
    const i = r.find(v) orelse return null;
    return .{ .region = i, .off = v - r.base[i] };
}

/// A kernel node's argument bytes, laid out 8-aligned, with the words that point into `r` (null: unreadable).
fn kernelPatch(gpa: std.mem.Allocator, d: *const Driver, node: graph.Node, r: *const Regions) !?KernelPatch {
    var p: abi.KernelNodeParams = undefined;
    try d.check(d.api.cuGraphKernelNodeGetParams_v2(node, &p), "cuGraphKernelNodeGetParams");
    if (p.extra != null) return null;
    const args = p.params orelse return null;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    var offsets: std.ArrayList(u32) = .empty;
    defer offsets.deinit(gpa);
    var words: std.ArrayList(Word) = .empty;
    defer words.deinit(gpa);
    var i: usize = 0;
    while (true) : (i += 1) {
        var off: usize = 0;
        var size: usize = 0;
        if (!(paramInfo(d, p, i, &off, &size) orelse return null)) break;
        const at = std.mem.alignForward(usize, bytes.items.len, 8);
        try bytes.appendNTimes(gpa, 0, at - bytes.items.len);
        const src: [*]const u8 = @ptrCast(args[i] orelse return null);
        try bytes.appendSlice(gpa, src[0..size]);
        try offsets.append(gpa, @intCast(at));
        var w: usize = 0;
        while (w + 8 <= size) : (w += 8) {
            const v = std.mem.readInt(u64, src[w..][0..8], .little);
            if (r.find(v)) |reg| try words.append(gpa, .{ .at = @intCast(at + w), .region = reg, .off = v - r.base[reg] });
        }
    }
    if (words.items.len == 0) return .{ .node = node, .p = p, .bytes = &.{}, .offsets = &.{}, .ptrs = &.{}, .words = &.{} };
    const owned = try gpa.alignedAlloc(u8, .@"16", bytes.items.len);
    @memcpy(owned, bytes.items);
    errdefer gpa.free(owned);
    const offs = try offsets.toOwnedSlice(gpa);
    errdefer gpa.free(offs);
    const ptrs = try gpa.alloc(?*anyopaque, offs.len);
    errdefer gpa.free(ptrs);
    return .{ .node = node, .p = p, .bytes = owned, .offsets = offs, .ptrs = ptrs, .words = try words.toOwnedSlice(gpa) };
}

/// The plan for graph `g` (captured on the sequence whose regions are `r`), or null when a node cannot be moved.
pub fn read(gpa: std.mem.Allocator, g: graph.Graph, r: *const Regions) !?Plan {
    const d = g.d;
    const n = try g.nodeCount();
    const nodes = try gpa.alloc(graph.Node, n);
    defer gpa.free(nodes);
    var kernels: std.ArrayList(KernelPatch) = .empty;
    var copies: std.ArrayList(CopyPatch) = .empty;
    var sets: std.ArrayList(SetPatch) = .empty;
    var ok = false;
    defer if (!ok) {
        for (kernels.items) |*x| x.deinit(gpa);
        kernels.deinit(gpa);
        copies.deinit(gpa);
        sets.deinit(gpa);
    };
    for (try g.nodes(nodes)) |node| {
        var kind: c_int = -1;
        try d.check(d.api.cuGraphNodeGetType(node, &kind), "cuGraphNodeGetType");
        switch (kind) {
            abi.node_empty, abi.node_wait_event, abi.node_event_record => {},
            abi.node_kernel => {
                const x = try kernelPatch(gpa, d, node, r) orelse return null;
                if (x.words.len > 0) try kernels.append(gpa, x);
            },
            abi.node_memcpy => {
                var p: abi.Memcpy3D = .{};
                try d.check(d.api.cuGraphMemcpyNodeGetParams(node, &p), "cuGraphMemcpyNodeGetParams");
                const c: CopyPatch = .{ .node = node, .p = p, .src = end(r, p.src_type, p.src_device), .dst = end(r, p.dst_type, p.dst_device) };
                if (c.src != null or c.dst != null) try copies.append(gpa, c);
            },
            abi.node_memset => {
                var p: abi.MemsetParams = .{};
                try d.check(d.api.cuGraphMemsetNodeGetParams(node, &p), "cuGraphMemsetNodeGetParams");
                if (end(r, abi.memory_device, p.dst)) |e| try sets.append(gpa, .{ .node = node, .p = p, .dst = e });
            },
            else => return null, // host, child graph, allocation or another kind: not moved
        }
    }
    ok = true;
    return .{ .kernels = try kernels.toOwnedSlice(gpa), .copies = try copies.toOwnedSlice(gpa), .sets = try sets.toOwnedSlice(gpa), .bound = r.* };
}

test "regions find the region and offset a pointer falls in" {
    var r: Regions = .{};
    r.add(0x1000, 0x100);
    r.add(0x8000, 0x10);
    try std.testing.expectEqual(@as(?u8, 0), r.find(0x10ff));
    try std.testing.expectEqual(@as(?u8, null), r.find(0x1100));
    try std.testing.expectEqual(@as(?u8, 1), r.find(0x8000));
    var s = r;
    try std.testing.expect(r.same(&s));
    s.add(0x20000, 0x10); // a region the capture never saw does not count
    try std.testing.expect(r.same(&s));
    s.base[1] = 0x9000;
    try std.testing.expect(!r.same(&s));
    const e = end(&r, abi.memory_device, 0x8004).?;
    try std.testing.expectEqual(@as(u64, 4), e.off);
    try std.testing.expectEqual(@as(?End, null), end(&r, 1, 0x8004)); // host memory is never moved
}
