//! A captured step as CUDA graph segments split at its cross-rank gathers, which run eager between them: a gather
//! between the nodes inside a graph costs several times one launched on the stream. The same work in the same order.
//! A plan captured for one sequence serves any other: every sequence lays out its memory alike, so the kernel
//! arguments that point into the capturing sequence's ranges are found at capture and moved by the base difference.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;

pub const Gather = struct { send: u64, recv: u64, count: usize, dt: cuda.nccl.DataType };

/// A sequence's two address ranges: its own allocation and its growing caches' reserved range.
pub const Bases = struct {
    mem: u64 = 0,
    mem_len: u64 = 0,
    grow: u64 = 0,
    grow_len: u64 = 0,

    fn region(b: Bases, v: u64) ?u1 {
        if (b.mem_len > 0 and v >= b.mem and v < b.mem + b.mem_len) return 0;
        if (b.grow_len > 0 and v >= b.grow and v < b.grow + b.grow_len) return 1;
        return null;
    }

    fn base(b: Bases, r: u1) u64 {
        return if (r == 0) b.mem else b.grow;
    }

    pub fn same(a: Bases, b: Bases) bool {
        return a.mem == b.mem and a.grow == b.grow;
    }
};

const Reloc = struct { at: u32, region: u1, off: u64 }; // a pointer at storage[at], ``off`` past its region's base

/// A kernel node whose arguments point into a sequence: its launch, its argument bytes and where the pointers sit.
const Patch = struct {
    node: cuda.graph.Node,
    p: abi.KernelNodeParams,
    storage: []align(16) u8,
    offsets: []u32,
    ptrs: []?*anyopaque,
    relocs: []Reloc,

    fn deinit(x: *Patch, gpa: std.mem.Allocator) void {
        gpa.free(x.storage);
        gpa.free(x.offsets);
        gpa.free(x.ptrs);
        gpa.free(x.relocs);
    }
};

const Segment = struct {
    exec: cuda.graph.Exec,
    tmpl: ?cuda.graph.Graph = null, // kept while ``patches`` name its nodes
    patches: []Patch = &.{},

    fn deinit(s: *Segment, gpa: std.mem.Allocator) void {
        s.exec.deinit();
        for (s.patches) |*x| x.deinit(gpa);
        if (s.patches.len > 0) gpa.free(s.patches);
        if (s.tmpl) |*g| g.deinit();
    }
};

pub const Item = union(enum) { graph: Segment, gather: Gather };

pub const Plan = struct {
    items: []Item,
    /// Every node was read: the plan serves any sequence once ``bind`` moved its pointers.
    shareable: bool,
    bound: Bases,
    last: u64 = 0, // the engine's use counter at its last launch (the least recent goes first)

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        for (p.items) |*it| switch (it.*) {
            .graph => |*g| g.deinit(gpa),
            .gather => {},
        };
        gpa.free(p.items);
        p.* = undefined;
    }

    /// The pointers moved to ``to``'s ranges (nothing to do when bound there already).
    pub fn bind(p: *Plan, to: Bases) !void {
        if (p.bound.same(to)) return;
        for (p.items) |*it| switch (it.*) {
            .graph => |*g| for (g.patches) |*x| {
                for (x.relocs) |r| std.mem.writeInt(u64, x.storage[r.at..][0..8], to.base(r.region) + r.off, .little);
                for (x.offsets, 0..) |o, i| x.ptrs[i] = &x.storage[o];
                x.p.params = x.ptrs.ptr;
                try g.exec.setKernelRaw(x.node, &x.p);
            },
            .gather => {},
        };
        p.bound = to;
    }

    pub fn launch(p: Plan, nccl: *cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream) !void {
        for (p.items) |it| switch (it) {
            .graph => |g| try g.exec.launchOn(stream),
            .gather => |g| try nccl.check(nccl.api.ncclAllGather(g.send, g.recv, g.count, g.dt, comm, stream.handle), "ncclAllGather"),
        };
    }
};

/// Argument ``i``'s offset and size: true, false past the last argument, null when the driver cannot say.
fn paramInfo(d: *const cuda.Driver, p: abi.KernelNodeParams, i: usize, off: *usize, size: *usize) ?bool {
    const r = if (p.func != null) d.api.cuFuncGetParamInfo(p.func, i, off, size) else if (p.kern != null) d.api.cuKernelGetParamInfo(p.kern, i, off, size) else return null;
    return r == abi.success;
}

/// What a capture records: each segment's graph, ended and instantiated at a gather (``split``) or at the end.
pub const Recorder = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    split: bool,
    seq: Bases = .{}, // the capturing sequence's ranges (none: a plan for this sequence only)
    shareable: bool = true,
    items: std.ArrayList(Item) = .empty,

    pub fn begin(r: *Recorder) !void {
        try cuda.graph.beginCapture(r.stream, .thread_local);
    }

    /// The kernel nodes whose arguments point into the capturing sequence (an unreadable node: the plan unshared).
    fn patches(r: *Recorder, g: cuda.graph.Graph) ![]Patch {
        var out: std.ArrayList(Patch) = .empty;
        errdefer {
            for (out.items) |*x| x.deinit(r.gpa);
            out.deinit(r.gpa);
        }
        const n = try g.nodeCount();
        const nodes = try r.gpa.alloc(cuda.graph.Node, n);
        defer r.gpa.free(nodes);
        for (try g.nodes(nodes)) |node| {
            const t = try g.nodeType(node);
            if (t == 5 or t == 6 or t == 7) continue; // empty and event nodes carry no addresses
            if (t == 1 or t == 2) { // a copy or memset: fine while it touches none of the sequence's memory
                var raw: [256]u64 = undefined;
                try g.copyParams(node, t, &raw);
                for (raw) |v| if (r.seq.region(v) != null) {
                    r.shareable = false;
                };
                if (!r.shareable) break;
                continue;
            }
            if (t != 0) {
                r.shareable = false;
                break;
            }
            const p = try g.kernelParams(node);
            const args = p.params orelse {
                r.shareable = false;
                break;
            };
            var offs: std.ArrayList(u32) = .empty;
            defer offs.deinit(r.gpa);
            var bytes: std.ArrayList(u8) = .empty;
            defer bytes.deinit(r.gpa);
            var relocs: std.ArrayList(Reloc) = .empty;
            defer relocs.deinit(r.gpa);
            var i: usize = 0;
            while (true) : (i += 1) {
                var off: usize = 0;
                var size: usize = 0;
                const ok = paramInfo(g.d, p, i, &off, &size) orelse {
                    r.shareable = false;
                    break;
                };
                if (!ok) break;
                const at: u32 = @intCast(std.mem.alignForward(usize, bytes.items.len, 8));
                try bytes.appendNTimes(r.gpa, 0, at - bytes.items.len);
                const src: [*]const u8 = @ptrCast(args[i].?);
                try bytes.appendSlice(r.gpa, src[0..size]);
                try offs.append(r.gpa, at);
                var w: usize = 0;
                while (w + 8 <= size) : (w += 8) {
                    const v = std.mem.readInt(u64, src[w..][0..8], .little);
                    if (r.seq.region(v)) |reg| try relocs.append(r.gpa, .{ .at = at + @as(u32, @intCast(w)), .region = reg, .off = v - r.seq.base(reg) });
                }
            }
            if (!r.shareable) break;
            if (relocs.items.len == 0) continue;
            const storage = try r.gpa.alignedAlloc(u8, .@"16", bytes.items.len);
            @memcpy(storage, bytes.items);
            const no = offs.items.len;
            try out.append(r.gpa, .{
                .node = node,
                .p = p,
                .storage = storage,
                .offsets = try offs.toOwnedSlice(r.gpa),
                .ptrs = try r.gpa.alloc(?*anyopaque, no),
                .relocs = try relocs.toOwnedSlice(r.gpa),
            });
        }
        if (!r.shareable) {
            for (out.items) |*x| x.deinit(r.gpa);
            out.deinit(r.gpa);
            return &.{};
        }
        return out.toOwnedSlice(r.gpa);
    }

    /// The capture so far as one segment (none when it captured nothing).
    fn cut(r: *Recorder) !void {
        var g = try cuda.graph.endCapture(r.stream);
        var keep = false;
        defer if (!keep) g.deinit();
        if (try g.nodeCount() == 0) return;
        const ps: []Patch = if (r.shareable and r.seq.mem_len > 0) try r.patches(g) else &.{};
        var seg: Segment = .{ .exec = try g.instantiate(), .patches = ps };
        errdefer seg.deinit(r.gpa);
        try seg.exec.upload(r.stream);
        if (ps.len > 0) {
            seg.tmpl = g;
            keep = true;
        }
        try r.items.append(r.gpa, .{ .graph = seg });
    }

    /// A gather inside the capture: the segment before it ends, the gather runs eager at replay, a new one begins.
    pub fn gather(r: *Recorder, g: Gather) !void {
        try r.cut();
        try r.items.append(r.gpa, .{ .gather = g });
        try r.begin();
    }

    pub fn finish(r: *Recorder) !Plan {
        try r.cut();
        return .{ .items = try r.items.toOwnedSlice(r.gpa), .shareable = r.shareable and r.seq.mem_len > 0, .bound = r.seq };
    }

    /// An abandoned capture: the capture ended and the segments made so far freed.
    pub fn abandon(r: *Recorder) void {
        if (cuda.graph.endCapture(r.stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        for (r.items.items) |*it| switch (it.*) {
            .graph => |*g| g.deinit(r.gpa),
            .gather => {},
        };
        r.items.deinit(r.gpa);
    }
};
