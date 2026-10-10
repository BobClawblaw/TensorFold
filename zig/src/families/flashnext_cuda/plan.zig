//! A captured step as CUDA graph segments split at its cross-rank gathers, which run eager between them: a gather
//! between the nodes inside a graph costs several times one launched on the stream. The same work in the same order.
const std = @import("std");
const cuda = @import("cuda");

pub const Gather = struct { send: u64, recv: u64, count: usize, dt: cuda.nccl.DataType };
pub const Item = union(enum) { graph: cuda.graph.Exec, gather: Gather };

pub const Plan = struct {
    items: []Item,

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        for (p.items) |*it| switch (it.*) {
            .graph => |*g| g.deinit(),
            .gather => {},
        };
        gpa.free(p.items);
        p.* = undefined;
    }

    pub fn launch(p: Plan, nccl: *cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream) !void {
        for (p.items) |it| switch (it) {
            .graph => |g| try g.launchOn(stream),
            .gather => |g| try nccl.check(nccl.api.ncclAllGather(g.send, g.recv, g.count, g.dt, comm, stream.handle), "ncclAllGather"),
        };
    }
};

/// What a capture records: each segment's graph, ended and instantiated at a gather (``split``) or at the end.
pub const Recorder = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    split: bool,
    items: std.ArrayList(Item) = .empty,

    pub fn begin(r: *Recorder) !void {
        try cuda.graph.beginCapture(r.stream, .thread_local);
    }

    /// The capture so far as one segment (none when it captured nothing).
    fn cut(r: *Recorder) !void {
        var g = try cuda.graph.endCapture(r.stream);
        defer g.deinit();
        if (try g.nodeCount() == 0) return;
        const ex = try g.instantiate();
        errdefer {
            var x = ex;
            x.deinit();
        }
        try ex.upload(r.stream);
        try r.items.append(r.gpa, .{ .graph = ex });
    }

    /// A gather inside the capture: the segment before it ends, the gather runs eager at replay, a new one begins.
    pub fn gather(r: *Recorder, g: Gather) !void {
        try r.cut();
        try r.items.append(r.gpa, .{ .gather = g });
        try r.begin();
    }

    pub fn finish(r: *Recorder) !Plan {
        try r.cut();
        return .{ .items = try r.items.toOwnedSlice(r.gpa) };
    }

    /// An abandoned capture: the capture ended and the segments made so far freed.
    pub fn abandon(r: *Recorder) void {
        if (cuda.graph.endCapture(r.stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        for (r.items.items) |*it| switch (it.*) {
            .graph => |*g| g.deinit(),
            .gather => {},
        };
        r.items.deinit(r.gpa);
    }
};
