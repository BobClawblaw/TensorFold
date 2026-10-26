//! MiniMax-M3.1-Flash native CUDA at TP (TF_MINIMAX_RANK, TF_MINIMAX_WORLD, NCCL_COMM_ID): rank 0 serves.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const mm_lanes = @import("cuda_lanes.zig");
const state = @import("cuda_state.zig");
const vision_mod = @import("cuda_vision.zig");

pub const model_type = "minimax_m3";
pub const formats: []const []const u8 = &.{"nvfp4-mix"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
/// Prompt chunks keep their arithmetic by chunk start: states keep and resume only there.
pub const snap_grid: u32 = prompt_rows;
pub const explains_open = true;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    vision: ?vision_mod.Limits = null, // --vision: the last rank's tower, every rank's image rows
};
/// An image takes at most 576 tokens; a request's images 4096 in all unless --vision-image-tokens says otherwise.
pub const default_image_tokens: u32 = 4096;
/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    peer_room: ?Peer = null,
    target: *anyopaque, // the prompt cache keeps this rank group's states (snap_*)
    vision: ?*vision_mod.Images = null, // the server's image offer (rank 0's processor and limits)
};

/// The tower's bytes on this rank: the last rank holds it (rows and patch staging on every rank are small).
pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    _ = a;
    _ = io;
    _ = dir;
    const p = place() catch return 0;
    return if (p.rank + 1 == p.world) 1_734_000_000 else 0;
}

/// This rank's place: TF_MINIMAX_RANK of TF_MINIMAX_WORLD (both, or neither for one rank).
pub const Place = struct { rank: usize, world: usize };

pub fn place() !Place {
    const r = std.c.getenv("TF_MINIMAX_RANK");
    const w = std.c.getenv("TF_MINIMAX_WORLD");
    if (r == null and w == null) return .{ .rank = 0, .world = 1 };
    if (r == null or w == null) return error.NoRank;
    const p: Place = .{ .rank = std.fmt.parseInt(usize, std.mem.span(r.?), 10) catch return error.NoRank, .world = std.fmt.parseInt(usize, std.mem.span(w.?), 10) catch return error.NoRank };
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > 4) return error.TooManyRanks;
    return p;
}

/// The weight bytes as the mix serves them (a folder may hold both expert forms; its files would count both).
pub fn tensorWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision: bool) !u64 {
    _ = vision;
    const c = try cfg.Config.read(a, io, dir);
    var thin = c;
    if (std.c.getenv("TF_MINIMAX_LAYERS")) |v| thin.layers = @min(c.layers, std.fmt.parseInt(u32, std.mem.span(v), 10) catch c.layers);
    const mix = try weights.Mix.read(a, io, dir, &c, if (std.c.getenv("TF_MINIMAX_NVFP4_LAYERS")) |v| std.mem.span(v) else null);
    return modelBytes(&thin, &mix);
}

fn modelBytes(c: *const cfg.Config, mix: *const weights.Mix) u64 {
    const h: u64 = c.hidden;
    const d: u64 = c.head_dim;
    const mx = struct {
        fn of(n: u64, k: u64) u64 {
            return n * k + n * k / 32;
        }
    }.of;
    const attn = mx((c.heads + 2 * c.kv_heads) * d + (c.index_heads + 1) * c.index_dim, h) + mx(h, c.heads * d);
    var total: u64 = 2 * @as(u64, c.vocab) * h * 2 + h * 2;
    const inter: u64 = c.moe_intermediate;
    const e: u64 = c.experts;
    for (0..c.layers) |i| {
        total += attn + 4 * h * 2 + 4 * d * 2;
        if (!c.layer[i].moe) {
            total += mx(2 * @as(u64, c.dense_intermediate), h) + mx(h, c.dense_intermediate);
            continue;
        }
        total += mx(2 * @as(u64, c.shared_intermediate), h) + mx(h, c.shared_intermediate) + e * h * 4 + e * 4;
        const mats = 3 * inter * h; // gate, up and down weights of one expert
        total += if (mix.nvfp4[i]) e * (mats / 2 + mats / 16 + 12) else e * (mats * 3 / 8 + 2 * 2 * (mats / 64));
    }
    return total;
}

/// Bytes this rank's weights take: its share of the split axes, the embeddings whole.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    const p = place() catch return total;
    const c = cfg.Config.read(a, io, dir) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    const embed: u64 = @as(u64, c.vocab) * c.hidden * 2;
    return @intCast(@as(u128, total - embed) * f[0] / f[1] + embed);
}

/// Slots for `context` plus a verify window, whole index blocks.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + engine.window_rows, 128);
}

pub const window_rows: u32 = engine.window_rows;

const Owned = struct {
    gpa: std.mem.Allocator,
    lib: ?cuda.nccl.Library = null,
    comm: cuda.nccl.Comm = null,
    group: ?cuda.tp_group.Group = null, // lib and comm (owned above) with RoCE writes for two ranks' small gathers
    e: *engine.Engine,
    link: ?follow.Link = null,
    lanes: mm_lanes.Lanes,
    place: Place,
};

const env = @import("sys").env;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels;
    const p = try place();
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    if (p.world > 1) try cuda.tp_watch.start(io, p.rank, p.world);
    return openRank(gpa, io, ctx, dir, o, p) catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
}

fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, p: Place) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = undefined, .lanes = undefined, .place = p };
    if (p.world > 1) {
        if (std.c.getenv("NCCL_RUNTIME_CONNECT") == null) _ = env.setenv("NCCL_RUNTIME_CONNECT", "0", 0);
        if (std.c.getenv("NCCL_CUMEM_HOST_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_HOST_ENABLE", "0", 0);
        if (std.c.getenv("NCCL_CUMEM_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_ENABLE", "0", 0);
        if (p.rank > 0 and std.c.getenv("NCCL_SOCKET_IFNAME") != null) _ = env.unsetenv("NCCL_COMM_ID");
        var lib = try cuda.nccl.Library.open();
        errdefer lib.close();
        var id: cuda.nccl.UniqueId = undefined;
        if (p.rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            cuda.tp_watch.sendId(&id.internal);
        } else id.internal = try cuda.tp_watch.takeId(io);
        var comm: cuda.nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
        own.lib = lib;
        own.comm = comm;
    }
    const cap = capacity(o.context);
    const nvfp4 = if (std.c.getenv("TF_MINIMAX_NVFP4_LAYERS")) |v| std.mem.span(v) else null;
    const thin = if (std.c.getenv("TF_MINIMAX_LAYERS")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.BadLayers else 0;
    // the release's DSpark head drafts when drafts are on and dspark/ is there (TF_MINIMAX_DRAFTER=0: copies only)
    const off = if (std.c.getenv("TF_MINIMAX_DRAFTER")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
    const has = blk: {
        const path = try std.fs.path.join(gpa, &.{ dir, "dspark", "config.json" });
        defer gpa.free(path);
        std.Io.Dir.cwd().access(io, path, .{}) catch break :blk false;
        break :blk true;
    };
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, p.rank, p.world, .{ .capacity = cap, .nvfp4 = nvfp4, .drafts = o.drafts, .layers = thin, .drafter = o.drafts and has and !off and thin == 0, .vision = o.vision });
    errdefer own.e.deinit();
    if (p.world > 1) {
        own.link = try follow.Link.init(gpa, ctx.d, &own.lib.?, own.comm, own.e.stream, @max(cap, engine.prompt_rows));
    }
    errdefer if (own.link) |*l| l.deinit();
    if (p.world > 1) {
        // small gathers by RoCE writes on two ranks, both or neither (TF_ROCE=0: NCCL; TF_ROCE_MAX_KB: the cap)
        own.group = .{ .lib = own.lib.?, .comm = own.comm, .rank = p.rank, .world = p.world };
        try own.group.?.openRoce(gpa, io, ctx.d, own.e.stream, try roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |v| std.mem.span(v) else null));
        own.e.exchange = .{ .nccl = .{ .lib = &own.lib.?, .comm = own.comm, .group = &own.group.? } };
    }
    errdefer if (own.group) |*g| if (g.fast) |f| f.deinit();
    // rank 0 admits by the smallest room of the ranks
    var peer: ?Peer = null;
    if (p.world > 1) {
        const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
        var rooms = try cuda.DeviceBuffer.alloc(ctx.d, 8 * p.world);
        defer rooms.free();
        const at = rooms.ptr + p.rank * 8;
        try rooms.upload(p.rank * 8, std.mem.asBytes(&mine));
        try own.lib.?.check(own.lib.?.api.ncclAllGather(at, rooms.ptr, 8, .u8, own.comm, own.e.stream.handle), "ncclAllGather");
        try own.e.stream.synchronize();
        const all = try gpa.alloc(u64, p.world);
        defer gpa.free(all);
        try rooms.download(0, std.mem.sliceAsBytes(all));
        for (all[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
            peer = .{ .bytes = r, .rank = q };
        };
        try cuda.tp_watch.ready(io, own.e.checkpoint_digest);
    }
    if (std.c.getenv("TF_MINIMAX_TIME_WINDOWS")) |v| { // dev: the lane cost tables' window and shared costs
        var ms: [engine.shared_rows + 1]f64 = undefined;
        try own.e.timeWindows(std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1024, &ms);
        if (p.rank == 0) for (ms[0..engine.shared_rows], 1..) |x, w| std.log.info("window of {d} rows: {d:.2} ms", .{ w, x });
        if (p.rank == 0) std.log.info("drafter block of 7: {d:.2} ms", .{ms[engine.shared_rows]});
    }
    if (p.rank != 0) {
        follow.follow(own.e, &own.link.?) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            cuda.tp_watch.fail(@errorName(err));
            std.c._exit(1);
        };
        cuda.tp_watch.ended();
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        release(own);
        std.process.exit(0);
    }
    const alone = if (std.c.getenv("TF_MINIMAX_SHARED")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
    if (own.link) |*l| l.asleep = cuda.tp_watch.open();
    own.lanes = .{ .x = .{ .e = own.e, .link = if (own.link) |*l| l else null }, .alone = alone };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = mm_lanes.max_rows,
        .stream_bytes = own.e.seqBytes(),
        .free_streams = 1,
        .ctx = own,
        .deinit = close,
        .peer_room = peer,
        .target = own,
        .vision = own.e.images,
    };
}

/// A state kept at `at` (rounded as every rank's copy): its bytes on this rank, the largest's.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    return own.e.stateBytes(at);
}

/// The prompt pass stands at `at` (a mark at a chunk start): every rank copies the state there under one new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const own: *Owned = @ptrCast(@alignCast(ptr));
    if (own.e.pos() != at) return error.NotAtMark;
    const k = try own.gpa.create(engine.Kept);
    errdefer own.gpa.destroy(k);
    k.* = try own.lanes.x.save(at);
    return k;
}

/// A restore outside a prompt pass would leave the other ranks behind: states resume inside it (Lanes.resumeFrom).
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    own.lanes.x.drop(k.id);
    own.gpa.destroy(k);
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    cuda.tp_watch.ended();
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    release(own);
}

/// TF_ROCE_MAX_KB in KiB (default 4 MiB, as GLM's): a rank's largest gather by RoCE writes.
fn roceBytes(text: ?[]const u8) !usize {
    const v = text orelse return 4096 << 10;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadRoceCap;
    if (n == 0 or n > std.math.maxInt(u32) / 1024) return error.BadRoceCap;
    return n * 1024;
}

fn release(own: *Owned) void {
    if (own.link) |*l| l.deinit();
    own.e.deinit();
    if (own.group) |*g| if (g.fast) |f| f.deinit();
    if (own.lib) |*lib| {
        _ = lib.api.ncclCommDestroy(own.comm);
        lib.close();
    }
    own.gpa.destroy(own);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "MiniMax-M3.1-Flash runs at TP over hosts: set TF_MINIMAX_RANK and TF_MINIMAX_WORLD on every host (or neither for one GPU)",
        error.NoCommId => "MiniMax-M3.1-Flash's ranks meet over NCCL: with TF_MINIMAX_WORLD 2 or more, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.TooManyRanks => "MiniMax-M3.1-Flash on CUDA runs on at most 4 ranks (one key head each): set TF_MINIMAX_WORLD to 4 or fewer",
        error.NoMix, error.BadMix => "the checkpoint's config.json names no nvfp4_mix (which layers' experts are NVFP4); convert it with tools/zig/minimax_m3_nvfp4_mix.py, or set TF_MINIMAX_NVFP4_LAYERS",
        error.UnexpectedMiniMaxTensor => "a tensor is not stored as MiniMax-M3.1-Flash's release stores it (the log names it)",
        error.NoVision => "images are not served by MiniMax-M3.1-Flash on CUDA yet",
        error.BadLayers => "TF_MINIMAX_LAYERS: a layer count from 1 to the model's 60 (a thin model for checks)",
        else => null,
    };
}

test "a place from the environment's shape: one rank without it, at most four, the rank inside its world" {
    try std.testing.expect(explain(null, error.TooManyRanks) != null);
    try std.testing.expect(explain(null, error.NoMix) != null);
    try std.testing.expectEqual(@as(usize, 32768 + 128), capacity(32768));
}

test "the model's bytes under a mix: every layer NVFP4 is the release's 231 GiB language model" {
    var c = cfg.Config{};
    for (0..c.layers) |i| c.layer[i] = .{ .sparse = true, .moe = i >= 3 };
    var all: weights.Mix = .{};
    for (0..c.layers) |i| all.nvfp4[i] = c.layer[i].moe;
    const bytes = modelBytes(&c, &all);
    try std.testing.expect(bytes > 230 * (1 << 30) and bytes < 233 * (1 << 30));
    const none: weights.Mix = .{};
    try std.testing.expect(modelBytes(&c, &none) < 190 * (1 << 30));
}

comptime {
    _ = state;
    _ = core;
}
