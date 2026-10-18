//! A stream's caches and a forward's scratch. Everything a stream keeps is indexed by position (window rings,
//! compressor projection rings) or by compressed group (the four sources' latents and index keys), so taking back
//! rejected draft rows only moves `len`.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Window ring rows a layer: the 128-key window behind the longest call's rows.
pub const ring = 4096;
/// Rows a forward takes at most (a prompt chunk).
pub const max_rows = ring - c.window;
/// Compressed picks a row: the indexer's top-k.
pub const picks = c.index_topk;
/// Rows that take the split attention (decode, windows, drafts), and its chunks of 64 keys at most a row.
pub const split_rows = 16;
pub const split_chunks = (c.window + picks + 15) / 16;
/// Scores a prompt call's indexer holds at once (f32): rows are taken in slices of this many keys' worth.
pub const score_budget = 64 << 20;

pub const Storage = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: std.ArrayList(mtl.Buffer) = .empty,

    pub fn alloc(self: *Storage, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(@max(bytes, 16), opts);
        errdefer b.deinit();
        try self.buffers.append(self.gpa, b);
        return b;
    }

    pub fn deinit(self: *Storage) void {
        for (self.buffers.items) |b| b.deinit();
        self.buffers.deinit(self.gpa);
    }
};

/// A KV source's compressed latents (bf16 [groups, 512], RoPE'd and fp4-rounded) and index keys (bf16 [groups, 128]).
pub const Source = struct { latents: mtl.Buffer, keys: mtl.Buffer, groups: usize };

pub const Cache = struct {
    gpa: std.mem.Allocator,
    memory: Storage,
    capacity: usize,
    len: usize = 0,
    /// Each layer's window keys (bf16 [ring, 512] by position).
    windows: [c.layers]mtl.Buffer,
    sources: [c.kv_sources.len]Source,
    /// The ratio-2 sources' projections (f32 [ring, 512] by position), pooled once a pair completes.
    proj_kv: [c.kv_sources.len]?mtl.Buffer,
    proj_score: [c.kv_sources.len]?mtl.Buffer,
    /// Compressed token ids by position, for the engram hashes.
    engram_ids: std.ArrayList(i32) = .empty,
    logits: mtl.Buffer,
    /// With the DSpark head: each stage's window keys of the target's positions (bf16 [ring, 512] by position).
    stage_rings: ?[c.stages]mtl.Buffer = null,
    /// Where the last call started and its rows (a window's keep takes back to a prefix of them).
    last_start: usize = 0,
    last_rows: usize = 0,
    /// The drafts held for the next window (the head's greedy chain).
    held: [c.draft_block]u32 = @splat(0),
    held_n: usize = 0,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, capacity: usize) !Cache {
        return initWith(gpa, device, capacity, false);
    }

    /// `drafts`: the DSpark head's rings.
    pub fn initWith(gpa: std.mem.Allocator, device: mtl.Device, capacity: usize, drafts: bool) !Cache {
        var out: Cache = .{ .gpa = gpa, .memory = .{ .gpa = gpa, .device = device }, .capacity = capacity, .windows = undefined, .sources = undefined, .proj_kv = @splat(null), .proj_score = @splat(null), .logits = undefined };
        errdefer out.memory.deinit();
        out.logits = try out.memory.alloc(c.vocab * 2);
        for (&out.windows) |*w| w.* = try out.memory.alloc(ring * c.head_dim * 2);
        for (&out.sources, c.kv_sources, 0..) |*s, layer, i| {
            const groups = capacity / c.ratios[layer] + 1;
            s.* = .{ .latents = try out.memory.alloc(groups * c.head_dim * 2), .keys = try out.memory.alloc(groups * c.index_dim * 2), .groups = groups };
            if (c.ratios[layer] > 1) {
                out.proj_kv[i] = try out.memory.alloc(ring * c.head_dim * 4);
                out.proj_score[i] = try out.memory.alloc(ring * c.head_dim * 4);
            }
        }
        try out.engram_ids.ensureTotalCapacity(gpa, capacity);
        if (drafts) {
            var rings: [c.stages]mtl.Buffer = undefined;
            for (&rings) |*r| r.* = try out.memory.alloc(ring * c.head_dim * 2);
            out.stage_rings = rings;
        }
        return out;
    }

    pub fn deinit(self: *Cache) void {
        self.engram_ids.deinit(self.gpa);
        self.memory.deinit();
    }

    /// Bytes a stream holds at `capacity` positions.
    pub fn bytes(capacity: usize) usize {
        var n: usize = c.vocab * 2 + c.layers * ring * c.head_dim * 2;
        for (c.kv_sources) |layer| {
            n += (capacity / c.ratios[layer] + 1) * (c.head_dim + c.index_dim) * 2;
            if (c.ratios[layer] > 1) n += 2 * ring * c.head_dim * 4;
        }
        return n + capacity * 4;
    }
};

pub const Scratch = struct {
    memory: Storage,
    rows: usize,
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    x: mtl.Buffer, // the streams, bf16 [rows, 4, 5120]
    x2: mtl.Buffer, // the other streams buffer: a fused expand writes it, then the two swap
    normed: mtl.Buffer,
    branch: mtl.Buffer,
    pre: [2]mtl.Buffer, // f32 [rows, 4]: the carried collapse, and this boundary's
    post: mtl.Buffer,
    comb: mtl.Buffer,
    hc_part: mtl.Buffer, // f32 [rows, 20 slices, 25]: a boundary's partial mix sums
    qr: mtl.Buffer,
    q: mtl.Buffer,
    kv: mtl.Buffer,
    o: mtl.Buffer,
    oa: mtl.Buffer,
    cproj: mtl.Buffer, // f32 [rows, 512]
    cscore: mtl.Buffer,
    latent: mtl.Buffer, // bf16 [rows, 512]
    iq: mtl.Buffer,
    iw: mtl.Buffer,
    ik: mtl.Buffer,
    scores: mtl.Buffer,
    /// Split attention's chunk partials for up to `split_rows` rows: outputs f32 [rows, 64, chunks, 512], (max, sum).
    attn_po: mtl.Buffer,
    attn_ml: mtl.Buffer,
    blocks: mtl.Buffer,
    keep: mtl.Buffer, // i32 [rows, 2048]: the candidate blocks, in order
    picks: mtl.Buffer, // i32 [rows, 512]
    router: mtl.Buffer,
    route_pick: mtl.Buffer,
    /// The shared expert's own gate, up and activation rows (bf16 [rows, 2304]) when it runs beside the routed ones.
    sh_gate: mtl.Buffer,
    sh_up: mtl.Buffer,
    sh_act: mtl.Buffer,
    /// u32 [rows * 8]: slot i's wo_a group (i % 8), the decode path's expert ids for the stacked wo_a.
    wo_ids: mtl.Buffer,
    route_weight: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    experts: mtl.Buffer,
    shared: mtl.Buffer,
    /// A prompt chunk's slots sorted by expert: each expert's first sorted slot, sorted -> slot, the sorted weights,
    /// the sorted slots' input rows, and the tensor units' group sums of a projection's input rows.
    offs: mtl.Buffer,
    order: mtl.Buffer,
    wsort: mtl.Buffer,
    xsort: mtl.Buffer,
    sums: mtl.Buffer,
    engram_rows: [c.engram_layers.len]mtl.Buffer, // bf16 [rows, 24 x 256] a layer, filled on the host
    /// Compressed groups this call completes, a KV source each (written on the host while encoding, read when the
    /// command buffer runs, so no two sources share one): their positions (for RoPE) and indices (cache rows).
    gpos: [c.kv_sources.len]mtl.Buffer,
    gidx: [c.kv_sources.len]mtl.Buffer,
    engram_kv: mtl.Buffer,
    logits: mtl.Buffer,
    /// DSpark: the target layers' stream means (bf16 [rows, 3 x 5120]) and their conditioning rows; a draft call's
    /// token ids (the pending token, then the drafts as the Markov steps pick them), its picks of its own five keys,
    /// and the argmax partials.
    mh: mtl.Buffer,
    /// Each row's image flag (i32: an image span's row) and code (forward.Vis).
    img: mtl.Buffer,
    img_code: mtl.Buffer,
    mainx: mtl.Buffer,
    draft_ids: mtl.Buffer,
    draft_picks: mtl.Buffer,
    markov_part: mtl.Buffer,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, rows: usize) !Scratch {
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.rows = rows;
        const m = &out.memory;
        const slots = rows * c.topk;
        out.ids = try m.alloc(rows * 4);
        out.pos = try m.alloc(rows * 4);
        out.x = try m.alloc(rows * c.hc * c.hidden * 2);
        out.x2 = try m.alloc(rows * c.hc * c.hidden * 2);
        out.normed = try m.alloc(rows * c.hidden * 2);
        out.branch = try m.alloc(rows * c.hidden * 2);
        for (&out.pre) |*p| p.* = try m.alloc(rows * c.hc * 4);
        out.post = try m.alloc(rows * c.hc * 4);
        out.comb = try m.alloc(rows * c.hc * c.hc * 4);
        out.hc_part = try m.alloc(rows * (c.hc * c.hidden / 1024) * (c.hc_mix + 1) * 4);
        out.qr = try m.alloc(rows * c.q_rank * 2);
        out.q = try m.alloc(rows * c.heads * c.head_dim * 2);
        out.kv = try m.alloc(rows * c.head_dim * 2);
        out.o = try m.alloc(rows * c.heads * c.head_dim * 2);
        out.oa = try m.alloc(rows * c.o_groups * c.o_rank * 2);
        out.cproj = try m.alloc(rows * c.head_dim * 4);
        out.cscore = try m.alloc(rows * c.head_dim * 4);
        out.latent = try m.alloc(rows * c.head_dim * 2);
        out.iq = try m.alloc(rows * c.index_heads * c.index_dim * 2);
        out.iw = try m.alloc(rows * c.index_heads * 2);
        out.ik = try m.alloc(rows * c.index_dim * 2);
        out.scores = try m.alloc(score_budget);
        out.attn_po = try m.alloc(@min(rows, split_rows) * @as(usize, c.heads) * split_chunks * c.head_dim * 4);
        out.attn_ml = try m.alloc(@min(rows, split_rows) * @as(usize, c.heads) * split_chunks * 8);
        out.blocks = try m.alloc(score_budget);
        out.keep = try m.alloc(rows * c.candidate_blocks * 4);
        out.picks = try m.alloc(rows * picks * 4);
        out.router = try m.alloc(rows * c.experts * 4);
        out.route_pick = try m.alloc(slots * 4);
        out.sh_gate = try m.alloc(rows * c.moe_inter * 2);
        out.sh_up = try m.alloc(rows * c.moe_inter * 2);
        out.sh_act = try m.alloc(rows * c.moe_inter * 2);
        out.wo_ids = try m.alloc(rows * c.o_groups * 4);
        for (out.wo_ids.slice(u32, rows * c.o_groups), 0..) |*v, i| v.* = @intCast(i % c.o_groups);
        out.route_weight = try m.alloc(slots * 4);
        out.gate = try m.alloc(slots * c.moe_inter * 2);
        out.up = try m.alloc(slots * c.moe_inter * 2);
        out.act = try m.alloc(slots * c.moe_inter * 2);
        out.experts = try m.alloc(slots * c.hidden * 2);
        out.shared = try m.alloc(rows * c.hidden * 2);
        out.offs = try m.alloc((c.experts + 1) * 4);
        out.order = try m.alloc(slots * 4);
        out.wsort = try m.alloc(slots * 4);
        out.xsort = try m.alloc(slots * c.hidden * 2);
        out.sums = try m.alloc(@max(rows * (c.heads * c.head_dim) / 64, slots * c.hidden / 64) * 4);
        for (&out.engram_rows) |*b| b.* = try m.alloc(rows * c.engram_cols * c.engram_dim * 2);
        for (&out.gpos, &out.gidx) |*p, *g| {
            p.* = try m.alloc(rows * 4);
            g.* = try m.alloc(rows * 4);
        }
        out.engram_kv = try m.alloc(rows * (c.hc + 1) * c.hidden * 2);
        out.logits = try m.alloc(rows * c.vocab * 2);
        out.mh = try m.alloc(rows * c.draft_targets.len * c.hidden * 2);
        out.img = try m.alloc(rows * 4);
        out.img_code = try m.alloc(rows * 4);
        out.mainx = try m.alloc(rows * c.hidden * 2);
        out.draft_ids = try m.alloc((c.draft_block + 1) * 4);
        out.draft_picks = try m.alloc(c.draft_block * c.draft_block * 4);
        for (out.draft_picks.slice(i32, c.draft_block * c.draft_block), 0..) |*v, i| v.* = @intCast(i % c.draft_block);
        out.markov_part = try m.alloc((c.vocab + 255) / 256 * 8);
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.memory.deinit();
    }
};
