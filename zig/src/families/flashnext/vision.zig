//! Qwen3.8 Flash Next's image tower on the GPU: an image's patches to the rows that replace its `<|image_pad|>` tokens.
//! transformers' Qwen4ExpVisionModel in bf16: a patch projection plus a learned 48x48 position table resampled to the
//! image (bilinear, corners aligned), 27 pre-LayerNorm blocks (axial 2-D rotary attention over the image's patches, a
//! tanh-GELU MLP), and the merger: a LayerNorm, four neighbouring patches as one row, a GELU MLP to 2560.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const st = @import("../../core/safetensors.zig");

// The tower this file reads; `check` refuses a checkpoint whose vision_config differs.
pub const hidden = 1152;
pub const heads = 16;
pub const head_dim = 72;
pub const depth = 27;
pub const mlp_inter = 4304;
pub const out_hidden = 2560; // the language model's width
pub const patch = 16;
pub const temporal = 2;
pub const merge = 2;
pub const grid_side = 48; // the position table's side: num_position_embeddings 2304
pub const patch_width = 3 * temporal * patch * patch; // 1536: a patch's values, channel by frame by row by column
pub const patches_per_token = merge * merge;
pub const merged = hidden * patches_per_token; // 4608: the merger's width
const eps: f32 = 1e-6;

pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, bytes: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }

    pub fn addr(r: Ref) [*]u8 {
        return r.buf.contents() + r.off;
    }
};

/// The checkpoint's vision_config is the tower this file reads (an empty deepstack list: rows enter at the embedding).
pub fn check(config_json: []const u8, arena: std.mem.Allocator) !void {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, config_json, .{});
    const v = (doc.object.get("vision_config") orelse return error.NoVisionTower).object;
    const want = .{ .{ "depth", depth }, .{ "hidden_size", hidden }, .{ "num_heads", heads }, .{ "intermediate_size", mlp_inter }, .{ "out_hidden_size", out_hidden }, .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "spatial_merge_size", merge }, .{ "num_position_embeddings", grid_side * grid_side } };
    inline for (want) |kv| {
        const got = v.get(kv[0]) orelse return error.UnsupportedVisionTower;
        if (got != .integer or got.integer != kv[1]) return error.UnsupportedVisionTower;
    }
    if (v.get("hidden_act")) |a| if (a != .string or !std.mem.eql(u8, a.string, "gelu_pytorch_tanh")) return error.UnsupportedVisionTower;
    if (v.get("deepstack_visual_indexes")) |d| if (d != .array or d.array.items.len != 0) return error.UnsupportedVisionTower;
}

const Kernels = struct {
    lib: mtl.Library,
    gemm: [3][2][2]mtl.Pipeline, // bf16 a b^T, fp32 a b^T, fp32 a b; by M aligned, N aligned
    bias: mtl.Pipeline,
    add: mtl.Pipeline,
    layer_norm: mtl.Pipeline,
    pos: mtl.Pipeline,
    qkv: mtl.Pipeline,
    softmax: mtl.Pipeline,
    heads_: mtl.Pipeline,
    gelu_tanh: mtl.Pipeline,
    gelu: mtl.Pipeline,

    fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        const src = try frags.source(device, gpa, sources.flashnext_vision);
        defer gpa.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        var k: Kernels = undefined;
        k.lib = lib;
        const kinds = [_][]const u8{ "bf16_nt", "f32_nt", "f32_nn" };
        for (kinds, 0..) |kind, ki| for (0..2) |m| for (0..2) |n| {
            var name: [64]u8 = undefined;
            const full = try std.fmt.bufPrint(&name, "fnv_gemm_{s}_{d}{d}", .{ kind, m, n });
            k.gemm[ki][m][n] = try mtl.Pipeline.init(device, lib, full, false);
        };
        inline for (.{ .{ "bias", "fnv_bias" }, .{ "add", "fnv_add" }, .{ "layer_norm", "fnv_layer_norm" }, .{ "pos", "fnv_pos" }, .{ "qkv", "fnv_qkv" }, .{ "softmax", "fnv_softmax" }, .{ "heads_", "fnv_heads" }, .{ "gelu_tanh", "fnv_gelu_tanh" }, .{ "gelu", "fnv_gelu" } }) |kv| {
            @field(k, kv[0]) = try mtl.Pipeline.init(device, lib, kv[1], false);
        }
        return k;
    }

    fn deinit(k: *Kernels) void {
        for (&k.gemm) |*a| for (a) |*b| for (b) |p| p.deinit();
        inline for (.{ "bias", "add", "layer_norm", "pos", "qkv", "softmax", "heads_", "gelu_tanh", "gelu" }) |f| @field(k, f).deinit();
        k.lib.deinit();
    }
};

const Block = struct { norm1_w: Ref, norm1_b: Ref, norm2_w: Ref, norm2_b: Ref, qkv_w: Ref, qkv_b: Ref, proj_w: Ref, proj_b: Ref, fc1_w: Ref, fc1_b: Ref, fc2_w: Ref, fc2_b: Ref };

const Weights = struct {
    buf: mtl.Buffer,
    patch_w: Ref, // [1152, 1536]: the convolution moved to the processor's (channel, frame, row, column) patch order
    patch_b: Ref,
    pos: Ref, // [2304, 1152]
    blocks: [depth]Block,
    m_norm_w: Ref,
    m_norm_b: Ref,
    m_fc1_w: Ref,
    m_fc1_b: Ref,
    m_fc2_w: Ref,
    m_fc2_b: Ref,
};

/// The tower: kernels, weights (bf16, about 0.9 GiB) and scratch for `max_tokens` image tokens at once.
pub const Vision = struct {
    gpa: std.mem.Allocator,
    k: Kernels,
    w: Weights,
    scratch: mtl.Buffer,
    max_patches: u32,
    chunk: u32, // query rows an attention pass scores at once (bounds the fp32 score buffer)
    patches: Ref, // [P, 1536] bf16
    cos: Ref, // [P, 72] f32
    sin: Ref,
    taps: Ref, // [P, 4] u32: the position table's rows a patch reads
    tap_w: Ref, // [P, 4] f32: their weights
    x: Ref, // [P, 1152] bf16: the residual stream
    xn: Ref, // [P, 1152]: a norm's output; the merger's [T, 4608] input
    att: Ref, // [P, 1152]: heads side by side; the merger's [T, 4608] hidden
    branch: Ref, // [P, 1152]
    qkv: Ref, // [P, 3456]
    q: Ref, // [16, P, 72] f32
    k_: Ref,
    v: Ref,
    s: Ref, // [16, chunk, P] f32
    o: Ref, // [16, chunk, 72] f32
    mlp: Ref, // [P, 4304] bf16
    bytes: usize,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, max_tokens: u32) !*Vision {
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.gpa = gpa;
        v.k = try Kernels.load(gpa, device);
        errdefer v.k.deinit();
        v.w = try loadWeights(gpa, device, dir);
        errdefer v.w.buf.deinit();
        const P: usize = @as(usize, max_tokens) * patches_per_token;
        v.max_patches = @intCast(P);
        v.chunk = @intCast(std.math.clamp((256 << 20) / (heads * P * 4) / 64 * 64, 64, 4096));
        const D = hidden;
        const parts = [_]usize{ P * patch_width * 2, P * head_dim * 4, P * head_dim * 4, P * 16, P * 16, P * D * 2, P * D * 2, P * D * 2, P * D * 2, P * 3 * D * 2, P * D * 4, P * D * 4, P * D * 4, @as(usize, heads) * v.chunk * P * 4, @as(usize, heads) * v.chunk * head_dim * 4, P * mlp_inter * 2 };
        var total: usize = 0;
        for (parts) |n| total += std.mem.alignForward(usize, n, 256);
        v.scratch = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer v.scratch.deinit();
        var at: usize = 0;
        const views = [_]*Ref{ &v.patches, &v.cos, &v.sin, &v.taps, &v.tap_w, &v.x, &v.xn, &v.att, &v.branch, &v.qkv, &v.q, &v.k_, &v.v, &v.s, &v.o, &v.mlp };
        for (views, parts) |view, n| {
            view.* = .{ .buf = v.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        v.bytes = total + v.w.buf.length();
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.scratch.deinit();
        v.w.buf.deinit();
        v.k.deinit();
        v.gpa.destroy(v);
    }

    /// Buffers to keep resident with the engine's.
    pub fn buffers(v: *const Vision) [2]mtl.Buffer {
        return .{ v.w.buf, v.scratch };
    }

    /// One image's patches ([gh * gw, 1536] fp32, the processor's order: 2x2 merge windows contiguous) into its tokens'
    /// rows `out` ([gh * gw / 4, 2560] bf16). The host writes the scratch: the caller has no work of this tower in
    /// flight. `stop`: after the embedding (0) or block `stop` (1-27), the residual stream is left in `x`.
    pub fn encode(v: *Vision, enc: mtl.ComputeEncoder, pixels: []const f32, gh: u32, gw: u32, out: Ref, stop: ?u32) !void {
        const P: u32 = gh * gw;
        if (P == 0 or gh % merge != 0 or gw % merge != 0) return error.ImageGrid;
        if (P > v.max_patches) return error.ImageTooLarge;
        if (pixels.len != @as(usize, P) * patch_width) return error.ImageGrid;
        const T: u32 = P / patches_per_token;
        const dst: [*]u16 = @ptrCast(@alignCast(v.patches.addr()));
        for (pixels, 0..) |p, i| dst[i] = bf16(p);
        hostTables(v, gh, gw);

        v.gemmLinear(enc, v.patches, v.w.patch_w, v.x, P, hidden, patch_width);
        v.biasAdd(enc, v.x, v.w.patch_b, P, hidden);
        enc.setPipeline(v.k.pos);
        const pb = [_]Ref{ v.x, v.w.pos, v.taps, v.tap_w };
        for (pb, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.dispatchThreads(mtl.Size.of(hidden, P, 1), mtl.Size.of(256, 1, 1));
        if (stop) |s| if (s == 0) return;
        for (v.w.blocks, 0..) |b, i| {
            v.block(enc, b, P);
            if (stop) |s| if (s == i + 1) return;
        }
        // the merger: the norm per patch, then each merge window's four rows read as one row of [T, 4608]
        v.layerNorm(enc, v.x, v.w.m_norm_w, v.w.m_norm_b, v.xn, P, hidden);
        v.gemmLinear(enc, v.xn, v.w.m_fc1_w, v.att, T, merged, merged);
        v.biasAdd(enc, v.att, v.w.m_fc1_b, T, merged);
        v.unary(enc, v.k.gelu, v.att, @as(usize, T) * merged);
        v.gemmLinear(enc, v.att, v.w.m_fc2_w, out, T, out_hidden, merged);
        v.biasAdd(enc, out, v.w.m_fc2_b, T, out_hidden);
    }

    fn block(v: *Vision, enc: mtl.ComputeEncoder, b: Block, P: u32) void {
        v.layerNorm(enc, v.x, b.norm1_w, b.norm1_b, v.xn, P, hidden);
        v.gemmLinear(enc, v.xn, b.qkv_w, v.qkv, P, 3 * hidden, hidden);
        v.biasAdd(enc, v.qkv, b.qkv_b, P, 3 * hidden);
        enc.setPipeline(v.k.qkv);
        const bufs = [_]Ref{ v.qkv, v.cos, v.sin, v.q, v.k_, v.v };
        for (bufs, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue(P, 6);
        enc.dispatchThreads(mtl.Size.of(head_dim / 2, @as(usize, P) * heads * 3, 1), mtl.Size.of(36, 4, 1));
        const hd: usize = head_dim;
        var q0: u32 = 0;
        while (q0 < P) : (q0 += v.chunk) {
            const lc = @min(v.chunk, P - q0);
            v.gemm(enc, 1, v.q.at(@as(usize, q0) * hd * 4), v.k_, v.s, lc, P, head_dim, head_dim, head_dim, P, heads, @as(usize, P) * hd, @as(usize, P) * hd, @as(usize, lc) * P);
            enc.setPipeline(v.k.softmax);
            enc.setBuffer(v.s.buf, v.s.off, 0);
            enc.setValue(P, 1);
            enc.setValue(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))), 2);
            enc.dispatchGroups(mtl.Size.of(@as(usize, heads) * lc, 1, 1), mtl.Size.of(256, 1, 1));
            v.gemm(enc, 2, v.s, v.v, v.o, lc, head_dim, P, P, head_dim, head_dim, heads, @as(usize, lc) * P, @as(usize, P) * hd, @as(usize, lc) * hd);
            enc.setPipeline(v.k.heads_);
            enc.setBuffer(v.o.buf, v.o.off, 0);
            enc.setBuffer(v.att.buf, v.att.off, 1);
            enc.setValue([2]u32{ lc, q0 }, 2);
            enc.dispatchThreads(mtl.Size.of(hidden, lc, 1), mtl.Size.of(256, 1, 1));
        }
        v.gemmLinear(enc, v.att, b.proj_w, v.branch, P, hidden, hidden);
        v.biasAdd(enc, v.branch, b.proj_b, P, hidden);
        v.add(enc, v.x, v.branch, @as(usize, P) * hidden);
        v.layerNorm(enc, v.x, b.norm2_w, b.norm2_b, v.xn, P, hidden);
        v.gemmLinear(enc, v.xn, b.fc1_w, v.mlp, P, mlp_inter, hidden);
        v.biasAdd(enc, v.mlp, b.fc1_b, P, mlp_inter);
        v.unary(enc, v.k.gelu_tanh, v.mlp, @as(usize, P) * mlp_inter);
        v.gemmLinear(enc, v.mlp, b.fc2_w, v.branch, P, hidden, mlp_inter);
        v.biasAdd(enc, v.branch, b.fc2_b, P, hidden);
        v.add(enc, v.x, v.branch, @as(usize, P) * hidden);
    }

    /// y [M, N] = x [M, K] w^T in bf16 (w as stored, [N, K]).
    fn gemmLinear(v: *Vision, enc: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: u32, N: u32, K: u32) void {
        v.gemm(enc, 0, x, w, y, M, N, K, K, K, N, 1, 0, 0, 0);
    }

    /// tfp::gemm_nax: d [batch, M, N] = a x b (kind 0: bf16 a b^T, 1: fp32 a b^T, 2: fp32 a b), batches `s*` apart.
    fn gemm(v: *Vision, enc: mtl.ComputeEncoder, kind: usize, a: Ref, b: Ref, d: Ref, M: u32, N: u32, K: u32, lda: u32, ldb: u32, ldd: u32, batch: u32, sa: usize, sb: usize, sd: usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        enc.setPipeline(v.k.gemm[kind][@intFromBool(M % 64 == 0)][@intFromBool(N % 128 == 0)]);
        enc.setBuffer(a.buf, a.off, 0);
        enc.setBuffer(b.buf, b.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, lda, ldb, ldd, tn, tm, 2, K / 256, sa, sb, sd, 0, 0 };
        for (vals, 0..) |x, i| p[i] = @intCast(x);
        enc.setBytes(std.mem.asBytes(&p), 2);
        enc.setBuffer(d.buf, d.off, 3);
        enc.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, batch) * 2), mtl.Size.of(32, 4, 2));
    }

    fn biasAdd(v: *Vision, enc: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, width: u32) void {
        enc.setPipeline(v.k.bias);
        enc.setBuffer(y.buf, y.off, 0);
        enc.setBuffer(b.buf, b.off, 1);
        enc.setValue(width, 2);
        enc.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    fn add(v: *Vision, enc: mtl.ComputeEncoder, x: Ref, y: Ref, n: usize) void {
        enc.setPipeline(v.k.add);
        enc.setBuffer(x.buf, x.off, 0);
        enc.setBuffer(y.buf, y.off, 1);
        enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn unary(_: *Vision, enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, x: Ref, n: usize) void {
        enc.setPipeline(pipe);
        enc.setBuffer(x.buf, x.off, 0);
        enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn layerNorm(v: *Vision, enc: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: u32, width: u32) void {
        enc.setPipeline(v.k.layer_norm);
        const bufs = [_]Ref{ x, w, b, y };
        for (bufs, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue(extern struct { rows: u32, width: u32, eps: f32 }{ .rows = rows, .width = width, .eps = eps }, 4);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }
};

/// fp32 to bf16, round to nearest even.
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (b & 0x7fffffff > 0x7f800000) return @truncate((b >> 16) | 0x40); // NaN stays NaN
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Patch `i`'s (row, column) in its image's patch grid: patches come in 2x2 merge windows, the windows row-major.
pub fn patchAt(i: u32, gw: u32) [2]u32 {
    const in_col = i % merge;
    const in_row = (i / merge) % merge;
    const windows_w = gw / merge;
    const block_col = (i / patches_per_token) % windows_w;
    const block_row = i / (patches_per_token * windows_w);
    return .{ block_row * merge + in_row, block_col * merge + in_col };
}

/// One axis's two bilinear taps into the 48-row table and their weights, corners aligned, edges clamped (fp32, as
/// transformers' _interpolation_axis_taps_weights).
pub fn axisTaps(index: u32, size: u32) struct { taps: [2]u32, w: [2]f32 } {
    const src: f32 = @as(f32, @floatFromInt(index)) * @as(f32, grid_side - 1) / @as(f32, @floatFromInt(@max(size, 2) - 1));
    const floor = @floor(src);
    const f: u32 = @intFromFloat(floor);
    return .{ .taps = .{ @min(f, grid_side - 1), @min(f + 1, grid_side - 1) }, .w = .{ @max(1 - @abs(src - floor), 0), @max(1 - @abs(src - floor - 1), 0) } };
}

/// Each patch's position-table taps and weights, and its axial rotary angles (18 frequencies 10000^(-2k/36) of its
/// row, 18 of its column, then the same again across a head's 72 values).
fn hostTables(v: *Vision, gh: u32, gw: u32) void {
    var inv: [head_dim / 4]f32 = undefined;
    for (&inv, 0..) |*f, k| f.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * k)) / @as(f32, head_dim / 2));
    const cs: [*]f32 = @ptrCast(@alignCast(v.cos.addr()));
    const sn: [*]f32 = @ptrCast(@alignCast(v.sin.addr()));
    const taps: [*]u32 = @ptrCast(@alignCast(v.taps.addr()));
    const tw: [*]f32 = @ptrCast(@alignCast(v.tap_w.addr()));
    const q = head_dim / 4;
    for (0..gh * gw) |i| {
        const at = patchAt(@intCast(i), gw);
        const h = axisTaps(at[0], gh);
        const w = axisTaps(at[1], gw);
        for (0..2) |a| for (0..2) |b| {
            taps[i * 4 + a * 2 + b] = h.taps[a] * grid_side + w.taps[b];
            tw[i * 4 + a * 2 + b] = h.w[a] * w.w[b];
        };
        for (0..q) |k| {
            const ah = @as(f32, @floatFromInt(at[0])) * inv[k];
            const aw = @as(f32, @floatFromInt(at[1])) * inv[k];
            for ([_]usize{ 0, 2 * q }) |base| {
                cs[i * head_dim + base + k] = @cos(ah);
                sn[i * head_dim + base + k] = @sin(ah);
                cs[i * head_dim + base + q + k] = @cos(aw);
                sn[i * head_dim + base + q + k] = @sin(aw);
            }
        }
    }
}

fn readAll(fd: c_int, dst: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const n = std.c.pread(fd, dst.ptr + done, dst.len - done, @intCast(off + done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
}

/// vision_tower.* from the checkpoint's shards into one buffer, the patch convolution moved to the patches' order.
fn loadWeights(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const Src = struct { path: [:0]const u8, off: u64, len: u64, dtype: st.DType, shape: []const usize };
    var names: std.StringHashMapUnmanaged(Src) = .empty;
    {
        const path = try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{dir}, 0);
        const f = try mtl.MappedFile.open(path);
        defer f.deinit();
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, f.bytes[0..f.size], .{ .allocate = .alloc_always });
        var files: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = doc.object.get("weight_map").?.object.iterator();
        while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, "vision_tower.")) try files.put(arena, kv.value_ptr.string, {});
        if (files.count() == 0) return error.NoVisionTower;
        for (files.keys()) |name| {
            const full = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
            const fd = std.c.open(full, .{ .ACCMODE = .RDONLY });
            if (fd < 0) return error.OpenFailed;
            defer _ = std.c.close(fd);
            var head: [8]u8 = undefined;
            try readAll(fd, &head, 0);
            const hlen = std.mem.readInt(u64, &head, .little);
            const header = try arena.alloc(u8, hlen);
            try readAll(fd, header, 8);
            const end: u64 = @intCast(std.c.lseek(fd, 0, std.c.SEEK.END));
            // read here, not by core/safetensors.zig: the patch convolution is 5-D
            const doc2 = try std.json.parseFromSliceLeaky(std.json.Value, arena, header, .{});
            var eit = doc2.object.iterator();
            while (eit.next()) |kv| {
                if (!std.mem.startsWith(u8, kv.key_ptr.*, "vision_tower.")) continue;
                const o = kv.value_ptr.object;
                const dtype = st.DType.parse(o.get("dtype").?.string) orelse return error.UnsupportedDType;
                const dims = o.get("shape").?.array.items;
                const shape = try arena.alloc(usize, dims.len);
                var n: u64 = dtype.size();
                for (dims, shape) |d, *s| {
                    s.* = @intCast(d.integer);
                    n *= s.*;
                }
                const offs = o.get("data_offsets").?.array.items;
                const b: u64 = @intCast(offs[0].integer);
                const e: u64 = @intCast(offs[1].integer);
                if (e < b or e - b != n or 8 + hlen + e > end) return error.BadSafetensors;
                try names.put(arena, kv.key_ptr.*, .{ .path = full, .off = 8 + hlen + b, .len = e - b, .dtype = dtype, .shape = shape });
            }
        }
    }
    const Want = struct { name: []const u8, shape: []const usize, ref: *Ref };
    var w: Weights = undefined;
    var wants: std.ArrayList(Want) = .empty;
    try wants.append(arena, .{ .name = "patch_embed.proj.weight", .shape = &.{ hidden, temporal, patch, patch, 3 }, .ref = &w.patch_w });
    try wants.append(arena, .{ .name = "patch_embed.proj.bias", .shape = &.{hidden}, .ref = &w.patch_b });
    try wants.append(arena, .{ .name = "pos_embed.weight", .shape = &.{ grid_side * grid_side, hidden }, .ref = &w.pos });
    for (&w.blocks, 0..) |*b, i| {
        const fields = .{ .{ "norm1.weight", &b.norm1_w, &[_]usize{hidden} }, .{ "norm1.bias", &b.norm1_b, &[_]usize{hidden} }, .{ "norm2.weight", &b.norm2_w, &[_]usize{hidden} }, .{ "norm2.bias", &b.norm2_b, &[_]usize{hidden} }, .{ "attn.qkv.weight", &b.qkv_w, &[_]usize{ 3 * hidden, hidden } }, .{ "attn.qkv.bias", &b.qkv_b, &[_]usize{3 * hidden} }, .{ "attn.proj.weight", &b.proj_w, &[_]usize{ hidden, hidden } }, .{ "attn.proj.bias", &b.proj_b, &[_]usize{hidden} }, .{ "mlp.linear_fc1.weight", &b.fc1_w, &[_]usize{ mlp_inter, hidden } }, .{ "mlp.linear_fc1.bias", &b.fc1_b, &[_]usize{mlp_inter} }, .{ "mlp.linear_fc2.weight", &b.fc2_w, &[_]usize{ hidden, mlp_inter } }, .{ "mlp.linear_fc2.bias", &b.fc2_b, &[_]usize{hidden} } };
        inline for (fields) |f| try wants.append(arena, .{ .name = try std.fmt.allocPrint(arena, "blocks.{d}.{s}", .{ i, f[0] }), .shape = f[2], .ref = f[1] });
    }
    try wants.append(arena, .{ .name = "merger.norm.weight", .shape = &.{hidden}, .ref = &w.m_norm_w });
    try wants.append(arena, .{ .name = "merger.norm.bias", .shape = &.{hidden}, .ref = &w.m_norm_b });
    try wants.append(arena, .{ .name = "merger.linear_fc1.weight", .shape = &.{ merged, merged }, .ref = &w.m_fc1_w });
    try wants.append(arena, .{ .name = "merger.linear_fc1.bias", .shape = &.{merged}, .ref = &w.m_fc1_b });
    try wants.append(arena, .{ .name = "merger.linear_fc2.weight", .shape = &.{ out_hidden, merged }, .ref = &w.m_fc2_w });
    try wants.append(arena, .{ .name = "merger.linear_fc2.bias", .shape = &.{out_hidden}, .ref = &w.m_fc2_b });
    const srcs = try arena.alloc(Src, wants.items.len);
    var total: usize = 0;
    for (wants.items, srcs) |want, *s| {
        const full = try std.fmt.allocPrint(arena, "vision_tower.{s}", .{want.name});
        s.* = names.get(full) orelse {
            std.log.err("flash next vision: the checkpoint has no tensor {s}", .{full});
            return error.MissingTensor;
        };
        if (s.dtype != .bf16 or !std.mem.eql(usize, s.shape, want.shape)) {
            std.log.err("flash next vision: {s} is {t} {any}; the tower reads bf16 {any}", .{ full, s.dtype, s.shape, want.shape });
            return error.UnexpectedTensor;
        }
        total = std.mem.alignForward(usize, total, 256) + s.len;
    }
    w.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer w.buf.deinit();
    var at: usize = 0;
    for (wants.items, srcs) |want, s| {
        at = std.mem.alignForward(usize, at, 256);
        want.ref.* = .{ .buf = w.buf, .off = at };
        const fd = std.c.open(s.path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        _ = std.c.fcntl(fd, 48, @as(c_int, 1)); // F_NOCACHE, as the language model's shards
        try readAll(fd, w.buf.contents()[at..][0..s.len], s.off);
        at += s.len;
    }
    // patch convolution [O, T, H, W, C] (MLX's layout) -> [O, C, T, H, W]: the processor's order within a patch
    const pw: [*]u16 = @ptrCast(@alignCast(w.patch_w.addr()));
    const tmp = try gpa.alloc(u16, patch_width);
    defer gpa.free(tmp);
    const spatial = temporal * patch * patch;
    for (0..hidden) |o| {
        const row = pw[o * patch_width ..][0..patch_width];
        @memcpy(tmp, row);
        for (0..spatial) |thw| for (0..3) |c| {
            row[c * spatial + thw] = tmp[thw * 3 + c];
        };
    }
    return w;
}

test "patches come in 2x2 merge windows, the windows row-major" {
    try std.testing.expectEqual([2]u32{ 0, 0 }, patchAt(0, 6));
    try std.testing.expectEqual([2]u32{ 0, 1 }, patchAt(1, 6));
    try std.testing.expectEqual([2]u32{ 1, 0 }, patchAt(2, 6));
    try std.testing.expectEqual([2]u32{ 1, 1 }, patchAt(3, 6));
    try std.testing.expectEqual([2]u32{ 0, 2 }, patchAt(4, 6));
    try std.testing.expectEqual([2]u32{ 2, 0 }, patchAt(12, 6));
}

test "bilinear taps: corners aligned, edges clamped" {
    const first = axisTaps(0, 14);
    try std.testing.expectEqual([2]u32{ 0, 1 }, first.taps);
    try std.testing.expectEqual([2]f32{ 1, 0 }, first.w);
    const last = axisTaps(13, 14);
    try std.testing.expectEqual([2]u32{ 47, 47 }, last.taps);
    try std.testing.expectEqual(@as(f32, 1), last.w[0] + last.w[1]);
    const mid = axisTaps(1, 3); // 23.5: halfway between rows 23 and 24
    try std.testing.expectEqual([2]u32{ 23, 24 }, mid.taps);
    try std.testing.expectEqual([2]f32{ 0.5, 0.5 }, mid.w);
}
