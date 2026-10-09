//! The Flash Next vision tower on CUDA (rank 0 encodes; rank 1 receives the features): the checkpoint's 27-block
//! ViT (hidden 1152, 16 heads of 72, GELU-tanh MLP 4304), its interpolated position table and 2-D rotary, and the
//! patch merger to the language model's 2560. The linears are cuBLASLt products (bf16 in, fp32 accumulate) and
//! the rest are fn_vision's kernels, rounding as the Python engine's torch tower rounds. Attention is full within
//! each image: per head, query chunks of a scores product, an fp32 softmax and the values product.

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kern.zig");

const H: i64 = 1152; // hidden
const HEADS: i64 = 16;
const HD: i64 = 72; // a head's width
const MLP: i64 = 4304;
const PATCH: i64 = 1536; // 3 channels x 2 frames x 16 x 16
const SIDE: i64 = 48; // the position table's side (2304 entries)
const MERGE: i64 = 2;
const OUT: i64 = 2560;
const DEPTH = 27;
const CHUNK: i64 = 2048; // query rows a scores product covers
const EPS: f32 = 1e-6;

const Block = struct { n1w: u64, n1b: u64, qkv_w: u64, qkv_b: u64, proj_w: u64, proj_b: u64, n2w: u64, n2b: u64, fc1_w: u64, fc1_b: u64, fc2_w: u64, fc2_b: u64 };

const LinKey = struct { m: u64, n: u64, k: u64, f32_out: bool };

pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lt: cuda.cublaslt.Library,
    mem: cuda.DeviceBuffer,
    blocks: [DEPTH]Block = undefined,
    patch_w: u64 = 0,
    patch_b: u64 = 0,
    pos_table: u64 = 0,
    mn_w: u64 = 0, mn_b: u64 = 0, m1_w: u64 = 0, m1_b: u64 = 0, m2_w: u64 = 0, m2_b: u64 = 0,
    inv_freq: cuda.DeviceBuffer,
    linears: std.AutoHashMap(LinKey, cuda.cublaslt.Linear),
    ws: cuda.DeviceBuffer,
    ws_len: usize = 64 << 20,
    /// The residual stream in fp32 between blocks (the blocks still compute in bf16): fewer roundings than torch's
    /// bf16 tower, the features nearer its exact arithmetic.
    precise: bool = true,

    /// The tower's tensors from ``dir``/model-visual.safetensors (bf16), onto the device in one allocation.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Tower {
        const path = try std.fs.path.join(gpa, &.{ dir, "model-visual.safetensors" });
        defer gpa.free(path);
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var head: [8]u8 = undefined;
        _ = try file.readPositionalAll(io, &head, 0);
        const hlen = std.mem.readInt(u64, &head, .little);
        const text = try gpa.alloc(u8, hlen);
        defer gpa.free(text);
        _ = try file.readPositionalAll(io, text, 8);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        var total: u64 = 0;
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            if (!std.mem.eql(u8, e.value_ptr.object.get("dtype").?.string, "BF16")) return error.UnsupportedVisionDType;
            const offs = e.value_ptr.object.get("data_offsets").?.array.items;
            total += std.mem.alignForward(u64, @intCast(offs[1].integer - offs[0].integer), 256);
        }
        var t: Tower = .{ .gpa = gpa, .d = d, .lt = try cuda.cublaslt.Library.open(), .mem = try cuda.DeviceBuffer.alloc(d, total),
            .inv_freq = try cuda.DeviceBuffer.alloc(d, 18 * 4), .linears = .init(gpa), .ws = undefined };
        errdefer t.deinit();
        t.ws = try cuda.DeviceBuffer.alloc(d, t.ws_len);
        var at: u64 = 0;
        var names = std.StringHashMap(u64).init(gpa);
        defer names.deinit();
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        it = parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const offs = e.value_ptr.object.get("data_offsets").?.array.items;
            const n: usize = @intCast(offs[1].integer - offs[0].integer);
            try buf.resize(gpa, n);
            _ = try file.readPositionalAll(io, buf.items, 8 + hlen + @as(u64, @intCast(offs[0].integer)));
            const ptr = t.mem.ptr + at;
            try d.check(d.api.cuMemcpyHtoD_v2(ptr, buf.items.ptr, n), "vision weights");
            try names.put(e.key_ptr.*, ptr);
            at += std.mem.alignForward(u64, n, 256);
        }
        const get = struct {
            fn f(m: *const std.StringHashMap(u64), comptime fmt: []const u8, args: anytype) !u64 {
                var nb: [128]u8 = undefined;
                const name = try std.fmt.bufPrint(&nb, "model.visual." ++ fmt, args);
                return m.get(name) orelse {
                    std.log.err("flash next vision: no tensor {s}", .{name});
                    return error.MissingVisionTensor;
                };
            }
        }.f;
        t.patch_w = try get(&names, "patch_embed.proj.weight", .{});
        t.patch_b = try get(&names, "patch_embed.proj.bias", .{});
        t.pos_table = try get(&names, "pos_embed.weight", .{});
        t.mn_w = try get(&names, "merger.norm.weight", .{});
        t.mn_b = try get(&names, "merger.norm.bias", .{});
        t.m1_w = try get(&names, "merger.linear_fc1.weight", .{});
        t.m1_b = try get(&names, "merger.linear_fc1.bias", .{});
        t.m2_w = try get(&names, "merger.linear_fc2.weight", .{});
        t.m2_b = try get(&names, "merger.linear_fc2.bias", .{});
        for (&t.blocks, 0..) |*bl, i| bl.* = .{
            .n1w = try get(&names, "blocks.{d}.norm1.weight", .{i}), .n1b = try get(&names, "blocks.{d}.norm1.bias", .{i}),
            .qkv_w = try get(&names, "blocks.{d}.attn.qkv.weight", .{i}), .qkv_b = try get(&names, "blocks.{d}.attn.qkv.bias", .{i}),
            .proj_w = try get(&names, "blocks.{d}.attn.proj.weight", .{i}), .proj_b = try get(&names, "blocks.{d}.attn.proj.bias", .{i}),
            .n2w = try get(&names, "blocks.{d}.norm2.weight", .{i}), .n2b = try get(&names, "blocks.{d}.norm2.bias", .{i}),
            .fc1_w = try get(&names, "blocks.{d}.mlp.linear_fc1.weight", .{i}), .fc1_b = try get(&names, "blocks.{d}.mlp.linear_fc1.bias", .{i}),
            .fc2_w = try get(&names, "blocks.{d}.mlp.linear_fc2.weight", .{i}), .fc2_b = try get(&names, "blocks.{d}.mlp.linear_fc2.bias", .{i}),
        };
        // the 2-D rotary's frequencies: 1 / 10000^(2i / 36), i < 18 (TensorFold sets them on the tower)
        var inv: [18]f32 = undefined;
        for (&inv, 0..) |*x, i| x.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / 36.0);
        try d.check(d.api.cuMemcpyHtoD_v2(t.inv_freq.ptr, &inv, inv.len * 4), "vision rotary");
        return t;
    }

    pub fn deinit(t: *Tower) void {
        var it = t.linears.valueIterator();
        while (it.next()) |l| l.deinit();
        t.linears.deinit();
        t.mem.free();
        t.inv_freq.free();
        t.ws.free();
        t.lt.close();
    }

    /// D[m, n] = X[m, k] . W[n, k]^T (fp32 or bf16 out), a plan per shape kept for the next image.
    fn linear(t: *Tower, k: *kern.K, x: u64, w: u64, d: u64, m: i64, n: i64, kk: i64, f32_out: bool) !void {
        const key: LinKey = .{ .m = @intCast(m), .n = @intCast(n), .k = @intCast(kk), .f32_out = f32_out };
        const gop = try t.linears.getOrPut(key);
        if (!gop.found_existing) {
            gop.value_ptr.* = cuda.cublaslt.Linear.init(&t.lt, key.m, key.n, key.k, if (f32_out) .f32 else .bf16, t.ws_len) catch |e| {
                _ = t.linears.remove(key);
                return e;
            };
        }
        try gop.value_ptr.run(x, w, d, t.ws.ptr, t.ws_len, k.stream.handle);
    }

    fn grid1(n: i64) cuda.Dim3 {
        return .{ .x = @intCast(@min(std.math.divCeil(i64, n, 256) catch unreachable, 65535)) };
    }

    /// A Linear with its bias (and GELU, act 1 tanh / 2 erf): the fp32 product into ``acc``, then the bias kernel.
    fn dense(t: *Tower, k: *kern.K, x: u64, w: u64, bias: u64, out: u64, acc: u64, m: i64, n: i64, kk: i64, act: i32) !void {
        try t.linear(k, x, w, acc, m, n, kk, true);
        const f = try k.ext(.vision, "fn_vis_bias_act");
        var a: cuda.Args = .{};
        a.add(acc); a.add(bias); a.add(out); a.add(@as(i64, m * n)); a.add(@as(i32, @intCast(n))); a.add(act);
        try k.go(f, grid1(m * n), 256, 0, &a);
    }

    fn layernorm(k: *kern.K, x: u64, w: u64, bias: u64, y: u64, rows: i64, cols: i64) !void {
        const f = try k.ext(.vision, "fn_vis_layernorm");
        var a: cuda.Args = .{};
        a.add(x); a.add(w); a.add(bias); a.add(y); a.add(@as(i32, @intCast(cols))); a.add(EPS);
        try k.go(f, .{ .x = @intCast(rows) }, 256, 0, &a);
    }

    fn add(k: *kern.K, x: u64, y: u64, n: i64) !void {
        const f = try k.ext(.vision, "fn_vis_add");
        var a: cuda.Args = .{};
        a.add(x); a.add(y); a.add(n);
        try k.go(f, grid1(n), 256, 0, &a);
    }

    /// Features for images of ``grids`` ([t, h, w] in patches; t = 1) from their patches (fp32, [patches, 1536] in the
    /// processor's merge-block order) into ``out`` ([patches / 4, 2560] bf16). Scratch is allocated per call.
    pub fn encode(t: *Tower, k: *kern.K, patches: []const f32, grids: []const [3]i64, out: u64) !void {
        const d = t.d;
        var N: i64 = 0;
        for (grids) |g| {
            if (g[0] != 1 or @mod(g[1], MERGE) != 0 or @mod(g[2], MERGE) != 0) return error.BadImageGrid;
            N += g[1] * g[2];
        }
        if (@as(i64, @intCast(patches.len)) != N * PATCH) return error.BadImagePatches;
        // host tables: each patch's (row, col) in merge-block order and its four position-table taps
        const pos = try t.gpa.alloc(i32, @intCast(N * 2));
        defer t.gpa.free(pos);
        const idx = try t.gpa.alloc(i32, @intCast(N * 4));
        defer t.gpa.free(idx);
        const wts = try t.gpa.alloc(f32, @intCast(N * 4));
        defer t.gpa.free(wts);
        var r0: usize = 0;
        for (grids) |g| {
            const h = g[1];
            const w = g[2];
            var i: i64 = 0;
            while (i < h * w) : (i += 1) {
                const in_col = @mod(i, MERGE);
                const in_row = @mod(@divTrunc(i, MERGE), MERGE);
                const block_col = @mod(@divTrunc(i, MERGE * MERGE), @divTrunc(w, MERGE));
                const block_row = @divTrunc(i, MERGE * MERGE * @divTrunc(w, MERGE));
                const row = block_row * MERGE + in_row;
                const col = block_col * MERGE + in_col;
                const p = r0 + @as(usize, @intCast(i));
                pos[p * 2] = @intCast(row);
                pos[p * 2 + 1] = @intCast(col);
                const hr = taps(row, h);
                const wc = taps(col, w);
                for (0..2) |a| for (0..2) |b| {
                    idx[p * 4 + a * 2 + b] = @intCast(hr.i[a] * SIDE + wc.i[b]);
                    wts[p * 4 + a * 2 + b] = hr.w[a] * wc.w[b];
                };
            }
            r0 += @intCast(h * w);
        }
        // scratch: x, normed, qkv, q/k/vt, attention out, mlp, the fp32 products
        const big = @max(@max(N * 3 * H, N * MLP), N * PATCH);
        var scratch = try cuda.DeviceBuffer.alloc(d, @intCast(N * H * 2 * 3 + big * 2 + big * 4 + N * 3 * H * 2 + CHUNK * N * 6 + CHUNK * HD * 4 + CHUNK * 4 + 1024 + N * H * 4 + N * 16 + N * 32 + N * H * 2));
        defer scratch.free();
        var at: u64 = scratch.ptr;
        const take = struct {
            fn f(p: *u64, n: i64) u64 {
                const r = p.*;
                p.* += std.mem.alignForward(u64, @intCast(n), 256);
                return r;
            }
        }.f;
        const x = take(&at, N * H * 2);
        const x32 = take(&at, N * H * 4);
        const xn = take(&at, N * H * 2);
        const att = take(&at, N * H * 2);
        const big16 = take(&at, big * 2);
        const acc = take(&at, big * 4);
        const qh = take(&at, N * H * 2);
        const kh = take(&at, N * H * 2);
        const vt = take(&at, N * H * 2);
        const sc = take(&at, CHUNK * N * 4);
        const pr = take(&at, CHUNK * N * 2);
        const oh = take(&at, CHUNK * HD * 4);
        const sums = take(&at, CHUNK * 4);
        const dpos = take(&at, N * 8);
        const didx = take(&at, N * 16);
        const dwts = take(&at, N * 16);
        try d.check(d.api.cuMemcpyHtoDAsync_v2(dpos, pos.ptr, pos.len * 4, k.stream.handle), "vision positions");
        try d.check(d.api.cuMemcpyHtoDAsync_v2(didx, idx.ptr, idx.len * 4, k.stream.handle), "vision taps");
        try d.check(d.api.cuMemcpyHtoDAsync_v2(dwts, wts.ptr, wts.len * 4, k.stream.handle), "vision tap weights");
        // patch embedding: fp32 patches cast to bf16, the conv as a linear, then the interpolated position table
        try d.check(d.api.cuMemcpyHtoDAsync_v2(acc, patches.ptr, patches.len * 4, k.stream.handle), "vision patches");
        {
            const f = try k.ext(.vision, "fn_vis_to_bf16");
            var a: cuda.Args = .{};
            a.add(acc); a.add(big16); a.add(@as(i64, N * PATCH));
            try k.go(f, grid1(N * PATCH), 256, 0, &a);
        }
        try t.dense(k, big16, t.patch_w, t.patch_b, x, acc, N, H, PATCH, 0);
        {
            const f = try k.ext(.vision, "fn_vis_pos_embed");
            var a: cuda.Args = .{};
            a.add(x); a.add(t.pos_table); a.add(didx); a.add(dwts); a.add(@as(i32, @intCast(N))); a.add(@as(i32, @intCast(H)));
            try k.go(f, grid1(N * H), 256, 0, &a);
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
        if (t.precise) try t.call1(k, "fn_vis_to_f32", x, x32, N * H);
        for (t.blocks) |bl| {
            if (t.precise) try t.ln32(k, x32, bl.n1w, bl.n1b, xn, N) else try layernorm(k, x, bl.n1w, bl.n1b, xn, N, H);
            try t.dense(k, xn, bl.qkv_w, bl.qkv_b, big16, acc, N, 3 * H, H, 0);
            {
                const f = try k.ext(.vision, "fn_vis_rope_split");
                var a: cuda.Args = .{};
                a.add(big16); a.add(dpos); a.add(t.inv_freq.ptr); a.add(qh); a.add(kh); a.add(vt);
                a.add(@as(i32, @intCast(N))); a.add(@as(i32, @intCast(HEADS))); a.add(@as(i32, @intCast(HD)));
                try k.go(f, grid1(N * H), 256, 0, &a);
            }
            // each image attends within itself (a segment of rows); per head, chunks of queries
            var s0: i64 = 0;
            for (grids) |g| {
                const n = g[1] * g[2];
                var h: i64 = 0;
                while (h < HEADS) : (h += 1) {
                    const q0 = qh + @as(u64, @intCast((h * N + s0) * HD * 2));
                    const k0 = kh + @as(u64, @intCast((h * N + s0) * HD * 2));
                    // vt is [H, D, N]: this image's columns start at s0 within each row of N
                    var c: i64 = 0;
                    while (c < n) : (c += CHUNK) {
                        const rows = @min(CHUNK, n - c);
                        try t.linear(k, q0 + @as(u64, @intCast(c * HD * 2)), k0, sc, rows, n, HD, true);
                        {
                            const f = try k.ext(.vision, "fn_vis_softmax");
                            var a: cuda.Args = .{};
                            a.add(sc); a.add(pr); a.add(sums); a.add(@as(i32, @intCast(n))); a.add(scale);
                            try k.go(f, .{ .x = @intCast(rows) }, 256, 0, &a);
                        }
                        try t.valuesProduct(k, pr, vt, h, s0, n, N, rows, oh);
                        {
                            const f = try k.ext(.vision, "fn_vis_head_out");
                            var a: cuda.Args = .{};
                            a.add(oh); a.add(sums); a.add(att + @as(u64, @intCast((s0 + c) * H * 2))); a.add(@as(i32, @intCast(rows)));
                            a.add(@as(i32, @intCast(HEADS))); a.add(@as(i32, @intCast(HD))); a.add(@as(i32, @intCast(h)));
                            try k.go(f, grid1(rows * HD), 256, 0, &a);
                        }
                    }
                }
                s0 += n;
            }
            try t.dense(k, att, bl.proj_w, bl.proj_b, xn, acc, N, H, H, 0);
            if (t.precise) try t.call1(k, "fn_vis_add32", x32, xn, N * H) else try add(k, x, xn, N * H);
            if (t.precise) try t.ln32(k, x32, bl.n2w, bl.n2b, xn, N) else try layernorm(k, x, bl.n2w, bl.n2b, xn, N, H);
            try t.dense(k, xn, bl.fc1_w, bl.fc1_b, big16, acc, N, MLP, H, 1);
            try t.dense(k, big16, bl.fc2_w, bl.fc2_b, xn, acc, N, H, MLP, 0);
            if (t.precise) try t.call1(k, "fn_vis_add32", x32, xn, N * H) else try add(k, x, xn, N * H);
        }
        // merger: LayerNorm per patch, four patches a row (they are adjacent), fc1 + exact GELU, fc2 to the model's width
        if (t.precise) try t.ln32(k, x32, t.mn_w, t.mn_b, xn, N) else try layernorm(k, x, t.mn_w, t.mn_b, xn, N, H);
        const M = @divExact(N, MERGE * MERGE);
        try t.dense(k, xn, t.m1_w, t.m1_b, big16, acc, M, 4 * H, 4 * H, 2);
        try t.dense(k, big16, t.m2_w, t.m2_b, out, acc, M, OUT, 4 * H, 0);
    }

    fn call1(t: *Tower, k: *kern.K, name: []const u8, a0: u64, a1: u64, n: i64) !void {
        _ = t;
        const f = try k.ext(.vision, name);
        var a: cuda.Args = .{};
        a.add(a0); a.add(a1); a.add(n);
        try k.go(f, grid1(n), 256, 0, &a);
    }

    fn ln32(t: *Tower, k: *kern.K, x: u64, w: u64, bias: u64, y: u64, rows: i64) !void {
        _ = t;
        const f = try k.ext(.vision, "fn_vis_layernorm32");
        var a: cuda.Args = .{};
        a.add(x); a.add(w); a.add(bias); a.add(y); a.add(@as(i32, @intCast(H))); a.add(EPS);
        try k.go(f, .{ .x = @intCast(rows) }, 256, 0, &a);
    }

    /// O[rows, 72] (fp32) = P[rows, n] . V^T where V^T is head h's columns s0 .. s0 + n of vt [H, 72, N]: a product with the
    /// weight's rows N apart (a plan for that leading dimension).
    fn valuesProduct(t: *Tower, k: *kern.K, p: u64, vt: u64, h: i64, s0: i64, n: i64, N: i64, rows: i64, out: u64) !void {
        if (n == N) return t.linear(k, p, vt + @as(u64, @intCast(h * HD * N * 2)), out, rows, HD, n, true);
        // several images: copy this image's columns into a contiguous [72, n] block first
        var tmp = try cuda.DeviceBuffer.alloc(t.d, @intCast(HD * n * 2));
        defer tmp.free();
        var r: i64 = 0;
        while (r < HD) : (r += 1) {
            const src = vt + @as(u64, @intCast(((h * HD + r) * N + s0) * 2));
            try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(tmp.ptr + @as(u64, @intCast(r * n * 2)), src, @intCast(n * 2), k.stream.handle), "vision values");
        }
        try t.linear(k, p, tmp.ptr, out, rows, HD, n, true);
        try k.stream.synchronize(); // tmp is freed on return
    }
};

/// The position table's bilinear taps for coordinate ``i`` of an axis of ``size`` patches (align_corners: the ends map
/// to 0 and 47), as transformers computes them in fp32.
const Taps = struct { i: [2]i64, w: [2]f32 };

fn taps(i: i64, size: i64) Taps {
    const src: f32 = @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(SIDE - 1)) / @as(f32, @floatFromInt(@max(size - 1, 1)));
    const fl = @floor(src);
    const base: i64 = @intFromFloat(fl);
    var out: Taps = undefined;
    for (0..2) |j| {
        out.i[j] = std.math.clamp(base + @as(i64, @intCast(j)), 0, SIDE - 1);
        const dist = @abs(src - fl - @as(f32, @floatFromInt(j)));
        out.w[j] = @max(1.0 - dist, 0.0);
    }
    return out;
}

/// An image prompt's rotary positions: ``pos`` [3, L] (t, h, w), the decode ``delta`` (the position the text after
/// the prompt resumes at, less the prompt's length) and the placeholder ``rows`` (the features' rows, in order).
pub const Positions = struct {
    pos: []i32,
    delta: i64,
    rows: []u32,

    pub fn deinit(p: *Positions, gpa: std.mem.Allocator) void {
        gpa.free(p.pos);
        gpa.free(p.rows);
    }
};

/// The image token ids (config.json: image_token_id, vision_start_token_id, vision_end_token_id).
pub const Ids = struct { image: u32 = 248056, start: u32 = 248053, end: u32 = 248054 };

/// Qwen3.5's rope index for images (the Python engine's media_positions): text counts up on all three axes; each
/// image's merged grid sits at the next position, rows on the h axis and columns on the w axis, and the text after it
/// resumes past its larger side. ``grids``: each image's [t, h, w] in patches (t = 1), in prompt order.
pub fn mediaPositions(gpa: std.mem.Allocator, tokens: []const u32, grids: []const [3]i64, ids: Ids) !Positions {
    const L = tokens.len;
    const pos = try gpa.alloc(i32, 3 * L);
    errdefer gpa.free(pos);
    var rows: std.ArrayList(u32) = .empty;
    errdefer rows.deinit(gpa);
    var cursor: usize = 0;
    var next_pos: i64 = 0;
    var used: usize = 0;
    while (true) {
        const begin = for (cursor..L) |i| {
            if (tokens[i] == ids.image) break i;
        } else break;
        if (used >= grids.len) return error.ImageWithoutGrid;
        const g = grids[used];
        used += 1;
        if (g[0] != 1 or @mod(g[1], MERGE) != 0 or @mod(g[2], MERGE) != 0 or g[1] <= 0 or g[2] <= 0) return error.BadImageGrid;
        const h: usize = @intCast(@divExact(g[1], MERGE));
        const w: usize = @intCast(@divExact(g[2], MERGE));
        const end = begin + h * w;
        if (begin == 0 or tokens[begin - 1] != ids.start or end >= L or tokens[end] != ids.end) return error.ImagePlaceholdersMismatch;
        for (tokens[begin..end]) |t| if (t != ids.image) return error.ImagePlaceholdersMismatch;
        for (cursor..begin) |i| {
            const v: i32 = @intCast(next_pos + @as(i64, @intCast(i - cursor)));
            for (0..3) |a| pos[a * L + i] = v;
        }
        const base: i64 = next_pos + @as(i64, @intCast(begin - cursor));
        for (0..h * w) |j| {
            const i = begin + j;
            pos[i] = @intCast(base);
            pos[L + i] = @intCast(base + @as(i64, @intCast(j / w)));
            pos[2 * L + i] = @intCast(base + @as(i64, @intCast(j % w)));
            try rows.append(gpa, @intCast(i));
        }
        next_pos = base + @as(i64, @intCast(@max(h, w)));
        cursor = end;
    }
    if (used < grids.len) return error.GridWithoutImage;
    for (cursor..L) |i| {
        const v: i32 = @intCast(next_pos + @as(i64, @intCast(i - cursor)));
        for (0..3) |a| pos[a * L + i] = v;
    }
    return .{ .pos = pos, .delta = next_pos - @as(i64, @intCast(cursor)), .rows = try rows.toOwnedSlice(gpa) };
}
