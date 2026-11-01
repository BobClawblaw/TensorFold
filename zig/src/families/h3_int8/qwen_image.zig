//! Qwen-Image-2.1's transformer over our Metal runtime: the denoising forward for the image rows.
//! Block projections and attention run in int8 on the M5 tensor units (bf16 with `int8` off); the kernels take the
//! model's sizes as compile-time defines written ahead of their source when the model loads.
//! The prompt's keys and values (one pass through the stack per prompt) are given; see tools/zig/qwen_image_case.py.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const kernels = @import("qwen_image_kernels");

pub const heads = 32;
pub const head_dim = 128;
pub const hidden = heads * head_dim;
pub const mlp_width = 3 * hidden;
pub const channels = 64;
pub const layers = 32;
pub const time_dim = 256;
pub const eps: f32 = 1e-6;
const tile = 128;
/// Channels of the MLP's wide rows that share one activation scale going into its last projection.
const wide_group = 1024;
/// Output columns one threadgroup of the int8 products owns (QI_TN in the kernels).
const ntile = 128;
/// Rows one threadgroup of the int8 products owns (QI_TM in the kernels).
const mtile = 128;
/// Threads of one int8 product's threadgroup (32 * QI_SG in the kernels).
const product_threads = 256;
/// Query rows one attention threadgroup owns (QI_TQ in the kernels).
const query_tile = 64;

pub fn toBf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

pub fn fromBf(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn round(x: f32) f32 {
    return fromBf(toBf(x));
}

/// A safetensors file mapped once and handed to Metal whole; a tensor is an offset into that one buffer.
pub const Tensors = struct {
    map: mtl.MappedFile,
    buffer: mtl.Buffer,
    data: usize,
    names: st.Header,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, device: mtl.Device, path: [*:0]const u8) !Tensors {
        const map = try mtl.MappedFile.open(path);
        errdefer map.deinit();
        if (map.size < 8) return error.BadSafetensors;
        const header_len: usize = @intCast(std.mem.readInt(u64, map.bytes[0..8], .little));
        if (header_len > map.size - 8) return error.BadSafetensors;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const names = try st.parseHeader(arena.allocator(), map.bytes[8..][0..header_len], map.size - 8 - header_len);
        const buffer = try device.bufferNoCopy(map.bytes.ptr, map.bytes.len, mtl.ResourceOptions.shared);
        return .{ .map = map, .buffer = buffer, .data = 8 + header_len, .names = names, .arena = arena };
    }

    pub fn deinit(self: *Tensors) void {
        self.buffer.deinit();
        self.arena.deinit();
        self.map.deinit();
    }

    /// The tensor `name` of `dtype` and `count` values: where it starts in the buffer.
    pub fn at(self: *const Tensors, name: []const u8, dtype: st.DType, count: usize) !Ref {
        const e = self.names.get(name) orelse {
            std.debug.print("no tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        const offset = self.data + e.begin;
        if (e.dtype != dtype or e.end - e.begin != count * dtype.size()) {
            std.debug.print("tensor {s}: unexpected type or size\n", .{name});
            return error.BadTensor;
        }
        return .{ .buffer = self.buffer, .offset = offset };
    }

    /// `at` for a tensor a kernel reads in place, which must start on a multiple of its value size.
    pub fn inPlace(self: *const Tensors, name: []const u8, dtype: st.DType, count: usize) !Ref {
        const ref = try self.at(name, dtype, count);
        if (ref.offset % dtype.size() != 0) {
            std.debug.print("tensor {s} is not aligned for the GPU\n", .{name});
            return error.BadTensor;
        }
        return ref;
    }

    pub fn values(self: *const Tensors, comptime T: type, ref: Ref, count: usize) []const T {
        return @as([*]const T, @ptrCast(@alignCast(self.map.bytes.ptr + ref.offset)))[0..count];
    }

    /// The tensor's values copied out, wherever it starts in the file.
    pub fn copy(self: *const Tensors, comptime T: type, gpa: std.mem.Allocator, ref: Ref, count: usize) ![]T {
        const out = try gpa.alloc(T, count);
        @memcpy(std.mem.sliceAsBytes(out), self.map.bytes[ref.offset..][0 .. count * @sizeOf(T)]);
        return out;
    }
};

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };

/// An int8 projection (K, N) and its scale per output channel.
const Quant = struct { w: mtl.Buffer, s: mtl.Buffer };

const Block = struct { to_q: Ref, to_k: Ref, to_v: Ref, to_out: Ref, norm_q: Ref, norm_k: Ref, gate: Ref, proj: Ref, out: Ref, keys: mtl.Buffer, vals: mtl.Buffer, q_q: Quant = undefined, q_k: Quant = undefined, q_v: Quant = undefined, q_out: Quant = undefined, q_gate: Quant = undefined, q_proj: Quant = undefined, q_mlp_out: Quant = undefined, k8: mtl.Buffer = undefined, ks: mtl.Buffer = undefined, v8: mtl.Buffer = undefined, vs: mtl.Buffer = undefined };

pub const Timing = struct { wall: f64, gpu: f64 };

/// Where a profiled forward's GPU time went.
pub const stages = [_][]const u8{ "projections", "norms", "heads", "scores", "softmax", "values", "rows", "gates", "mlp projections", "swiglu", "quantize", "mlp in", "mlp out" };

const Pass = struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pipelines: [kernels.names.len]mtl.Pipeline,
    weights: Tensors,
    rows: usize,
    text: usize,
    /// Prompt and image rows, padded to whole tiles; the padding holds zero keys and values that take no weight.
    keys: usize,
    blocks: [layers]Block,
    img_in: Ref,
    proj_out: Ref,
    time_1: Ref,
    time_2: Ref,
    modulation: Ref,
    norm_out: Ref,
    cos: mtl.Buffer,
    sin: mtl.Buffer,
    // rows of bf16: the stream, a normed copy, projections, head-major tiles, scores, and the MLP's wide rows
    x: mtl.Buffer,
    n: mtl.Buffer,
    qr: mtl.Buffer,
    kr: mtl.Buffer,
    vr: mtl.Buffer,
    qh: mtl.Buffer,
    scores: mtl.Buffer,
    oh: mtl.Buffer,
    y: mtl.Buffer,
    wide_a: mtl.Buffer,
    wide_b: mtl.Buffer,
    wide_h: mtl.Buffer,
    latents: mtl.Buffer,
    velocity: mtl.Buffer,
    scale_1: mtl.Buffer,
    gate_1: mtl.Buffer,
    scale_2: mtl.Buffer,
    gate_2: mtl.Buffer,
    scale_out: mtl.Buffer,
    // one row each, for the modulation's own projections
    mod_emb: mtl.Buffer,
    mod_a: mtl.Buffer,
    mod_temb: mtl.Buffer,
    mod_table: mtl.Buffer,
    /// int8 rows and their scales: the stream's width, then the MLP's.
    q8: mtl.Buffer,
    xs: mtl.Buffer,
    w8: mtl.Buffer,
    wxs: mtl.Buffer,
    /// int8 weights and activations for the block projections (the float path when off).
    int8: bool = true,
    /// int8 queries (heads, padded rows, 128) and their scales, for the single-pass attention.
    qh8: mtl.Buffer,
    qhs: mtl.Buffer,
    /// Scores and values in int8 with an online softmax; off, bf16 scores, a softmax pass and a values product.
    attention8: bool = true,
    /// With `profile` set each stage runs in a command buffer of its own and its GPU seconds add up in `spent`.
    profile: bool = false,
    /// Whole-tile products where the shapes allow; off, every product takes the fragment kernel.
    tiles: bool = true,
    spent: [stages.len]f64 = @splat(0),

    /// `weights_path`: the transposed projections. `rows` image rows after `text` prompt rows.
    pub fn load(gpa: std.mem.Allocator, weights_path: [*:0]const u8, rows: usize, text: usize, int8: bool) !*Model {
        const device = try mtl.Device.init();
        if (!device.tensorUnits()) return error.NeedsTensorUnits;
        const self = try gpa.create(Model);
        self.profile = false;
        self.int8 = int8;
        self.tiles = true;
        self.spent = @splat(0);
        self.gpa = gpa;
        self.device = device;
        self.queue = try device.queue();
        // the kernels take the model's sizes as compile-time values
        const source = try std.fmt.allocPrint(gpa, "#define QI_HIDDEN {d}\n#define QI_MLP {d}\n#define QI_WIDE_GROUP {d}\n#define QI_HEADS {d}\n#define QI_ROWS {d}\n#define QI_KEYS {d}\n{s}", .{ hidden, mlp_width, wide_group, heads, rows, text + rows, kernels.source });
        defer gpa.free(source);
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        for (kernels.names, 0..) |name, i| self.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
        self.weights = try Tensors.open(gpa, device, weights_path);
        self.rows = rows;
        self.text = text;
        self.keys = (text + rows + tile - 1) / tile * tile;
        const w = &self.weights;
        const square = hidden * hidden;
        var name: [96]u8 = undefined;
        for (&self.blocks, 0..) |*b, i| {
            b.to_q = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_q.weight", .{i}), .bf16, square);
            b.to_k = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_k.weight", .{i}), .bf16, square);
            b.to_v = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_v.weight", .{i}), .bf16, square);
            b.to_out = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_out.0.weight", .{i}), .bf16, square);
            b.norm_q = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_q.weight", .{i}), .bf16, head_dim);
            b.norm_k = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_k.weight", .{i}), .bf16, head_dim);
            b.gate = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.img_mlp.gate_layer.weight", .{i}), .bf16, hidden * mlp_width);
            b.proj = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.img_mlp.proj.weight", .{i}), .bf16, hidden * mlp_width);
            b.out = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.img_mlp.out.weight", .{i}), .bf16, hidden * mlp_width);
            b.keys = try self.rowsOf(heads * head_dim * self.keys);
            b.vals = try self.rowsOf(heads * head_dim * self.keys);
        }
        self.img_in = try w.inPlace("img_in.weight", .bf16, channels * hidden);
        self.proj_out = try w.inPlace("proj_out.weight", .bf16, channels * hidden);
        self.time_1 = try w.inPlace("time_text_embed.timestep_embedder.linear_1.weight", .bf16, time_dim * hidden);
        self.time_2 = try w.inPlace("time_text_embed.timestep_embedder.linear_2.weight", .bf16, square);
        self.modulation = try w.inPlace("modulation.1.weight", .bf16, 4 * square);
        self.norm_out = try w.inPlace("norm_out.linear.weight", .bf16, square);
        self.cos = try device.buffer(rows * head_dim / 2 * 4, mtl.ResourceOptions.shared);
        self.sin = try device.buffer(rows * head_dim / 2 * 4, mtl.ResourceOptions.shared);
        // a tile kernel reads its left operand's rows up to the next whole tile, so every such buffer has a spare one
        inline for (.{ "x", "n", "qr", "kr", "vr", "qh", "oh", "y" }) |f| @field(self, f) = try self.rowsOf((rows + tile) * hidden);
        inline for (.{ "wide_a", "wide_b", "wide_h" }) |f| @field(self, f) = try self.rowsOf((rows + tile) * mlp_width);
        self.scores = try self.rowsOf((heads * rows + tile) * self.keys);
        self.latents = try self.rowsOf((rows + tile) * channels);
        self.velocity = try self.rowsOf(rows * channels);
        inline for (.{ "scale_1", "gate_1", "scale_2", "gate_2", "scale_out" }) |f| @field(self, f) = try self.rowsOf(hidden);
        self.mod_emb = try self.rowsOf(time_dim);
        self.mod_a = try self.rowsOf(hidden);
        self.mod_temb = try self.rowsOf(hidden);
        self.mod_table = try self.rowsOf(4 * hidden);
        const padded = (rows + tile - 1) / tile * tile;
        self.q8 = try device.buffer(padded * hidden, mtl.ResourceOptions.shared);
        self.xs = try device.buffer(padded * 4, mtl.ResourceOptions.shared);
        self.w8 = try device.buffer(padded * mlp_width, mtl.ResourceOptions.shared);
        self.wxs = try device.buffer(padded * (mlp_width / wide_group) * 4, mtl.ResourceOptions.shared);
        self.attention8 = int8;
        self.qh8 = try device.buffer(heads * padded * head_dim, mtl.ResourceOptions.shared);
        self.qhs = try device.buffer(heads * padded * 4, mtl.ResourceOptions.shared);
        if (int8) for (&self.blocks) |*b| {
            b.k8 = try device.buffer(heads * self.keys * head_dim, mtl.ResourceOptions.shared);
            b.v8 = try device.buffer(heads * self.keys * head_dim, mtl.ResourceOptions.shared);
            b.ks = try device.buffer(heads * self.keys * 4, mtl.ResourceOptions.shared);
            b.vs = try device.buffer(heads * self.keys * 4, mtl.ResourceOptions.shared);
            @memset(b.k8.slice(u8, heads * self.keys * head_dim), 0);
            @memset(b.v8.slice(u8, heads * self.keys * head_dim), 0);
            @memset(b.ks.slice(f32, heads * self.keys), 0);
            @memset(b.vs.slice(f32, heads * self.keys), 0);
        };
        if (int8) for (&self.blocks) |*b| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = self.queue.commandBuffer();
            const enc = cb.compute(.serial);
            b.q_q = try self.quantize(enc, b.to_q, hidden, hidden);
            b.q_k = try self.quantize(enc, b.to_k, hidden, hidden);
            b.q_v = try self.quantize(enc, b.to_v, hidden, hidden);
            b.q_out = try self.quantize(enc, b.to_out, hidden, hidden);
            b.q_gate = try self.quantize(enc, b.gate, hidden, mlp_width);
            b.q_proj = try self.quantize(enc, b.proj, hidden, mlp_width);
            b.q_mlp_out = try self.quantize(enc, b.out, mlp_width, hidden);
            enc.end();
            cb.commit();
            cb.wait();
        };
        return self;
    }

    /// `w` (k, n) in int8 with a scale per output channel, written by the GPU.
    fn quantize(self: *const Model, enc: mtl.ComputeEncoder, w: Ref, k: usize, n: usize) !Quant {
        const q = Quant{ .w = try self.device.buffer(k * n, mtl.ResourceOptions.shared), .s = try self.device.buffer(n * 4, mtl.ResourceOptions.shared) };
        enc.setPipeline(self.pipeline("qi_quant_weight"));
        enc.setBuffer(w.buffer, w.offset, 0);
        enc.setValue([2]i32{ @intCast(k), @intCast(n) }, 1);
        enc.setBuffer(q.w, 0, 2);
        enc.setBuffer(q.s, 0, 3);
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
        return q;
    }

    fn paddedRows(self: *const Model) usize {
        return (self.rows + tile - 1) / tile * tile;
    }

    /// The stream normed, scaled and quantized: `q8` and one scale a row in `xs`.
    fn normScaleQ8(self: *const Model, enc: mtl.ComputeEncoder, scale: mtl.Buffer) void {
        enc.setPipeline(self.pipeline("qi_norm_scale_q8"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(scale, 0, 1);
        enc.setValue([2]i32{ @intCast(self.rows), hidden }, 2);
        enc.setValue([1]f32{eps}, 3);
        enc.setBuffer(self.q8, 0, 4);
        enc.setBuffer(self.xs, 0, 5);
        enc.dispatchGroups(mtl.Size.of(self.paddedRows(), 1, 1), mtl.Size.of(32, 1, 1));
    }

    /// `x += gate * y`, then the stream normed, scaled and quantized as `normScaleQ8` does, in one pass.
    fn gateNormQ8(self: *const Model, enc: mtl.ComputeEncoder, gate: mtl.Buffer, scale: mtl.Buffer) void {
        enc.setPipeline(self.pipeline("qi_gate_norm_q8"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(self.y, 0, 1);
        enc.setBuffer(gate, 0, 2);
        enc.setBuffer(scale, 0, 3);
        enc.setValue([2]i32{ @intCast(self.rows), hidden }, 4);
        enc.setValue([1]f32{eps}, 5);
        enc.setBuffer(self.q8, 0, 6);
        enc.setBuffer(self.xs, 0, 7);
        enc.dispatchGroups(mtl.Size.of(self.paddedRows(), 1, 1), mtl.Size.of(32, 1, 1));
    }

    /// Rows of `width` bf16 values to int8 with a scale per `group` channels.
    fn quantRows(self: *const Model, enc: mtl.ComputeEncoder, x: mtl.Buffer, q: mtl.Buffer, scales: mtl.Buffer, width: usize, group: usize) void {
        enc.setPipeline(self.pipeline("qi_quant_rows"));
        enc.setBuffer(x, 0, 0);
        enc.setValue([3]i32{ @intCast(self.rows), @intCast(width), @intCast(group) }, 1);
        enc.setBuffer(q, 0, 2);
        enc.setBuffer(scales, 0, 3);
        enc.dispatchGroups(mtl.Size.of(width / group, self.paddedRows(), 1), mtl.Size.of(32, 1, 1));
    }

    /// d = x w for int8 rows of `k` values with a scale per `group`, and an int8 projection (k, n).
    fn i8Linear(self: *const Model, enc: mtl.ComputeEncoder, x: mtl.Buffer, scales: mtl.Buffer, w: Quant, d: mtl.Buffer, k: usize, n: usize, group: usize) void {
        std.debug.assert(n == hidden and ((k == hidden and group == hidden) or (k == mlp_width and group == wide_group)));
        enc.setPipeline(if (k == hidden) self.pipeline("qi_i8_linear") else self.pipeline("qi_i8_linear_wide"));
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(scales, 0, 1);
        enc.setBuffer(w.w, 0, 2);
        enc.setBuffer(w.s, 0, 3);
        enc.setBuffer(d, 0, 4);
        enc.dispatchGroups(mtl.Size.of(n / ntile, (self.rows + mtile - 1) / mtile, 1), mtl.Size.of(product_threads, 1, 1));
    }

    fn rowsOf(self: *Model, count: usize) !mtl.Buffer {
        return self.device.buffer(count * 2, mtl.ResourceOptions.shared);
    }

    /// The image rows' rotary tables, (rows, 64) each.
    pub fn setRotary(self: *Model, cos: []const f32, sin: []const f32) void {
        @memcpy(self.cos.slice(f32, cos.len), cos);
        @memcpy(self.sin.slice(f32, sin.len), sin);
    }

    /// Block `index`'s prompt keys (heads, 128, text) and values (heads, text, 128), copied ahead of the image rows'.
    pub fn setPrompt(self: *Model, index: usize, keys: []const u16, vals: []const u16) void {
        const total = self.keys;
        const k = self.blocks[index].keys.slice(u16, heads * head_dim * total);
        const v = self.blocks[index].vals.slice(u16, heads * head_dim * total);
        for (0..heads) |h| {
            for (0..head_dim) |c| @memcpy(k[(h * head_dim + c) * total ..][0..self.text], keys[(h * head_dim + c) * self.text ..][0..self.text]);
            @memcpy(v[h * total * head_dim ..][0 .. self.text * head_dim], vals[h * self.text * head_dim ..][0 .. self.text * head_dim]);
        }
        if (!self.int8) return;
        // the same rows in int8, a scale for each (head, row)
        const b = &self.blocks[index];
        const k8 = b.k8.slice(i8, heads * head_dim * total);
        const v8 = b.v8.slice(i8, heads * head_dim * total);
        const ks = b.ks.slice(f32, heads * total);
        const vs = b.vs.slice(f32, heads * total);
        for (0..heads) |h| for (0..self.text) |t| {
            var ktop: f32 = 1e-12;
            var vtop: f32 = 1e-12;
            for (0..head_dim) |c| {
                ktop = @max(ktop, @abs(fromBf(keys[(h * head_dim + c) * self.text + t])));
                vtop = @max(vtop, @abs(fromBf(vals[(h * self.text + t) * head_dim + c])));
            }
            ks[h * total + t] = ktop / 127.0;
            vs[h * total + t] = vtop / 127.0;
            for (0..head_dim) |c| {
                k8[(h * total + t) * head_dim + c] = @intFromFloat(@round(fromBf(keys[(h * head_dim + c) * self.text + t]) * 127.0 / ktop));
                v8[(h * total + t) * head_dim + c] = @intFromFloat(@round(fromBf(vals[(h * self.text + t) * head_dim + c]) * 127.0 / vtop));
            }
        };
    }

    fn lap(self: *Model, pass: *Pass, comptime stage: []const u8) void {
        if (self.int8) pass.enc.barrier();
        if (!self.profile) return;
        pass.enc.end();
        pass.cb.commit();
        pass.cb.wait();
        inline for (stages, 0..) |name, i| if (comptime std.mem.eql(u8, name, stage)) {
            self.spent[i] += pass.cb.gpuSeconds();
        };
        pass.cb = self.queue.commandBuffer();
        pass.enc = pass.cb.compute(if (self.int8) .concurrent else .serial);
    }

    fn pipeline(self: *const Model, comptime name: []const u8) mtl.Pipeline {
        inline for (kernels.names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return self.pipelines[i];
        @compileError("no kernel " ++ name);
    }

    /// d[z] = a[z] b[z] for `batch` z: a (m, k), b (k, n); `ld` holds the leading dimensions, then the strides between z.
    fn gemm(self: *const Model, enc: mtl.ComputeEncoder, a: Ref, b: Ref, d: Ref, m: usize, n: usize, k: usize, ld: [6]i64, batch: usize) void {
        enc.setPipeline(self.pipeline("qi_gemm"));
        enc.setBuffer(a.buffer, a.offset, 0);
        enc.setBuffer(b.buffer, b.offset, 1);
        enc.setValue([3]i32{ @intCast(m), @intCast(n), @intCast(k) }, 2);
        enc.setValue(ld, 3);
        enc.setBuffer(d.buffer, d.offset, 4);
        enc.dispatchGroups(mtl.Size.of((n + 127) / 128, (m + 63) / 64, batch), mtl.Size.of(32, 4, 1));
    }

    /// `gemm` on whole tiles: n and k multiples of 128.
    fn tileGemm(self: *const Model, enc: mtl.ComputeEncoder, a: Ref, b: Ref, d: Ref, m: usize, n: usize, k: usize, ld: [6]i64, batch: usize) void {
        enc.setPipeline(self.pipeline("qi_tile_gemm"));
        enc.setBuffer(a.buffer, a.offset, 0);
        enc.setBuffer(b.buffer, b.offset, 1);
        enc.setValue([3]i32{ @intCast(m), @intCast(n), @intCast(k) }, 2);
        enc.setValue(ld, 3);
        enc.setBuffer(d.buffer, d.offset, 4);
        enc.dispatchGroups(mtl.Size.of(n / tile, (m + tile - 1) / tile, batch), mtl.Size.of(256, 1, 1));
    }

    /// d = a w for rows of `k` values and a projection stored (k, n).
    fn linear(self: *const Model, enc: mtl.ComputeEncoder, a: mtl.Buffer, w: Ref, d: mtl.Buffer, k: usize, n: usize) void {
        const ld = [6]i64{ @intCast(k), @intCast(n), @intCast(n), 0, 0, 0 };
        if (self.tiles and k % tile == 0 and n % tile == 0) {
            self.tileGemm(enc, .{ .buffer = a }, w, .{ .buffer = d }, self.rows, n, k, ld, 1);
        } else {
            self.gemm(enc, .{ .buffer = a }, w, .{ .buffer = d }, self.rows, n, k, ld, 1);
        }
    }

    fn normScale(self: *const Model, enc: mtl.ComputeEncoder, scale: mtl.Buffer) void {
        enc.setPipeline(self.pipeline("qi_norm_scale"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(scale, 0, 1);
        enc.setValue([1]i32{hidden}, 2);
        enc.setValue([1]f32{eps}, 3);
        enc.setBuffer(self.n, 0, 4);
        enc.dispatchGroups(mtl.Size.of(self.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }

    fn gateAdd(self: *const Model, enc: mtl.ComputeEncoder, gate: mtl.Buffer) void {
        enc.setPipeline(self.pipeline("qi_gate_add"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(self.y, 0, 1);
        enc.setBuffer(gate, 0, 2);
        enc.setValue([2]i32{ @intCast(self.rows), hidden }, 3);
        enc.dispatchThreads(mtl.Size.of(hidden, self.rows, 1), mtl.Size.of(64, 16, 1));
    }

    /// One row through a projection on the GPU: `from` (k values) to `to` (n values), waited for.
    fn projectRow(self: *const Model, w: Ref, from: mtl.Buffer, to: mtl.Buffer, k: usize, n: usize) void {
        const cb = self.queue.commandBuffer();
        const enc = cb.compute(.serial);
        self.gemm(enc, .{ .buffer = from }, w, .{ .buffer = to }, 1, n, k, .{ @intCast(k), @intCast(n), @intCast(n), 0, 0, 0 }, 1);
        enc.end();
        cb.commit();
        cb.wait();
    }

    fn silu(values: []u16) void {
        for (values) |*h| {
            const v = fromBf(h.*);
            h.* = toBf(v * round(1.0 / (1.0 + @exp(-v))));
        }
    }

    /// The modulation every block shares at noise level `sigma`, and the output norm's scale.
    fn modulate(self: *Model, sigma: f32) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const emb = self.mod_emb.slice(u16, time_dim);
        const half = time_dim / 2;
        for (0..half) |i| {
            const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, half));
            const angle = sigma * 1000.0 * freq;
            emb[i] = toBf(@cos(angle));
            emb[half + i] = toBf(@sin(angle));
        }
        self.projectRow(self.time_1, self.mod_emb, self.mod_a, time_dim, hidden);
        silu(self.mod_a.slice(u16, hidden));
        self.projectRow(self.time_2, self.mod_a, self.mod_temb, hidden, hidden);
        silu(self.mod_temb.slice(u16, hidden));
        self.projectRow(self.modulation, self.mod_temb, self.mod_table, hidden, 4 * hidden);
        self.projectRow(self.norm_out, self.mod_temb, self.mod_a, hidden, hidden);
        const table = self.mod_table.slice(u16, 4 * hidden);
        const outs = .{ self.scale_1, self.gate_1, self.scale_2, self.gate_2 };
        inline for (outs, 0..) |buffer, part| {
            const to = buffer.slice(u16, hidden);
            for (to, table[part * hidden ..][0..hidden]) |*o, v| o.* = toBf(if (part % 2 == 0) 1.0 + fromBf(v) else std.math.tanh(fromBf(v)));
        }
        for (self.scale_out.slice(u16, hidden), self.mod_a.slice(u16, hidden)) |*o, v| o.* = toBf(1.0 + fromBf(v));
    }

    /// The velocity (rows, 64) for `latents` (rows, 64) at noise level `sigma`.
    pub fn forward(self: *Model, latents: []const f32, sigma: f32, velocity: []f32) !Timing {
        const started = mtl.clock.seconds();
        self.modulate(sigma);
        for (self.latents.slice(u16, latents.len), latents) |*o, v| o.* = toBf(v);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const rows = self.rows;
        const total = self.keys;
        var pass: Pass = undefined;
        pass.cb = self.queue.commandBuffer();
        pass.enc = pass.cb.compute(if (self.int8) .concurrent else .serial);
        var gpu: f64 = 0;
        self.linear(pass.enc, self.latents, self.img_in, self.x, channels, hidden);
        if (self.int8) pass.enc.barrier();
        for (&self.blocks, 0..) |*b, index| {
            if (self.int8) {
                // the block before left its MLP's rows in y: its gated add runs with this block's first norm
                if (index == 0) self.normScaleQ8(pass.enc, self.scale_1) else self.gateNormQ8(pass.enc, self.gate_2, self.scale_1);
                self.lap(&pass, "norms");
                self.i8Linear(pass.enc, self.q8, self.xs, b.q_q, self.qr, hidden, hidden, hidden);
                self.i8Linear(pass.enc, self.q8, self.xs, b.q_k, self.kr, hidden, hidden, hidden);
                self.i8Linear(pass.enc, self.q8, self.xs, b.q_v, self.vr, hidden, hidden, hidden);
            } else {
                self.normScale(pass.enc, self.scale_1);
                self.lap(&pass, "norms");
                self.linear(pass.enc, self.n, b.to_q, self.qr, hidden, hidden);
                self.linear(pass.enc, self.n, b.to_k, self.kr, hidden, hidden);
                self.linear(pass.enc, self.n, b.to_v, self.vr, hidden, hidden);
            }
            self.lap(&pass, "projections");
            if (self.attention8) {
                const padded = self.paddedRows();
                pass.enc.setPipeline(self.pipeline("qi_heads_q8"));
                pass.enc.setBuffer(self.qr, 0, 0);
                pass.enc.setBuffer(self.kr, 0, 1);
                pass.enc.setBuffer(self.vr, 0, 2);
                pass.enc.setBuffer(b.norm_q.buffer, b.norm_q.offset, 3);
                pass.enc.setBuffer(b.norm_k.buffer, b.norm_k.offset, 4);
                pass.enc.setBuffer(self.cos, 0, 5);
                pass.enc.setBuffer(self.sin, 0, 6);
                pass.enc.setValue([5]i32{ @intCast(rows), heads, @intCast(self.text), @intCast(padded), @intCast(total) }, 7);
                pass.enc.setValue([1]f32{eps}, 8);
                pass.enc.setBuffer(self.qh8, 0, 9);
                pass.enc.setBuffer(self.qhs, 0, 10);
                pass.enc.setBuffer(b.k8, 0, 11);
                pass.enc.setBuffer(b.ks, 0, 12);
                pass.enc.setBuffer(b.v8, 0, 13);
                pass.enc.setBuffer(b.vs, 0, 14);
                pass.enc.dispatchGroups(mtl.Size.of(padded, heads, 1), mtl.Size.of(32, 1, 1));
                self.lap(&pass, "heads");
                pass.enc.setPipeline(self.pipeline("qi_attention_i8"));
                pass.enc.setBuffer(self.qh8, 0, 0);
                pass.enc.setBuffer(self.qhs, 0, 1);
                pass.enc.setBuffer(b.k8, 0, 2);
                pass.enc.setBuffer(b.ks, 0, 3);
                pass.enc.setBuffer(b.v8, 0, 4);
                pass.enc.setBuffer(b.vs, 0, 5);
                pass.enc.setValue([1]f32{1.0 / @sqrt(@as(f32, head_dim))}, 6);
                pass.enc.setBuffer(self.qr, 0, 7);
                pass.enc.dispatchGroups(mtl.Size.of((rows + query_tile - 1) / query_tile, heads, 1), mtl.Size.of(256, 1, 1));
                self.lap(&pass, "scores");
            } else {
                pass.enc.setPipeline(self.pipeline("qi_heads"));
                pass.enc.setBuffer(self.qr, 0, 0);
                pass.enc.setBuffer(self.kr, 0, 1);
                pass.enc.setBuffer(self.vr, 0, 2);
                pass.enc.setBuffer(b.norm_q.buffer, b.norm_q.offset, 3);
                pass.enc.setBuffer(b.norm_k.buffer, b.norm_k.offset, 4);
                pass.enc.setBuffer(self.cos, 0, 5);
                pass.enc.setBuffer(self.sin, 0, 6);
                pass.enc.setValue([4]i32{ @intCast(rows), heads, @intCast(self.text), @intCast(total) }, 7);
                pass.enc.setValue([1]f32{eps}, 8);
                pass.enc.setBuffer(self.qh, 0, 9);
                pass.enc.setBuffer(b.keys, 0, 10);
                pass.enc.setBuffer(b.vals, 0, 11);
                pass.enc.dispatchThreads(mtl.Size.of(rows, heads, 1), mtl.Size.of(32, 32, 1));
                self.lap(&pass, "heads");
                // scores = q k^T a head at a time, then probabilities in place, then probabilities times values
                const score_ld = [6]i64{ head_dim, @intCast(total), @intCast(total), @intCast(rows * head_dim), @intCast(head_dim * total), @intCast(rows * total) };
                if (self.tiles) self.tileGemm(pass.enc, .{ .buffer = self.qh }, .{ .buffer = b.keys }, .{ .buffer = self.scores }, rows, total, head_dim, score_ld, heads) else self.gemm(pass.enc, .{ .buffer = self.qh }, .{ .buffer = b.keys }, .{ .buffer = self.scores }, rows, total, head_dim, score_ld, heads);
                self.lap(&pass, "scores");
                pass.enc.setPipeline(self.pipeline("qi_softmax"));
                pass.enc.setBuffer(self.scores, 0, 0);
                pass.enc.setValue([2]i32{ @intCast(self.text + rows), @intCast(total) }, 1);
                pass.enc.setValue([1]f32{1.0 / @sqrt(@as(f32, head_dim))}, 2);
                pass.enc.dispatchGroups(mtl.Size.of(heads * rows, 1, 1), mtl.Size.of(32, 1, 1));
                self.lap(&pass, "softmax");
                const value_ld = [6]i64{ @intCast(total), head_dim, head_dim, @intCast(rows * total), @intCast(total * head_dim), @intCast(rows * head_dim) };
                if (self.tiles) self.tileGemm(pass.enc, .{ .buffer = self.scores }, .{ .buffer = b.vals }, .{ .buffer = self.oh }, rows, head_dim, total, value_ld, heads) else self.gemm(pass.enc, .{ .buffer = self.scores }, .{ .buffer = b.vals }, .{ .buffer = self.oh }, rows, head_dim, total, value_ld, heads);
                self.lap(&pass, "values");
                pass.enc.setPipeline(self.pipeline("qi_rows"));
                pass.enc.setBuffer(self.oh, 0, 0);
                pass.enc.setValue([2]i32{ @intCast(rows), heads }, 1);
                pass.enc.setBuffer(self.qr, 0, 2);
                pass.enc.dispatchThreads(mtl.Size.of(rows, heads, 1), mtl.Size.of(32, 32, 1));
                self.lap(&pass, "rows");
            }
            if (self.int8) {
                self.quantRows(pass.enc, self.qr, self.q8, self.xs, hidden, hidden);
                self.lap(&pass, "quantize");
                self.i8Linear(pass.enc, self.q8, self.xs, b.q_out, self.y, hidden, hidden, hidden);
                self.lap(&pass, "projections");
                self.gateNormQ8(pass.enc, self.gate_1, self.scale_2);
                self.lap(&pass, "norms");
                pass.enc.setPipeline(self.pipeline("qi_i8_swiglu"));
                pass.enc.setBuffer(self.q8, 0, 0);
                pass.enc.setBuffer(self.xs, 0, 1);
                pass.enc.setBuffer(b.q_gate.w, 0, 2);
                pass.enc.setBuffer(b.q_gate.s, 0, 3);
                pass.enc.setBuffer(b.q_proj.w, 0, 4);
                pass.enc.setBuffer(b.q_proj.s, 0, 5);
                pass.enc.setBuffer(self.wide_h, 0, 6);
                pass.enc.dispatchGroups(mtl.Size.of(mlp_width / ntile, (self.rows + mtile - 1) / mtile, 1), mtl.Size.of(product_threads, 1, 1));
                self.lap(&pass, "mlp in");
                self.quantRows(pass.enc, self.wide_h, self.w8, self.wxs, mlp_width, wide_group);
                self.lap(&pass, "quantize");
                self.i8Linear(pass.enc, self.w8, self.wxs, b.q_mlp_out, self.y, mlp_width, hidden, wide_group);
                self.lap(&pass, "mlp out");
                if (index == layers - 1) self.gateAdd(pass.enc, self.gate_2);
                self.lap(&pass, "gates");
                continue;
            }
            self.linear(pass.enc, self.qr, b.to_out, self.y, hidden, hidden);
            self.lap(&pass, "projections");
            self.gateAdd(pass.enc, self.gate_1);
            self.lap(&pass, "gates");
            self.normScale(pass.enc, self.scale_2);
            self.lap(&pass, "norms");
            self.linear(pass.enc, self.n, b.gate, self.wide_a, hidden, mlp_width);
            self.linear(pass.enc, self.n, b.proj, self.wide_b, hidden, mlp_width);
            self.lap(&pass, "mlp projections");
            pass.enc.setPipeline(self.pipeline("qi_swiglu"));
            pass.enc.setBuffer(self.wide_a, 0, 0);
            pass.enc.setBuffer(self.wide_b, 0, 1);
            pass.enc.setValue([1]i64{@intCast(rows * mlp_width)}, 2);
            pass.enc.setBuffer(self.wide_h, 0, 3);
            pass.enc.dispatchThreads(mtl.Size.of(1024, (rows * mlp_width + 1023) / 1024, 1), mtl.Size.of(1024, 1, 1));
            self.lap(&pass, "swiglu");
            self.linear(pass.enc, self.wide_h, b.out, self.y, mlp_width, hidden);
            self.lap(&pass, "mlp projections");
            self.gateAdd(pass.enc, self.gate_2);
            self.lap(&pass, "gates");
        }
        self.normScale(pass.enc, self.scale_out);
        if (self.int8) pass.enc.barrier();
        self.linear(pass.enc, self.n, self.proj_out, self.velocity, hidden, channels);
        pass.enc.end();
        pass.cb.commit();
        pass.cb.wait();
        gpu += pass.cb.gpuSeconds();
        if (pass.cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        for (velocity, self.velocity.slice(u16, velocity.len)) |*o, v| o.* = fromBf(v);
        return .{ .wall = mtl.clock.seconds() - started, .gpu = gpu };
    }
};

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(f32, 1.0), fromBf(toBf(1.0)));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf(@bitCast(@as(u32, 0x3f818000))));
}
