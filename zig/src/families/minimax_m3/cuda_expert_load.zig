//! A rank's share of a MoE layer's routed experts: the release's NVFP4 packed as such, or the 3-bit refit.

const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");

const X = cuda.experts.Nvfp4Experts;
const A3 = cuda.experts3.Affine3Experts;
const Loader = weights.Loader;
const prefix = weights.prefix;

/// An expert's stored NVFP4 matrix: codes [n, k/2] (low nibble first), e4m3 scales [n, k/16], the dividing global.
const Fp4 = struct { codes: []const u8, scales: []const u8, global: f32, n: usize, k: usize };

fn fp4(L: *Loader, i: u32, e: usize, comptime w: []const u8, n: usize, k: usize) !Fp4 {
    const codes = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.experts.{d}." ++ w ++ ".weight_packed", .{ i, e }, .u8, &.{ n, k / 2 });
    const scales = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.experts.{d}." ++ w ++ ".weight_scale", .{ i, e }, .f8_e4m3, &.{ n, k / 16 });
    const g = try L.get(prefix ++ "model.layers.{d}.block_sparse_moe.experts.{d}." ++ w ++ ".weight_global_scale", .{ i, e });
    if (g.dtype != .f32 or g.numel() != 1) return error.UnexpectedMiniMaxTensor;
    const global: f32 = @bitCast(std.mem.readInt(u32, g.bytes[0..4], .little));
    if (!(global > 0)) return error.UnexpectedMiniMaxTensor;
    return .{ .codes = codes.bytes, .scales = scales.bytes, .global = global, .n = n, .k = k };
}

/// Layer i's NVFP4 experts: gate and up by the rank's rows, down by its columns; each global as its reciprocal.
pub fn nvfp4(L: *Loader, i: u32) !cuda.experts.Layer {
    const c = &L.w.config;
    const d: usize = c.hidden;
    const cut = L.w.share.moe;
    const ni = cut.len();
    const count: usize = c.experts;
    const inter: usize = c.moe_intermediate;
    const up_len = X.bytes(1, ni, d, 2) / 4;
    const down_len = X.bytes(1, d, ni, 1) / 4;
    const up = try L.gpa.alloc(u32, count * up_len);
    defer L.gpa.free(up);
    const down = try L.gpa.alloc(u32, count * down_len);
    defer L.gpa.free(down);
    const up_scale = try L.gpa.alloc(f32, count * 2);
    defer L.gpa.free(up_scale);
    const down_scale = try L.gpa.alloc(f32, count);
    defer L.gpa.free(down_scale);
    const codes = try L.gpa.alloc(u8, d * ni / 2);
    defer L.gpa.free(codes);
    const scales = try L.gpa.alloc(u8, d * ni / 16);
    defer L.gpa.free(scales);
    for (0..count) |e| {
        const g = try fp4(L, i, e, "w1", inter, d);
        const u = try fp4(L, i, e, "w3", inter, d);
        const dn = try fp4(L, i, e, "w2", d, inter);
        // gate and up: the rank's rows; down: every row, the rank's columns
        X.packGateUp(up[e * up_len ..][0..up_len], .{ g.codes[cut.lo * d / 2 ..][0 .. ni * d / 2], g.scales[cut.lo * d / 16 ..][0 .. ni * d / 16] }, .{ u.codes[cut.lo * d / 2 ..][0 .. ni * d / 2], u.scales[cut.lo * d / 16 ..][0 .. ni * d / 16] }, ni, d);
        for (0..d) |r| {
            @memcpy(codes[r * ni / 2 ..][0 .. ni / 2], dn.codes[r * inter / 2 + cut.lo / 2 ..][0 .. ni / 2]);
            @memcpy(scales[r * ni / 16 ..][0 .. ni / 16], dn.scales[r * inter / 16 + cut.lo / 16 ..][0 .. ni / 16]);
        }
        X.packOne(down[e * down_len ..][0..down_len], codes, scales, d, ni);
        up_scale[e * 2] = 1.0 / g.global;
        up_scale[e * 2 + 1] = 1.0 / u.global;
        down_scale[e] = 1.0 / dn.global;
    }
    const up_p = try L.put(std.mem.sliceAsBytes(up));
    const down_p = try L.put(std.mem.sliceAsBytes(down));
    const us_p = try L.put(std.mem.sliceAsBytes(up_scale));
    const ds_p = try L.put(std.mem.sliceAsBytes(down_scale));
    return .{ .format = .nvfp4, .up = up_p, .down = down_p, .up_scale = us_p, .down_scale = ds_p, .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit, .oai = true };
}

/// An affine matrix as stored: u32 words [E][n][k * 3 / 32], bf16 scales and biases [E][n][k / 64].
const Affine = struct { words: []const u8, scales: []const u8, biases: []const u8, n: usize, k: usize };

fn affine(L: *Loader, i: u32, comptime proj: []const u8, count: usize, n: usize, k: usize) !Affine {
    const w = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.experts_affine." ++ proj ++ ".weight", .{i}, .u32, &.{ count, n, k * 3 / 32 });
    const s = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.experts_affine." ++ proj ++ ".scales", .{i}, .bf16, &.{ count, n, k / 64 });
    const b = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.experts_affine." ++ proj ++ ".biases", .{i}, .bf16, &.{ count, n, k / 64 });
    return .{ .words = w.bytes, .scales = s.bytes, .biases = b.bytes, .n = n, .k = k };
}

/// Rows [lo, hi), inputs [klo, khi) of experts [e0, e0 + ec) of `a` uploaded as a pack's words, scales, biases.
fn stage(L: *Loader, a: Affine, e0: usize, ec: usize, lo: usize, hi: usize, klo: usize, khi: usize, bufs: *[3]cuda.DeviceBuffer) !void {
    const n = hi - lo;
    const k = khi - klo;
    const wrow = a.k * 3 / 32 * 4; // a row's word bytes as stored
    const wcut = k * 3 / 32 * 4;
    const grow = a.k / 64 * 2;
    const gcut = k / 64 * 2;
    const words = try L.gpa.alloc(u8, ec * n * wcut);
    defer L.gpa.free(words);
    const scales = try L.gpa.alloc(u8, ec * n * gcut);
    defer L.gpa.free(scales);
    const biases = try L.gpa.alloc(u8, ec * n * gcut);
    defer L.gpa.free(biases);
    for (0..ec) |j| for (0..n) |r| {
        const src = ((e0 + j) * a.n + lo + r);
        const dst = j * n + r;
        @memcpy(words[dst * wcut ..][0..wcut], a.words[src * wrow + klo * 3 / 32 * 4 ..][0..wcut]);
        @memcpy(scales[dst * gcut ..][0..gcut], a.scales[src * grow + klo / 64 * 2 ..][0..gcut]);
        @memcpy(biases[dst * gcut ..][0..gcut], a.biases[src * grow + klo / 64 * 2 ..][0..gcut]);
    };
    try bufs[0].upload(0, words);
    try bufs[1].upload(0, scales);
    try bufs[2].upload(0, biases);
}

/// Layer i's 3-bit experts: the rank's gate and up rows as one pair table, its down columns, chunks of 16.
pub fn affine3(L: *Loader, i: u32) !cuda.experts.Layer {
    const c = &L.w.config;
    const d: usize = c.hidden;
    const cut = L.w.share.moe;
    const ni = cut.len();
    const count: usize = c.experts;
    const inter: usize = c.moe_intermediate;
    const gu = try affine(L, i, "gate_up_proj", count, 2 * inter, d);
    const dn = try affine(L, i, "down_proj", count, d, inter);
    const chunk: usize = 16;
    const pair_table = try cuda.DeviceBuffer.alloc(L.t.d, A3.pairBytes(count, ni, d));
    try L.w.buffers.append(L.gpa, pair_table);
    L.w.bytes += pair_table.len;
    const down_table = try cuda.DeviceBuffer.alloc(L.t.d, A3.bytes(count, d, ni));
    try L.w.buffers.append(L.gpa, down_table);
    L.w.bytes += down_table.len;
    const s_up = A3.scratchBytes(chunk, ni, d);
    const s_dn = A3.scratchBytes(chunk, d, ni);
    var scratch: [2]cuda.DeviceBuffer = .{ try cuda.DeviceBuffer.alloc(L.t.d, @max(s_up[0], s_dn[0])), undefined };
    defer scratch[0].free();
    scratch[1] = try cuda.DeviceBuffer.alloc(L.t.d, @max(s_up[1], s_dn[1]));
    defer scratch[1].free();
    const stage_rows = @max(ni * d, d * ni);
    var bufs: [3]cuda.DeviceBuffer = .{ try cuda.DeviceBuffer.alloc(L.t.d, chunk * stage_rows * 3 / 32 * 4), undefined, undefined };
    defer bufs[0].free();
    bufs[1] = try cuda.DeviceBuffer.alloc(L.t.d, chunk * stage_rows / 64 * 2);
    defer bufs[1].free();
    bufs[2] = try cuda.DeviceBuffer.alloc(L.t.d, chunk * stage_rows / 64 * 2);
    defer bufs[2].free();
    const sc: [2]u64 = .{ scratch[0].ptr, scratch[1].ptr };
    var e0: usize = 0;
    while (e0 < count) : (e0 += chunk) {
        const ec = @min(chunk, count - e0);
        const pair_at = pair_table.ptr + e0 * A3.pairBytes(1, ni, d);
        try stage(L, gu, e0, ec, cut.lo, cut.hi, 0, d, &bufs); // gate rows
        try L.t.a3.packInto(L.t.a4, L.t.stream, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, sc, pair_at, ec, ni, d, 2, 0);
        try L.t.stream.synchronize();
        try stage(L, gu, e0, ec, inter + cut.lo, inter + cut.hi, 0, d, &bufs); // up rows
        try L.t.a3.packInto(L.t.a4, L.t.stream, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, sc, pair_at, ec, ni, d, 2, 1);
        try L.t.stream.synchronize();
        try stage(L, dn, e0, ec, 0, d, cut.lo, cut.hi, &bufs); // down: every row, the rank's columns
        try L.t.a3.packInto(L.t.a4, L.t.stream, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, sc, down_table.ptr + e0 * A3.bytes(1, d, ni), ec, d, ni, 1, 0);
        try L.t.stream.synchronize();
    }
    return .{ .format = .affine3, .up = pair_table.ptr, .down = down_table.ptr, .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit, .swiglu = true, .oai = true };
}
