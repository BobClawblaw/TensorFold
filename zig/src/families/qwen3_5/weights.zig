//! Views into the checkpoint's buffers: the head aliases the packed input embedding when tied, else lm_head.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const c = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = struct { weight: Tensor, scales: Tensor, biases: Tensor, inputs: usize, outputs: usize };
pub const Delta = struct { qkv: Linear, z: Linear, a: Linear, b: Linear, out: Linear, conv: Tensor, norm: Tensor, a_log: Tensor, dt_bias: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    embedding: Linear,
    lm_head: ?Linear, // untied checkpoints only
    norm: Tensor,
    blocks: []Layer,
    a_logs: mtl.Buffer, // f32 [layers, value heads]: each DeltaNet layer's A_log, converted when the checkpoint's is bf16

    pub fn head(self: *const Weights) Linear {
        return self.lm_head orelse self.embedding;
    }

    pub fn deinit(self: *Weights) void {
        self.a_logs.deinit();
        self.gpa.free(self.blocks);
    }
};

fn expect(ckpt: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    const t = try ckpt.get(name);
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("{s}: unexpected Qwen tensor dtype or shape", .{name});
        return error.UnexpectedQwenTensor;
    }
    return t;
}

fn tensor(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    return expect(ckpt, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), dtype, shape);
}

fn projection(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, outputs: usize, inputs: usize) !Linear {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    return .{
        .weight = try tensor(ckpt, name, ".weight", .u32, &.{ outputs, inputs / 8 }),
        .scales = try tensor(ckpt, name, ".scales", .bf16, &.{ outputs, inputs / c.group }),
        .biases = try tensor(ckpt, name, ".biases", .bf16, &.{ outputs, inputs / c.group }),
        .inputs = inputs,
        .outputs = outputs,
    };
}

/// A layer's A_log as f32 at `row` of `a_logs`: the 2B stores it f32, the 27B bf16 (the kernel reads f32 either way).
fn aLog(ckpt: *const ck.Checkpoint, prefix: []const u8, a_logs: mtl.Buffer, row: usize, heads: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const t = try ckpt.get(try std.fmt.bufPrint(&buf, "{s}linear_attn.A_log", .{prefix}));
    if (t.rank != 1 or t.shape[0] != heads or (t.dtype != .f32 and t.dtype != .bf16)) return error.UnexpectedQwenTensor;
    const dst: [*]f32 = @ptrCast(@alignCast(a_logs.contents() + row * heads * 4));
    const src = t.buffer.contents() + t.offset;
    for (0..heads) |h| {
        dst[h] = if (t.dtype == .f32) @as(*align(1) const f32, @ptrCast(src + h * 4)).* else @bitCast(@as(u32, @as(*align(1) const u16, @ptrCast(src + h * 2)).*) << 16);
    }
    return .{ .buffer = a_logs, .offset = row * heads * 4, .bytes = heads * 4, .dtype = .f32, .shape = .{ heads, 1, 1, 1 }, .rank = 1 };
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, ckpt: *const ck.Checkpoint, g: c.Geometry) !Weights {
    const prefix = "language_model.model.";
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    var w = Weights{
        .gpa = gpa,
        .embedding = try projection(ckpt, prefix, "embed_tokens", c.vocab, g.hidden),
        .lm_head = null,
        .norm = try tensor(ckpt, prefix, "norm.weight", .bf16, &.{g.hidden}),
        .blocks = try gpa.alloc(Layer, g.layers),
        .a_logs = try device.buffer(g.layers * g.linear_v_heads * 4, opts),
    };
    errdefer w.deinit();
    const has_head = ckpt.has("language_model.lm_head.weight");
    if (has_head == g.tied) return if (g.tied) error.UnexpectedUntiedQwenHead else error.MissingQwenHead;
    if (!g.tied) w.lm_head = try projection(ckpt, "language_model.", "lm_head", c.vocab, g.hidden);
    for (w.blocks, 0..) |*block, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "{s}layers.{d}.", .{ prefix, i });
        block.* = .{
            .input_norm = try tensor(ckpt, p, "input_layernorm.weight", .bf16, &.{g.hidden}),
            .post_norm = try tensor(ckpt, p, "post_attention_layernorm.weight", .bf16, &.{g.hidden}),
            .gate = try projection(ckpt, p, "mlp.gate_proj", g.intermediate, g.hidden),
            .up = try projection(ckpt, p, "mlp.up_proj", g.intermediate, g.hidden),
            .down = try projection(ckpt, p, "mlp.down_proj", g.hidden, g.intermediate),
            .mixer = undefined,
        };
        if (c.linear(i)) {
            block.mixer = .{ .delta = .{
                .qkv = try projection(ckpt, p, "linear_attn.in_proj_qkv", g.convDim(), g.hidden),
                .z = try projection(ckpt, p, "linear_attn.in_proj_z", g.vInner(), g.hidden),
                .a = try projection(ckpt, p, "linear_attn.in_proj_a", g.linear_v_heads, g.hidden),
                .b = try projection(ckpt, p, "linear_attn.in_proj_b", g.linear_v_heads, g.hidden),
                .out = try projection(ckpt, p, "linear_attn.out_proj", g.hidden, g.vInner()),
                .conv = try tensor(ckpt, p, "linear_attn.conv1d.weight", .bf16, &.{ g.convDim(), c.conv_taps, 1 }),
                .norm = try tensor(ckpt, p, "linear_attn.norm.weight", .bf16, &.{c.linear_dim}),
                .a_log = try aLog(ckpt, p, w.a_logs, i, g.linear_v_heads),
                .dt_bias = try tensor(ckpt, p, "linear_attn.dt_bias", .bf16, &.{g.linear_v_heads}),
            } };
        } else {
            block.mixer = .{ .attention = .{
                .q = try projection(ckpt, p, "self_attn.q_proj", 2 * g.qInner(), g.hidden),
                .k = try projection(ckpt, p, "self_attn.k_proj", g.kvInner(), g.hidden),
                .v = try projection(ckpt, p, "self_attn.v_proj", g.kvInner(), g.hidden),
                .out = try projection(ckpt, p, "self_attn.o_proj", g.hidden, g.qInner()),
                .q_norm = try tensor(ckpt, p, "self_attn.q_norm.weight", .bf16, &.{c.head_dim}),
                .k_norm = try tensor(ckpt, p, "self_attn.k_norm.weight", .bf16, &.{c.head_dim}),
            } };
        }
    }
    return w;
}
