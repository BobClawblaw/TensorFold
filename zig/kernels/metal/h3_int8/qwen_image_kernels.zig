//! Qwen-Image-2.1's kernels: one Metal source, compiled when the model loads.
pub const source = @embedFile("qwen_image.metal");
pub const names = [_][:0]const u8{ "qi_gemm", "qi_tile_gemm", "qi_norm_scale", "qi_heads", "qi_softmax", "qi_rows", "qi_gate_add", "qi_swiglu", "qi_quant_weight", "qi_quant_rows", "qi_norm_scale_q8", "qi_gate_norm_q8", "qi_i8_linear", "qi_i8_linear_wide", "qi_i8_swiglu", "qi_heads_q8", "qi_attention_i8" };
