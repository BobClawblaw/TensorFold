//! MiniMax H3's kernels: Qwen-Image's source (the int8 products) followed by this family's, compiled at load.
const qi = @import("qwen_image_kernels");
pub const source = qi.source ++ @embedFile("h3.metal");
/// The block projections' int8 rounding and adapter merge: no case-dependent defines, compiled before the case.
pub const weights_source = @embedFile("weights.metal");
pub const weight_names = [_][:0]const u8{ "h3_quant_weight_t", "h3_lora_merge", "h3_quant_merged_t" };
pub const names = [_][:0]const u8{ "h3_rows_in", "h3_norm_q8", "h3_gate_add", "h3_heads_q8", "h3_tile_scales", "h3_pool", "h3_tile_scores", "h3_topk", "h3_coarse", "h3_attention_tiles", "h3_value_sums", "h3_attention_w8", "h3_gate_mix", "h3_mix_quant", "h3_i8_in", "h3_i8_out", "h3_final_norm", "h3_rows_out", "qi_quant_rows", "qi_i8_linear_wide", "qi_i8_swiglu", "h3_i8_swiglu" };
