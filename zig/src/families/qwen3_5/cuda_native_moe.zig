//! The qwen3_5_moe model type (Qwen3.6-35B-A3B) on the same CUDA engine: routed experts in place of the dense MLP.
const native = @import("cuda_native.zig");

pub const model_type = "qwen3_5_moe";
pub const formats = native.formats;
pub const default_context = native.default_context;
pub const max_segments = native.max_segments;
pub const prompt_rows = native.prompt_rows;
/// Drafts come from the checkpoint's MTP layer, not a separate draft model.
pub const takes_drafter = false;
pub const open = native.open;
pub const openDrafter = native.openDrafter;
pub const explain = native.explain;
