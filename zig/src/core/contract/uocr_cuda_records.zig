//! Unlimited-OCR-2's CUDA launches (uocr_vision.cu, uocr_decoder.cu), from kernel-contract-limits on nvcc 13.0.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{};
