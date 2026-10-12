//! Activation rows in float64, the reference every backend's activation kernels are checked against.
const std = @import("std");

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// silu(gate) * up.
pub fn swiglu(gate: f64, up: f64) f64 {
    return gate * sigmoid(gate) * up;
}

/// A bf16 output of fp32 arithmetic through expf: half a bf16 ulp plus 2^-18 of |want|.
pub fn tolerance(want: f64) f64 {
    return @abs(want) * (std.math.pow(f64, 2, -8) + std.math.pow(f64, 2, -18)) + 1e-30;
}

test "activations against hand values" {
    try std.testing.expectEqual(@as(f64, 0.5), sigmoid(0));
    try std.testing.expectApproxEqRel(@as(f64, 2 * sigmoid(2) * 3), swiglu(2, 3), 1e-15);
    try std.testing.expect(tolerance(1) > tolerance(0.5));
}
