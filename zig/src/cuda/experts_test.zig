//! Host regressions of experts.zig's packers and launch geometry.
const std = @import("std");
const experts = @import("experts.zig");
const Fp8Experts = experts.Fp8Experts;
const Nvfp4Experts = experts.Nvfp4Experts;

test "a packed fp8 expert puts each word where fp8/experts.py's pack does" {
    const gpa = std.testing.allocator;
    const n = 128;
    const k = 128;
    const w = try gpa.alloc(u8, n * k);
    defer gpa.free(w);
    for (w, 0..) |*b, i| b.* = @truncate(i * 31 + 7);
    const out = try gpa.alloc(u32, n * k / 4);
    defer gpa.free(out);
    Fp8Experts.packOne(out, w, n, k);
    // (cb 1, g 2, gq 5, t 3, h 1, j 2): row 32 + 16 + 5 = 53, word 2 * 8 + 4 + 3 = 23
    const idx = (((((1 * 4 + 2) * 8 + 5) * 4 + 3) * 2 + 1) * 4) + 2;
    const words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, w));
    try std.testing.expectEqual(words[53 * (k / 4) + 23], out[idx]);
}

test "NVFP4 decode CTA geometry is explicit, bounded and defaults to four warps" {
    try std.testing.expectEqual(@as(usize, 4), try Nvfp4Experts.decodeWarps(null));
    try std.testing.expectEqual(@as(usize, 1), try Nvfp4Experts.decodeWarps("1"));
    try std.testing.expectEqual(@as(usize, 4), try Nvfp4Experts.decodeWarps("4"));
    for ([_][]const u8{ "0", "2", "8", "", "garbage" }) |v| try std.testing.expectError(error.BadExpertWarps, Nvfp4Experts.decodeWarps(v));
}
