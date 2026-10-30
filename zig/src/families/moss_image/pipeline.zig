//! MOSS-Image's sampling: one Euler step under the pipeline's classifier-free guidance.
const model = @import("model.zig");

/// The card's guidance scale.
pub const guidance: f32 = 3.0;
/// The card's: the block the negative branch skips.
pub const negative_skip: usize = 22;

pub const Spent = struct { wall: f64, gpu: f64, host: f64 };

/// x from `sigma` to `next` by uncond + g (cond - uncond) in fp32, as the pipeline combines them; cond alone at g 1.
pub fn step(m: *model.Model, prompt: *const model.Prompt, negative: ?*const model.Prompt, x: []f32, timestep: f32, sigma: f32, next: f32, g: f32, cond: []f32, uncond: []f32) !Spent {
    const c = try m.forward(prompt, x, timestep, null, cond);
    const by = next - sigma;
    if (g == 1) {
        for (x, cond) |*o, pc| o.* += pc * by;
        return .{ .wall = c.wall, .gpu = c.gpu, .host = c.host };
    }
    const u = try m.forward(negative orelse return error.NoNegativePrompt, x, timestep, negative_skip, uncond);
    for (x, cond, uncond) |*o, pc, pu| o.* += (pu + g * (pc - pu)) * by;
    return .{ .wall = c.wall + u.wall, .gpu = c.gpu + u.gpu, .host = c.host + u.host };
}
