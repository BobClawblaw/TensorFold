//! tf-moss-image-noise: MOSS-Image's native starting latents against torch.randn's, bit for bit.
const std = @import("std");
const tf = @import("tensorfold");
const mi = tf.moss_image;

const usage =
    \\tf-moss-image-noise REFERENCE
    \\REFERENCE (moss_image_case.py noise) holds torch.randn's starting latents for several seeds and sizes, named
    \\latents.SEED.WIDTHxHEIGHT. Each must equal the native draw for that seed and size bit for bit; else it fails.
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var reference = try tf.safetensors.File.open(a, io, args[1]);
    defer reference.close(io);
    var checked: usize = 0;
    var differ: usize = 0;
    var values: usize = 0;
    var it = reference.names.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        var parts = std.mem.splitScalar(u8, name, '.');
        if (!std.mem.eql(u8, parts.next() orelse "", "latents")) continue;
        const seed = try std.fmt.parseInt(u64, parts.next() orelse return error.BadName, 10);
        var size = std.mem.splitScalar(u8, parts.next() orelse return error.BadName, 'x');
        const width = try std.fmt.parseInt(usize, size.next() orelse return error.BadName, 10);
        const height = try std.fmt.parseInt(usize, size.next() orelse return error.BadName, 10);
        const g = try mi.config.grid(width, height);
        const ours = try mi.noise.latents(a, seed, g.rows, g.columns, mi.config.channels);
        const t = reference.get(name).?;
        const same = t.dtype == .f32 and std.mem.eql(u8, t.bytes, std.mem.sliceAsBytes(ours));
        if (!same) {
            differ += 1;
            std.debug.print("seed {d} at {d}x{d}: DIFFERENT from torch.randn's\n", .{ seed, width, height });
        }
        checked += 1;
        values += ours.len;
    }
    std.debug.print("{d} draws ({d} values); {d} differ\n", .{ checked, values, differ });
    const pass = checked > 0 and differ == 0;
    std.debug.print("{s}\n", .{if (pass) "PASS every draw equal bit for bit" else "FAIL"});
    return if (pass) 0 else 1;
}
