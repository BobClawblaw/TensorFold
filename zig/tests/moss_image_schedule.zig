//! tf-moss-image-schedule: MOSS-Image's native schedule against the publisher's scheduler's, bit for bit.
const std = @import("std");
const tf = @import("tensorfold");
const mi = tf.moss_image;

const usage =
    \\tf-moss-image-schedule MODEL_DIR REFERENCE
    \\MODEL_DIR is the OpenMOSS-Team/MOSS-Image-Base folder; REFERENCE (moss_image_case.py schedule) holds the
    \\publisher's MOSSImageScheduler's timesteps and sigmas for 1 to N steps. Each step count's native schedule, from
    \\MODEL_DIR's scheduler/scheduler_config.json, must equal them bit for bit; else it fails.
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ args[1], "scheduler", "scheduler_config.json" }), a, .limited(1 << 20));
    const config = try mi.schedule.parse(a, text);
    var reference = try tf.safetensors.File.open(a, io, args[2]);
    defer reference.close(io);
    var checked: usize = 0;
    var differ: usize = 0;
    var shortened: usize = 0;
    var steps: usize = 1;
    var name: [32]u8 = undefined;
    while (reference.get(try std.fmt.bufPrint(&name, "timesteps.{d}", .{steps}))) |_| : (steps += 1) {
        const s = try mi.schedule.make(a, config, steps);
        const same = equal(&reference, try std.fmt.bufPrint(&name, "timesteps.{d}", .{steps}), s.timesteps) and
            equal(&reference, try std.fmt.bufPrint(&name, "sigmas.{d}", .{steps}), s.sigmas);
        if (!same) {
            differ += 1;
            if (differ <= 5) std.debug.print("{d} steps: DIFFERENT from the scheduler's\n", .{steps});
        }
        if (s.timesteps.len != steps) shortened += 1;
        checked += 1;
    }
    std.debug.print("{d} step counts (1 to {d}); {d} with repeats dropped; {d} differ\n", .{ checked, checked, shortened, differ });
    const pass = checked > 0 and differ == 0;
    std.debug.print("{s}\n", .{if (pass) "PASS every schedule equal bit for bit" else "FAIL"});
    return if (pass) 0 else 1;
}

/// Whether the reference's float32 tensor `name` holds exactly `ours`.
fn equal(f: *const tf.safetensors.File, name: []const u8, ours: []const f32) bool {
    const t = f.get(name) orelse return false;
    return t.dtype == .f32 and std.mem.eql(u8, t.bytes, std.mem.sliceAsBytes(ours));
}
