//! A served model's long-lived buffers in one residency set, kept wired by the server's idle keepalive.
const std = @import("std");
const mtl = @import("metal");

/// macOS unwires GPU memory after ~1-2 s idle and a request re-wires it all: keepalive ticks use the set.
pub const Resident = struct {
    set: mtl.ResidencySet,
    work: mtl.Queue,
    scratch: ?mtl.Buffer = null, // the keepalive's tick copies within it (`ticked`, or TF_KEEPALIVE_WORK=1)

    /// `buffers` join one set that `work` holds; the set is requested resident at once.
    pub fn init(device: mtl.Device, work: mtl.Queue, buffers: []const mtl.Buffer) !Resident {
        return initTicked(device, work, buffers, false);
    }

    /// `init`, the keepalive's ticks copying 16 bytes inside the set when `ticked`: an empty command buffer need not
    /// reach the GPU, and Flash Next's first prompt pass after an idle second still waited ~1.2 s in its queue while
    /// the set came back (2026-10-12): with work in every tick none did.
    pub fn initTicked(device: mtl.Device, work: mtl.Queue, buffers: []const mtl.Buffer, ticked: bool) !Resident {
        const set = try device.residencySet(buffers.len + 1);
        errdefer set.deinit();
        for (buffers) |b| set.add(b);
        const asked = if (std.c.getenv("TF_KEEPALIVE_WORK")) |v| v[0] == '1' else false;
        const scratch: ?mtl.Buffer = if (ticked or asked) try device.buffer(64, mtl.ResourceOptions.shared) else null;
        if (scratch) |b| set.add(b);
        set.commit();
        set.requestResidency();
        work.addResidencySet(set);
        return .{ .set = set, .work = work, .scratch = scratch };
    }

    /// More buffers into the set (a slot's memory taken after load), requested resident with the rest.
    pub fn add(r: *Resident, buffers: []const mtl.Buffer) void {
        for (buffers) |b| r.set.add(b);
        r.set.commit();
        r.set.requestResidency();
    }

    /// The idle keepalive's target: a tiny command buffer on `work` using the set; `r` must not move.
    pub fn target(r: *const Resident) mtl.keepalive.Target {
        return .{ .queue = r.work, .sets = @as(*const [1]mtl.ResidencySet, &r.set), .scratch = r.scratch };
    }

    pub fn deinit(r: *Resident) void {
        r.work.removeResidencySet(r.set);
        r.set.deinit();
        if (r.scratch) |b| b.deinit();
    }
};

/// FLASHNEXT_RESIDENCY=weights: a Flash Next host's set holds the checkpoint's tensors only, not its working buffers.
pub fn weightsOnly() bool {
    const v = std.c.getenv("FLASHNEXT_RESIDENCY") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "weights");
}
