//! macOS memory pressure for the shared lane host.
extern "c" fn sysctlbyname(name: [*:0]const u8, old: ?*anyopaque, old_len: ?*usize, new: ?*anyopaque, new_len: usize) c_int;

/// The kernel's memory pressure level (1 normal, 2 warning, 4 critical); null where the OS gives none.
pub fn pressure() ?u32 {
    if (@import("builtin").os.tag != .macos) return null;
    var level: u32 = 0;
    var len: usize = @sizeOf(u32);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, null, 0) != 0 or len != @sizeOf(u32)) return null;
    return level;
}

