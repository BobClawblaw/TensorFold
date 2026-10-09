//! A sequence's caches that grow with its position: one address range reserved for the whole window (pointers and
//! captured graphs stay valid), physical memory mapped into each region as positions reach it, zeroed on mapping (a
//! reused sequence never reads another request's rows), and charged to one budget per engine (a GB10 shares its
//! memory with the host: a growth past the budget is refused, never left to the kernel's OOM killer).

const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;

/// Positions a region grows by at a time (the Python engine's STEP).
pub const step_positions: i64 = 8192;

/// Bytes every sequence may map together, shared by the engine's sequences.
pub const Budget = struct {
    cap: u64,
    used: u64 = 0,
};

/// A region of the reserved range: ``bytes_per`` bytes for every ``per`` positions (a pooled index: 4).
pub const Region = struct { off: u64, cap: u64, bytes_per: u64, per: i64 = 1, mapped: u64 = 0 };

const Chunk = struct { at: u64, size: u64, handle: abi.MemHandle };

pub const Range = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    base: u64 = 0,
    size: u64 = 0,
    gran: u64,
    regions: std.ArrayList(Region) = .empty,
    chunks: std.ArrayList(Chunk) = .empty,
    budget: *Budget,

    /// The mapping granularity of device memory (regions start at multiples of it).
    pub fn granularity(d: *const cuda.Driver) !u64 {
        var g: usize = 0;
        const prop: abi.MemAllocationProp = .{};
        try d.check(d.api.cuMemGetAllocationGranularity(&g, &prop, 0), "cuMemGetAllocationGranularity");
        return g;
    }

    /// Lays out a region of ``cap`` bytes (rounded up to the granularity); call before ``reserve``. Its address is
    /// ``base + off`` once reserved.
    pub fn add(self: *Range, cap: u64, bytes_per: u64, per: i64) !u64 {
        const off = self.size;
        try self.regions.append(self.gpa, .{ .off = off, .cap = cap, .bytes_per = bytes_per, .per = per });
        self.size += std.mem.alignForward(u64, @max(cap, 1), self.gran);
        return off;
    }

    pub fn reserve(self: *Range) !void {
        try self.d.check(self.d.api.cuMemAddressReserve(&self.base, self.size, self.gran, 0, 0), "cuMemAddressReserve");
    }

    /// The bytes ``ensure(positions)`` would map (a growth both ranks agree on before a forward needs it).
    pub fn wants(self: *const Range, positions: i64) u64 {
        const want_pos = std.mem.alignForward(u64, @intCast(@max(positions, 1)), @intCast(step_positions));
        var bytes: u64 = 0;
        for (self.regions.items) |r| {
            const units = std.math.divCeil(u64, want_pos, @intCast(r.per)) catch unreachable;
            const need = @min(r.cap, std.mem.alignForward(u64, units * r.bytes_per, self.gran));
            if (need > r.mapped) bytes += std.mem.alignForward(u64, need - r.mapped, self.gran);
        }
        return bytes;
    }

    /// Every region backed through ``positions`` (in steps of step_positions), new memory zeroed on ``stream``.
    pub fn ensure(self: *Range, positions: i64, stream: abi.Stream) !void {
        const want_pos = std.mem.alignForward(u64, @intCast(@max(positions, 1)), @intCast(step_positions));
        for (self.regions.items) |*r| {
            const units = std.math.divCeil(u64, want_pos, @intCast(r.per)) catch unreachable;
            const need = @min(r.cap, std.mem.alignForward(u64, units * r.bytes_per, self.gran));
            if (need <= r.mapped) continue;
            const grow = std.mem.alignForward(u64, need - r.mapped, self.gran);
            if (self.budget.used + grow > self.budget.cap) return error.OutOfDeviceMemory;
            const at = self.base + r.off + r.mapped;
            var h: abi.MemHandle = 0;
            const prop: abi.MemAllocationProp = .{};
            if (self.d.api.cuMemCreate(&h, grow, &prop, 0) != abi.success) return error.OutOfDeviceMemory;
            errdefer _ = self.d.api.cuMemRelease(h);
            try self.d.check(self.d.api.cuMemMap(at, grow, 0, h, 0), "cuMemMap");
            errdefer _ = self.d.api.cuMemUnmap(at, grow);
            const access: abi.MemAccessDesc = .{};
            try self.d.check(self.d.api.cuMemSetAccess(at, grow, &access, 1), "cuMemSetAccess");
            try self.d.check(self.d.api.cuMemsetD8Async(at, 0, grow, stream), "zero grown cache");
            try self.chunks.append(self.gpa, .{ .at = at, .size = grow, .handle = h });
            self.budget.used += grow;
            r.mapped += grow;
        }
    }

    /// Every chunk unmapped and released (the range stays reserved, for the next request).
    pub fn shrink(self: *Range) void {
        for (self.chunks.items) |c| {
            _ = self.d.api.cuMemUnmap(c.at, c.size);
            _ = self.d.api.cuMemRelease(c.handle);
            self.budget.used -= c.size;
        }
        self.chunks.clearRetainingCapacity();
        for (self.regions.items) |*r| r.mapped = 0;
    }

    /// ``shrink`` but each region's first chunk (its first step of positions, which every request maps) stays,
    /// zeroed on ``stream``: a reused sequence starts without mapping its caches again.
    pub fn trim(self: *Range, stream: abi.Stream) !void {
        var kept: usize = 0;
        for (self.regions.items) |*r| r.mapped = 0;
        for (self.chunks.items) |c| {
            const r = for (self.regions.items) |*x| {
                if (c.at == self.base + x.off) break x;
            } else null;
            if (r) |first| {
                try self.d.check(self.d.api.cuMemsetD8Async(c.at, 0, c.size, stream), "zero kept cache");
                first.mapped = c.size;
                self.chunks.items[kept] = c;
                kept += 1;
                continue;
            }
            _ = self.d.api.cuMemUnmap(c.at, c.size);
            _ = self.d.api.cuMemRelease(c.handle);
            self.budget.used -= c.size;
        }
        self.chunks.shrinkRetainingCapacity(kept);
    }

    pub fn deinit(self: *Range) void {
        self.shrink();
        if (self.base != 0) _ = self.d.api.cuMemAddressFree(self.base, self.size);
        self.regions.deinit(self.gpa);
        self.chunks.deinit(self.gpa);
    }
};
