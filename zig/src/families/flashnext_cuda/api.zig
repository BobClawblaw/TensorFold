//! Flash Next (qwen4_exp) on CUDA, two ranks: the pieces the weight loader, the forward and the native host share.
//!
//! Weights are a name -> device tensor store whose names are the Python engine's prepared-weight names (its pack
//! index: "layers.3.gdn.proj.parts.0.weight", "head.parts.0.scale", ...), so the store the checkpoint loader builds
//! and the one read from a Python pack (development) are interchangeable, and each tensor can be checked byte for byte.
//! Kernels: the Triton set (aot.json + cubins, captured from the Python engine) and the extension fatbins embedded by
//! the build (zig/kernels/cuda/flashnext, SASS-equal to the Python extensions).

const std = @import("std");
const cuda = @import("cuda");

pub const DType = enum { bf16, f16, f32, i32, i64, u8, i8, i16, f8e4m3, bool_ };

pub fn dtypeBytes(t: DType) usize {
    return switch (t) {
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        .i64 => 8,
        .u8, .i8, .f8e4m3, .bool_ => 1,
    };
}

pub fn parseDType(s: []const u8) ?DType {
    const map = std.StaticStringMap(DType).initComptime(.{ .{ "bfloat16", .bf16 }, .{ "float16", .f16 },
        .{ "float32", .f32 }, .{ "int32", .i32 }, .{ "int64", .i64 }, .{ "uint8", .u8 }, .{ "int8", .i8 },
        .{ "int16", .i16 }, .{ "float8_e4m3fn", .f8e4m3 }, .{ "bool", .bool_ } });
    return map.get(s);
}

/// A device tensor: address, element type, shape (row-major, contiguous unless ``stride`` says otherwise).
pub const Tensor = struct {
    ptr: u64,
    dtype: DType,
    shape: [5]i64 = .{ 1, 1, 1, 1, 1 },
    nd: u8 = 0,

    pub fn numel(t: Tensor) usize {
        var n: usize = 1;
        for (t.shape[0..t.nd]) |d| n *= @intCast(d);
        return n;
    }
    pub fn bytes(t: Tensor) usize {
        return t.numel() * dtypeBytes(t.dtype);
    }
    pub fn dim(t: Tensor, i: usize) i64 {
        return t.shape[i];
    }
};

/// Prepared weights by name, all on the device. ``deinit`` frees what ``owned`` holds.
pub const Store = struct {
    gpa: std.mem.Allocator,
    map: std.StringHashMap(Tensor),
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Host-side facts the loaders record (vocabulary offset, draft ids, n-gram table geometry), by name.
    ints: std.StringHashMap(i64),
    /// Host data by name (owned): "ngram.json" (the n-gram table's geometry, shard files and offsets, lookup
    /// table; the format tests/cuda/fn_ngram.zig opens), "draft_ids" (this rank's draft vocabulary, i32 LE),
    /// "ngram_root" (optional: the folder shard paths under /cache/tf/ resolve in).
    host: std.StringHashMap([]u8),

    pub fn init(gpa: std.mem.Allocator) Store {
        return .{ .gpa = gpa, .map = .init(gpa), .ints = .init(gpa), .host = .init(gpa) };
    }

    pub fn deinit(self: *Store) void {
        for (self.owned.items) |*b| b.free();
        self.owned.deinit(self.gpa);
        var it = self.map.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.map.deinit();
        var it2 = self.ints.keyIterator();
        while (it2.next()) |k| self.gpa.free(k.*);
        self.ints.deinit();
        var it3 = self.host.iterator();
        while (it3.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            self.gpa.free(e.value_ptr.*);
        }
        self.host.deinit();
    }

    pub fn put(self: *Store, name: []const u8, t: Tensor) !void {
        try self.map.put(try self.gpa.dupe(u8, name), t);
    }

    pub fn get(self: *const Store, name: []const u8) !Tensor {
        return self.map.get(name) orelse {
            std.log.err("flash next: no weight named {s}", .{name});
            return error.MissingWeight;
        };
    }

    pub fn getf(self: *const Store, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [256]u8 = undefined;
        return self.get(try std.fmt.bufPrint(&buf, fmt, args));
    }
};

/// The device, its stream and the two ranks' communicator.
pub const Ctx = struct {
    d: *const cuda.Driver,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    nccl: *cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: u8,
    world: u8 = 2,
};

/// The kernels: Triton variants (picked as Triton picks them) and the extension modules.
pub const Kernels = struct {
    triton: cuda.aot.Set,
    ext: [ext_names.len]cuda.Module,

    pub const ext_names = [_][]const u8{ "fn_experts", "fn_experts_prefill", "fn_gdn_prefill", "fn_qmm", "fn_qmm_prefill", "fn_gdn", "fn_gdn_io" };

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: cuda.abi.Device, triton_dir: []const u8) !Kernels {
        var k: Kernels = .{ .triton = try cuda.aot.Set.load(gpa, io, d, device, triton_dir), .ext = undefined };
        const images = [_][]const u8{ cuda.kernels.fn_experts, cuda.kernels.fn_experts_prefill, cuda.kernels.fn_gdn_prefill,
            cuda.kernels.fn_qmm, cuda.kernels.fn_qmm_prefill, cuda.kernels.fn_gdn, cuda.kernels.fn_gdn_io };
        for (images, 0..) |img, i| k.ext[i] = try cuda.Module.load(d, img);
        return k;
    }

    pub fn deinit(self: *Kernels) void {
        for (&self.ext) |*m| m.unload();
        self.triton.deinit();
    }
};
