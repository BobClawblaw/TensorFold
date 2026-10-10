//! Flash Next port, step 1: one Triton launch captured from the Python engine (inputs and outputs of every memory
//! region its pointers reach, aliased pointers sharing a region), replayed from the cubin and compared byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

fn readFile(gpu: Gpu, dir: []const u8, name: []const u8, limit: usize) ![]u8 {
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(limit));
}

const Region = struct { buf: cuda.DeviceBuffer, pad: usize, bytes: usize, name: []const u8 };

pub fn tritonRegions(gpu: Gpu, dir: []const u8) !void {
    const gpa = gpu.gpa;
    const text = try readFile(gpu, dir, "manifest.json", 1 << 24);
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const meta_text = try readFile(gpu, dir, "kernel.json", 1 << 22);
    defer gpa.free(meta_text);
    const meta = try cuda.triton.parseMeta(gpa, meta_text);
    defer meta.deinit();
    const cubin_raw = try readFile(gpu, dir, "kernel.cubin", 1 << 28);
    defer gpa.free(cubin_raw);
    const cubin = try gpa.alignedAlloc(u8, .@"16", cubin_raw.len);
    defer gpa.free(cubin);
    @memcpy(cubin, cubin_raw);
    const name_z = try gpa.dupeSentinel(u8, meta.value.name, 0);
    defer gpa.free(name_z);
    var kernel = try cuda.triton.Kernel.load(gpu.d, gpu.ctx.device, cubin, meta.value, name_z);
    defer kernel.unload();

    var regions: std.StringHashMap(Region) = .init(gpa);
    defer {
        var it = regions.valueIterator();
        while (it.next()) |r| r.buf.free();
        regions.deinit();
    }
    var arr_it = root.get("arrays").?.object.iterator();
    while (arr_it.next()) |e| {
        const bytes: usize = @intCast(e.value_ptr.object.get("bytes").?.integer);
        const align256: usize = @intCast(e.value_ptr.object.get("align256").?.integer);
        var buf = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 512);
        errdefer buf.free();
        const pad = (align256 + 256 - @as(usize, @intCast(buf.ptr % 256))) % 256;
        const file = try std.fmt.allocPrint(gpa, "{s}.bin", .{e.key_ptr.*});
        defer gpa.free(file);
        const host = try readFile(gpu, dir, file, 1 << 31);
        defer gpa.free(host);
        try check.expect(host.len == bytes, "{s}: {d} bytes on disk, manifest says {d}", .{ file, host.len, bytes });
        try buf.upload(pad, host);
        try regions.put(e.key_ptr.*, .{ .buf = buf, .pad = pad, .bytes = bytes, .name = e.key_ptr.* });
    }

    var args: cuda.Args = .{};
    for (root.get("args").?.array.items) |item| {
        const o = item.object;
        const kind = o.get("kind").?.string;
        if (std.mem.eql(u8, kind, "ptr")) {
            const r = regions.get(o.get("array").?.string).?;
            const off: usize = @intCast(o.get("offset").?.integer);
            args.add(@as(u64, r.buf.ptr + r.pad + off));
        } else if (std.mem.eql(u8, kind, "i32")) {
            args.add(@as(i32, @intCast(o.get("value").?.integer)));
        } else if (std.mem.eql(u8, kind, "i64")) {
            args.add(@as(i64, o.get("value").?.integer));
        } else if (std.mem.eql(u8, kind, "f32")) {
            args.add(@as(f32, @bitCast(@as(u32, @intCast(o.get("bits").?.integer)))));
        } else return check.expect(false, "unknown argument kind {s}", .{kind});
    }
    const g = root.get("grid").?.array.items;
    const dims: cuda.Dim3 = .{ .x = @intCast(g[0].integer), .y = @intCast(g[1].integer), .z = @intCast(g[2].integer) };
    var divisible: std.ArrayList(u32) = .empty;
    defer divisible.deinit(gpa);
    for (root.get("divisible16").?.array.items) |i| try divisible.append(gpa, @intCast(i.integer));

    var scratch_g = try cuda.DeviceBuffer.alloc(gpu.d, kernel.globalScratchBytes(dims));
    defer scratch_g.free();
    const prof_bytes = @as(usize, dims.x) * dims.y * dims.z * meta.value.num_ctas * meta.value.profile_scratch_size;
    var scratch_p = try cuda.DeviceBuffer.alloc(gpu.d, prof_bytes);
    defer scratch_p.free();

    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    try kernel.launchOn(dims, stream, &args, .{ .global = scratch_g.ptr, .profile = scratch_p.ptr }, divisible.items);
    try stream.synchronize();

    var changed: usize = 0;
    var it = regions.valueIterator();
    while (it.next()) |r| {
        const want_file = try std.fmt.allocPrint(gpa, "expected_{s}.bin", .{r.name});
        defer gpa.free(want_file);
        const want = try readFile(gpu, dir, want_file, 1 << 31);
        defer gpa.free(want);
        const got = try gpa.alloc(u8, r.bytes);
        defer gpa.free(got);
        try r.buf.download(r.pad, got);
        try check.sameBytes(r.name, got, want);
        const before = try readFile(gpu, dir, try std.fmt.allocPrint(gpa, "{s}.bin", .{r.name}), 1 << 31);
        defer gpa.free(before);
        if (!std.mem.eql(u8, before, want)) changed += 1;
    }
    check.pass("BITEXACT {s}: {d} regions equal ({d} written), grid ({d},{d},{d})", .{ meta.value.name, regions.count(), changed, dims.x, dims.y, dims.z });
}
