//! `tf-minimax-cuda <dir> <ids.json> <layers> <out> [nvfp4]`: each layer's rows and the logits, for the reference.

const std = @import("std");
const cuda = @import("cuda");
const mm3 = @import("minimax_m3_cuda");

const usage = "usage: tf-minimax-cuda <model dir> <ids.json> <layers> <out dir> [nvfp4 layers: ranges, all or none]\n";

const Dump = struct {
    d: *const cuda.Driver,
    io: std.Io,
    stream: cuda.Stream,
    dir: std.Io.Dir,
    gpa: std.mem.Allocator,
    hidden: usize,
    chunk: usize = 0,

    fn call(ptr: *anyopaque, layer: usize, resid: u64, rows: usize) anyerror!void {
        const self: *Dump = @ptrCast(@alignCast(ptr));
        try self.stream.synchronize();
        const bytes = try self.gpa.alloc(u8, rows * self.hidden * 4);
        defer self.gpa.free(bytes);
        try self.d.check(self.d.api.cuMemcpyDtoH_v2(bytes.ptr, resid, bytes.len), "cuMemcpyDtoH");
        var name: [32]u8 = undefined;
        try self.dir.writeFile(self.io, .{ .sub_path = try std.fmt.bufPrint(&name, "r{d}.c{d}.f32", .{ layer, self.chunk }), .data = bytes });
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5 and args.len != 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const layers = try std.fmt.parseInt(u32, args[3], 10);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 24));
    var doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    const list = switch (doc.value) {
        .array => |arr| arr,
        .object => |o| (o.get("ids") orelse return error.BadIds).array,
        else => return error.BadIds,
    };
    const ids = try a.alloc(i32, list.items.len);
    for (list.items, ids) |v, *id| id.* = @intCast(v.integer);
    if (ids.len == 0) return error.BadIds;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var m = try mm3.forward.Modules.load(&driver, &ctx);
    defer m.deinit();
    var c = try mm3.config.Config.read(gpa, io, args[1]);
    c.layers = layers;
    const share = try mm3.shard.Share.of(&c, 0, 1);
    const mix = try mm3.weights.Mix.read(gpa, io, args[1], &c, if (args.len == 6) args[5] else null);
    const which = try a.alloc(u32, layers);
    for (which, 0..) |*l, i| l.* = @intCast(i);
    var t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var w = try mm3.weights.load(gpa, io, m.tools(&driver, stream), args[1], c, share, mix, which);
    defer w.deinit();
    std.debug.print("loaded {d} layers, {d:.2} GiB, {d} NVFP4 of them, in {d:.1} s\n", .{ layers, @as(f64, @floatFromInt(w.bytes)) / (1 << 30), mix.count(&c), @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) * 1e-9 });
    const context: usize = if (std.c.getenv("TF_MINIMAX_PROF_POS") != null) 32768 else 4096;
    // TF_MINIMAX_PROF_DECODE=1: a decode window's buffers (ids of at most 16 rows), as a verify runs them
    const decode = std.c.getenv("TF_MINIMAX_PROF_DECODE") != null;
    var b = try mm3.state.Buffers.init(&driver, &w, if (decode) mm3.state.window_rows else @min(ids.len, mm3.state.prompt_rows), !decode, context, 1, &m.lane, &.{}, false);
    defer b.deinit();
    var st = try mm3.state.State.init(gpa, &driver, &c, share.kv.len(), context);
    defer st.deinit();
    try std.Io.Dir.cwd().createDirPath(io, args[4]);
    var out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);
    var dump: Dump = .{ .d = &driver, .io = io, .stream = stream, .dir = out, .gpa = gpa, .hidden = c.hidden };
    const profiling = std.c.getenv("TF_MINIMAX_PROF") != null;
    var prof = try mm3.forward.Prof.init(&driver);
    defer prof.deinit();
    // TF_MINIMAX_PROF_POS: positions already in the cache (stale), so the chunk's attention sees a long context
    if (std.c.getenv("TF_MINIMAX_PROF_POS")) |v| st.pos = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    const f: mm3.forward.Forward = .{ .d = &driver, .m = &m, .w = &w, .stream = stream, .b = &b, .watch = if (profiling) null else .{ .ctx = &dump, .call = Dump.call }, .prof = if (profiling) &prof else null };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var start: usize = 0;
    while (start < ids.len) : (start += mm3.state.prompt_rows) {
        const n = @min(ids.len - start, mm3.state.prompt_rows);
        dump.chunk = start / mm3.state.prompt_rows;
        const seg = [_]mm3.forward.Seg{.{ .st = &st, .row0 = 0, .rows = n }};
        try f.run(ids[start..][0..n], &seg);
        try mm3.forward.Forward.commit(&seg, &.{n});
    }
    try stream.synchronize();
    if (profiling) {
        prof.print();
        if (decode) { // steady state: ten more windows at the next positions, timed whole
            prof = try mm3.forward.Prof.init(&driver);
            const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
            const g: mm3.forward.Forward = .{ .d = &driver, .m = &m, .w = &w, .stream = stream, .b = &b };
            for (0..10) |_| {
                const seg = [_]mm3.forward.Seg{.{ .st = &st, .row0 = 0, .rows = ids.len }};
                try g.run(ids, &seg);
                try mm3.forward.Forward.commit(&seg, &.{ids.len});
            }
            try stream.synchronize();
            std.debug.print("window of {d} rows: {d:.2} ms each (no timers)\n", .{ ids.len, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t1)) * 1e-7 });
        }
    }
    std.debug.print("forward of {d} rows in {d:.2} s\n", .{ ids.len, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) * 1e-9 });
    const logits = try gpa.alloc(u8, share.vocab.len() * 2);
    defer gpa.free(logits);
    try driver.check(driver.api.cuMemcpyDtoH_v2(logits.ptr, b.logits, logits.len), "cuMemcpyDtoH");
    try out.writeFile(io, .{ .sub_path = "logits.bf16", .data = logits });
    var best: f32 = -std.math.inf(f32);
    var bid: usize = 0;
    for (0..share.vocab.len()) |i| {
        const v: f32 = @bitCast(@as(u32, std.mem.readInt(u16, logits[2 * i ..][0..2], .little)) << 16);
        if (v > best) {
            best = v;
            bid = i;
        }
    }
    std.debug.print("greedy {d} ({d:.3})\n", .{ bid, best });
    return 0;
}
