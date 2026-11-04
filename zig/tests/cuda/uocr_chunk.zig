//! tf-cuda-test uocr-chunk: a prompt prefilled in chunks of odd sizes gives the bits of the whole, then the same decode.
const std = @import("std");
const cuda = @import("cuda");
const uocr = @import("uocr_cuda");
const check = @import("check.zig");

const cfg = uocr.config;
const Decoder = uocr.decoder.Decoder;

/// Chunk sizes checked against the served ones (2048 rows at most): odd sizes cuBLASLt plans for apart, and 1.
const cuts = [_]usize{ 1021, 397, 37, 1 };
/// Decode steps after the prompt, fed the reference's own reply tokens.
const steps = 8;

const Case = struct { prompt_token_ids: []u32, token_ids: []u32, pages: [][]const u8 };

/// MODEL, the pages' folder and a reference capture (tools/zig/uocr_reference.py capture: its prompt and reply ids).
pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    if (args.len != 3) {
        std.debug.print("usage: tf-cuda-test uocr-chunk MODEL PAGES CAPTURE.json\n", .{});
        return error.TestFailed;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const json = try cwd.readFileAlloc(gpu.io, args[2], a, .limited(64 << 20));
    const case = try std.json.parseFromSliceLeaky(Case, a, json, .{ .ignore_unknown_fields = true });
    const n = case.prompt_token_ids.len;
    const cap = std.mem.alignForward(usize, n + steps, cfg.block);
    const m = try uocr.serve.Model.load(gpu.gpa, gpu.io, gpu.d, args[0], cap);
    defer m.deinit();
    const images = try a.alloc(@import("lanes").stream.Image, case.pages.len);
    var at: u32 = 0;
    for (case.pages, images) |name, *img| {
        const bytes = try cwd.readFileAlloc(gpu.io, try std.fs.path.join(a, &.{ args[1], name }), a, .limited(1 << 28));
        const p = try m.prepare(a, bytes, std.math.maxInt(u32));
        img.* = .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .at = at, .tokens = p.tokens, .hash = p.hash };
        at += p.tokens;
    }
    const rows = try m.encodeImages(a, images);
    const whole = try logitsAt(gpu, &m.decoder, case, rows, a);
    var failed: usize = 0;
    for (cuts) |cut| {
        var dec = try Decoder.init(gpu.gpa, gpu.d, m.s, &m.kernels, &m.gemms, m.weights, cap, cut);
        defer dec.deinit();
        const got = try logitsAt(gpu, &dec, case, rows, a);
        const same = std.mem.eql(u16, whole, got);
        if (!same) failed += 1;
        std.debug.print("{s} uocr chunk {d} rows: {d}-token prompt and {d} decode steps, logits {s} the served chunks' (2048 rows at most)\n", .{ if (same) "PASS" else "FAIL", cut, n, steps, if (same) "bit-identical to" else "differ from" });
    }
    if (failed > 0) return error.TestFailed;
}

/// The prompt through `dec`, then `steps` reply tokens: every forward's logits, end to end.
fn logitsAt(gpu: check.Gpu, dec: *Decoder, case: Case, rows: []const u16, a: std.mem.Allocator) ![]u16 {
    var p = try uocr.prompt.positions(gpu.gpa, try gpu.gpa.dupe(u32, case.prompt_token_ids));
    defer p.deinit(gpu.gpa);
    try dec.prefill(p, rows);
    const out = try a.alloc(u16, (steps + 1) * cfg.vocab);
    try dec.logits(out[0..cfg.vocab]);
    var step = uocr.prompt.Step.init(p);
    for (case.token_ids[0..steps], 1..) |id, i| {
        const c = step.next(id);
        try dec.step(id, c.tok, c.page, c.qpage);
        try dec.logits(out[i * cfg.vocab ..][0..cfg.vocab]);
    }
    return out;
}
