//! Flash Next's n-gram rows on the host (ngram.py and host_table.FP8Table): hashed row ids for tokens after a
//! history (EOS resets the n-grams, int64 products wrap, floor modulo), and their table rows as bf16 bits. The
//! tables are shared; each stream keeps its own ``History``.

const std = @import("std");


pub const NGram = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    n: usize,
    per: usize,
    heads: usize,
    eos: i64,
    width: usize,
    sizes: []i64,
    offsets: []i64,
    mult: []i64,
    starts: []i64,
    lut: [256]u16,
    files: []std.Io.File,
    offs: []u64,
    tables: [][]const u8 = &.{},        // each shard's rows, memory-mapped (``map``)
    history: [8]i64 = undefined,
    hlen: usize = 0,
    vocab_offset: i64 = 0,

    pub fn ints(gpa: std.mem.Allocator, v: std.json.Value) ![]i64 {
        const a = v.array.items;
        const out = try gpa.alloc(i64, a.len);
        for (a, 0..) |x, i| out[i] = x.integer;
        return out;
    }

    /// ``text``: the geometry (ngram.json); ``root``: the folder shard paths under /cache/tf/ resolve in (others as given).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, text: []const u8, root: []const u8) !NGram {
        const p = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        const o = p.value.object;
        var g: NGram = .{ .gpa = gpa, .io = io, .n = @intCast(o.get("n").?.integer), .per = @intCast(o.get("per_ngram").?.integer),
            .heads = @intCast(o.get("heads").?.integer), .eos = o.get("eos").?.integer, .width = @intCast(o.get("width").?.integer),
            .sizes = try ints(gpa, o.get("head_sizes").?), .offsets = try ints(gpa, o.get("head_offsets").?),
            .mult = try ints(gpa, o.get("multipliers").?), .starts = try ints(gpa, o.get("starts").?), .lut = undefined,
            .files = undefined, .offs = undefined, .vocab_offset = o.get("vocab_offset").?.integer };
        for (o.get("lut").?.array.items, 0..) |x, i| g.lut[i] = @intCast(x.integer);
        const shards = o.get("shards").?.array.items;
        g.files = try gpa.alloc(std.Io.File, shards.len);
        g.offs = try gpa.alloc(u64, shards.len);
        for (shards, 0..) |s, i| {
            const f = s.object.get("file").?.string;           // a container path under /cache/tf
            const host = if (std.mem.startsWith(u8, f, "/cache/tf/")) try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, f["/cache/tf/".len..] }) else f;
            g.files[i] = try std.Io.Dir.cwd().openFile(io, host, .{});
            g.offs[i] = @intCast(s.object.get("offset").?.integer);
        }
        for (o.get("initial_history").?.array.items, 0..) |x, i| g.history[i] = x.integer;
        g.hlen = o.get("initial_history").?.array.items.len;
        return g;
    }

    /// Maps every shard file once (read-only, shared with the page cache) and, with ``prefetch``, touches every page
    /// of the rows from eight threads so no decode step faults on a table page.
    pub fn map(self: *NGram, prefetch: bool) !void {
        const shards = self.files.len;
        self.tables = try self.gpa.alloc([]const u8, shards);
        var bases = std.AutoHashMap(std.posix.fd_t, []align(std.heap.page_size_min) u8).init(self.gpa);
        defer bases.deinit();
        for (0..shards) |i| {
            const fd = self.files[i].handle;
            const whole = bases.get(fd) orelse blk: {
                const len = try self.files[i].length(self.io);
                const m = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
                try bases.put(fd, m);
                break :blk m;
            };
            const rows: usize = @intCast(self.starts[i + 1] - self.starts[i]);
            self.tables[i] = whole[self.offs[i]..][0 .. rows * self.width];
        }
        if (!prefetch) return;
        const Touch = struct {
            fn run(tables: []const []const u8, first: usize, step: usize, sum: *u64) void {
                var acc: u64 = 0;
                var i = first;
                while (i < tables.len) : (i += step) {
                    var at: usize = 0;
                    while (at < tables[i].len) : (at += 4096) acc +%= tables[i][at];
                }
                sum.* = acc;
            }
        };
        var threads: [8]std.Thread = undefined;
        var sums: [8]u64 = @splat(0);
        for (0..8) |t| threads[t] = try std.Thread.spawn(.{}, Touch.run, .{ self.tables, t, 8, &sums[t] });
        for (threads) |t| t.join();
    }

    /// A fresh stream's history (the n-gram start state).
    pub fn start(self: *const NGram) History {
        var h: History = .{ .len = self.hlen };
        @memcpy(h.h[0..self.hlen], self.history[0..self.hlen]);
        return h;
    }

    /// Row ids [tokens.len * heads] for ``tokens`` after ``hist``.
    pub fn ids(self: *const NGram, hist: *const History, tokens: []const i64, out: []i64) !void {
        const gpa = self.gpa;
        const w = hist.len + tokens.len;
        const seq = try gpa.alloc(i64, w);
        defer gpa.free(seq);
        @memcpy(seq[0..hist.len], hist.h[0..hist.len]);
        @memcpy(seq[hist.len..], tokens);
        const seg = try gpa.alloc(i64, w);
        defer gpa.free(seg);
        var last_eos: i64 = -1;                     // the latest EOS strictly before p
        for (0..w) |p| {
            seg[p] = @as(i64, @intCast(p)) - (last_eos + 1);
            if (seq[p] == self.eos) last_eos = @intCast(p);
        }
        const first_row = hist.len;
        for (first_row..w) |p| {
            const r = p - first_row;
            var col: usize = 0;
            var ngram: usize = 2;
            while (ngram <= self.n) : (ngram += 1) {
                var mixed: i64 = self.shifted(seq, seg, p, 0) *% self.mult[0];
                var q: usize = 1;
                while (q < ngram) : (q += 1) mixed ^= self.shifted(seq, seg, p, q) *% self.mult[q];
                const first = (ngram - 2) * self.per;
                for (0..self.per) |h| {
                    out[r * self.heads + col] = @mod(mixed, self.sizes[first + h]) + self.offsets[first + h];
                    col += 1;
                }
            }
        }
    }

    fn shifted(self: *const NGram, seq: []const i64, seg: []const i64, p: usize, shift: usize) i64 {
        if (p < shift or seg[p] < @as(i64, @intCast(shift))) return self.eos;
        return seq[p - shift];
    }

    /// The rows of ``row_ids`` as bf16 bits, ``width`` values each.
    pub fn gather(self: *const NGram, row_ids: []const i64, out: []u16) !void {
        var raw: [4096]u8 = undefined;
        for (row_ids, 0..) |id, i| {
            var s: usize = 0;
            while (s + 1 < self.starts.len and self.starts[s + 1] <= id) s += 1;
            const local: u64 = @intCast(id - self.starts[s]);
            const row: []const u8 = if (self.tables.len > 0) self.tables[s][local * self.width ..][0..self.width] else blk: {
                const n = try self.files[s].readPositionalAll(self.io, raw[0..self.width], self.offs[s] + local * self.width);
                if (n != self.width) return error.ShortRead;
                break :blk raw[0..self.width];
            };
            for (0..self.width) |j| out[i * self.width + j] = self.lut[row[j]];
        }
    }


};

/// One stream's n-gram history: the last n - 1 committed tokens.
pub const History = struct {
    h: [8]i64 = undefined,
    len: usize = 0,

    /// After a commit of ``tokens``.
    pub fn advance(self: *History, tokens: []const i64) void {
        for (tokens) |t| {
            var i: usize = 0;
            while (i + 1 < self.len) : (i += 1) self.h[i] = self.h[i + 1];
            if (self.len > 0) self.h[self.len - 1] = t;
        }
    }
};
