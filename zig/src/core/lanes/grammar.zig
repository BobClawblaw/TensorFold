//! Structured output through libtfgrammar (xgrammar, zig/src/grammar/tf_grammar.cc), loaded when the first
//! structured request arrives: the server's compiler over the model's tokenizer, a request's compiled grammar (the
//! compiler caches it by its text, so the server's check and the engine's compile share it) and a reply's matcher.

const std = @import("std");

const Fns = struct {
    open: *const fn ([*]const u8, usize, c_int, [*]const i32, c_int, [*]u8, usize) callconv(.c) ?*anyopaque,
    close: *const fn (*anyopaque) callconv(.c) void,
    words: *const fn (*anyopaque) callconv(.c) c_int,
    compile: *const fn (*anyopaque, c_int, [*]const u8, usize, [*]u8, usize) callconv(.c) ?*anyopaque,
    free: *const fn (*anyopaque) callconv(.c) void,
    matcher: *const fn (*anyopaque) callconv(.c) ?*anyopaque,
    matcher_free: *const fn (*anyopaque) callconv(.c) void,
    accept: *const fn (*anyopaque, i32) callconv(.c) c_int,
    rollback: *const fn (*anyopaque, c_int) callconv(.c) c_int,
    terminated: *const fn (*anyopaque) callconv(.c) c_int,
    fill: *const fn (*anyopaque, [*]i32, c_int) callconv(.c) c_int,
};

var lib: ?Fns = null;
var lib_failed = false;

fn load() ?Fns {
    if (lib) |l| return l;
    if (lib_failed) return null;
    var dl = std.DynLib.open("libtfgrammar.so") catch {
        lib_failed = true;
        return null;
    };
    var f: Fns = undefined;
    const info = @typeInfo(Fns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, "tfg_" ++ name) orelse {
            lib_failed = true;
            return null;
        };
    }
    lib = f;
    return f;
}

pub const Kind = enum(u8) { json, json_schema, regex, choice, grammar };

/// One tokenizer's grammar compiler (thread-safe in xgrammar; callers still serialize compiles, as Python does).
pub const Compiler = struct {
    f: Fns,
    h: *anyopaque,
    words: usize,
    mutex: std.Io.Mutex = .init,

    /// ``tokenizer_json``: tokenizer.json's text; ``vocab``: the logits' width; ``stops``: the model's eos ids.
    pub fn open(tokenizer_json: []const u8, vocab: u32, stops: []const u32, err: *[512]u8) !Compiler {
        const f = load() orelse return error.NoGrammarLibrary;
        err[0] = 0;
        const h = f.open(tokenizer_json.ptr, tokenizer_json.len, @intCast(vocab), @ptrCast(stops.ptr), @intCast(stops.len), err, err.len) orelse return error.GrammarSetup;
        return .{ .f = f, .h = h, .words = @intCast(f.words(h)) };
    }

    pub fn close(c: *Compiler) void {
        c.f.close(c.h);
    }

    /// The compiled grammar, or error.Grammar with xgrammar's words in ``err``.
    pub fn compile(c: *Compiler, io: std.Io, kind: Kind, text: []const u8, err: *[512]u8) !Compiled {
        err[0] = 0;
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        const g = c.f.compile(c.h, @intFromEnum(kind), text.ptr, text.len, err, err.len) orelse return error.Grammar;
        return .{ .f = c.f, .h = g };
    }
};

pub const Compiled = struct {
    f: Fns,
    h: *anyopaque,

    pub fn free(g: Compiled) void {
        g.f.free(g.h);
    }

    pub fn matcher(g: Compiled) !Matcher {
        return .{ .f = g.f, .h = g.f.matcher(g.h) orelse return error.Grammar };
    }
};

/// A reply's place in its grammar.
pub const Matcher = struct {
    f: Fns,
    h: *anyopaque,

    pub fn free(m: Matcher) void {
        m.f.matcher_free(m.h);
    }

    /// Whether the grammar takes ``token`` next (it then has).
    pub fn accept(m: Matcher, token: u32) !bool {
        return switch (m.f.accept(m.h, @intCast(token))) {
            1 => true,
            0 => false,
            else => error.Grammar,
        };
    }

    pub fn rollback(m: Matcher, n: usize) !void {
        if (n > 0 and m.f.rollback(m.h, @intCast(n)) != 0) return error.Grammar;
    }

    /// The grammar has taken its stop token.
    pub fn terminated(m: Matcher) bool {
        return m.f.terminated(m.h) == 1;
    }

    /// The next token's allowed bits (token t: bit t % 32 of word t / 32).
    pub fn fill(m: Matcher, words: []u32) !void {
        if (m.f.fill(m.h, @ptrCast(words.ptr), @intCast(words.len)) != 0) return error.Grammar;
    }
};

/// A request's grammar for the engine: the server's compiler, what to compile, the token it starts after.
pub const Structure = struct {
    compiler: *Compiler,
    kind: Kind,
    text: []const u8,
    after: ?u32 = null,
};
