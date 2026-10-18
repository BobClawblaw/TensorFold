//! DeepSeek's DSML tool-call markup, V4's and V4.1's dialects: finding a block and reading its invokes.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tp = @import("tool_parse.zig");
const Allocator = std.mem.Allocator;
const strip = reply_text.pyStrip;

/// DeepSeek's DSML markup: V4's tags, and V4.1's (a "calls" block, and a space after the marker in every tag).
pub const Dsml = struct { open: []const u8, close: []const u8, invoke_open: []const u8, invoke_close: []const u8, param_open: []const u8, param_close: []const u8 };

fn dsmlDialect(comptime sep: []const u8, comptime block: []const u8) Dsml {
    const m = "\u{ff5c}DSML\u{ff5c}" ++ sep;
    return .{
        .open = "<" ++ m ++ block ++ ">",
        .close = "</" ++ m ++ block ++ ">",
        .invoke_open = "<" ++ m ++ "invoke name=\"",
        .invoke_close = "</" ++ m ++ "invoke>",
        .param_open = "<" ++ m ++ "parameter name=\"",
        .param_close = "</" ++ m ++ "parameter>",
    };
}

pub const dialects = [_]Dsml{ dsmlDialect("", "tool_calls"), dsmlDialect(" ", "calls") };

/// The first DSML block opener of any dialect at or after `pos`, and its dialect.
pub fn find(text: []const u8, pos: usize) ?struct { at: usize, d: *const Dsml } {
    var best: ?usize = null;
    var which: *const Dsml = &dialects[0];
    for (&dialects) |*d| if (tp.find(text, d.open, pos)) |at| if (best == null or at < best.?) {
        best = at;
        which = d;
    };
    return if (best) |at| .{ .at = at, .d = which } else null;
}

/// Every invoke of a DSML block, or Invalid when anything in it is not a well-formed invoke.
pub fn calls(a: Allocator, block: []const u8, d: *const Dsml) tp.Error![]tp.Call {
    var out: std.ArrayList(tp.Call) = .empty;
    var at: usize = 0;
    var pos: usize = 0;
    while (tp.find(block, d.invoke_open, pos)) |s| {
        pos = s + 1;
        const name_start = s + d.invoke_open.len;
        const q = std.mem.indexOfScalarPos(u8, block, name_start, '"') orelse continue;
        if (!std.mem.startsWith(u8, block[q..], "\">")) continue;
        const e = tp.find(block, d.invoke_close, q + 2) orelse continue;
        if (!tp.allSpace(block[at..s])) return error.Invalid;
        at = e + d.invoke_close.len;
        pos = at;
        const body = block[q + 2 .. e];
        const arguments = try json.newObject(a);
        var p: usize = 0;
        var kept: usize = 0;
        var residue: std.ArrayList(u8) = .empty;
        while (tp.find(body, d.param_open, p)) |ps| {
            p = ps + 1;
            const ns = ps + d.param_open.len;
            const nq = std.mem.indexOfScalarPos(u8, body, ns, '"') orelse continue;
            const flag = body[nq..];
            const is_string = std.mem.startsWith(u8, flag, "\" string=\"true\">");
            if (!is_string and !std.mem.startsWith(u8, flag, "\" string=\"false\">")) continue;
            const v_start = nq + (if (is_string) "\" string=\"true\">".len else "\" string=\"false\">".len);
            const pe = tp.find(body, d.param_close, v_start) orelse continue;
            const value = body[v_start..pe];
            try residue.appendSlice(a, body[kept..ps]);
            kept = pe + d.param_close.len;
            p = kept;
            if (is_string) {
                try arguments.put(a, body[ns..nq], .{ .string = value });
            } else switch (try json.parseText(a, value)) {
                .ok => |v| try arguments.put(a, body[ns..nq], v),
                .err => return error.Invalid,
            }
        }
        try residue.appendSlice(a, body[kept..]);
        if (!tp.allSpace(residue.items)) return error.Invalid;
        try out.append(a, .{ .name = strip(block[name_start..q]), .arguments = .{ .object = arguments } });
    }
    if (out.items.len == 0 or !tp.allSpace(block[at..])) return error.Invalid;
    return out.items;
}
