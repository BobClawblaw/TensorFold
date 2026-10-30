//! The Metal engines that are not token families: Drama 3's speech (fish_omni) and MOSS-Image's pictures.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub fn handles(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "fish_omni") or std.mem.eql(u8, model_type, "moss_image");
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (std.mem.eql(u8, o.model_type, "fish_omni")) return @import("drama_open.zig").openMetal(a, gpa, io, o, problem);
    return @import("moss_image_host.zig").openServe(a, gpa, io, o, problem);
}
