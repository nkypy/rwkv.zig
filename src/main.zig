const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const rwkv = @import("rwkv");

// for simd
const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;

fn softmax(x: []f32) void {
    assert(x.len > 0);
    // max of x for numerical stability
    var max: f32 = x[0];
    for (x[1..]) |val| {
        if (val > max) {
            max = val;
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (x) |*val| {
        val.* = std.math.exp(val.* - max);
        sum += val.*;
    }
    // normalize
    for (x) |*val| {
        val.* /= sum;
    }
}

fn argmax(x: []f32) usize {
    assert(x.len > 0);
    var max: f32 = x[0];
    var maxi: usize = 0;
    for (1..x.len) |i| {
        if (x[i] > max) {
            max = x[i];
            maxi = i;
        }
    }
    return maxi;
}

fn sample(x: []f32) usize {
    assert(x.len > 0);
    const random = prng.random();
    const r = random.float(f32);

    var cdf: f32 = 0.0;
    for (x, 0..) |val, i| {
        cdf += val;
        if (r < cdf) {
            return i;
        }
    }
    return x.len - 1;
}

const IndexedF32 = struct {
    index: u32,
    value: f32,

    fn desc(_: void, a: IndexedF32, b: IndexedF32) bool {
        return a.value > b.value;
    }
};

/// Top-p (nucleus) sampling. Samples from the smallest set of tokens whose
/// cumulative probability mass exceeds the probability p.
fn sample_top_p(logits: []f32, p: f32, logits_index: []IndexedF32) usize {
    assert(logits.len > 0);
    assert(p > 0.0 and p <= 1.0);
    assert(logits.len == logits_index.len);

    // elements smaller than (1 - p) / (n - 1) cannot be part of the result
    // and can be filtered out directly
    const cutoff: f32 = (1 - p) / (@as(f32, @floatFromInt(logits.len)) - 1);
    var num_to_sort: usize = 0;
    for (0..logits.len) |i| {
        assert(i < std.math.maxInt(u32));
        if (logits[i] >= cutoff) {
            logits_index[num_to_sort].value = logits[i];
            logits_index[num_to_sort].index = @intCast(i);
            num_to_sort += 1;
        }
    }
    assert(num_to_sort > 0);

    // sort the remaining elements
    std.sort.pdq(IndexedF32, logits_index[0..num_to_sort], {}, IndexedF32.desc);

    // find the cutoff index
    var cumulative_prob: f32 = 0.0;
    var cutoff_index: usize = num_to_sort - 1; // default to last element
    for (0..num_to_sort) |i| {
        cumulative_prob += logits_index[i].value;
        if (cumulative_prob > p) {
            cutoff_index = i;
            break;
        }
    }

    // sample from the cutoff index
    const random = prng.random();
    const r = random.float(f32) * cumulative_prob;
    var cdf: f32 = 0.0;
    for (0..cutoff_index + 1) |i| {
        cdf += logits_index[i].value;
        if (r < cdf) {
            return logits_index[i].index;
        }
    }
    return logits_index[cutoff_index].index;
}

const usage_text: []const u8 =
    \\Usage:   rwkv [options]
    \\Example: rwkv -n 256 -i "Once upon a time"
    \\Options:
    \\ -h, --help                print this help message
    \\ -m, --model <path>        path to the model file, default to "model.bin"
    \\ -i, --input <string>      input text for the prompt, default "User: Hello\nAssistant: <think></think>"
    \\ -t, --temperature <float> temperature, default 1.0 (0.0, 1]
    \\ -p, --top-p <float>       p value in top-p (nucleus) sampling. default 0.9, 0 || 1 = off
    \\ -n, --seq-len <int>       number of steps to run for, default 256. 0 = max_seq_len
    \\ -s, --seed <int>          random seed, default to time
    \\ -z, --tokenizer <path>    path to the tokenizer to use, default to "tokenizer.bin"
    \\ -v, --verbose             print model info and tokens/s
    \\
;

var prng: std.Random.DefaultPrng = undefined;
var verbose: bool = false;
fn log(comptime format: []const u8, args: anytype) void {
    if (verbose) {
        std.debug.print(format, args);
    }
}

pub fn main(init: std.process.Init) !void {
    // This is appropriate for anything that lives as long as the process.
    const allocator: Allocator = init.arena.allocator();
    // In order to do I/O operations need an `Io` instance.
    const io = init.io;

    var model_path: []const u8 = "model.bin";
    var input: ?[]const u8 = null;
    var temperature: f32 = 1.0;
    var top_p: f32 = 0.9;
    var seq_len: usize = 0;
    var seed: u64 = @bitCast(Io.Clock.now(.real, io).toSeconds());
    var tokenizer_path: []const u8 = "tokenizer.bin";
    prng = std.Random.DefaultPrng.init(seed);

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    const args = try init.minimal.args.toSlice(allocator);

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try stdout.writeAll(usage_text);
            try stdout.flush();
            return std.process.cleanExit(io);
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for model\n", .{});
                std.process.exit(1);
            }
            model_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--input")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for input\n", .{});
                std.process.exit(1);
            }
            input = args[arg_i];
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temperature")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for temperature\n", .{});
                std.process.exit(1);
            }
            temperature = std.fmt.parseFloat(f32, args[arg_i]) catch |err| {
                std.debug.print("unable to parse --temperature argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--top-p")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for top-p\n", .{});
                std.process.exit(1);
            }
            top_p = std.fmt.parseFloat(f32, args[arg_i]) catch |err| {
                std.debug.print("unable to parse --top-p argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
            top_p = std.math.clamp(top_p, 0.0, 1.0);
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--seq-len")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for seq-len\n", .{});
                std.process.exit(1);
            }
            seq_len = std.fmt.parseInt(usize, args[arg_i], 10) catch |err| {
                std.debug.print("unable to parse --seq-len argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--seed")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for seed\n", .{});
                std.process.exit(1);
            }
            seed = std.fmt.parseInt(u64, args[arg_i], 10) catch |err| {
                std.debug.print("unable to parse --seed argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
            prng = std.Random.DefaultPrng.init(seed);
        } else if (std.mem.eql(u8, arg, "-z") or std.mem.eql(u8, arg, "--tokenizer")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for tokenizer\n", .{});
                std.process.exit(1);
            }
            tokenizer_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else {
            std.debug.print("error: unknown argument '{s}'\n", .{arg});
            try stdout.writeAll(usage_text);
            try stdout.flush();
            return std.process.cleanExit(io);
        }
    }

    log("model_path: \"{s}\"\n", .{model_path});
    if (input) |in| {
        log("input: \"{s}\"\n", .{in});
    } else {
        log("input: \"\"\n", .{});
    }
    log("temperature: {d}\n", .{temperature});
    log("top_p: {d}\n", .{top_p});
    log("seq_len: {d}\n", .{seq_len});
    log("seed: {d}\n", .{seed});
    log("tokenizer_path: \"{s}\"\n", .{tokenizer_path});
    log("\n", .{});

    log("loading model...\n", .{});
    var rwkv_model = try rwkv.Model.fromFile(io, model_path, allocator);
    defer rwkv_model.deinit(allocator);
    log("model header: {any}\n", .{rwkv_model.header});
    log("\n", .{});

    log("loading tokenizer...\n", .{});
    const tokenizer = try rwkv.Tokenizer.fromFile(io, tokenizer_path, allocator);
    defer tokenizer.deinit(allocator);

    var state = try rwkv.RunState.init(allocator, &rwkv_model);
    defer state.deinit(allocator);

    @memset(state.last_x, 0.0);
    @memset(state.wkv_state, 0.0);

    var prompt: ?[]u32 = null;
    var prompt_len: usize = 0;
    defer if (prompt) |p| allocator.free(p);
    if (input) |in| {
        const encoded_input = try tokenizer.encode(in, allocator);
        prompt_len = encoded_input.len;
        prompt = encoded_input;
    }

    var next: usize = undefined;
    const start_time = Io.Clock.now(.real, io);

    seq_len = if (seq_len == 0) 256 else seq_len;

    const vocab_size: usize = @intCast(rwkv_model.header.vocab_size);
    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);
    const logits_indexed = try allocator.alloc(IndexedF32, vocab_size);
    defer allocator.free(logits_indexed);

    var pos: usize = 0;

    if (prompt_len > 0) {
        log("prefilling {} tokens...\n", .{prompt_len});
        rwkv_model.forward(prompt.?, &state, logits);
        pos = prompt_len;

        if (!verbose) {
            if (prompt) |p| {
                for (p) |tok_id| {
                    try stdout.print("{s}", .{tokenizer.tokens[tok_id]});
                }
            }
            try stdout.flush();
        }
    } else {
        pos = 0;
        prompt_len = 1;
        prompt = try allocator.alloc(u32, 1);
        prompt.?[0] = 0;
        rwkv_model.forward(prompt.?, &state, logits);
        pos = 1;
    }

    while (pos < seq_len) : (pos += 1) {
        if (temperature == 0.0) {
            next = argmax(logits);
        } else {
            if (temperature != 1.0) {
                for (logits) |*val| val.* /= temperature;
            }
            softmax(logits);
            next = if (top_p == 0.0 or top_p == 1.0)
                sample(logits)
            else
                sample_top_p(logits, top_p, logits_indexed);
        }

        if (next == 0) {
            try stdout.print("\n---Meet EOS!---\n", .{});
            break;
        }

        const token_str = tokenizer.tokens[next];
        try stdout.print("{s}", .{token_str});
        try stdout.flush();

        rwkv_model.forward(&[_]u32{@intCast(next)}, &state, logits);
    }

    const end_time = Io.Clock.now(.real, io);
    const time = end_time.toNanoseconds() - start_time.toNanoseconds();
    const tokens_per_ms = @as(f64, @floatFromInt(pos - 1)) / @as(f64, @floatFromInt(@divFloor(time, std.time.ns_per_ms)));
    const tokens_per_sec: u32 = @intFromFloat(tokens_per_ms * 1000.0);

    log("\n\n{d} tokens per second\n", .{tokens_per_sec});
}

test "softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    softmax(&x);
    var sum: f32 = 0.0;
    for (0..x.len) |i| {
        sum += x[i];
    }
    // simple assertion instead of direct float comparison
    try std.testing.expect(@abs(sum - 1.0) < 0.0001);
}
