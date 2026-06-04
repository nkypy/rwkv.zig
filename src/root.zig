//! By convention, root.zig is the root source file when making a package.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const MAGIC_NUMBER: u64 = 0x00632E37766B7772; // rwkv7.c\0

const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const simd_align: comptime_int = @alignOf(@Vector(DEFAULT_VECTOR_WIDTH, f32));
const simd_alignment = std.mem.Alignment.of(@Vector(DEFAULT_VECTOR_WIDTH, f32));

pub const Tokenizer = struct {
    const Self = @This();

    tokens: [][]u8,
    scores: []f32,
    max_token_len: u32,

    pub fn fromFile(io: Io, path: []const u8, allocator: Allocator) !Self {
        var token_file = try Io.Dir.cwd().openFile(io, path, .{});
        defer token_file.close(io);
        var buffer: [4096]u8 = undefined;
        var file_reader = token_file.reader(io, &buffer);
        const tokenizer = try Tokenizer.init(&file_reader, allocator);

        return tokenizer;
    }

    fn init(reader: *Io.File.Reader, allocator: Allocator) !Self {
        var tokenizer: Self = undefined;

        var max_token_len: [4]u8 = undefined;
        try reader.interface.readSliceAll(&max_token_len);

        const vocab_size = std.mem.readInt(u32, &max_token_len, .little);

        tokenizer.tokens = try allocator.alloc([]u8, vocab_size);
        tokenizer.scores = try allocator.alloc(f32, vocab_size);
        tokenizer.max_token_len = vocab_size;

        for (0..vocab_size) |i| {
            var score_bytes: [4]u8 = undefined;
            try reader.interface.readSliceAll(&score_bytes);

            const score_u32 = std.mem.readInt(u32, &score_bytes, .little);
            tokenizer.scores[i] = @bitCast(score_u32);

            var token_len_bytes: [4]u8 = undefined;
            try reader.interface.readSliceAll(&token_len_bytes);

            const token_len = std.mem.readInt(u32, &token_len_bytes, .little);
            if (token_len == 0) {
                tokenizer.tokens[i] = &[_]u8{};
                continue;
            }
            tokenizer.tokens[i] = try allocator.alloc(u8, token_len);
            try reader.interface.readSliceAll(tokenizer.tokens[i]);
        }

        return tokenizer;
    }

    pub fn deinit(self: *const Self, allocator: Allocator) void {
        for (self.tokens) |token| {
            allocator.free(token);
        }
        allocator.free(self.tokens);
        allocator.free(self.scores);
    }

    pub fn encode(self: *const Self, text: []const u8, allocator: Allocator) ![]u32 {
        var tokens = try allocator.alloc(u32, 0);
        var i: usize = 0;
        while (i < text.len) {
            var best_token: u32 = 0;
            var best_score: f32 = -std.math.floatMax(f32);
            var best_token_len: u32 = 0;
            for (0..self.max_token_len) |j| {
                const token = self.tokens[j];
                const token_len = token.len;
                if (i + token_len > text.len) {
                    continue;
                }
                if (std.mem.eql(u8, token, text[i .. i + token_len])) {
                    if (self.scores[j] > best_score) {
                        best_score = self.scores[j];
                        best_token = @intCast(j);
                        best_token_len = @intCast(token_len);
                    }
                }
            }
            if (best_token_len == 0) {
                return error.TokenNotFound;
            }
            tokens = try allocator.realloc(tokens, tokens.len + 1);
            tokens[tokens.len - 1] = best_token;
            i += best_token_len;
        }
        return tokens;
    }
};

pub const RunState = struct {
    const Self = @This();

    last_x: []align(simd_align) f32, // (n_layer, 2, n_embd)
    wkv_state: []align(simd_align) f32, // (n_layer, n_head, head_size, head_size)
    scratch: []align(simd_align) f32, // intermediate buffer

    pub fn init(allocator: Allocator, model: *const Model) !Self {
        const n_layer: usize = @intCast(model.header.n_layer);
        const n_embd: usize = @intCast(model.header.n_embd);
        const head_size: usize = @intCast(model.header.head_size);

        return Self{
            .last_x = try allocator.alignedAlloc(f32, simd_alignment, n_layer * 2 * n_embd),
            .wkv_state = try allocator.alignedAlloc(f32, simd_alignment, n_layer * @divTrunc(n_embd, head_size) * head_size * head_size),
            .scratch = try allocator.alignedAlloc(f32, simd_alignment, 32 * n_embd),
        };
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.last_x);
        allocator.free(self.wkv_state);
        allocator.free(self.scratch);
        self.* = undefined;
    }
};

/// Matches the C rwkv7.c header layout exactly.
const Header = extern struct {
    magic_number: u64,
    quant: i32,
    head_size: i32,
    n_embd: i32,
    n_layer: i32,
    vocab_size: i32,
    w_lora_r: i32,
    a_lora_r: i32,
    g_lora_r: i32,
    v_lora_r: i32,
    de: i32,
    dea: i32,
    s_lora_r: i32,
};

/// Per-block weights, matching C's block_weights layout.
pub const BlockWeights = struct {
    ln1_weight: [*]f32,
    ln1_bias: [*]f32,
    ln2_weight: [*]f32,
    ln2_bias: [*]f32,
    att_x_r: [*]f32,
    att_x_w: [*]f32,
    att_x_k: [*]f32,
    att_x_v: [*]f32,
    att_x_a: [*]f32,
    att_x_g: [*]f32,
    att_w0: [*]f32,
    att_r_k: [*]f32,
    att_w1_T: [*]f32,
    att_w2_T: [*]f32,
    att_a1_T: [*]f32,
    att_a2_T: [*]f32,
    att_a0: [*]f32,
    att_g1_T: [*]f32,
    att_g2_T: [*]f32,
    att_v2_T: [*]f32, // only i > 0
    att_v1_T: [*]f32, // only i > 0
    att_v0: [*]f32, // only i > 0
    att_k_k: [*]f32,
    att_k_a: [*]f32,
    att_receptance_weight: [*]f32,
    att_key_weight: [*]f32,
    att_value_weight: [*]f32,
    att_output_weight: [*]f32,
    att_ln_x_weight: [*]f32,
    att_ln_x_bias: [*]f32,
    ffn_x_k: [*]f32,
    ffn_key_weight: [*]f32,
    ffn_value_weight: [*]f32,
};

/// RWKV7 model
pub const Model = struct {
    const Self = @This();

    header: Header,
    emb_weight: [*]f32,
    blocks: []BlockWeights,
    ln_out_weight: [*]f32,
    ln_out_bias: [*]f32,
    head_weight: [*]f32,
    model_data: []align(std.heap.page_size_min) u8,

    pub fn fromFile(io: Io, path: []const u8, allocator: Allocator) !Self {
        var model_file = try Io.Dir.cwd().openFile(io, path, .{});
        defer model_file.close(io);

        var buffer: [4096]u8 = undefined;
        var file_reader = model_file.reader(io, &buffer);

        const header_bytes = try allocator.alloc(u8, @sizeOf(Header));
        defer allocator.free(header_bytes);
        try file_reader.interface.readSliceAll(header_bytes);
        const header: Header = std.mem.bytesToValue(Header, header_bytes);

        assert(header.magic_number == MAGIC_NUMBER);
        assert(header.quant == 0);

        const model_data: []align(std.heap.page_size_min) u8 = blk: {
            const size = (try model_file.stat(io)).size;
            const weights_size: usize = size - @sizeOf(Header);
            const alignment = comptime std.mem.Alignment.fromByteUnits(std.heap.page_size_min);
            const weights_bytes = try allocator.alignedAlloc(u8, alignment, weights_size);
            try file_reader.interface.readSliceAll(weights_bytes);
            break :blk weights_bytes;
        };

        const model = try Model.init(header, model_data, allocator);
        return model;
    }

    fn init(header: Header, data: []align(std.heap.page_size_min) u8, allocator: Allocator) !Self {
        var model: Self = undefined;

        model.header = header;
        model.model_data = data;
        const head_size: usize = @intCast(header.head_size);
        const vocab_size: usize = @intCast(header.vocab_size);
        const n_embd: usize = @intCast(header.n_embd);
        const n_layer: usize = @intCast(header.n_layer);
        const w_lora_r: usize = @intCast(header.w_lora_r);
        const a_lora_r: usize = @intCast(header.a_lora_r);
        const g_lora_r: usize = @intCast(header.g_lora_r);
        const v_lora_r: usize = @intCast(header.v_lora_r);
        const n_head: usize = n_embd / head_size;

        model.blocks = try allocator.alloc(BlockWeights, n_layer);

        var ptr: [*]f32 = @ptrCast(@alignCast(data));
        model.emb_weight = ptr;
        ptr += vocab_size * n_embd;
        // blocks_0 ln0 weight/bias (merged into emb_weight below)
        const ln0_weight = ptr;
        ptr += n_embd;
        const ln0_bias = ptr;
        ptr += n_embd;

        for (0..n_layer) |i| {
            const b = &model.blocks[i];
            b.ln1_weight = ptr;
            ptr += n_embd;
            b.ln1_bias = ptr;
            ptr += n_embd;
            b.ln2_weight = ptr;
            ptr += n_embd;
            b.ln2_bias = ptr;
            ptr += n_embd;
            b.att_x_r = ptr;
            ptr += n_embd;
            b.att_x_w = ptr;
            ptr += n_embd;
            b.att_x_k = ptr;
            ptr += n_embd;
            b.att_x_v = ptr;
            ptr += n_embd;
            b.att_x_a = ptr;
            ptr += n_embd;
            b.att_x_g = ptr;
            ptr += n_embd;
            b.att_w0 = ptr;
            ptr += n_embd;
            b.att_r_k = ptr;
            ptr += n_head * head_size;
            b.att_w1_T = ptr;
            ptr += n_embd * w_lora_r;
            b.att_w2_T = ptr;
            ptr += w_lora_r * n_embd;
            b.att_a1_T = ptr;
            ptr += n_embd * a_lora_r;
            b.att_a2_T = ptr;
            ptr += a_lora_r * n_embd;
            b.att_a0 = ptr;
            ptr += n_embd;
            b.att_g1_T = ptr;
            ptr += n_embd * g_lora_r;
            b.att_g2_T = ptr;
            ptr += g_lora_r * n_embd;
            if (i != 0) {
                b.att_v2_T = ptr;
                ptr += v_lora_r * n_embd;
                b.att_v1_T = ptr;
                ptr += n_embd * v_lora_r;
                b.att_v0 = ptr;
                ptr += n_embd;
            }
            b.att_k_k = ptr;
            ptr += n_embd;
            b.att_k_a = ptr;
            ptr += n_embd;
            b.att_receptance_weight = ptr;
            ptr += n_embd * n_embd;
            b.att_key_weight = ptr;
            ptr += n_embd * n_embd;
            b.att_value_weight = ptr;
            ptr += n_embd * n_embd;
            b.att_output_weight = ptr;
            ptr += n_embd * n_embd;
            b.att_ln_x_weight = ptr;
            ptr += n_embd;
            b.att_ln_x_bias = ptr;
            ptr += n_embd;
            b.ffn_x_k = ptr;
            ptr += n_embd;
            b.ffn_key_weight = ptr;
            ptr += n_embd * n_embd * 4;
            b.ffn_value_weight = ptr;
            ptr += n_embd * 4 * n_embd;
        }
        model.ln_out_weight = ptr;
        ptr += n_embd;
        model.ln_out_bias = ptr;
        ptr += n_embd;
        model.head_weight = ptr;
        ptr += n_embd * vocab_size;

        // Merge ln0 into embedding weights (same as C: layer_norm on all vocab embeddings)
        for (0..vocab_size) |vi| {
            const emb = model.emb_weight[vi * n_embd .. (vi + 1) * n_embd];
            var mean: f32 = 0.0;
            for (emb) |v| mean += v;
            mean /= @floatFromInt(n_embd);
            var variance: f32 = 0.0;
            for (emb) |v| {
                const d = v - mean;
                variance += d * d;
            }
            variance /= @floatFromInt(n_embd);
            const scale = 1.0 / std.math.sqrt(variance + 1e-5);
            for (0..n_embd) |j| {
                emb[j] = (emb[j] - mean) * scale * ln0_weight[j] + ln0_bias[j];
            }
        }

        // Transpose all LoRA weight matrices in-place
        {
            const max_lora_dim = @max(n_embd * w_lora_r, @max(w_lora_r * n_embd, @max(n_embd * a_lora_r, @max(a_lora_r * n_embd, @max(n_embd * g_lora_r, @max(g_lora_r * n_embd, @max(n_embd * v_lora_r, v_lora_r * n_embd)))))));
            const tmp = try allocator.alloc(f32, max_lora_dim);
            defer allocator.free(tmp);
            for (0..n_layer) |i| {
                const b = &model.blocks[i];
                matTranspose(b.att_w1_T, n_embd, w_lora_r, tmp);
                matTranspose(b.att_w2_T, w_lora_r, n_embd, tmp);
                matTranspose(b.att_a1_T, n_embd, a_lora_r, tmp);
                matTranspose(b.att_a2_T, a_lora_r, n_embd, tmp);
                matTranspose(b.att_g1_T, n_embd, g_lora_r, tmp);
                matTranspose(b.att_g2_T, g_lora_r, n_embd, tmp);
                if (i != 0) {
                    matTranspose(b.att_v1_T, n_embd, v_lora_r, tmp);
                    matTranspose(b.att_v2_T, v_lora_r, n_embd, tmp);
                }
            }
        }

        return model;
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.model_data);
        allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn forward(self: *const Self, token_list: []const u32, state: *RunState, logits: []f32) void {
        const c = self.header;
        const seq_len = token_list.len;
        const n_embd: usize = @intCast(c.n_embd);
        const n_layer: usize = @intCast(c.n_layer);

        const x = state.scratch[12 * n_embd .. 13 * n_embd];
        const x_ = state.scratch[13 * n_embd .. 14 * n_embd];
        const dx = state.scratch[14 * n_embd .. 15 * n_embd];
        const v0 = state.scratch[15 * n_embd .. 16 * n_embd];

        for (0..seq_len) |t| {
            const token = token_list[t];
            @memcpy(x, self.emb_weight[token * n_embd .. (token + 1) * n_embd]);

            v0[0] = std.math.nan(f32); // Use NaN to track initialization inside time_mixing

            for (0..n_layer) |i| {
                const b = &self.blocks[i];
                layer_norm(x_, x, b.ln1_weight[0..n_embd], b.ln1_bias[0..n_embd], 1e-5);

                const last_x_offset = i * 2 * n_embd;
                const state_offset = i * @as(usize, @intCast(@divTrunc(c.n_embd, c.head_size))) * @as(usize, @intCast(c.head_size)) * @as(usize, @intCast(c.head_size));

                time_mixing(dx, x_, v0, state.last_x[last_x_offset .. last_x_offset + n_embd], state.wkv_state[state_offset..], self, b, state.scratch);
                vec_add(x, x, dx);

                layer_norm(x_, x, b.ln2_weight[0..n_embd], b.ln2_bias[0..n_embd], 1e-5);

                const last_x_ffn_offset = i * 2 * n_embd + n_embd;
                channel_mixing(dx, x_, state.last_x[last_x_ffn_offset .. last_x_ffn_offset + n_embd], b, state.scratch);
                vec_add(x, x, dx);
            }

            if (t == seq_len - 1) {
                layer_norm(x, x, self.ln_out_weight[0..n_embd], self.ln_out_bias[0..n_embd], 1e-5);
                mat_mul_vec(logits, x, self.head_weight[0 .. n_embd * @as(usize, @intCast(c.vocab_size))]);
            }
        }
    }
};

pub fn wkv_kernel(
    y: []f32,
    model: *const Model,
    bw: *const BlockWeights,
    state: []f32,
    r: []const f32,
    w: []const f32,
    k: []const f32,
    v: []const f32,
    kk: []f32,
    a: []const f32,
) void {
    const c = model.header;
    const n_embd: usize = @intCast(c.n_embd);
    const head_size: usize = @intCast(c.head_size);
    const n_head: usize = n_embd / head_size;

    for (0..n_head) |i| {
        const head_state = state[i * head_size * head_size .. (i + 1) * head_size * head_size];
        const head_kk = kk[i * head_size .. (i + 1) * head_size];
        const head_y = y[i * head_size .. (i + 1) * head_size];
        const head_r = r[i * head_size .. (i + 1) * head_size];
        const head_w = w[i * head_size .. (i + 1) * head_size];
        const head_k = k[i * head_size .. (i + 1) * head_size];
        const head_v = v[i * head_size .. (i + 1) * head_size];
        const head_a = a[i * head_size .. (i + 1) * head_size];

        const ln_w = bw.att_ln_x_weight[i * head_size .. (i + 1) * head_size];
        const ln_b = bw.att_ln_x_bias[i * head_size .. (i + 1) * head_size];
        const r_k = bw.att_r_k[i * head_size .. (i + 1) * head_size];

        var kk_norm = vec_dot_product(head_kk, head_kk);
        kk_norm = std.math.sqrt(kk_norm);
        vec_scale(head_kk, head_kk, 1.0 / @max(kk_norm, 1e-12));

        {
            var tmp_buf: [1024]f32 = undefined;
            const smk = tmp_buf[0..head_size];
            mat_mul_vec(smk, head_kk, head_state);

            const kma = tmp_buf[head_size .. 2 * head_size];
            vec_hadamard(kma, head_kk, head_a);

            var tmp2_buf: [65536]f32 = undefined;
            const t = tmp2_buf[0 .. head_size * head_size];
            vec_out_product(t, smk, kma);

            const vmk = tmp2_buf[head_size * head_size .. 2 * head_size * head_size];
            vec_out_product(vmk, head_v, head_k);

            for (0..head_size) |j| {
                const state_row = head_state[j * head_size .. (j + 1) * head_size];
                vec_hadamard(state_row, state_row, head_w);
            }

            vec_sub(head_state, head_state, t);
            vec_add(head_state, head_state, vmk);
        }

        mat_mul_vec(head_y, head_r, head_state);

        layer_norm(head_y, head_y, ln_w, ln_b, 64e-5);

        {
            var tmp_buf: [1024]f32 = undefined;
            const kmrk = tmp_buf[0..head_size];
            vec_hadamard(kmrk, head_k, r_k);
            const y_sum_ = vec_dot_product(head_r, kmrk);

            const hu = tmp_buf[head_size .. 2 * head_size];
            vec_scale(hu, head_v, y_sum_);
            vec_add(head_y, head_y, hu);
        }
    }
}

pub fn time_mixing(
    dx: []f32,
    x: []const f32, // n_embd
    v0: []f32, // n_embd
    last_x: []f32, // n_embd
    state: []f32,
    model: *const Model,
    bw: *const BlockWeights,
    scratch: []f32,
) void {
    const c = model.header;
    const n_embd: usize = @intCast(c.n_embd);

    // allocate from scratch
    const x_lerp = scratch[0..n_embd];
    const r = scratch[n_embd .. 2 * n_embd];
    const w = scratch[2 * n_embd .. 3 * n_embd];
    const k = scratch[3 * n_embd .. 4 * n_embd];
    const v = scratch[4 * n_embd .. 5 * n_embd];
    const kk = scratch[5 * n_embd .. 6 * n_embd];
    const a = scratch[6 * n_embd .. 7 * n_embd];
    const g = scratch[7 * n_embd .. 8 * n_embd];
    const w_sigmoid = scratch[8 * n_embd .. 9 * n_embd];
    const v_sigmoid = scratch[9 * n_embd .. 10 * n_embd];
    const a_minus_1 = scratch[10 * n_embd .. 11 * n_embd];
    const y = scratch[11 * n_embd .. 12 * n_embd];

    // r = Wr @ xr
    lerp(x_lerp, last_x, x, bw.att_x_r[0..n_embd]);
    mat_mul_vec(r, x_lerp, bw.att_receptance_weight[0 .. n_embd * n_embd]);

    // w = np.exp(-sigmoid(...) / np.e**0.5)
    lerp(x_lerp, last_x, x, bw.att_x_w[0..n_embd]);
    const w_lora_r: usize = @intCast(c.w_lora_r);
    const w1_T = bw.att_w1_T[0 .. n_embd * w_lora_r];
    const w2_T = bw.att_w2_T[0 .. w_lora_r * n_embd];
    lora(w_sigmoid, x_lerp, w1_T, w2_T, .TANH);
    vec_add(w_sigmoid, w_sigmoid, bw.att_w0[0..n_embd]);
    vec_sigm(w_sigmoid);
    for (0..n_embd) |i| {
        w[i] = std.math.exp(-w_sigmoid[i] / 1.6487212707);
    } // 1.6487... = sqrt(e)

    // k = Wk @ xk
    lerp(x_lerp, last_x, x, bw.att_x_k[0..n_embd]);
    mat_mul_vec(k, x_lerp, bw.att_key_weight[0 .. n_embd * n_embd]);

    // v = Wv @ xv
    lerp(x_lerp, last_x, x, bw.att_x_v[0..n_embd]);
    mat_mul_vec(v, x_lerp, bw.att_value_weight[0 .. n_embd * n_embd]);

    if (std.math.isNan(v0[0])) {
        @memcpy(v0, v);
    } else {
        const v_lora_r: usize = @intCast(c.v_lora_r);
        const v1_T = bw.att_v1_T[0 .. n_embd * v_lora_r];
        const v2_T = bw.att_v2_T[0 .. v_lora_r * n_embd];
        lora(v_sigmoid, x_lerp, v1_T, v2_T, .NONE);
        vec_add(v_sigmoid, v_sigmoid, bw.att_v0[0..n_embd]);
        vec_sigm(v_sigmoid);
        lerp(v, v0, v, v_sigmoid);
    }

    // kk = k * k_k
    vec_hadamard(kk, k, bw.att_k_k[0..n_embd]);

    // a = sigmoid(...)
    lerp(x_lerp, last_x, x, bw.att_x_a[0..n_embd]);
    const a_lora_r: usize = @intCast(c.a_lora_r);
    const a1_T = bw.att_a1_T[0 .. n_embd * a_lora_r];
    const a2_T = bw.att_a2_T[0 .. a_lora_r * n_embd];
    lora(a, x_lerp, a1_T, a2_T, .NONE);
    vec_add(a, a, bw.att_a0[0..n_embd]);
    vec_sigm(a);

    // g = sigmoid(xg @ Wg1) @ Wg2
    lerp(x_lerp, last_x, x, bw.att_x_g[0..n_embd]);
    const g_lora_r: usize = @intCast(c.g_lora_r);
    const g1_T = bw.att_g1_T[0 .. n_embd * g_lora_r];
    const g2_T = bw.att_g2_T[0 .. g_lora_r * n_embd];
    lora(g, x_lerp, g1_T, g2_T, .SIGM);

    // k += k * (a-1) * k_a
    @memcpy(a_minus_1, a);
    vec_bias(a_minus_1, a_minus_1, -1.0);
    vec_hadamard(a_minus_1, a_minus_1, bw.att_k_a[0..n_embd]);
    vec_hadamard(a_minus_1, k, a_minus_1);
    vec_add(k, k, a_minus_1);

    // wkv_kernel
    wkv_kernel(y, model, bw, state, r, w, k, v, kk, a);

    // dx = Wo @ (y * g)
    vec_hadamard(y, y, g);
    mat_mul_vec(dx, y, bw.att_output_weight[0 .. n_embd * n_embd]);

    // last_x = x
    @memcpy(last_x, x);
}

pub fn channel_mixing(
    dx: []f32,
    x: []const f32, // n_embd
    last_x: []f32, // n_embd
    bw: *const BlockWeights,
    scratch: []f32,
) void {
    const n_embd = x.len;

    const xk = scratch[0..n_embd];
    const k = scratch[n_embd .. 5 * n_embd]; // 4 * n_embd

    lerp(xk, last_x, x, bw.ffn_x_k[0..n_embd]);
    mat_mul_vec(k, xk, bw.ffn_key_weight[0 .. n_embd * n_embd * 4]);

    for (0..4 * n_embd) |i| {
        const relu_k = @max(k[i], 0.0);
        k[i] = relu_k * relu_k;
    }

    mat_mul_vec(dx, k, bw.ffn_value_weight[0 .. n_embd * 4 * n_embd]);

    @memcpy(last_x, x);
}

pub fn vec_add(xout: []f32, a: []const f32, b: []const f32) void {
    assert(xout.len == a.len and a.len == b.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= xout.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const bv: V = b[i..][0..DEFAULT_VECTOR_WIDTH].*;
        xout[i..][0..DEFAULT_VECTOR_WIDTH].* = av + bv;
    }
    while (i < xout.len) : (i += 1) {
        xout[i] = a[i] + b[i];
    }
}

pub fn vec_sub(xout: []f32, a: []const f32, b: []const f32) void {
    assert(xout.len == a.len and a.len == b.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= xout.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const bv: V = b[i..][0..DEFAULT_VECTOR_WIDTH].*;
        xout[i..][0..DEFAULT_VECTOR_WIDTH].* = av - bv;
    }
    while (i < xout.len) : (i += 1) {
        xout[i] = a[i] - b[i];
    }
}

pub fn vec_hadamard(xout: []f32, a: []const f32, b: []const f32) void {
    assert(xout.len == a.len and a.len == b.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= xout.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const bv: V = b[i..][0..DEFAULT_VECTOR_WIDTH].*;
        xout[i..][0..DEFAULT_VECTOR_WIDTH].* = av * bv;
    }
    while (i < xout.len) : (i += 1) {
        xout[i] = a[i] * b[i];
    }
}

pub fn vec_bias(xout: []f32, a: []const f32, b: f32) void {
    assert(xout.len == a.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    const bv: V = @splat(b);
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= xout.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        xout[i..][0..DEFAULT_VECTOR_WIDTH].* = av + bv;
    }
    while (i < xout.len) : (i += 1) {
        xout[i] = a[i] + b;
    }
}

pub fn vec_scale(xout: []f32, a: []const f32, b: f32) void {
    assert(xout.len == a.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    const bv: V = @splat(b);
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= xout.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        xout[i..][0..DEFAULT_VECTOR_WIDTH].* = av * bv;
    }
    while (i < xout.len) : (i += 1) {
        xout[i] = a[i] * b;
    }
}

pub fn vec_dot_product(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    var acc: V = @splat(@as(f32, 0.0));
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= a.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const bv: V = b[i..][0..DEFAULT_VECTOR_WIDTH].*;
        acc = @mulAdd(V, av, bv, acc);
    }
    var ret = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        ret += a[i] * b[i];
    }
    return ret;
}

pub fn vec_out_product(xout: []f32, a: []const f32, b: []const f32) void {
    assert(xout.len == a.len * b.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    for (0..a.len) |i| {
        const av: V = @splat(a[i]);
        var j: usize = 0;
        while (j + DEFAULT_VECTOR_WIDTH <= b.len) : (j += DEFAULT_VECTOR_WIDTH) {
            const bv: V = b[j..][0..DEFAULT_VECTOR_WIDTH].*;
            xout[i * b.len + j ..][0..DEFAULT_VECTOR_WIDTH].* = av * bv;
        }
        while (j < b.len) : (j += 1) {
            xout[i * b.len + j] = a[i] * b[j];
        }
    }
}

pub fn vec_sum(x: []const f32) f32 {
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    var acc: V = @splat(@as(f32, 0.0));
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= x.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const xv: V = x[i..][0..DEFAULT_VECTOR_WIDTH].*;
        acc += xv;
    }
    var ret = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) {
        ret += x[i];
    }
    return ret;
}

pub fn lerp(xout: []f32, a: []const f32, b: []const f32, mu: []const f32) void {
    assert(xout.len == a.len and a.len == b.len and b.len == mu.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= xout.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const bv: V = b[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const muv: V = mu[i..][0..DEFAULT_VECTOR_WIDTH].*;
        xout[i..][0..DEFAULT_VECTOR_WIDTH].* = @mulAdd(V, muv, av - bv, bv);
    }
    while (i < xout.len) : (i += 1) {
        xout[i] = b[i] + mu[i] * (a[i] - b[i]);
    }
}

pub fn vec_sigm(x: []f32) void {
    for (x) |*v| {
        v.* = 1.0 / (1.0 + std.math.exp(-v.*));
    }
}

pub fn vec_tanh(x: []f32) void {
    for (x) |*v| {
        v.* = std.math.tanh(v.*);
    }
}

pub fn mat_mul_vec(xout: []f32, x: []const f32, w: []const f32) void {
    const d = xout.len;
    const n = x.len;
    assert(w.len == d * n);
    for (0..d) |i| {
        xout[i] = vec_dot_product(w[i * n .. (i + 1) * n], x);
    }
}

pub fn layer_norm(xout: []f32, x: []const f32, weight: []const f32, bias: []const f32, eps: f32) void {
    const len = x.len;
    const x_mean = vec_sum(x) / @as(f32, @floatFromInt(len));

    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    const mean_v: V = @splat(x_mean);

    var acc: V = @splat(@as(f32, 0.0));
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= len) : (i += DEFAULT_VECTOR_WIDTH) {
        const xv: V = x[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const diff = xv - mean_v;
        acc = @mulAdd(V, diff, diff, acc);
    }
    var x_var = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        const diff = x[i] - x_mean;
        x_var += diff * diff;
    }
    x_var /= @as(f32, @floatFromInt(len));

    const scale = 1.0 / std.math.sqrt(x_var + eps);
    const scale_v: V = @splat(scale);

    i = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= len) : (i += DEFAULT_VECTOR_WIDTH) {
        const xv: V = x[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const wv: V = weight[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const bv: V = bias[i..][0..DEFAULT_VECTOR_WIDTH].*;
        xout[i..][0..DEFAULT_VECTOR_WIDTH].* = @mulAdd(V, (xv - mean_v) * scale_v, wv, bv);
    }
    while (i < len) : (i += 1) {
        xout[i] = (x[i] - x_mean) * scale * weight[i] + bias[i];
    }
}

pub const LoraAct = enum { NONE, TANH, SIGM };

/// Lora computation
pub fn lora(xout: []f32, x: []const f32, weight_1: []const f32, weight_2: []const f32, act: LoraAct) void {
    const lora_rank = weight_1.len / x.len;
    var tmp_buf: [4096]f32 = undefined;
    const tmp = tmp_buf[0..lora_rank];

    mat_mul_vec(tmp, x, weight_1);

    switch (act) {
        .NONE => {},
        .TANH => vec_tanh(tmp),
        .SIGM => vec_sigm(tmp),
    }

    mat_mul_vec(xout, tmp, weight_2);
}

/// In-place matrix transpose: mat(rows, cols) -> mat(cols, rows)
fn matTranspose(mat: [*]f32, rows: usize, cols: usize, tmp: []f32) void {
    const n = rows * cols;
    assert(n <= tmp.len);
    @memcpy(tmp[0..n], mat[0..n]);
    for (0..rows) |i| {
        for (0..cols) |j| {
            mat[j * rows + i] = tmp[i * cols + j];
        }
    }
}
