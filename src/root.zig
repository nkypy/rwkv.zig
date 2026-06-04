//! RWKV-7 inference engine — model loading, forward pass, and SIMD-optimized compute kernels.
//!
//! Architecture overview:
//!   RWKV-7 is an RNN-based language model that replaces the quadratic attention of
//!   Transformers with a linear-time recurrence. Each block has two sub-layers:
//!     1. Time-mixing (attention analogue) — multi-head WKV recurrence with LoRA gates
//!     2. Channel-mixing (FFN analogue) — squared-ReLU feed-forward network
//!
//! Weight format:
//!   Binary file with a 56-byte Header, followed by fp32 weights in per-block order.
//!   Compatible with rwkv7.c (https://github.com/KevlarKanou/rwkv7.c).
//!
//! SIMD strategy:
//!   All vector kernels use Zig's `@Vector` builtins. The main loop processes
//!   DEFAULT_VECTOR_WIDTH elements (8 on AVX2, 4 on NEON) per iteration, with a
//!   scalar tail for remaining elements. FMA is emitted via `@mulAdd`.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

/// Binary file magic number — matches rwkv7.c format.
const MAGIC_NUMBER: u64 = 0x00632E37766B7772; // "rwkv7.c\0" in little-endian

/// SIMD vector width: 8 on AVX2 (256-bit), 4 on SSE/NEON (128-bit), fallback 4.
const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const simd_align: comptime_int = @alignOf(@Vector(DEFAULT_VECTOR_WIDTH, f32));
const simd_alignment = std.mem.Alignment.of(@Vector(DEFAULT_VECTOR_WIDTH, f32));

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// BPE tokenizer that loads from a binary vocab file.
/// Binary format: [vocab_size:u32] then for each token: [score:u32] [len:u32] [data:u8*len]
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

    /// Greedy longest-match encoding: at each position, find the vocab token with
    /// the highest score that matches the input text starting at that position.
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

// ---------------------------------------------------------------------------
// Thread pool for parallel mat_mul_vec
// ---------------------------------------------------------------------------

/// Persistent thread pool using phase counter for synchronization.
/// Workers spin on a phase counter; when it flips, work is available.
pub const ThreadPool = struct {
    const State = enum(u8) { idle = 0, work = 1, shutdown = 2 };

    xout_ptr: [*]f32 = undefined,
    x_ptr: [*]const f32 = undefined,
    w_ptr: [*]const f32 = undefined,
    n_cols: usize = 0,
    d_rows: usize = 0,

    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),
    finished: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    total_workers: usize,
    threads: []std.Thread,

    pub fn init(allocator: Allocator, n_threads: usize) !*ThreadPool {
        const pool = try allocator.create(ThreadPool);
        pool.* = .{
            .total_workers = n_threads,
            .threads = try allocator.alloc(std.Thread, n_threads),
        };
        for (0..n_threads) |i| {
            pool.threads[i] = try std.Thread.spawn(.{}, workerLoop, .{ pool, i });
        }
        return pool;
    }

    pub fn deinit(self: *ThreadPool, allocator: Allocator) void {
        self.state.store(@intFromEnum(State.shutdown), .release);
        for (self.threads) |t| t.join();
        allocator.free(self.threads);
        allocator.destroy(self);
    }

    fn workerLoop(pool: *ThreadPool, worker_id: usize) void {
        while (true) {
            // Spin-wait with periodic yield to avoid burning too much CPU
            var spin: u32 = 0;
            while (pool.state.load(.acquire) == @intFromEnum(State.idle)) {
                spin +%= 1;
                if (spin > 1024) {
                    std.Thread.yield() catch {};
                    spin = 0;
                }
            }
            if (pool.state.load(.acquire) == @intFromEnum(State.shutdown)) return;

            // Read work parameters and compute my portion
            const d = pool.d_rows;
            const n = pool.n_cols;
            const chunk = d / pool.total_workers;
            const start = worker_id * chunk;
            const end = if (worker_id == pool.total_workers - 1) d else start + chunk;

            const xout = pool.xout_ptr;
            const x = pool.x_ptr;
            const w = pool.w_ptr;
            for (start..end) |i| {
                xout[i] = vec_dot_product(w[i * n .. (i + 1) * n], x[0..n]);
            }

            _ = pool.finished.fetchAdd(1, .release);
        }
    }

    pub fn matmul(self: *ThreadPool, xout: []f32, x: []const f32, w: []const f32) void {
        const d = xout.len;
        const n = x.len;

        // Publish work parameters
        self.xout_ptr = xout.ptr;
        self.x_ptr = x.ptr;
        self.w_ptr = w.ptr;
        self.n_cols = n;
        self.d_rows = d;
        self.finished.store(0, .release);

        // Signal workers
        self.state.store(@intFromEnum(State.work), .release);

        // Thread 0 does rows [0, chunk)
        const chunk = d / self.total_workers;
        const x_slice = x[0..n];
        for (0..chunk) |i| {
            xout[i] = vec_dot_product(w[i * n .. (i + 1) * n], x_slice);
        }

        // Wait for workers to finish
        while (self.finished.load(.acquire) < self.total_workers - 1) {}

        // Reset to idle
        self.state.store(@intFromEnum(State.idle), .release);
    }
};

// ---------------------------------------------------------------------------
// RunState — persistent hidden state across token generations
// ---------------------------------------------------------------------------

/// Mutable inference state that persists across forward calls.
/// All buffers are SIMD-aligned for efficient vector loads/stores.
pub const RunState = struct {
    const Self = @This();

    /// Previous token's input per layer, used for lerp mixing.
    /// Layout: (n_layer, 2, n_embd) — index 0 for time-mixing, 1 for channel-mixing.
    last_x: []align(simd_align) f32,

    /// WKV recurrence state per layer per head.
    /// Layout: (n_layer, n_head, head_size, head_size)
    wkv_state: []align(simd_align) f32,

    /// Scratch buffer shared across time_mixing and channel_mixing (called sequentially).
    /// Size: 32 * n_embd floats, sub-allocated into ~12 named regions.
    scratch: []align(simd_align) f32,

    /// Optional thread pool for parallel mat_mul_vec.
    pool: ?*ThreadPool = null,

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

    pub fn initThreadPool(self: *Self, allocator: Allocator, n_threads: usize) !void {
        self.pool = try ThreadPool.init(allocator, n_threads);
        global_pool = self.pool;
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        if (self.pool) |p| p.deinit(allocator);
        allocator.free(self.last_x);
        allocator.free(self.wkv_state);
        allocator.free(self.scratch);
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Model header and per-block weights
// ---------------------------------------------------------------------------

/// 56-byte file header — must match the C rwkv7.c packed struct exactly.
const Header = extern struct {
    magic_number: u64,
    quant: i32, // 0 = fp32, non-zero = quantized (not supported)
    head_size: i32, // attention head dimension (e.g. 64)
    n_embd: i32, // embedding dimension (e.g. 768)
    n_layer: i32, // number of transformer blocks
    vocab_size: i32, // vocabulary size (e.g. 65536)
    w_lora_r: i32, // LoRA rank for time-decay (w) gate
    a_lora_r: i32, // LoRA rank for bonus (a) gate
    g_lora_r: i32, // LoRA rank for output gate (g)
    v_lora_r: i32, // LoRA rank for value blending gate
    de: i32, // dynamic expansion flag (not used in this impl)
    dea: i32, // dynamic expansion attention flag
    s_lora_r: i32, // LoRA rank for dynamic expansion
};

/// All weight pointers for a single transformer block.
/// Pointers point into the memory-mapped model_data buffer.
pub const BlockWeights = struct {
    // Layer norms
    ln1_weight: [*]f32,
    ln1_bias: [*]f32,
    ln2_weight: [*]f32,
    ln2_bias: [*]f32,

    // Time-mixing lerp coefficients: x_mixed = lerp(last_x, x, coeff)
    att_x_r: [*]f32, // receptance mixing
    att_x_w: [*]f32, // time-decay mixing
    att_x_k: [*]f32, // key mixing
    att_x_v: [*]f32, // value mixing
    att_x_a: [*]f32, // bonus mixing
    att_x_g: [*]f32, // output gate mixing

    // Time-mixing parameters
    att_w0: [*]f32, // base time-decay bias
    att_r_k: [*]f32, // receptance-key interaction per head

    // LoRA weight pairs (stored transposed for mat_mul_vec)
    att_w1_T: [*]f32, // time-decay LoRA down-projection
    att_w2_T: [*]f32, // time-decay LoRA up-projection
    att_a1_T: [*]f32, // bonus LoRA down-projection
    att_a2_T: [*]f32, // bonus LoRA up-projection
    att_a0: [*]f32, // bonus bias
    att_g1_T: [*]f32, // output gate LoRA down-projection
    att_g2_T: [*]f32, // output gate LoRA up-projection

    // Value blending LoRA — only present for layer > 0
    att_v2_T: [*]f32,
    att_v1_T: [*]f32,
    att_v0: [*]f32,

    // Key normalization and decay
    att_k_k: [*]f32, // per-element key normalization scale
    att_k_a: [*]f32, // key bonus decay coefficient

    // Dense projection matrices (n_embd x n_embd)
    att_receptance_weight: [*]f32,
    att_key_weight: [*]f32,
    att_value_weight: [*]f32,
    att_output_weight: [*]f32,

    // Per-head group norm inside wkv_kernel
    att_ln_x_weight: [*]f32,
    att_ln_x_bias: [*]f32,

    // Channel-mixing (FFN)
    ffn_x_k: [*]f32, // lerp coefficient for FFN input mixing
    ffn_key_weight: [*]f32, // (4*n_embd, n_embd) — squared-ReLU expansion
    ffn_value_weight: [*]f32, // (n_embd, 4*n_embd) — projection back
};

// ---------------------------------------------------------------------------
// Model — loaded weights + inference methods
// ---------------------------------------------------------------------------

pub const Model = struct {
    const Self = @This();

    header: Header,
    emb_weight: [*]f32, // (vocab_size, n_embd) — with LN0 already merged in
    blocks: []BlockWeights, // per-block weight pointers
    ln_out_weight: [*]f32, // final layer norm
    ln_out_bias: [*]f32,
    head_weight: [*]f32, // (vocab_size, n_embd) — language model head
    model_data: []align(std.heap.page_size_min) u8, // raw weight buffer

    pub fn fromFile(io: Io, path: []const u8, allocator: Allocator) !Self {
        var model_file = try Io.Dir.cwd().openFile(io, path, .{});
        defer model_file.close(io);

        var buffer: [4096]u8 = undefined;
        var file_reader = model_file.reader(io, &buffer);

        // Read 56-byte header
        const header_bytes = try allocator.alloc(u8, @sizeOf(Header));
        defer allocator.free(header_bytes);
        try file_reader.interface.readSliceAll(header_bytes);
        const header: Header = std.mem.bytesToValue(Header, header_bytes);

        assert(header.magic_number == MAGIC_NUMBER);
        assert(header.quant == 0); // quantized models not supported

        // Memory-map all remaining weight data (page-aligned for large model files)
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

        // Walk through the flat weight buffer and assign pointers to each block.
        // Order matches rwkv7.c load_model exactly.
        var ptr: [*]f32 = @ptrCast(@alignCast(data));
        model.emb_weight = ptr;
        ptr += vocab_size * n_embd;
        // Block-0's LN0 weights — will be merged into emb_weight below
        const ln0_weight = ptr;
        ptr += n_embd;
        const ln0_bias = ptr;
        ptr += n_embd;

        for (0..n_layer) |i| {
            const b = &model.blocks[i];
            // Pre-attention layer norm
            b.ln1_weight = ptr;
            ptr += n_embd;
            b.ln1_bias = ptr;
            ptr += n_embd;
            // Post-attention layer norm
            b.ln2_weight = ptr;
            ptr += n_embd;
            b.ln2_bias = ptr;
            ptr += n_embd;
            // Time-mixing lerp coefficients
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
            // Time-decay base bias
            b.att_w0 = ptr;
            ptr += n_embd;
            // Receptance-key interaction (per head)
            b.att_r_k = ptr;
            ptr += n_head * head_size;
            // LoRA projections: w (time-decay), a (bonus), g (output gate)
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
            // Value blending LoRA — absent in layer 0 (v_lora only for i > 0)
            if (i != 0) {
                b.att_v2_T = ptr;
                ptr += v_lora_r * n_embd;
                b.att_v1_T = ptr;
                ptr += n_embd * v_lora_r;
                b.att_v0 = ptr;
                ptr += n_embd;
            }
            // Key normalization and bonus-decay coefficients
            b.att_k_k = ptr;
            ptr += n_embd;
            b.att_k_a = ptr;
            ptr += n_embd;
            // Dense projections (n_embd x n_embd each)
            b.att_receptance_weight = ptr;
            ptr += n_embd * n_embd;
            b.att_key_weight = ptr;
            ptr += n_embd * n_embd;
            b.att_value_weight = ptr;
            ptr += n_embd * n_embd;
            b.att_output_weight = ptr;
            ptr += n_embd * n_embd;
            // Per-head group norm inside wkv_kernel
            b.att_ln_x_weight = ptr;
            ptr += n_embd;
            b.att_ln_x_bias = ptr;
            ptr += n_embd;
            // Channel-mixing (FFN) weights
            b.ffn_x_k = ptr;
            ptr += n_embd;
            b.ffn_key_weight = ptr;
            ptr += n_embd * n_embd * 4;
            b.ffn_value_weight = ptr;
            ptr += n_embd * 4 * n_embd;
        }
        // Final layer norm and LM head
        model.ln_out_weight = ptr;
        ptr += n_embd;
        model.ln_out_bias = ptr;
        ptr += n_embd;
        model.head_weight = ptr;
        ptr += n_embd * vocab_size;

        // Merge LN0 into embedding table — this is a one-time preprocessing step
        // that avoids an extra layer_norm call during every forward pass.
        // Equivalent to C's: layer_norm(emb_weight, emb_weight, ln0_w, ln0_b, ...)
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

        // Transpose all LoRA weight matrices in-place.
        // The model file stores LoRA weights in PyTorch's (in, out) layout,
        // but mat_mul_vec expects (out, in) row-major layout.
        // A single tmp buffer is reused across all transposes.
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

    /// Run the full RWKV-7 forward pass for a sequence of tokens.
    /// Processes one token at a time (equivalent to C's seq_len=1 per step),
    /// which is correct for the RNN recurrence — each token only depends on
    /// the current input and the accumulated hidden state.
    pub fn forward(self: *const Self, token_list: []const u32, state: *RunState, logits: []f32) void {
        const c = self.header;
        const seq_len = token_list.len;
        const n_embd: usize = @intCast(c.n_embd);
        const n_layer: usize = @intCast(c.n_layer);

        // Scratch layout (12 regions of n_embd each, used by time_mixing/channel_mixing):
        //   [12..13) = x    (current token embedding)
        //   [13..14) = x_   (layer-normed x)
        //   [14..15) = dx   (sub-layer output delta)
        //   [15..16) = v0   (first-layer value, used for v_lora blending)
        const x = state.scratch[12 * n_embd .. 13 * n_embd];
        const x_ = state.scratch[13 * n_embd .. 14 * n_embd];
        const dx = state.scratch[14 * n_embd .. 15 * n_embd];
        const v0 = state.scratch[15 * n_embd .. 16 * n_embd];

        for (0..seq_len) |t| {
            const token = token_list[t];
            // Look up embedding (LN0 already merged during model load)
            @memcpy(x, self.emb_weight[token * n_embd .. (token + 1) * n_embd]);

            // NaN sentinel: time_mixing uses this to detect first-layer initialization.
            // Resets per token (matching C's local v0 in forward).
            v0[0] = std.math.nan(f32);

            for (0..n_layer) |i| {
                const b = &self.blocks[i];

                // --- Time-mixing sub-layer ---
                layer_norm(x_, x, b.ln1_weight[0..n_embd], b.ln1_bias[0..n_embd], 1e-5);

                const last_x_offset = i * 2 * n_embd;
                const state_offset = i * @as(usize, @intCast(@divTrunc(c.n_embd, c.head_size))) * @as(usize, @intCast(c.head_size)) * @as(usize, @intCast(c.head_size));

                time_mixing(dx, x_, v0, state.last_x[last_x_offset .. last_x_offset + n_embd], state.wkv_state[state_offset..], self, b, state.scratch);
                vec_add(x, x, dx); // residual connection

                // --- Channel-mixing sub-layer ---
                layer_norm(x_, x, b.ln2_weight[0..n_embd], b.ln2_bias[0..n_embd], 1e-5);

                const last_x_ffn_offset = i * 2 * n_embd + n_embd;
                channel_mixing(dx, x_, state.last_x[last_x_ffn_offset .. last_x_ffn_offset + n_embd], b, state.scratch);
                vec_add(x, x, dx); // residual connection
            }

            // Compute logits only for the last token in the sequence
            if (t == seq_len - 1) {
                layer_norm(x, x, self.ln_out_weight[0..n_embd], self.ln_out_bias[0..n_embd], 1e-5);
                mat_mul_vec(logits, x, self.head_weight[0 .. n_embd * @as(usize, @intCast(c.vocab_size))]);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// wkv_kernel — multi-head WKV recurrence (the core of RWKV-7)
// ---------------------------------------------------------------------------

/// Implements the WKV (Weighted Key-Value) recurrence from the RWKV-7 paper:
///   S = S * diag(w) - S @ kk @ (kk * a).mT + v * k.mT   (state update)
///   y = group_norm(S @ r) + (r . (k * r_k)) * v           (output)
///
/// This is equivalent to linear attention with a learned time-decay (w),
/// but computed as an O(n) recurrence rather than O(n^2) attention.
pub fn wkv_kernel(
    y: []f32,
    model: *const Model,
    bw: *const BlockWeights,
    state: []f32,
    r: []const f32, // receptance
    w: []const f32, // time-decay (exp of learned sigmoid)
    k: []const f32, // key (modified with a-1 bonus)
    v: []const f32, // value (optionally blended with v0)
    kk: []f32, // normalized key (k * k_k, then L2-normalized)
    a: []const f32, // bonus gate (sigmoid output)
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

        // L2-normalize kk: kk /= max(||kk||, 1e-12)
        var kk_norm = vec_dot_product(head_kk, head_kk);
        kk_norm = std.math.sqrt(kk_norm);
        vec_scale(head_kk, head_kk, 1.0 / @max(kk_norm, 1e-12));

        // State update: S = S * w.mT - S@kk * (kk*a).mT + v * k.mT
        // Decomposed to avoid non-contiguous column-vector access.
        {
            var tmp_buf: [1024]f32 = undefined;
            const smk = tmp_buf[0..head_size]; // S @ kk
            mat_mul_vec(smk, head_kk, head_state);

            const kma = tmp_buf[head_size .. 2 * head_size]; // kk * a
            vec_hadamard(kma, head_kk, head_a);

            var tmp2_buf: [65536]f32 = undefined;
            const t = tmp2_buf[0 .. head_size * head_size]; // (S@kk) outer (kk*a)
            vec_out_product(t, smk, kma);

            const vmk = tmp2_buf[head_size * head_size .. 2 * head_size * head_size]; // v outer k
            vec_out_product(vmk, head_v, head_k);

            // S = S * diag(w) — multiply each row of S by the decay vector
            for (0..head_size) |j| {
                const state_row = head_state[j * head_size .. (j + 1) * head_size];
                vec_hadamard(state_row, state_row, head_w);
            }

            vec_sub(head_state, head_state, t); // S -= (S@kk) * (kk*a).mT
            vec_add(head_state, head_state, vmk); // S += v * k.mT
        }

        // y = S @ r
        mat_mul_vec(head_y, head_r, head_state);

        // y = group_norm(y, ln_w, ln_b)
        layer_norm(head_y, head_y, ln_w, ln_b, 64e-5);

        // y += (r . (k * r_k)) * v — bonus output from receptance-key interaction
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

// ---------------------------------------------------------------------------
// time_mixing — attention sub-layer
// ---------------------------------------------------------------------------

/// Time-mixing: the RWKV-7 attention analogue.
///
/// For each of {r, w, k, v, a, g}, the input is a lerp of the current token (x)
/// and the previous token (last_x) using learned mixing coefficients.
/// Then LoRA projections and the WKV recurrence produce the output.
///
/// The v_lora mechanism blends the current v with a persistent v0 (from layer 0)
/// using a learned sigmoid gate — this gives the model a form of "long-term memory"
/// that persists across all layers.
pub fn time_mixing(
    dx: []f32,
    x: []const f32, // layer-normed input (n_embd)
    v0: []f32, // persistent value from layer 0 (n_embd), NaN on first call
    last_x: []f32, // previous token's input (n_embd)
    state: []f32, // WKV recurrence state for this layer
    model: *const Model,
    bw: *const BlockWeights,
    scratch: []f32, // shared scratch buffer (12 * n_embd)
) void {
    const c = model.header;
    const n_embd: usize = @intCast(c.n_embd);

    // Scratch sub-allocation (each region is n_embd floats):
    const x_lerp = scratch[0..n_embd]; // lerped input (reused per gate)
    const r = scratch[n_embd .. 2 * n_embd]; // receptance
    const w = scratch[2 * n_embd .. 3 * n_embd]; // time-decay
    const k = scratch[3 * n_embd .. 4 * n_embd]; // key
    const v = scratch[4 * n_embd .. 5 * n_embd]; // value
    const kk = scratch[5 * n_embd .. 6 * n_embd]; // normalized key
    const a = scratch[6 * n_embd .. 7 * n_embd]; // bonus gate
    const g = scratch[7 * n_embd .. 8 * n_embd]; // output gate
    const w_sigmoid = scratch[8 * n_embd .. 9 * n_embd]; // w before exp
    const v_sigmoid = scratch[9 * n_embd .. 10 * n_embd]; // v blending gate
    const a_minus_1 = scratch[10 * n_embd .. 11 * n_embd]; // temp for k bonus
    const y = scratch[11 * n_embd .. 12 * n_embd]; // WKV output

    // r = Wr @ lerp(last_x, x, x_r)
    lerp(x_lerp, last_x, x, bw.att_x_r[0..n_embd]);
    mat_mul_vec(r, x_lerp, bw.att_receptance_weight[0 .. n_embd * n_embd]);

    // w = exp(-sigmoid(tanh(xw @ Ww1) @ Ww2 + w0) / sqrt(e))
    lerp(x_lerp, last_x, x, bw.att_x_w[0..n_embd]);
    const w_lora_r: usize = @intCast(c.w_lora_r);
    const w1_T = bw.att_w1_T[0 .. n_embd * w_lora_r];
    const w2_T = bw.att_w2_T[0 .. w_lora_r * n_embd];
    lora(w_sigmoid, x_lerp, w1_T, w2_T, .TANH);
    vec_add(w_sigmoid, w_sigmoid, bw.att_w0[0..n_embd]);
    vec_sigm(w_sigmoid);
    for (0..n_embd) |i| {
        w[i] = std.math.exp(-w_sigmoid[i] / 1.6487212707); // 1.6487... = sqrt(e)
    }

    // k = Wk @ lerp(last_x, x, x_k)
    lerp(x_lerp, last_x, x, bw.att_x_k[0..n_embd]);
    mat_mul_vec(k, x_lerp, bw.att_key_weight[0 .. n_embd * n_embd]);

    // v = Wv @ lerp(last_x, x, x_v)
    lerp(x_lerp, last_x, x, bw.att_x_v[0..n_embd]);
    mat_mul_vec(v, x_lerp, bw.att_value_weight[0 .. n_embd * n_embd]);

    // v_lora: blend v with persistent v0 using a sigmoid gate.
    // On the very first call (layer 0 of first token), v0 is NaN → just copy v.
    // On subsequent layers, v = lerp(v0, v, sigmoid(lora(x_lerp) + v0_bias)).
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

    // kk = k * k_k (element-wise key normalization scale)
    vec_hadamard(kk, k, bw.att_k_k[0..n_embd]);

    // a = sigmoid(xa @ Wa1 @ Wa2 + a0) — bonus gate
    lerp(x_lerp, last_x, x, bw.att_x_a[0..n_embd]);
    const a_lora_r: usize = @intCast(c.a_lora_r);
    const a1_T = bw.att_a1_T[0 .. n_embd * a_lora_r];
    const a2_T = bw.att_a2_T[0 .. a_lora_r * n_embd];
    lora(a, x_lerp, a1_T, a2_T, .NONE);
    vec_add(a, a, bw.att_a0[0..n_embd]);
    vec_sigm(a);

    // g = sigmoid(xg @ Wg1) @ Wg2 — output gate
    lerp(x_lerp, last_x, x, bw.att_x_g[0..n_embd]);
    const g_lora_r: usize = @intCast(c.g_lora_r);
    const g1_T = bw.att_g1_T[0 .. n_embd * g_lora_r];
    const g2_T = bw.att_g2_T[0 .. g_lora_r * n_embd];
    lora(g, x_lerp, g1_T, g2_T, .SIGM);

    // k += k * (a - 1) * k_a — apply bonus decay to keys
    @memcpy(a_minus_1, a);
    vec_bias(a_minus_1, a_minus_1, -1.0);
    vec_hadamard(a_minus_1, a_minus_1, bw.att_k_a[0..n_embd]);
    vec_hadamard(a_minus_1, k, a_minus_1);
    vec_add(k, k, a_minus_1);

    // WKV recurrence — updates state in-place and produces y
    wkv_kernel(y, model, bw, state, r, w, k, v, kk, a);

    // dx = Wo @ (y * g) — output projection with gating
    vec_hadamard(y, y, g);
    mat_mul_vec(dx, y, bw.att_output_weight[0 .. n_embd * n_embd]);

    // Save current input for next token's lerp
    @memcpy(last_x, x);
}

// ---------------------------------------------------------------------------
// channel_mixing — FFN sub-layer
// ---------------------------------------------------------------------------

/// Channel-mixing: squared-ReLU feed-forward network.
///   xk = lerp(last_x, x, ffn_x_k)
///   k = (ReLU(Wk @ xk))^2
///   dx = Wv @ k
pub fn channel_mixing(
    dx: []f32,
    x: []const f32, // layer-normed input (n_embd)
    last_x: []f32, // previous token's input (n_embd)
    bw: *const BlockWeights,
    scratch: []f32,
) void {
    const n_embd = x.len;

    const xk = scratch[0..n_embd];
    const k = scratch[n_embd .. 5 * n_embd]; // 4 * n_embd for expanded FFN

    // xk = lerp(last_x, x, ffn_x_k)
    lerp(xk, last_x, x, bw.ffn_x_k[0..n_embd]);

    // k = Wk @ xk — project to 4x hidden dim
    mat_mul_vec(k, xk, bw.ffn_key_weight[0 .. n_embd * n_embd * 4]);

    // k = (ReLU(k))^2 — squared ReLU activation
    for (0..4 * n_embd) |i| {
        const relu_k = @max(k[i], 0.0);
        k[i] = relu_k * relu_k;
    }

    // dx = Wv @ k — project back to hidden dim
    mat_mul_vec(dx, k, bw.ffn_value_weight[0 .. n_embd * 4 * n_embd]);

    // Save current input for next token's lerp
    @memcpy(last_x, x);
}

// ---------------------------------------------------------------------------
// SIMD vector primitives
// ---------------------------------------------------------------------------
// All functions below process DEFAULT_VECTOR_WIDTH elements per iteration
// using Zig's @Vector type, which compiles to single SIMD instructions
// (e.g., vaddps, vmulps, vfmaddps on x86-64 AVX2+FMA).

/// xout = a + b (element-wise)
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

/// xout = a - b (element-wise)
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

/// xout = a * b (Hadamard / element-wise product)
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

/// xout = a + b (broadcast scalar b across all elements)
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

/// xout = a * b (broadcast scalar b across all elements)
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

/// Dot product: sum(a[i] * b[i]). Uses FMA accumulation for accuracy and speed.
pub fn vec_dot_product(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    var acc: V = @splat(@as(f32, 0.0));
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= a.len) : (i += DEFAULT_VECTOR_WIDTH) {
        const av: V = a[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const bv: V = b[i..][0..DEFAULT_VECTOR_WIDTH].*;
        acc = @mulAdd(V, av, bv, acc); // acc += a * b (fused multiply-add)
    }
    var ret = @reduce(.Add, acc); // horizontal sum of accumulator
    while (i < a.len) : (i += 1) {
        ret += a[i] * b[i];
    }
    return ret;
}

/// Outer product: xout[i,j] = a[i] * b[j]. Inner loop is SIMD-vectorized.
pub fn vec_out_product(xout: []f32, a: []const f32, b: []const f32) void {
    assert(xout.len == a.len * b.len);
    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    for (0..a.len) |i| {
        const av: V = @splat(a[i]); // broadcast a[i] across vector
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

/// Sum of all elements. Uses SIMD accumulation + horizontal reduce.
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

/// Linear interpolation: xout = b + mu * (a - b).
/// Uses FMA: @mulAdd(mu, a - b, b) compiles to vfmaddps on FMA-capable hardware.
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

/// Element-wise sigmoid: x[i] = 1 / (1 + exp(-x[i])).
/// Not SIMD-vectorized (matches C — exp() is the bottleneck).
pub fn vec_sigm(x: []f32) void {
    for (x) |*v| {
        v.* = 1.0 / (1.0 + std.math.exp(-v.*));
    }
}

/// Element-wise tanh. Not SIMD-vectorized (matches C).
pub fn vec_tanh(x: []f32) void {
    for (x) |*v| {
        v.* = std.math.tanh(v.*);
    }
}

/// Optional global thread pool pointer, set once at init time.
/// mat_mul_vec checks this to decide whether to parallelize.
var global_pool: ?*ThreadPool = null;

/// Matrix-vector multiply: xout = W @ x, where W is (d, n) row-major.
/// Each output element is a SIMD-accelerated dot product of a W row with x.
/// Uses thread pool when available and d is large enough to justify overhead.
pub fn mat_mul_vec(xout: []f32, x: []const f32, w: []const f32) void {
    const d = xout.len;
    const n = x.len;
    assert(w.len == d * n);
    // Use parallel path only for large matrices (threshold: 256 rows)
    if (global_pool) |pool| {
        if (d >= 256) {
            pool.matmul(xout, x, w);
            return;
        }
    }
    for (0..d) |i| {
        xout[i] = vec_dot_product(w[i * n .. (i + 1) * n], x);
    }
}

/// Layer normalization (actually group norm in RWKV — applied per-head).
///   xout = (x - mean) / sqrt(var + eps) * weight + bias
/// Three passes: mean, variance (FMA), then transform (FMA).
pub fn layer_norm(xout: []f32, x: []const f32, weight: []const f32, bias: []const f32, eps: f32) void {
    const len = x.len;
    const x_mean = vec_sum(x) / @as(f32, @floatFromInt(len));

    const V = @Vector(DEFAULT_VECTOR_WIDTH, f32);
    const mean_v: V = @splat(x_mean);

    // Pass 2: compute variance using FMA accumulation
    var acc: V = @splat(@as(f32, 0.0));
    var i: usize = 0;
    while (i + DEFAULT_VECTOR_WIDTH <= len) : (i += DEFAULT_VECTOR_WIDTH) {
        const xv: V = x[i..][0..DEFAULT_VECTOR_WIDTH].*;
        const diff = xv - mean_v;
        acc = @mulAdd(V, diff, diff, acc); // acc += diff^2
    }
    var x_var = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        const diff = x[i] - x_mean;
        x_var += diff * diff;
    }
    x_var /= @as(f32, @floatFromInt(len));

    const scale = 1.0 / std.math.sqrt(x_var + eps);
    const scale_v: V = @splat(scale);

    // Pass 3: normalize, scale, shift — single FMA chain
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

// ---------------------------------------------------------------------------
// LoRA helper
// ---------------------------------------------------------------------------

pub const LoraAct = enum { NONE, TANH, SIGM };

/// Low-Rank Adaptation: xout = W2 @ act(W1 @ x).
/// Used for time-decay (w), bonus (a), output gate (g), and value blend (v) gates.
pub fn lora(xout: []f32, x: []const f32, weight_1: []const f32, weight_2: []const f32, act: LoraAct) void {
    const lora_rank = weight_1.len / x.len;
    var tmp_buf: [4096]f32 = undefined;
    const tmp = tmp_buf[0..lora_rank];

    mat_mul_vec(tmp, x, weight_1); // down-project: (lora_rank,)

    switch (act) {
        .NONE => {},
        .TANH => vec_tanh(tmp),
        .SIGM => vec_sigm(tmp),
    }

    mat_mul_vec(xout, tmp, weight_2); // up-project: (n_embd,)
}

// ---------------------------------------------------------------------------
// Matrix transpose
// ---------------------------------------------------------------------------

/// In-place matrix transpose: mat(rows, cols) -> mat(cols, rows).
/// Uses a caller-provided temporary buffer to avoid heap allocation.
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
