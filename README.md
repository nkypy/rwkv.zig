# rwkv.zig

[English](#english) | [中文](#中文)

---

## English

A high-performance inference implementation of [RWKV-7](https://github.com/BlinkDL/RWKV-LM) in Zig, with SIMD vectorization.

### Features

- RWKV-7 architecture inference (prefill + decode)
- AVX2/FMA SIMD vectorization using Zig `@Vector` builtins
- Custom binary model format compatible with [rwkv7.c](https://github.com/KevlarKanou/rwkv7.c)
- BPE tokenizer with binary format
- Top-p (nucleus) sampling with temperature control
- Single-file, zero-dependency implementation

### Performance

Benchmarked on Intel i5-7360U @ 2.30GHz (RWKV7-0.1B, fp32):

| Implementation | Decode Speed |
|---|---|
| rwkv7.c (AVX2) | 3.5 token/s |
| **rwkv.zig (SIMD)** | **11-22 token/s** |

### Prerequisites

- Zig 0.17.0 or later

### Quick Start

```bash
# 1. Export model from .pth to .bin format
uv run --python 3.12 export.py model.pth model.bin rwkv_vocab_v20230424.txt tokenizer.bin

# 2. Build and run
zig build run -Doptimize=ReleaseFast -- -m model.bin -z tokenizer.bin -i "Once upon a time" -n 256 -t 1.0
```

### CLI Options

```
-m, --model <path>        Model file path (default: model.bin)
-z, --tokenizer <path>    Tokenizer file path (default: tokenizer.bin)
-i, --input <string>      Input prompt
-n, --seq-len <int>       Max generation length (default: 256)
-t, --temperature <float> Sampling temperature (default: 1.0, 0 = greedy)
-p, --top-p <float>       Top-p sampling threshold (default: 0.9)
-s, --seed <int>          Random seed
-v, --verbose             Show model info and tokens/s
```

### Project Structure

```
src/
  root.zig   - Core RWKV-7 inference engine (model loading, forward pass, SIMD kernels)
  main.zig   - CLI entry point (argument parsing, sampling, generation loop)
export.py    - Model export tool (.pth -> .bin, compatible with rwkv7.c format)
build.zig    - Zig build configuration
```

### Model Export

The `export.py` script converts RWKV-7 `.pth` checkpoints to the binary format used by both rwkv.zig and [rwkv7.c](https://github.com/KevlarKanou/rwkv7.c). The format includes a 56-byte header followed by fp32 weights in per-block layout.

### License

Apache License 2.0

---

## 中文

基于 Zig 的 [RWKV-7](https://github.com/BlinkDL/RWKV-LM) 高性能推理实现，支持 SIMD 向量化。

### 特性

- RWKV-7 架构推理（预填充 + 解码）
- 使用 Zig `@Vector` 内建函数实现 AVX2/FMA SIMD 向量化
- 兼容 [rwkv7.c](https://github.com/KevlarKanou/rwkv7.c) 的二进制模型格式
- 自定义 BPE 分词器二进制格式
- Top-p（核采样）采样，支持温度控制
- 单文件、零依赖实现

### 性能

在 Intel i5-7360U @ 2.30GHz 上测试（RWKV7-0.1B, fp32）：

| 实现 | 解码速度 |
|---|---|
| rwkv7.c (AVX2) | 3.5 token/s |
| **rwkv.zig (SIMD)** | **11-22 token/s** |

### 环境要求

- Zig 0.17.0 或更高版本

### 快速开始

```bash
# 1. 将 .pth 模型导出为 .bin 格式
uv run --python 3.12 export.py model.pth model.bin rwkv_vocab_v20230424.txt tokenizer.bin

# 2. 编译并运行
zig build run -Doptimize=ReleaseFast -- -m model.bin -z tokenizer.bin -i "Once upon a time" -n 256 -t 1.0
```

### 命令行参数

```
-m, --model <path>        模型文件路径（默认：model.bin）
-z, --tokenizer <path>    分词器文件路径（默认：tokenizer.bin）
-i, --input <string>      输入提示词
-n, --seq-len <int>       最大生成长度（默认：256）
-t, --temperature <float> 采样温度（默认：1.0，0 = 贪心解码）
-p, --top-p <float>       Top-p 采样阈值（默认：0.9）
-s, --seed <int>          随机种子
-v, --verbose             显示模型信息和 tokens/s
```

### 项目结构

```
src/
  root.zig   - RWKV-7 推理核心（模型加载、前向传播、SIMD 计算核）
  main.zig   - CLI 入口（参数解析、采样、生成循环）
export.py    - 模型导出工具（.pth -> .bin，兼容 rwkv7.c 格式）
build.zig    - Zig 构建配置
```

### 模型导出

`export.py` 脚本将 RWKV-7 的 `.pth` 检查点转换为 rwkv.zig 和 [rwkv7.c](https://github.com/KevlarKanou/rwkv7.c) 共用的二进制格式。格式包含 56 字节头部和按 block 排列的 fp32 权重。

### 许可证

Apache License 2.0
