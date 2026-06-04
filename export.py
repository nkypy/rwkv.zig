# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==1.26.4",
#     "rwkv==0.8.32",
#     "torch==2.2.2",
# ]
#
# [[tool.uv.index]]
# url = "http://mirrors.aliyun.com/pypi/simple/"
# default = true
# ///

import os

os.environ["RWKV_V7_ON"] = "1"

import sys
import gc
import struct
import urllib.request


# https://modelscope.cn/models/Blink_DL/rwkv7-g1/files
MODEL_URL = "https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master/rwkv7-g1d-0.1b-20260129-ctx8192.pth"
VOCAB_URL = "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/refs/heads/main/rwkv_pip_package/src/rwkv/rwkv_vocab_v20230424.txt"


def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().reshape(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def export_model(model, export_path):
    quant = 0
    # r  w  k  v  7  .  c
    # 72 77 6B 76 37 2E 63
    magic_number = 0x00632E37766B7772

    w_lora_rank = model["blocks.0.att.w1"].size()[1]
    a_lora_rank = model["blocks.0.att.a1"].size()[1]
    g_lora_rank = model["blocks.0.att.g1"].size()[1]
    v_lora_rank = model["blocks.1.att.v1"].size()[1]

    de = "blocks.0.ffn.s_emb.weight" in model
    dea = False
    s_lora_rank = model["blocks.0.ffn.s1"].size()[1] if de else 0

    export_model = open(export_path, "wb")
    header = struct.pack(
        "Liiiiiiiiiiii",
        magic_number,
        quant,
        head_size,
        n_embd,
        n_layer,
        vocab_size,
        w_lora_rank,
        a_lora_rank,
        g_lora_rank,
        v_lora_rank,
        de,
        dea,
        s_lora_rank,
    )
    export_model.write(header)

    weights = [
        model["emb.weight"],
        model["blocks.0.ln0.weight"],
        model["blocks.0.ln0.bias"],
    ]

    for i in range(n_layer):
        weights.extend(
            [
                model[f"blocks.{i}.ln1.weight"],
                model[f"blocks.{i}.ln1.bias"],
                model[f"blocks.{i}.ln2.weight"],
                model[f"blocks.{i}.ln2.bias"],
                model[f"blocks.{i}.att.x_r"],
                model[f"blocks.{i}.att.x_w"],
                model[f"blocks.{i}.att.x_k"],
                model[f"blocks.{i}.att.x_v"],
                model[f"blocks.{i}.att.x_a"],
                model[f"blocks.{i}.att.x_g"],
                model[f"blocks.{i}.att.w0"],
                model[f"blocks.{i}.att.r_k"],
                model[f"blocks.{i}.att.w1"],
                model[f"blocks.{i}.att.w2"],
                model[f"blocks.{i}.att.a1"],
                model[f"blocks.{i}.att.a2"],
                model[f"blocks.{i}.att.a0"],
                model[f"blocks.{i}.att.g1"],
                model[f"blocks.{i}.att.g2"],
            ]
        )
        if i != 0:
            weights.extend(
                [
                    model[f"blocks.{i}.att.v2"],
                    model[f"blocks.{i}.att.v1"],
                    model[f"blocks.{i}.att.v0"],
                ]
            )
        weights.extend(
            [
                model[f"blocks.{i}.att.k_k"],
                model[f"blocks.{i}.att.k_a"],
                model[f"blocks.{i}.att.receptance.weight"],
                model[f"blocks.{i}.att.key.weight"],
                model[f"blocks.{i}.att.value.weight"],
                model[f"blocks.{i}.att.output.weight"],
                model[f"blocks.{i}.att.ln_x.weight"],
                model[f"blocks.{i}.att.ln_x.bias"],
                model[f"blocks.{i}.ffn.x_k"],
                model[f"blocks.{i}.ffn.key.weight"],
                model[f"blocks.{i}.ffn.value.weight"],
            ]
        )
        if de:
            weights.extend(
                [
                    model[f"blocks.{i}.ffn.s1"],
                    model[f"blocks.{i}.ffn.s2"],
                    model[f"blocks.{i}.ffn.s0"],
                    model[f"blocks.{i}.ffn.s_emb_x.weight"],
                ]
            )

    weights.extend(
        [
            model["ln_out.weight"],
            model["ln_out.bias"],
            model["head.weight"],
        ]
    )

    for w in weights:
        serialize_fp32(export_model, w)
    export_model.close()

    if de:
        export_extra = open(export_path + ".extra", "wb")
        extra = []
        for i in range(n_layer):
            extra.extend(
                [
                    model[f"blocks.{i}.ffn.s_emb.weight"],
                ]
            )
        for w in extra:
            serialize_fp32(export_extra, w)
        export_extra.close()


def export_tokenizer(tokenizer, export_path):
    with open(tokenizer, "r") as f:
        lines = f.readlines()
    print(f"max_token_len {len(lines) + 1}")
    with open(export_path, "wb") as f:
        # zero score and length for empty token
        f.write(struct.pack("I", len(lines) + 1))
        f.write(struct.pack("I", 0))  # score
        f.write(struct.pack("I", 0))  # length
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            f.write(struct.pack("I", len(x)))  # score
            f.write(struct.pack("I", len(x)))  # length
            f.write(x)  # data


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python export.py <rwkv_model> <export_model> <tokenizer> <export_tokenizer>"
        )
        sys.exit(1)

    rwkv_model = os.path.abspath(sys.argv[1])
    if not os.path.exists(rwkv_model):
        print(f"File {rwkv_model} not found")
        print(f"Downloading model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, rwkv_model)

    if not os.path.exists(sys.argv[3]):
        print(f"File {sys.argv[3]} not found")
        print(f"Downloading tokenizer from {VOCAB_URL}...")
        urllib.request.urlretrieve(VOCAB_URL, sys.argv[3])

    if not os.path.exists(sys.argv[4]):
        print(f"Exporting tokenizer {os.path.abspath(sys.argv[4])}...")
        export_tokenizer(os.path.abspath(sys.argv[3]), os.path.abspath(sys.argv[4]))
        print(f"Tokenizer export done: {os.path.abspath(sys.argv[4])}")
    else:
        print(f"{os.path.abspath(sys.argv[4])} exists")

    if not os.path.exists(sys.argv[2]):
        import torch
        from rwkv.model import RWKV

        print(f"Loading model {rwkv_model}...")
        model = RWKV(model=rwkv_model[:-4], strategy="cpu fp32")
        head_size = model.args.head_size
        n_embd = model.args.n_embd
        n_layer = model.args.n_layer
        vocab_size = model.args.vocab_size
        del model
        gc.collect()
        model = torch.load(rwkv_model, map_location="cpu", weights_only=True)

        print(f"Exporting model {os.path.abspath(sys.argv[2])}...")
        export_model(model, os.path.abspath(sys.argv[2]))
        print(f"Model export done: {os.path.abspath(sys.argv[2])}")
    else:
        print(f"{os.path.abspath(sys.argv[2])} exists")
