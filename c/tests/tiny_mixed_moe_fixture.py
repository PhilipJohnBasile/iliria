"""Build a MINUSCULE GLM-5.2-shaped model dir with a MIXED int4/int2 routed-expert
container, loadable by the real, unmodified C engine (glm.c's expert_load infers the
per-tensor bit width from the packed byte count -- see tools/quant_container.py).

Same architecture family as tiny_serve_fixture.py (MLA + sigmoid router + shared
expert, no DSA weights, no MTP), extended so that each MoE layer's routed experts are
pre-quantized with a MIX of formats instead of left as plain F32:
  - HALF the routed experts (even index) -> int4 (glm.c fmt 2)
  - HALF the routed experts (odd index)  -> int2 (glm.c fmt 3)
The router's e_score_correction_bias is set to an overwhelming +50/-50 split so the
top-K selection is DETERMINISTIC and GUARANTEED to span both formats on every decode
step, regardless of the (random) router weights or hidden state: this is what actually
exercises the mixed-format block-builder guard in moe(), rather than leaving it to
routing luck. Non-expert tensors (attention, dense first layer, router, shared expert,
norms, embed/lm_head) are left as plain F32 and quantized on load via argv ebits/dbits,
exactly like tiny_serve_fixture.py -- so a single fixture also exercises three formats
at once (e.g. int8 shared expert + int4/int2 routed experts) when run with the engine's
default ebits=8.

Used by test_mixed_format_moe.py to run the real `glm` binary's SCORE mode CPU-vs-Metal.
"""

import importlib.util
import json
import os
import struct
import sys

import numpy as np

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qc = _load_module("quant_container", os.path.join(TOOLS_DIR, "quant_container.py"))

VOCAB_BYTES = 256
SPECIALS = [
    "<|endoftext|>", "[gMASK]", "<sop>", "<|user|>",
    "<|assistant|>", "<|observation|>", "<think>", "</think>",
]
VOCAB_SIZE = 272

N_ROUTED_EXPERTS = 8
TOPK = 4                          # half of n_routed_experts
INT4_EXPERTS = (0, 1, 2, 3)       # always-selected half: two int4 + two int2
INT2_EXPERTS = (4, 5, 6, 7)
ALWAYS_SELECTED = INT4_EXPERTS[:2] + INT2_EXPERTS[:2]   # (0,1,4,5): guaranteed mixed top-4
BIAS_MAGNITUDE = 50.0              # >> any sigmoid(logit) in (0,1): bias alone decides top-K


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def build_config():
    return {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": 128,
        "intermediate_size": 64,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 4,
        "first_k_dense_replace": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_routed_experts": N_ROUTED_EXPERTS,
        "num_experts_per_tok": TOPK,
        "n_shared_experts": 1,
        "q_lora_rank": 64,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "index_topk": 4096,
        "index_head_dim": 16,
        "index_n_heads": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "rms_norm_eps": 1e-5,
        "eos_token_id": VOCAB_BYTES,
        "max_position_embeddings": 4096,
    }


def _tensors(cfg, rng):
    """name -> (f32 ndarray | (packed_u8, scales_f32) for pre-quantized experts)."""
    D = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    qk_head = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    kvr = cfg["kv_lora_rank"]
    t = {}

    def w(name, *shape, scale=0.05):
        t[name] = (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(name, *shape):
        t[name] = np.ones(shape, dtype=np.float32)

    w("model.embed_tokens.weight", cfg["vocab_size"], D)
    w("lm_head.weight", cfg["vocab_size"], D)
    ones("model.norm.weight", D)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        ones(p + "input_layernorm.weight", D)
        ones(p + "post_attention_layernorm.weight", D)
        w(p + "self_attn.q_a_proj.weight", cfg["q_lora_rank"], D)
        ones(p + "self_attn.q_a_layernorm.weight", cfg["q_lora_rank"])
        w(p + "self_attn.q_b_proj.weight", H * qk_head, cfg["q_lora_rank"])
        w(p + "self_attn.kv_a_proj_with_mqa.weight",
          kvr + cfg["qk_rope_head_dim"], D)
        ones(p + "self_attn.kv_a_layernorm.weight", kvr)
        w(p + "self_attn.kv_b_proj.weight",
          H * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), kvr)
        w(p + "self_attn.o_proj.weight", D, H * cfg["v_head_dim"])
        if i < cfg["first_k_dense_replace"]:
            w(p + "mlp.gate_proj.weight", cfg["intermediate_size"], D)
            w(p + "mlp.up_proj.weight", cfg["intermediate_size"], D)
            w(p + "mlp.down_proj.weight", D, cfg["intermediate_size"])
        else:
            w(p + "mlp.gate.weight", cfg["n_routed_experts"], D)
            # Deliberately overwhelming bias: sigmoid(logit) in (0,1), so a +/-50 split
            # alone determines the top-K winners regardless of router weights/hidden
            # state -- the routed selection is (0,1,4,5) on EVERY decode step, EVERY
            # MoE layer, guaranteeing (not merely likely-ing) a mixed int4/int2 block.
            bias = np.full(cfg["n_routed_experts"], -BIAS_MAGNITUDE, dtype=np.float32)
            for e in ALWAYS_SELECTED:
                bias[e] = BIAS_MAGNITUDE
            t[p + "mlp.gate.e_score_correction_bias"] = bias
            sI = cfg["moe_intermediate_size"] * cfg["n_shared_experts"]
            w(p + "mlp.shared_experts.gate_proj.weight", sI, D)
            w(p + "mlp.shared_experts.up_proj.weight", sI, D)
            w(p + "mlp.shared_experts.down_proj.weight", D, sI)
            for e in range(cfg["n_routed_experts"]):
                q = p + f"mlp.experts.{e}."
                bits = 4 if e in INT4_EXPERTS else 2
                for proj, (O, I) in (
                    ("gate_proj", (cfg["moe_intermediate_size"], D)),
                    ("up_proj", (cfg["moe_intermediate_size"], D)),
                    ("down_proj", (D, cfg["moe_intermediate_size"])),
                ):
                    raw = (rng.standard_normal((O, I)) * 0.3).astype(np.float32)
                    packed, scales = qc.quantize(raw, bits)
                    t[q + f"{proj}.weight"] = packed
                    t[q + f"{proj}.weight.qs"] = scales
    return t


def write_tokenizer(path):
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    added = [{"content": s, "id": VOCAB_BYTES + i, "special": True}
             for i, s in enumerate(SPECIALS)]
    doc = {"model": {"type": "BPE", "vocab": vocab, "merges": [],
                     "ignore_merges": True},
           "added_tokens": added}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)


def write_safetensors(path, tensors):
    """Same convention as quant_container.st_save: F32 tensors first (keeps every
    offset 4-byte aligned), names sorted within each dtype group."""
    order = sorted(tensors, key=lambda n: (tensors[n].dtype != np.float32, n))
    header, blobs, off = {}, [], 0
    for name in order:
        a = tensors[name]
        dt = "F32" if a.dtype == np.float32 else "U8"
        raw = a.tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for raw in blobs:
            f.write(raw)


def build(model_dir, seed=1234):
    """Create the fixture in model_dir (a pathlib.Path); returns model_dir."""
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config()
    with open(model_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    write_tokenizer(model_dir / "tokenizer.json")
    write_safetensors(model_dir / "model.safetensors",
                      _tensors(cfg, np.random.default_rng(seed)))
    return model_dir


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "glm_tiny_mixed")
    build(out)
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"built {out} ({total/1e6:.2f} MB); routed experts {INT4_EXPERTS}=int4 "
          f"{INT2_EXPERTS}=int2, guaranteed top-{TOPK} selection={ALWAYS_SELECTED}")
