"""Build a DENSE-ONLY, single-layer, real-GLM-5.2-ATTENTION-DIMS fixture model dir, with a
FULL DSA indexer on its one layer, loadable by the real, unmodified C engine.

Why this fixture is NOT "tiny" the way tiny_serve_fixture.py's is: the S>4 Metal-prefill-
attention gate in glm.c's attention() hardcodes GLM-5.2's real attention shape as part of
its own precondition --

    D==6144 && H==64 && c->kv_lora==512 && c->qk_nope==192 && c->qk_rope==64 && vh==256
    && l->kv_b.fmt==2

-- so NO fixture smaller than these exact dims can ever reach that kernel; a "tiny" 128-
hidden fixture like tiny_serve_fixture.py's is architecturally incapable of exercising it.
This module builds the smallest model that CAN: real attention dims (hidden_size=6144,
num_attention_heads=64, q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
qk_rope_head_dim=64, v_head_dim=256 -- all mandatory, unshrinkable), but otherwise minimal:
ONE layer, DENSE only (first_k_dense_replace >= num_hidden_layers, so no MoE/expert weights
are needed at all -- attention()'s S>4 gate does not care what the MLP is), tiny vocab
(byte-level, 272 tokens), small dense-MLP width. A FULL DSA indexer (indexer_types=["full"])
sits on that one layer, mirroring GLM-5.2's own layer 0.

On-disk size is dominated entirely by the mandatory attention dims (o_proj alone is a real
6144x16384 matrix, ~100M params) -- roughly 175M f32 params, ~700MB on disk. Still many
orders of magnitude smaller than the 744B production model ("no 744B" per the spec), but not
"tiny" in the sub-2MB sense of tiny_serve_fixture.py; this is the necessary, minimal cost of
a fixture that can actually drive the repaired gate on its real code path, per the reviewer-
endorsed requirement that the greedy-parity regression (c/tests/test_score_greedy_parity.py)
must reach it on both the CPU and Metal legs, not merely exercise a stand-in.

Reuses tiny_serve_fixture.py's bytes_to_unicode/write_tokenizer/write_safetensors verbatim
(generic, dimension-independent helpers) -- only build_config() and the tensor set differ.
"""

import importlib.util
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tsf = _load_module("tiny_serve_fixture", os.path.join(_THIS_DIR, "tiny_serve_fixture.py"))
write_tokenizer = _tsf.write_tokenizer
write_safetensors = _tsf.write_safetensors
VOCAB_BYTES = _tsf.VOCAB_BYTES
VOCAB_SIZE = _tsf.VOCAB_SIZE

# Mandatory: the EXACT dims attention()'s S>4 Metal-prefill gate checks (glm.c). None of
# these may shrink without making the fixture unable to reach the kernel under test.
HIDDEN = 6144
N_HEADS = 64
Q_LORA = 2048
KV_LORA = 512
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256

# Free to minimize: does not affect the gate.
DENSE_INTER = 256
INDEX_HD = 32
INDEX_NH = 2
INDEX_TOPK = 4096   # generous vs. any short test context, so dsa_may_select stays 0 (engage)


def build_config():
    return {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN,
        "intermediate_size": DENSE_INTER,
        "moe_intermediate_size": 32,          # unused (dense-only model), any valid value
        "num_hidden_layers": 1,
        "first_k_dense_replace": 1,           # == num_hidden_layers: ALL layers dense, no MoE weights
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_HEADS,
        "n_routed_experts": 1,                # unused, schema-valid placeholder
        "num_experts_per_tok": 1,
        "n_shared_experts": 0,
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA,
        "qk_nope_head_dim": QK_NOPE,
        "qk_rope_head_dim": QK_ROPE,
        "v_head_dim": V_HEAD,
        "index_topk": INDEX_TOPK,
        "index_head_dim": INDEX_HD,
        "index_n_heads": INDEX_NH,
        "indexer_types": ["full"],            # layer 0 = FULL DSA, mirrors GLM-5.2
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "rms_norm_eps": 1e-5,
        "eos_token_id": VOCAB_BYTES,
        "max_position_embeddings": 8192,
    }


def _tensors(cfg, rng):
    D = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    qk_head = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    kvr = cfg["kv_lora_rank"]
    I = cfg["intermediate_size"]
    inh, ihd = cfg["index_n_heads"], cfg["index_head_dim"]
    t = {}

    def w(name, *shape, scale=0.02):
        t[name] = (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(name, *shape):
        t[name] = np.ones(shape, dtype=np.float32)

    def zeros(name, *shape):
        t[name] = np.zeros(shape, dtype=np.float32)

    w("model.embed_tokens.weight", cfg["vocab_size"], D)
    w("lm_head.weight", cfg["vocab_size"], D)
    ones("model.norm.weight", D)

    p = "model.layers.0."
    ones(p + "input_layernorm.weight", D)
    ones(p + "post_attention_layernorm.weight", D)
    w(p + "self_attn.q_a_proj.weight", cfg["q_lora_rank"], D)
    ones(p + "self_attn.q_a_layernorm.weight", cfg["q_lora_rank"])
    w(p + "self_attn.q_b_proj.weight", H * qk_head, cfg["q_lora_rank"])
    w(p + "self_attn.kv_a_proj_with_mqa.weight", kvr + cfg["qk_rope_head_dim"], D)
    ones(p + "self_attn.kv_a_layernorm.weight", kvr)
    w(p + "self_attn.kv_b_proj.weight", H * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), kvr)
    w(p + "self_attn.o_proj.weight", D, H * cfg["v_head_dim"])
    # FULL DSA indexer (layer 0) -- see model_init's PI(...) tensor names.
    w(p + "self_attn.indexer.wq_b.weight", inh * ihd, cfg["q_lora_rank"])
    w(p + "self_attn.indexer.wk.weight", ihd, D)
    w(p + "self_attn.indexer.weights_proj.weight", inh, D)
    ones(p + "self_attn.indexer.k_norm.weight", ihd)
    zeros(p + "self_attn.indexer.k_norm.bias", ihd)
    # Dense MLP (first_k_dense_replace covers this layer -- no routed/shared experts at all).
    w(p + "mlp.gate_proj.weight", I, D)
    w(p + "mlp.up_proj.weight", I, D)
    w(p + "mlp.down_proj.weight", D, I)
    return t


def build(model_dir, seed=7):
    """Create the fixture in model_dir (a pathlib.Path); returns model_dir. Idempotent-ish:
    skips regeneration if config.json already matches (cheap re-run for repeated test
    invocations against the same cache dir -- this fixture is ~700MB, not free to rebuild)."""
    cfg = build_config()
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        try:
            if json.loads(cfg_path.read_text()) == cfg and (model_dir / "model.safetensors").exists():
                return model_dir
        except Exception:
            pass
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    write_tokenizer(model_dir / "tokenizer.json")
    write_safetensors(model_dir / "model.safetensors", _tensors(cfg, np.random.default_rng(seed)))
    return model_dir


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "glm_score_gate_fixture")
    build(out)
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"built {out} ({total/1e6:.1f} MB)")
