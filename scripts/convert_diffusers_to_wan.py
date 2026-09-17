"""diffusers WanTransformer3DModel state dict -> original Wan `WanModel` key layout.

Inverse of diffusers/scripts/convert_wan_to_diffusers.py (TRANSFORMER_KEYS_RENAME_DICT).
Usage:
  python convert_diffusers_to_wan.py <in.safetensors> <out.safetensors>
  python convert_diffusers_to_wan.py --verify   # convert the diffusers teacher and diff vs ckpt/Wan2.1-T2V-1.3B
"""
import os, sys, glob, torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENAME = [  # (diffusers substring, wan substring) -- order matters for the norm2/norm3 swap
    ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
    ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
    ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
    ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
    ("condition_embedder.time_proj", "time_projection.1"),
    ("ffn.net.0.proj", "ffn.0"),
    ("ffn.net.2", "ffn.2"),
    ("attn1.to_q", "self_attn.q"), ("attn1.to_k", "self_attn.k"), ("attn1.to_v", "self_attn.v"),
    ("attn1.to_out.0", "self_attn.o"), ("attn1.norm_q", "self_attn.norm_q"), ("attn1.norm_k", "self_attn.norm_k"),
    ("attn2.to_q", "cross_attn.q"), ("attn2.to_k", "cross_attn.k"), ("attn2.to_v", "cross_attn.v"),
    ("attn2.to_out.0", "cross_attn.o"), ("attn2.norm_q", "cross_attn.norm_q"), ("attn2.norm_k", "cross_attn.norm_k"),
    (".norm2.", ".norm__tmp."), (".norm3.", ".norm2."), (".norm__tmp.", ".norm3."),
]


def convert(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        if nk == "scale_shift_table":
            nk = "head.modulation"
        elif nk.startswith("proj_out."):
            nk = nk.replace("proj_out.", "head.head.")
        elif ".scale_shift_table" in nk:
            nk = nk.replace(".scale_shift_table", ".modulation")
        for a, b in RENAME:
            nk = nk.replace(a, b)
        out[nk] = v.contiguous()
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--verify":
        src = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/*/transformer/*.safetensors"))
        sd = {}
        for f in src:
            sd.update(load_file(f))
        conv = convert(sd)
        ref = load_file(os.path.join(ROOT, "ckpt", "Wan2.1-T2V-1.3B", "diffusion_pytorch_model.safetensors"))
        print("converted", len(conv), "ref", len(ref))
        missing = set(ref) - set(conv); extra = set(conv) - set(ref)
        print("missing in converted:", sorted(missing)[:10], "extra:", sorted(extra)[:10])
        worst = 0.0
        for k in ref:
            if k in conv:
                a, b = conv[k].float(), ref[k].float()
                if a.shape != b.shape:
                    print("SHAPE MISMATCH", k, a.shape, b.shape); continue
                worst = max(worst, (a - b).abs().max().item())
        print("max |diff| over all shared tensors:", worst, "(ref dtype", next(iter(ref.values())).dtype,
              "src dtype", next(iter(sd.values())).dtype, ")")
    else:
        sd = load_file(sys.argv[1])
        conv = convert(sd)
        save_file(conv, sys.argv[2])
        print("wrote", sys.argv[2], len(conv), "tensors")
