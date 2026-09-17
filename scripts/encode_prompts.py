"""Encode every prompt we may ever use with Wan's UMT5-XXL (diffusers weights, streamed to
GPU in bf16) and cache the un-padded [L, 4096] embeddings -> prompts/prompt_emb.pt.

Pool: 30 pilot prompts + official negative prompt + the full VBench list (946, for stage 3).
Zero-padding to 512 happens at load time (src/wan_env.py:CachedTextEncoder), which is
exactly what Wan's own WanTextEncoder does (`u[v:] = 0`).
"""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prompts_io import load_pilot_prompts, NEG   # noqa: E402
from transformers import UMT5EncoderModel, AutoTokenizer
from huggingface_hub import snapshot_download

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUT = os.path.join(ROOT, "prompts", "prompt_emb.pt")
emb = torch.load(OUT) if os.path.exists(OUT) else {}

want = [p for p, _ in load_pilot_prompts()] + [NEG]
vb = os.path.join(ROOT, "..", "tas", "data", "vbench_all.txt")
want += [l.strip() for l in open(vb) if l.strip()]
todo = list(dict.fromkeys(p for p in want if p not in emb))
print("cached", len(emb), "to encode", len(todo), flush=True)
if todo:
    tok = AutoTokenizer.from_pretrained(os.path.join(snapshot_download(MODEL), "tokenizer"))
    te = UMT5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder",
                                          dtype=torch.bfloat16, device_map="cuda").eval()
    with torch.no_grad():
        for i, p in enumerate(todo):
            ids = tok(p, padding="max_length", max_length=512, truncation=True,
                      add_special_tokens=True, return_tensors="pt")
            mask = ids.attention_mask.cuda()
            h = te(ids.input_ids.cuda(), attention_mask=mask).last_hidden_state[0]
            L = int(mask.sum())
            emb[p] = h[:L].to("cpu", torch.bfloat16).clone()
            if i < 40 or i % 100 == 0:
                print(f"  {L:3d} tok | {p[:80]}", flush=True)
    torch.save(emb, OUT)
print("saved", OUT, len(emb), "prompts,", os.path.getsize(OUT) / 1e6, "MB")
