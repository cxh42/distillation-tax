"""Cross-check: does diffusers' WanPipeline also give a blank frame at CFG=1?  (T-nocfg smoke looked gray)"""
import os, sys, torch, numpy as np, imageio
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prompts_io import NEG
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler
M = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
emb = torch.load(os.path.join(ROOT, "prompts", "prompt_emb.pt"))
def pad(e): return torch.cat([e, e.new_zeros(512 - e.shape[0], e.shape[1])])[None].cuda()
tr = WanTransformer3DModel.from_pretrained(M, subfolder="transformer", dtype=torch.bfloat16)
vae = AutoencoderKLWan.from_pretrained(M, subfolder="vae", dtype=torch.float32)
sch = UniPCMultistepScheduler.from_config(UniPCMultistepScheduler.from_pretrained(M, subfolder="scheduler").config, flow_shift=5.0)
pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=tr, vae=vae, scheduler=sch).to("cuda")
pipe.set_progress_bar_config(disable=True)
p = "a dog running happily"
for cfg, steps in [(1.0, 50), (5.0, 50)]:
    g = torch.Generator("cuda").manual_seed(0)
    kw = dict(prompt=None, prompt_embeds=pad(emb[p]), num_frames=81, height=480, width=832, generator=g,
              num_inference_steps=steps, guidance_scale=cfg, output_type="np")
    if cfg > 1: kw["negative_prompt_embeds"] = pad(emb[NEG])
    v = pipe(**kw).frames[0]
    fr = (np.clip(v, 0, 1) * 255).astype("uint8")
    print(f"cfg={cfg} steps={steps}: frame mean {fr.mean():.1f} std {fr.std():.1f}", flush=True)
    imageio.mimwrite(os.path.join(ROOT, "corpus", "smoke", f"diffusers_cfg{cfg}.mp4"), fr, fps=16, codec="libx264", quality=8, macro_block_size=1)
