"""Single inference harness for every configuration of the Distillation-Tax study.

All variants -- bidirectional teacher (any #steps / CFG), bidirectional DMD students,
causal multi-step AR-diffusion, causal few-step students -- run through the *same*
codebase (third_party/Causal-Forcing, which vendors Wan2.1 + CausVid + Self-Forcing),
the same VAE, the same attention kernel (torch sdpa) and the same cached UMT5-XXL
prompt embeddings.  This removes implementation differences as a confound.

The text encoder is never loaded here: `scripts/encode_prompts.py` encodes every
prompt once with the (identical) diffusers UMT5-XXL weights and caches the
un-padded [L, 4096] bf16 tensors; `CachedTextEncoder` zero-pads to 512 exactly like
Wan's own `WanTextEncoder` does.
"""
import os, sys, math, torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CF = os.path.join(ROOT, "third_party", "Causal-Forcing")
CKPT = os.path.join(ROOT, "ckpt")
EMB_FILE = os.path.join(ROOT, "prompts", "prompt_emb.pt")
TEXT_LEN = 512
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_io import NEG   # noqa: E402

# Common generation config (configs/common_generation.yaml) -- every variant uses this.
H, W, FRAMES, FPS = 480, 832, 81, 16
LAT_F, LAT_H, LAT_W, LAT_C = 21, 60, 104, 16


def _enter_cf():
    """Causal-Forcing uses cwd-relative paths (wan_models/...), so chdir there."""
    if CF not in sys.path:
        sys.path.insert(0, CF)
    link = os.path.join(CF, "wan_models", "Wan2.1-T2V-1.3B")
    if not os.path.exists(link):
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(os.path.join(CKPT, "Wan2.1-T2V-1.3B"), link)
    os.chdir(CF)


_enter_cf()
from omegaconf import OmegaConf                                   # noqa: E402
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper   # noqa: E402
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


class CachedTextEncoder(torch.nn.Module):
    def __init__(self, emb_file=EMB_FILE, device="cuda"):
        super().__init__()
        self.emb = torch.load(emb_file)
        self.dev = device

    def forward(self, text_prompts):
        out = []
        for p in text_prompts:
            e = self.emb[p]                                   # [L, 4096], real tokens only
            out.append(torch.cat([e, e.new_zeros(TEXT_LEN - e.shape[0], e.shape[1])]))
        return {"prompt_embeds": torch.stack(out).to(self.dev, torch.bfloat16)}

    def n_tokens(self, p):
        return int(self.emb[p].shape[0])


# --------------------------------------------------------------------------- variants
# Every entry is fully explicit so RESULTS.md can quote it.  `family` decides the
# sampler:  'bi_diff'  = bidirectional multi-step UniPC (teacher family)
#           'bi_dmd'   = bidirectional few-step x0-prediction + re-noise (DMD students)
#           'ar_diff'  = causal chunk-wise multi-step UniPC with KV cache (AR-diffusion)
#           'ar_dmd'   = causal chunk-wise few-step DMD with KV cache (causal students)
VARIANTS = {
    # ----- teacher family: base Wan2.1-T2V-1.3B weights, Wan's own default sampler
    "T-full":   dict(family="bi_diff", ckpt=None, steps=50, cfg=5.0, shift=5.0),
    "T-few":    dict(family="bi_diff", ckpt=None, steps=4,  cfg=5.0, shift=5.0),
    "T-nocfg":  dict(family="bi_diff", ckpt=None, steps=50, cfg=1.0, shift=5.0),
    "T-cfg3":   dict(family="bi_diff", ckpt=None, steps=50, cfg=3.0, shift=5.0),
    "T-cfg7.5": dict(family="bi_diff", ckpt=None, steps=50, cfg=7.5, shift=5.0),
    # ----- bidirectional DMD students (isolate distillation without causalisation)
    "S-bi-causvid": dict(family="bi_dmd", ckpt="causvid/bidirectional_checkpoint2/model.pt", key="generator",
                         step_list=[1000, 757, 522], shift=8.0, warp=False),
    "S-bi-fastwan": dict(family="bi_dmd", ckpt="fastwan/wan_format.safetensors", key=None,
                         step_list=[1000, 757, 522], shift=8.0, warp=False),
    # ----- causal multi-step (isolate causalisation without distillation)
    "AR-diff":      dict(family="ar_diff", ckpt="causal-forcing/chunkwise/ar_diffusion.pt", key="generator",
                         steps=50, cfg=5.0, shift=5.0, block=3),
    "AR-diff-cfg3": dict(family="ar_diff", ckpt="causal-forcing/chunkwise/ar_diffusion.pt", key="generator",
                         steps=50, cfg=3.0, shift=5.0, block=3),
    "AR-diff-cfg7.5": dict(family="ar_diff", ckpt="causal-forcing/chunkwise/ar_diffusion.pt", key="generator",
                           steps=50, cfg=7.5, shift=5.0, block=3),
    # ----- causal few-step students (each with its authors' own inference schedule)
    "S-causvid": dict(family="ar_dmd", ckpt="causvid/autoregressive_checkpoint/model.pt", key="generator",
                      step_list=[1000, 757, 522], shift=8.0, warp=False, block=3),
    "S-sf":      dict(family="ar_dmd", ckpt="self-forcing/checkpoints/self_forcing_dmd.pt", key="generator_ema",
                      step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    "S-cf":      dict(family="ar_dmd", ckpt="causal-forcing/chunkwise/causal_forcing.pt", key="generator",
                      step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    # ----- second, within-one-lab 2x2: NVIDIA Causal-rCM release (Wan2.1-1.3B, chunk pattern c3-3, bf16)
    # rCM few-step sampling = x0-prediction + fresh re-noise on a TrigFlow time grid; with shift=1 our
    # FlowMatchScheduler sigma(t)=t/1000 reproduces it (timesteps rounded to integers, <0.06% error).
    "AR-diff-rcm":  dict(family="ar_diff", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF_Diffusion_c3-3.pt", key=None, convert="rcm",
                         steps=50, cfg=5.0, shift=5.0, block=3),
    "S-rcm-tfdcm":  dict(family="ar_dmd", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM_c3-3.pt", key=None, convert="rcm",
                         step_list=[999, 937, 833, 625], shift=1.0, warp=False, block=3),      # sigma_max 1600 -> 1600/1601; mid_t 15/16 5/6 5/8
    "S-rcm-sfdmd":  dict(family="ar_dmd", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt", key=None, convert="rcm",
                         step_list=[999, 937, 833, 625], shift=1.0, warp=False, block=3),
    "S-bi-rcm":     dict(family="bi_dmd", ckpt="rcm/rCM_Wan2.1_T2V_1.3B_480p.pt", key=None, convert="rcm",
                         step_list=[988, 934, 853, 609], shift=1.0, warp=False, init_scale=0.98765),  # sigma_max 80; TrigFlow mid_t 1.5,1.4,1.0
    # ----- round 3 (2026-09-17): chunk-size axis (B), loss axis (C), extra causal cells
    # B: frame-wise (chunk = 1 latent frame) versions of the causal cells
    "S-cf-fw":         dict(family="ar_dmd", ckpt="causal-forcing/framewise/causal_forcing.pt", key="generator_ema",
                            step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=1),
    "AR-diff-fw":      dict(family="ar_diff", ckpt="causal-forcing/framewise/ar_diffusion.pt", key="generator",
                            steps=50, cfg=5.0, shift=5.0, block=1),
    "S-rcm-sfdmd-fw":  dict(family="ar_dmd", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt", key=None, convert="rcm",
                            step_list=[999, 937, 833, 625], shift=1.0, warp=False, block=1),
    "S-rcm-tfdcm-fw":  dict(family="ar_dmd", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM_c1-1.pt", key=None, convert="rcm",
                            step_list=[999, 937, 833, 625], shift=1.0, warp=False, block=1),
    "AR-diff-rcm-fw":  dict(family="ar_diff", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF_Diffusion_c1-1.pt", key=None, convert="rcm",
                            steps=50, cfg=5.0, shift=5.0, block=1),
    # C: other distillation losses in the causal self-rollout recipe (Self-Forcing lab)
    "S-sf-sid":        dict(family="ar_dmd", ckpt="self-forcing/checkpoints/self_forcing_sid.pt", key="generator_ema",
                            step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    "S-sf-sid2":       dict(family="ar_dmd", ckpt="self-forcing/checkpoints/self_forcing_sid_v2.pt", key="generator_ema",
                            step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    "S-sf-gan":        dict(family="ar_dmd", ckpt="self-forcing/checkpoints/self_forcing_gan.pt", key="generator_ema",
                            step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    # extra causal cells: CF-lab consistency distillation and ODE-regression init (both non-DMD), CausVid ODE init + warp-4-step DMD
    "S-cf-cd":         dict(family="ar_dmd", ckpt="causal-forcing/chunkwise/causal_cd.pt", key="generator",
                            step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    "S-cf-ode":        dict(family="ar_dmd", ckpt="causal-forcing/chunkwise/causal_ode.pt", key="generator",
                            step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    "S-causvid-ode":   dict(family="ar_dmd", ckpt="causvid/wan_causal_ode_checkpoint_model_003000/model.pt", key="generator",
                            step_list=[1000, 757, 522], shift=8.0, warp=False, block=3),
    "S-causvid-w4":    dict(family="ar_dmd", ckpt="causvid/autoregressive_checkpoint_warp_4step_cfg2/model.pt", key="generator",
                            step_list=[1000, 750, 500, 250], shift=8.0, warp=True, block=3),
    # NVIDIA: diffusion-forcing causal (undistilled) and the continuous-time CM variants
    "AR-diff-rcm-df":  dict(family="ar_diff", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_DF_Diffusion_c3-3.pt", key=None, convert="rcm",
                            steps=50, cfg=5.0, shift=5.0, block=3),
    "S-rcm-tfscm":     dict(family="ar_dmd", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-sCM_c3-3.pt", key=None, convert="rcm",
                            step_list=[999, 937, 833, 625], shift=1.0, warp=False, block=3),
    "S-rcm-sfdmd-scm": dict(family="ar_dmd", ckpt="rcm/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-sCM-init_SF-DMD_c3-3_step4.pt", key=None, convert="rcm",
                            step_list=[999, 937, 833, 625], shift=1.0, warp=False, block=3),
}


def _load_generator_weights(gen, spec):
    if spec["ckpt"] is None:
        return
    path = os.path.join(CKPT, spec["ckpt"])
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
        sd = {("model." + k if not k.startswith("model.") else k): v for k, v in sd.items()}
    else:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if spec.get("key"):
            k = spec["key"]
            if k not in sd:
                k = "generator_ema" if k == "generator" and "generator_ema" in sd else "generator"
                print(f"[wan_env] key {spec['key']!r} missing in {path}; using {k!r}")
            sd = sd[k]
    if spec.get("convert") == "rcm":
        sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.") and not k.startswith("net.accum_")}
        w = sd["patch_embedding.weight"]                      # rcm uses nn.Linear over (c kt kh kw); Wan uses Conv3d
        sd["patch_embedding.weight"] = w.reshape(w.shape[0], 16, 1, 2, 2)
        sd = {"model." + k: v.to(torch.float32) for k, v in sd.items()}
    sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in sd.items()}
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert not missing, missing[:5]


class Runner:
    """Holds one variant on the GPU; `generate(prompt, seed)` -> uint8 [F, H, W, 3]."""

    def __init__(self, name, device="cuda"):
        self.name, self.spec, self.dev = name, VARIANTS[name], device
        spec = self.spec
        causal = spec["family"].startswith("ar")
        self.gen = WanDiffusionWrapper(timestep_shift=spec["shift"], is_causal=causal)
        _load_generator_weights(self.gen, spec)
        self.gen.to(dtype=torch.bfloat16, device=device).eval()
        self.te = CachedTextEncoder(device=device)
        self.vae = WanVAEWrapper().to(dtype=torch.bfloat16, device=device)
        self.pipe = None
        if causal:
            cfg = OmegaConf.load(os.path.join(CF, "configs", "default_config.yaml"))
            cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(
                num_train_timestep=1000, timestep_shift=spec["shift"], negative_prompt=NEG,
                guidance_scale=spec.get("cfg", 1.0), num_frame_per_block=spec["block"],
                denoising_step_list=spec.get("step_list", [1000]), warp_denoising_step=spec.get("warp", False),
                independent_first_frame=False, context_noise=0, i2v=False)))
            if spec["family"] == "ar_dmd":
                from pipeline import CausalInferencePipeline
                self.pipe = CausalInferencePipeline(cfg, device, generator=self.gen, text_encoder=self.te, vae=self.vae)
            else:
                from pipeline import CausalDiffusionInferencePipeline
                self.pipe = CausalDiffusionInferencePipeline(cfg, device, generator=self.gen, text_encoder=self.te, vae=self.vae)
                self.pipe.sampling_steps = spec["steps"]
        elif spec["family"] == "bi_dmd":
            sched = self.gen.get_scheduler()
            steps = torch.tensor(spec["step_list"], dtype=torch.long)
            if spec.get("warp"):
                ts = torch.cat((sched.timesteps.cpu(), torch.tensor([0.0])))
                steps = ts[1000 - steps]
            self.step_list = steps

    # ----------------------------------------------------------------- samplers
    @torch.no_grad()
    def _bi_diff(self, noise, cond, uncond, steps, cfg, shift):
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched.set_timesteps(steps, device=noise.device, shift=shift)
        lat = noise
        for t in sched.timesteps:
            ts = t * torch.ones([1, LAT_F], device=noise.device, dtype=torch.float32)
            v_c, _ = self.gen(lat, cond, ts)
            if cfg > 1.0:
                v_u, _ = self.gen(lat, uncond, ts)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = v_c
            lat = sched.step(v.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False)[0].squeeze(0)
        return lat

    @torch.no_grad()
    def _bi_dmd(self, noise, cond, g):
        sched = self.gen.get_scheduler()
        x = noise * self.spec.get("init_scale", 1.0)
        for i, t in enumerate(self.step_list):
            ts = torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * int(t)
            _, x0 = self.gen(noisy_image_or_video=x, conditional_dict=cond, timestep=ts)
            if i < len(self.step_list) - 1:
                nt = int(self.step_list[i + 1]) * torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device)
                eps = torch.randn(x0.shape, generator=g, device=noise.device, dtype=x0.dtype)
                x = sched.add_noise(x0.flatten(0, 1), eps.flatten(0, 1), nt.flatten(0, 1)).unflatten(0, x0.shape[:2])
        return x0

    @torch.no_grad()
    def generate(self, prompt, seed, return_latent=False):
        g = torch.Generator(self.dev).manual_seed(seed)
        noise = torch.randn([1, LAT_F, LAT_C, LAT_H, LAT_W], generator=g, device=self.dev, dtype=torch.bfloat16)
        spec = self.spec
        if spec["family"] == "bi_diff":
            cond = self.te([prompt]); uncond = self.te([NEG]) if spec["cfg"] > 1 else None
            lat = self._bi_diff(noise, cond, uncond, spec["steps"], spec["cfg"], spec["shift"])
            video = self.vae.decode_to_pixel(lat)
        elif spec["family"] == "bi_dmd":
            lat = self._bi_dmd(noise, self.te([prompt]), g)
            video = self.vae.decode_to_pixel(lat)
        elif spec["family"] == "ar_diff":
            video, lat = self.pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True)
        else:
            torch.manual_seed(seed)   # the causal DMD loop draws its re-noise from the global RNG
            video, lat = self.pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True)
        video = (video.float() * 0.5 + 0.5).clamp(0, 1) if spec["family"].startswith("bi") else video.float()
        frames = (video[0].permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
        return (frames, lat) if return_latent else frames


def write_mp4(frames, path, fps=FPS):
    import imageio
    frames = np.ascontiguousarray(frames)
    # ffmpeg's demuxer skips an "ID3" tag if the raw stream starts with those bytes (a first pixel of RGB
    # 73,68,51 does exactly that and truncates the whole video) -> nudge one LSB.  Happened once in ~4k videos.
    if bytes(frames.reshape(-1)[:3]) == b"ID3":
        frames = frames.copy(); frames[0, 0, 0, 0] ^= 1
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
