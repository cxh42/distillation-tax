# Variant table (verbatim from `src/wan_env.py:VARIANTS`, 32 entries; smoke-only cells `S-causvid-w4`, `S-rcm-tfscm`, `S-rcm-sfdmd-scm`, `AR-diff-fw`, `AR-diff-rcm-fw`, `AR-diff-rcm-df` were registered but not run at scale)

```python
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
```

* `bi_diff`  bidirectional Wan2.1-1.3B base weights, Wan's own FlowUniPC sampler, `steps`, CFG `cfg`, timestep shift `shift`
* `bi_dmd`   bidirectional few-step sampler (x0-prediction, re-noise with fresh noise) with the authors' step list / shift; rCM uses its TrigFlow grid (`init_scale`)
* `ar_diff`  causal chunk-wise (block latent frames) multi-step UniPC with KV cache (CausalDiffusionInferencePipeline)
* `ar_dmd`   causal chunk-wise few-step sampler with KV cache (CausalInferencePipeline), authors' schedule; `warp=True` maps the step list through the shifted schedule as Self-Forcing does
* `convert='rcm'` strips the `net.` prefix and reshapes the Linear patch embedding of NVIDIA checkpoints to Wan's Conv3d layout
* `S-bi-fastwan` runs FastWan's weights with *dense* attention (trained with VSA 0.8; `to_gate_compress` dropped) -> secondary cell only
