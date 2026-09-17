# Variant table (verbatim from src/wan_env.py:VARIANTS)

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
    # ----- causal few-step students (each with its authors' own inference schedule)
    "S-causvid": dict(family="ar_dmd", ckpt="causvid/autoregressive_checkpoint/model.pt", key="generator",
                      step_list=[1000, 757, 522], shift=8.0, warp=False, block=3),
    "S-sf":      dict(family="ar_dmd", ckpt="self-forcing/checkpoints/self_forcing_dmd.pt", key="generator_ema",
                      step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
    "S-cf":      dict(family="ar_dmd", ckpt="causal-forcing/chunkwise/causal_forcing.pt", key="generator",
                      step_list=[1000, 750, 500, 250], shift=5.0, warp=True, block=3),
}
```

* family bi_diff = bidirectional Wan2.1-1.3B base weights, UniPC (Wan's own FlowUniPC), `steps`, CFG `cfg`, timestep shift `shift`
* family bi_dmd  = bidirectional few-step DMD sampler (x0-prediction, re-noise with fresh noise), authors' step list / shift
* family ar_diff = causal chunk-wise (3 latent frames) multi-step UniPC with KV cache; Causal-Forcing `ar_diffusion.pt` = teacher-forcing-trained causal Wan (no distillation)
* family ar_dmd  = causal chunk-wise few-step DMD with KV cache (CausalInferencePipeline), authors' own schedule
* S-bi-fastwan runs FastWan's weights with *dense* attention; the model was trained with VSA (sparsity 0.8) and its `to_gate_compress` branch is dropped -> secondary S-bi only; primary S-bi is CausVid bidirectional (same codebase/recipe as S-causvid).
