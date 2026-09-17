#!/bin/bash
# Stage-0 downloads. Idempotent (hf download resumes).
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
dl() { $PY -c "
from huggingface_hub import hf_hub_download
import sys
p = hf_hub_download('$1', '$2', local_dir='$3')
print('OK', p, flush=True)
"; }
set -x
# original-format Wan2.1-1.3B (needed by the causal codebases: config + VAE + DiT init); T5 skipped, we use cached embeddings
dl Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth ckpt/Wan2.1-T2V-1.3B
dl Wan-AI/Wan2.1-T2V-1.3B diffusion_pytorch_model.safetensors ckpt/Wan2.1-T2V-1.3B
dl Wan-AI/Wan2.1-T2V-1.3B config.json ckpt/Wan2.1-T2V-1.3B
dl Wan-AI/Wan2.1-T2V-1.3B google/umt5-xxl/tokenizer.json ckpt/Wan2.1-T2V-1.3B
dl Wan-AI/Wan2.1-T2V-1.3B google/umt5-xxl/spiece.model ckpt/Wan2.1-T2V-1.3B
dl Wan-AI/Wan2.1-T2V-1.3B google/umt5-xxl/special_tokens_map.json ckpt/Wan2.1-T2V-1.3B
dl Wan-AI/Wan2.1-T2V-1.3B google/umt5-xxl/tokenizer_config.json ckpt/Wan2.1-T2V-1.3B
# students
dl gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt ckpt/self-forcing
dl zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt ckpt/causal-forcing
dl zhuhz22/Causal-Forcing chunkwise/ar_diffusion.pt ckpt/causal-forcing
dl tianweiy/CausVid bidirectional_checkpoint2/model.pt ckpt/causvid
dl tianweiy/CausVid autoregressive_checkpoint/model.pt ckpt/causvid
dl FastVideo/FastWan2.1-T2V-1.3B-Diffusers transformer/diffusion_pytorch_model.safetensors ckpt/fastwan
dl FastVideo/FastWan2.1-T2V-1.3B-Diffusers transformer/config.json ckpt/fastwan
dl FastVideo/FastWan2.1-T2V-1.3B-Diffusers scheduler/scheduler_config.json ckpt/fastwan
dl FastVideo/FastWan2.1-T2V-1.3B-Diffusers model_index.json ckpt/fastwan
echo ALL_DONE
