#!/bin/bash
# All checkpoints used in the paper (HF repo, file, local dir).  ~120 GB total.
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$(dirname "$0")/.."
PY=${PY:-python}
dl() { $PY -c "from huggingface_hub import hf_hub_download; print('OK', hf_hub_download('$1', '$2', local_dir='$3'), flush=True)"; }
for f in Wan2.1_VAE.pth diffusion_pytorch_model.safetensors config.json google/umt5-xxl/tokenizer.json google/umt5-xxl/spiece.model google/umt5-xxl/special_tokens_map.json google/umt5-xxl/tokenizer_config.json; do dl Wan-AI/Wan2.1-T2V-1.3B $f ckpt/Wan2.1-T2V-1.3B; done
for f in checkpoints/self_forcing_dmd.pt checkpoints/self_forcing_sid.pt checkpoints/self_forcing_sid_v2.pt checkpoints/self_forcing_gan.pt; do dl gdhe17/Self-Forcing $f ckpt/self-forcing; done
for f in chunkwise/causal_forcing.pt chunkwise/ar_diffusion.pt chunkwise/causal_cd.pt chunkwise/causal_ode.pt framewise/causal_forcing.pt framewise/ar_diffusion.pt; do dl zhuhz22/Causal-Forcing $f ckpt/causal-forcing; done
for f in bidirectional_checkpoint2/model.pt autoregressive_checkpoint/model.pt wan_causal_ode_checkpoint_model_003000/model.pt autoregressive_checkpoint_warp_4step_cfg2/model.pt; do dl tianweiy/CausVid $f ckpt/causvid; done
for f in transformer/diffusion_pytorch_model.safetensors transformer/config.json scheduler/scheduler_config.json model_index.json; do dl FastVideo/FastWan2.1-T2V-1.3B-Diffusers $f ckpt/fastwan; done
for f in rCM_Wan2.1_T2V_1.3B_480p.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF_Diffusion_c3-3.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM_c3-3.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt \
         Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM_c1-1.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF_Diffusion_c1-1.pt \
         Causal_rCM_Wan2.1_T2V_1.3B_480p_DF_Diffusion_c3-3.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-sCM_c3-3.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-sCM-init_SF-DMD_c3-3_step4.pt; do dl worstcoder/rcm-Wan $f ckpt/rcm; done
$PY scripts/convert_diffusers_to_wan.py ckpt/fastwan/transformer/diffusion_pytorch_model.safetensors ckpt/fastwan/wan_format.safetensors
echo ALL_DONE
