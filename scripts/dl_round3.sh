#!/bin/bash
# Round-3 downloads (user now has 300 GB): chunk-size axis (B), loss axis (C), extra causal-undistilled / CM cells.
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
dl() { for try in 1 2 3 4 5; do $PY -c "
from huggingface_hub import hf_hub_download
print('OK', hf_hub_download('$1', '$2', local_dir='$3'), flush=True)" && return 0; echo "retry $try $2"; sleep 20; done; }
# B: chunk size = 1 frame
dl zhuhz22/Causal-Forcing framewise/causal_forcing.pt ckpt/causal-forcing
dl zhuhz22/Causal-Forcing framewise/ar_diffusion.pt ckpt/causal-forcing
dl worstcoder/rcm-Wan Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt ckpt/rcm
dl worstcoder/rcm-Wan Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM_c1-1.pt ckpt/rcm
dl worstcoder/rcm-Wan Causal_rCM_Wan2.1_T2V_1.3B_480p_TF_Diffusion_c1-1.pt ckpt/rcm
# C: loss families in the causal branch (Self-Forcing lab)
dl gdhe17/Self-Forcing checkpoints/self_forcing_sid.pt ckpt/self-forcing
dl gdhe17/Self-Forcing checkpoints/self_forcing_sid_v2.pt ckpt/self-forcing
dl gdhe17/Self-Forcing checkpoints/self_forcing_gan.pt ckpt/self-forcing
# extra cells: CF-lab consistency distillation + ODE init (causal, non-DMD), CausVid ODE init, NVIDIA DF / sCM
dl zhuhz22/Causal-Forcing chunkwise/causal_cd.pt ckpt/causal-forcing
dl zhuhz22/Causal-Forcing chunkwise/causal_ode.pt ckpt/causal-forcing
dl tianweiy/CausVid wan_causal_ode_checkpoint_model_003000/model.pt ckpt/causvid
dl tianweiy/CausVid autoregressive_checkpoint_warp_4step_cfg2/model.pt ckpt/causvid
dl worstcoder/rcm-Wan Causal_rCM_Wan2.1_T2V_1.3B_480p_DF_Diffusion_c3-3.pt ckpt/rcm
dl worstcoder/rcm-Wan Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-sCM_c3-3.pt ckpt/rcm
dl worstcoder/rcm-Wan Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-sCM-init_SF-DMD_c3-3_step4.pt ckpt/rcm
echo ALL_DONE
