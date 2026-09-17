#!/bin/bash
# Second, within-one-lab 2x2 (NVIDIA Causal-rCM release, Wan2.1-1.3B, chunk pattern c3-3), 4 x 2.84 GB = 11.4 GB:
#   TF_Diffusion_c3-3            causal, multi-step, undistilled   (second AR-diff cell, different lab)
#   TF-dCM_c3-3                  causal + consistency distillation, teacher-forcing only (no self-rollout, not DMD)
#   TF-dCM-init_SF-DMD_c3-3_step4  + self-forcing DMD refinement    (causal + self-rollout)
#   rCM_Wan2.1_T2V_1.3B_480p     bidirectional consistency distillation (bi-distilled cell, non-DMD loss)
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
for f in Causal_rCM_Wan2.1_T2V_1.3B_480p_TF_Diffusion_c3-3.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM_c3-3.pt Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt rCM_Wan2.1_T2V_1.3B_480p.pt; do
  for try in 1 2 3 4 5; do
    $PY -c "
from huggingface_hub import hf_hub_download
print('OK', hf_hub_download('worstcoder/rcm-Wan', '$f', local_dir='ckpt/rcm'), flush=True)" && break
    echo "retry $try $f"; sleep 20
  done
done
echo ALL_DONE
