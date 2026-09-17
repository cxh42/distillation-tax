#!/bin/bash
# Re-create third_party/ at the exact commits used in the paper and apply our sdpa patch.
set -e
cd "$(dirname "$0")/../third_party"
clone() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" fetch -q --depth 1 origin "$3" && git -C "$2" checkout -q "$3"; }
clone https://github.com/thu-ml/Causal-Forcing Causal-Forcing b4ffbd491b4f941e25bde66f425384c31f8dc9af
clone https://github.com/guandeh17/Self-Forcing  Self-Forcing  33593df3e81fa3ec10239271dd2c100facac6de1
clone https://github.com/tianweiy/CausVid        CausVid       adb6a5ecd07666b4d0290042915c8406e6d5ce22
clone https://github.com/NVlabs/rcm              rcm           ed3cb14dd936f92cdc9f9381af7369991509b41f
git -C Causal-Forcing apply --check ../patches/causal-forcing-attention-sdpa.patch 2>/dev/null && git -C Causal-Forcing apply ../patches/causal-forcing-attention-sdpa.patch
echo "third_party ready (Causal-Forcing patched: flash_attention -> torch sdpa fallback)"
