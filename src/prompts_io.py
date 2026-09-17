"""Prompt-set loading (light-weight; no torch)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDIR = os.path.join(ROOT, "prompts")
# official Wan2.1 negative prompt (wan/configs/shared_config.py)
NEG = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，'
       'JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，'
       '手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')


def load_pilot_prompts():
    """-> list of (prompt, subset_tag); tag 'general' or the distill-sensitive category."""
    out = []
    for l in open(os.path.join(PDIR, "general.txt")):
        if l.strip():
            out.append((l.strip(), "general"))
    for l in open(os.path.join(PDIR, "distill_sensitive.txt")):
        if l.strip():
            tag, p = l.rstrip("\n").split("\t")
            out.append((p.strip(), tag))
    return out


def prompt_id(i):
    return f"P{i:02d}"
