"""Blind sheets for the small human check of the diversity metric (guide stage 4).

For each prompt: rows = the four 2x2 cells in a random order (labelled A-D), cols = seeds, tile = the
middle frame (composition diversity) with the first frame inset top-left (so motion is visible too).
The rater ranks rows from most to least diverse; the row->variant key is written to key.json.

  python scripts/human_eval_sheets.py --cells T-full,AR-diff,S-bi-causvid,S-causvid --seeds 0-3 --out figures/human_eval
"""
import os, sys, json, argparse, random, numpy as np, imageio
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prompts_io import load_pilot_prompts, prompt_id

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="T-full,AR-diff,S-bi-causvid,S-causvid")
ap.add_argument("--seeds", default="0-3"); ap.add_argument("--out", default="figures/human_eval"); ap.add_argument("--w", type=int, default=240)
a = ap.parse_args()
cells = a.cells.split(","); s0, s1 = map(int, a.seeds.split("-")); seeds = list(range(s0, s1 + 1))
out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
rng = random.Random(2026); key = {}
plist = load_pilot_prompts()
for i, (prompt, tag) in enumerate(plist):
    pid = prompt_id(i)
    order = cells[:]; rng.shuffle(order); key[pid] = dict(zip("ABCD", order))
    h = int(a.w * 480 / 832); rows = []
    for lab, v in zip("ABCD", order):
        row = Image.new("RGB", (a.w * len(seeds) + 30, h), "white")
        ImageDraw.Draw(row).text((6, h // 2 - 6), lab, fill="black")
        for j, s in enumerate(seeds):
            p = os.path.join(ROOT, "corpus", v, f"{pid}_s{s}.mp4")
            if not os.path.exists(p):
                continue
            fr = imageio.mimread(p, memtest=False)
            mid = Image.fromarray(fr[len(fr) // 2]).resize((a.w, h))
            first = Image.fromarray(fr[0]).resize((a.w // 3, h // 3))
            mid.paste(first, (0, 0))
            row.paste(mid, (30 + j * a.w, 0))
        rows.append(row)
    W = rows[0].width; sheet = Image.new("RGB", (W, 20 + sum(r.height + 4 for r in rows)), "white")
    ImageDraw.Draw(sheet).text((6, 4), f"{pid} [{tag}]  {prompt}", fill="black")
    y = 20
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height + 4
    sheet.save(os.path.join(out, f"{pid}.jpg"), quality=88)
json.dump(key, open(os.path.join(out, "key.json"), "w"), indent=1)
open(os.path.join(out, "INSTRUCTIONS.md"), "w").write(
    "# Human check of the cross-seed diversity metric\n\nEach sheet shows one prompt; rows A-D are four generators in a random order, "
    "columns are different random seeds (tile = middle frame, inset = first frame).\n\n"
    "For each sheet, rank the rows from MOST to LEAST diverse across seeds (composition, subject, camera), e.g. `P04: B > A > D > C`.\n"
    "Write one line per sheet into `ratings.txt`. Do not look at key.json until done.\n")
print("wrote", len(key), "sheets to", out)
