"""Contact sheet: rows = videos, cols = frames.   python sheet.py out.jpg a.mp4 b.mp4 ... [--label x,y] [--n 6]"""
import sys, argparse, numpy as np, imageio
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("videos", nargs="+")
ap.add_argument("--labels", default=None); ap.add_argument("--n", type=int, default=6); ap.add_argument("--w", type=int, default=256)
a = ap.parse_args()
labels = a.labels.split(",") if a.labels else [v.split("/")[-2] + "/" + v.split("/")[-1].replace(".mp4", "") for v in a.videos]
rows = []
for v, lab in zip(a.videos, labels):
    fr = imageio.mimread(v, memtest=False)
    idx = np.linspace(0, len(fr) - 1, a.n).round().astype(int)
    h = int(a.w * fr[0].shape[0] / fr[0].shape[1])
    tiles = [Image.fromarray(fr[i]).resize((a.w, h)) for i in idx]
    row = Image.new("RGB", (a.w * a.n, h + 16), "white")
    for j, t in enumerate(tiles):
        row.paste(t, (j * a.w, 16))
    ImageDraw.Draw(row).text((4, 2), f"{lab}   frames {list(idx)}", fill="black")
    rows.append(row)
W = max(r.width for r in rows); Hs = sum(r.height for r in rows)
sheet = Image.new("RGB", (W, Hs), "white")
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height
sheet.save(a.out, quality=88)
print("wrote", a.out, sheet.size)
