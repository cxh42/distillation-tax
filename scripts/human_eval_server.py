"""Blind PAIRWISE human evaluation (Chinese UI, autoplaying videos), served locally.

  python scripts/human_eval_server.py [--port 8765]   ->  http://127.0.0.1:8765

Design: 6 cells, 30 prompts.  For each prompt 3 disjoint pairs (a perfect matching of the 6 cells); the 5
round-robin matchings of K6 cover all 15 cell pairs, each pair 6 times over the 30 prompts -> 90 comparisons.
Each comparison shows left/right blocks of 4 seeds (2x2, autoplay, muted, loop) and asks three 3-way questions:
  多样性 (which block differs more across its 4 seeds), 运动 (more natural real motion), 画质 (better image quality).
Left/right assignment is randomised.  Answers -> figures/human_eval/pairs.json; the blind key -> pairs_key.json.
"""
import os, sys, json, random, argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prompts_io import load_pilot_prompts, prompt_id

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--cells", default="T-full,AR-diff,S-bi-causvid,S-causvid,S-sf,S-rcm-tfdcm")
ap.add_argument("--seeds", default="0-3")
args = ap.parse_args()
CELLS = args.cells.split(","); s0, s1 = map(int, args.seeds.split("-")); SEEDS = list(range(s0, s1 + 1))
assert len(CELLS) == 6
OUT = os.path.join(ROOT, "figures", "human_eval"); os.makedirs(OUT, exist_ok=True)
ANS = os.path.join(OUT, "pairs.json"); KEY = os.path.join(OUT, "pairs_key.json")

# round-robin (circle method) perfect matchings of K6 -> 5 rounds x 3 disjoint pairs, all 15 pairs once
def matchings(n=6):
    ids = list(range(n)); out = []
    for r in range(n - 1):
        pairs = [(ids[0], ids[-1])] + [(ids[i], ids[-1 - i]) for i in range(1, n // 2)]
        out.append(pairs); ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return out

plist = load_pilot_prompts()
rng = random.Random(20260917)
rounds = matchings()
ZH = {"general": "通用", "fast": "快速运动", "multi": "多物体交互", "rare": "罕见组合", "count": "数量守恒", "physics": "物理事件"}
items, key = [], {}
for i, (p, tag) in enumerate(plist):
    pid = prompt_id(i)
    for (a, b) in rounds[i % 5]:
        left, right = (CELLS[a], CELLS[b]) if rng.random() < 0.5 else (CELLS[b], CELLS[a])
        cid = f"{pid}_{CELLS[a]}_vs_{CELLS[b]}"
        key[cid] = dict(left=left, right=right)
        items.append(dict(id=cid, pid=pid, prompt=p, tag=ZH.get(tag, tag),
                          left=[f"/corpus/{left}/{pid}_s{s}.mp4" for s in SEEDS], right=[f"/corpus/{right}/{pid}_s{s}.mp4" for s in SEEDS]))
rng.shuffle(items)
json.dump(key, open(KEY, "w"), indent=1)
answers = json.load(open(ANS)) if os.path.exists(ANS) else {}

PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>盲评：两两比较</title>
<style>
 body{font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f5f5f4;color:#111}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;z-index:5;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 .bar{height:6px;background:#e5e5e5;border-radius:3px;flex:1;min-width:160px}.bar>div{height:6px;background:#2a78d6;border-radius:3px}
 main{padding:12px 20px 40px;max-width:1500px;margin:0 auto}
 .prompt{font-size:17px;margin:4px 0 10px}.prompt b{color:#2a78d6}
 .pair{display:flex;gap:16px}.side{flex:1;background:#fff;border:2px solid #e2e2e2;border-radius:12px;padding:10px}
 .side h3{margin:0 0 8px;font-size:18px}.side.l h3{color:#2a78d6}.side.r h3{color:#eb6834}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.grid video{width:100%;border-radius:6px;background:#000;display:block}
 .qs{margin-top:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
 .q{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:10px}.q .t{font-weight:600;margin-bottom:4px}.q .h{font-size:12px;color:#666;margin-bottom:8px;min-height:32px}
 .opts{display:flex;gap:6px}.opts button{flex:1;padding:10px 4px;font-size:15px;border-radius:8px;border:1px solid #bbb;background:#fafafa;cursor:pointer}
 .opts button.on{background:#2a78d6;color:#fff;border-color:#2a78d6}.opts button.on.r{background:#eb6834;border-color:#eb6834}
 .btns{display:flex;gap:10px;margin-top:12px;align-items:center;flex-wrap:wrap}
 button.big{font-size:15px;padding:9px 18px;border-radius:8px;border:1px solid #bbb;background:#fff;cursor:pointer}
 button.primary{background:#2a78d6;color:#fff;border-color:#2a78d6}button:disabled{opacity:.45;cursor:not-allowed}
 .note{color:#666;font-size:13px}.done{font-size:20px;padding:40px;text-align:center}
</style></head><body>
<header><b>两两盲评</b><span id="pos"></span><div class="bar"><div id="fill" style="width:0%"></div></div><span class="note">已保存 <span id="saved">0</span>/<span id="total"></span></span></header>
<main>
<div class="prompt">[<span id="tag"></span>] <b id="ptext"></b> <span class="note">— 每边 4 个视频 = 同一 prompt 的 4 个随机种子</span></div>
<div class="pair">
 <div class="side l"><h3>左（L）</h3><div class="grid" id="gl"></div></div>
 <div class="side r"><h3>右（R）</h3><div class="grid" id="gr"></div></div>
</div>
<div class="qs">
 <div class="q"><div class="t">① 跨种子多样性：哪边 4 个视频彼此差别更大？</div><div class="h">看构图 / 主体 / 镜头是否各不相同。画面崩坏的也要评：崩坏方式各不相同也算“不同”。</div><div class="opts" data-q="diversity"><button data-v="L">左</button><button data-v="tie">差不多</button><button data-v="R" class="r">右</button></div></div>
 <div class="q"><div class="t">② 运动：哪边的运动更真实自然？</div><div class="h">有真实位移、符合 prompt；抖动、糊、崩坏、或几乎静止都算不自然。</div><div class="opts" data-q="motion"><button data-v="L">左</button><button data-v="tie">差不多</button><button data-v="R" class="r">右</button></div></div>
 <div class="q"><div class="t">③ 画质：哪边画面更好？</div><div class="h">清晰、结构正常、颜色正常（过艳/灰蒙/噪点/结构崩坏都扣分）。</div><div class="opts" data-q="quality"><button data-v="L">左</button><button data-v="tie">差不多</button><button data-v="R" class="r">右</button></div></div>
</div>
<div class="btns">
 <button class="big" id="prev">上一题</button><button class="big" id="replay">重播</button>
 <button class="big primary" id="next" disabled>保存并下一题 →</button>
 <span class="note">键盘：① 1/2/3　② 4/5/6　③ 7/8/9（左/差不多/右），回车保存</span>
</div>
</main>
<script>
const ITEMS = __ITEMS__; let ANS = __ANS__;
let idx = 0, cur = {};
const $ = s => document.querySelector(s);
$("#total").textContent = ITEMS.length;
function firstUnrated(){ for (let i=0;i<ITEMS.length;i++) if(!ANS[ITEMS[i].id]) return i; return ITEMS.length; }
function vids(list){ return list.map(v => `<video src="${v}" muted loop autoplay playsinline preload="auto"></video>`).join(""); }
function render(){
  if (idx >= ITEMS.length){ $("main").innerHTML = '<div class="done">全部完成，谢谢！结果在 figures/human_eval/pairs.json</div>'; return; }
  const it = ITEMS[idx]; cur = ANS[it.id] ? {...ANS[it.id]} : {};
  $("#pos").textContent = `第 ${idx+1} / ${ITEMS.length} 题`; $("#fill").style.width = (100*idx/ITEMS.length)+"%";
  $("#tag").textContent = it.tag; $("#ptext").textContent = it.prompt; $("#saved").textContent = Object.keys(ANS).length;
  $("#gl").innerHTML = vids(it.left); $("#gr").innerHTML = vids(it.right);
  paint();
}
function paint(){
  document.querySelectorAll(".opts").forEach(o => { const q = o.dataset.q; o.querySelectorAll("button").forEach(b => b.classList.toggle("on", cur[q] === b.dataset.v)); });
  $("#next").disabled = !(cur.diversity && cur.motion && cur.quality);
}
document.querySelectorAll(".opts button").forEach(b => b.onclick = () => { cur[b.parentElement.dataset.q] = b.dataset.v; paint(); });
$("#replay").onclick = () => document.querySelectorAll("video").forEach(v => { v.currentTime = 0; v.play(); });
$("#prev").onclick = () => { if (idx > 0){ idx--; render(); } };
$("#next").onclick = async () => {
  const it = ITEMS[idx]; const rec = { id: it.id, pid: it.pid, ...cur, ts: Date.now() };
  const r = await fetch("/save", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(rec) });
  if (!r.ok){ alert("保存失败"); return; }
  ANS[it.id] = rec; idx++; render();
};
const KEYS = {"1":["diversity","L"],"2":["diversity","tie"],"3":["diversity","R"],"4":["motion","L"],"5":["motion","tie"],"6":["motion","R"],"7":["quality","L"],"8":["quality","tie"],"9":["quality","R"]};
document.addEventListener("keydown", e => { if (KEYS[e.key]){ cur[KEYS[e.key][0]] = KEYS[e.key][1]; paint(); } if (e.key === "Enter" && !$("#next").disabled) $("#next").click(); });
idx = firstUnrated(); render();
</script></body></html>"""


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)).replace("__ANS__", json.dumps(answers)).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body); return
        if self.path.startswith("/corpus/"):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        if self.path != "/save":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", 0)); rec = json.loads(self.rfile.read(n))
        answers[rec["id"]] = rec
        json.dump(answers, open(ANS, "w"), indent=1, ensure_ascii=False)
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


print(f"{len(items)} comparisons ({len(CELLS)} cells, seeds {SEEDS}); {len(answers)} already answered; key -> {KEY}")
print(f"open  http://127.0.0.1:{args.port}")
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
