"""A4 semantic / compositional adherence.

  vqa_yes     BLIP-2 (OPT-2.7b, cached locally) yes/no VQA on 8 uniformly spaced frames:
              P(yes)/(P(yes)+P(no)) from the first answer token, averaged over the prompt's
              pre-registered questions (prompts/vqa_questions.json) and frames.  Counts and
              rare compositions are only checkable this way.
  vqascore    VQAScore-style generic question with the same model:
              "Does this image show {prompt}?"  (weaker than CLIP-FlanT5-XXL, which we cannot
              download on the metered connection; noted in RESULTS.md).
  clip_t      CLIP ViT-L/14 text-image cosine averaged over 16 frames.
"""
import json, os, numpy as np, torch, torch.nn.functional as F

_m = {}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QFILE = os.path.join(ROOT, "prompts", "vqa_questions.json")


def _blip(device):
    if "blip" not in _m:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        m = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", dtype=torch.float16).to(device).eval()
        pr = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        tok = pr.tokenizer
        ids = dict(yes=[tok(" yes", add_special_tokens=False).input_ids[0], tok("yes", add_special_tokens=False).input_ids[0]],
                   no=[tok(" no", add_special_tokens=False).input_ids[0], tok("no", add_special_tokens=False).input_ids[0]])
        _m["blip"] = (m, pr, ids)
    return _m["blip"]


def _clipL(device):
    if "clipL" not in _m:
        from transformers import CLIPModel, CLIPProcessor
        _m["clipL"] = (CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).eval().to(device),
                       CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"))
    return _m["clipL"]


def _idx(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


@torch.no_grad()
def blip_yes_prob(frames, question, device="cuda"):
    m, pr, ids = _blip(device)
    x = pr(images=list(frames), text=[f"Question: {question} Answer:"] * len(frames), return_tensors="pt", padding=True).to(device)
    x["pixel_values"] = x["pixel_values"].half()
    logits = m(**x).logits[:, -1].float()
    p = logits.softmax(-1)
    py = p[:, ids["yes"]].sum(-1); pn = p[:, ids["no"]].sum(-1)
    return (py / (py + pn + 1e-9)).cpu().numpy()          # per frame


@torch.no_grad()
def clip_t(frames, prompt, device="cuda"):
    m, pr = _clipL(device)
    x = pr(images=list(frames), text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    ie = F.normalize(m.get_image_features(pixel_values=x["pixel_values"].half()).pooler_output.float(), dim=-1)
    te = F.normalize(m.get_text_features(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).pooler_output.float(), dim=-1)
    return float((ie @ te.T).mean())


@torch.no_grad()
def semantic_metrics(frames, prompt, pid, device="cuda"):
    qs = json.load(open(QFILE))
    fr8 = frames[_idx(len(frames), 8)]
    out = dict(clip_t=clip_t(frames[_idx(len(frames), 16)], prompt, device),
               vqascore=float(blip_yes_prob(fr8, f"Does this image show {prompt}?", device).mean()))
    if pid in qs:
        per_q = [float(blip_yes_prob(fr8, q, device).mean()) for q in qs[pid]]
        out["vqa_yes"] = float(np.mean(per_q)); out["vqa_per_q"] = per_q
        out["vqa_min_q"] = float(np.min(per_q))     # the hardest attribute (counts / rare combos)
    return out
