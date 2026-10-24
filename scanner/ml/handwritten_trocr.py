"""Offline handwritten-prescription reader for MediScan (Model B).
Put in scanner/ml/.  HandwrittenTrOCR(model_dir).read_prescription(photo, matcher) -> dict.
Model B reads ONE handwritten word (or short phrase) per image, so a text FINDER (EasyOCR detector, no recognition) first cuts the page
into word boxes, the boxes are grouped into text lines, every box is read by Model B, and each line is turned into a medicine entry."""
import re
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import torch
from PIL import Image

FORMS = {"tab", "tab.", "tablet", "cap", "cap.", "capsule", "syp", "syp.", "syr", "syr.", "syrup", "inj", "inj.", "susp", "drop", "drops", "oint", "cream", "gel"}
RE_FREQ = re.compile(r"^(\d(?:\s?[-+]\s?\d){1,3}|od|bd|tds|tid|qid|hs|sos|prn|stat)$", re.I)       # 1+0+1, 1-0-1, OD, BD, TDS ...
RE_DURATION = re.compile(r"^(\d+)\s*(day|days|d|week|weeks|wk|month|months)$", re.I)
RE_DOSE = re.compile(r"^(\d+(?:\.\d+)?)\s*(mg|mcg|gm|g|ml|iu|%)?$", re.I)


def group_into_lines(boxes, tol=0.55):
    """boxes: [(x0,y0,x1,y1)]. Words whose vertical centres are close (relative to box height) belong to one line.
    Returns a list of lines, each a list of box indexes ordered left to right; lines ordered top to bottom."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    lines = []                                   # each: dict(cy, h, idx)
    for i in order:
        x0, y0, x1, y1 = boxes[i]; cy, h = (y0 + y1) / 2, max(1, y1 - y0)
        best = None
        for L in lines:
            if abs(L["cy"] - cy) <= tol * max(h, L["h"]):
                best = L; break
        if best is None: lines.append(dict(cy=cy, h=h, idx=[i]))
        else:
            best["idx"].append(i); n = len(best["idx"]); best["cy"] = (best["cy"] * (n - 1) + cy) / n; best["h"] = max(best["h"], h)
    lines.sort(key=lambda L: L["cy"])
    return [sorted(L["idx"], key=lambda i: boxes[i][0]) for L in lines]


def parse_line(words: List[str], matcher: Optional[Callable] = None, min_score=0.0):
    """words: the words of one line, left to right. matcher(text) -> list of candidates [{'name':..., 'score': 0-100, ...}] (best first).
    Finds the brand/medicine word(s); the rest are form / dose / frequency / duration."""
    form = dose = freq = duration = None; cands = []
    toks = [w.strip() for w in words if w and w.strip()]
    rest = []; UNITS = {"mg", "mcg", "gm", "g", "ml", "iu", "%"}; PER = {"day", "days", "d", "week", "weeks", "wk", "month", "months"}
    i = 0
    while i < len(toks):
        w = toks[i]; lw = w.lower().strip(".,:;"); nxt = toks[i + 1].lower().strip(".,:;") if i + 1 < len(toks) else ""
        if lw in FORMS: form = form or w.strip(".,:;")
        elif lw == "x": pass                                                    # filler in "x 5 days"
        elif RE_FREQ.match(w.replace(" ", "")) or RE_FREQ.match(lw): freq = freq or w
        elif RE_DURATION.match(lw): duration = duration or w
        elif re.fullmatch(r"\d+", lw) and nxt in PER: duration = duration or f"{w} {toks[i + 1]}"; i += 1      # "7 days" written as two words
        elif RE_DOSE.match(lw) and i > 0:
            dose = dose or (f"{w} {toks[i + 1]}" if nxt in UNITS else w); i += 1 if nxt in UNITS else 0
        else: rest.append((i, w))
        i += 1
    best = None
    for k, (i, w) in enumerate(rest):                        # try single words and neighbour pairs ("Napa Extra")
        for text in (w, f"{w} {rest[k+1][1]}" if k + 1 < len(rest) else None):
            if not text or len(text) < 3 or matcher is None: continue
            c = matcher(text) or []
            if c and (best is None or c[0].get("score", 0) > best["cands"][0].get("score", 0)):
                best = dict(read=text, cands=c)
    if best and best["cands"][0].get("score", 0) >= min_score:
        cands = best["cands"][:3]; read = best["read"]
    else:
        read = rest[0][1] if rest else (toks[0] if toks else ""); cands = []
    return dict(read=read, candidates=cands, form=form, dose=dose, frequency=freq, duration=duration, text=" ".join(toks))


class HandwrittenTrOCR:
    def __init__(self, model_dir, device=None, pad=6):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = TrOCRProcessor.from_pretrained(str(model_dir), use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).eval()
        if self.dev == "cuda": self.model = self.model.half()
        self.model = self.model.to(self.dev); self.pad, self._det = pad, None

    def _detector(self):
        if self._det is None:
            import easyocr
            self._det = easyocr.Reader(["en"], gpu=(self.dev == "cuda"), recognizer=False, verbose=False)
        return self._det

    @staticmethod
    def _tidy(t): return re.sub(r"\s+", " ", re.sub(r'["@~?^|`]', " ", t)).strip()      # keeps the writer's capital/small letters

    @torch.no_grad()
    def read_crops(self, crops, beams=4):
        if not crops: return []
        pv = self.proc.image_processor([c.convert("RGB") for c in crops], return_tensors="pt").pixel_values.to(self.dev)
        if self.dev == "cuda": pv = pv.half()
        tk = self.proc.tokenizer
        out = self.model.generate(pv, max_length=48, num_beams=beams, output_scores=True, return_dict_in_generate=True,
                                  decoder_start_token_id=tk.sep_token_id, eos_token_id=tk.sep_token_id, pad_token_id=tk.pad_token_id)
        texts = [self._tidy(t) for t in tk.batch_decode(out.sequences, skip_special_tokens=True)]
        sc = getattr(out, "sequences_scores", None); conf = torch.exp(sc).tolist() if sc is not None else [1.0] * len(texts)
        return list(zip(texts, conf))

    def find_boxes(self, img):
        horiz, free = self._detector().detect(np.array(img.convert("RGB")))
        boxes = [(int(a), int(c), int(b), int(d)) for a, b, c, d in (horiz[0] if horiz else [])]
        for pts in (free[0] if free else []):
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]; boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        return [b for b in boxes if b[2] - b[0] > 10 and b[3] - b[1] > 10]

    def read_prescription(self, photo, matcher=None, beams=4, min_score=0.0):
        """Returns {'ok', 'lines': [{'text','words':[{'text','conf','box'}]}], 'medicines': [parse_line result + 'line'], 'message'}."""
        from PIL import ImageOps
        img = photo if isinstance(photo, Image.Image) else Image.open(photo)
        img = ImageOps.exif_transpose(img).convert("RGB"); W, H = img.size
        boxes = self.find_boxes(img)
        if not boxes: return dict(ok=False, message="No handwriting found in the image", lines=[], medicines=[])
        crops = [img.crop((max(0, x0 - self.pad), max(0, y0 - self.pad), min(W, x1 + self.pad), min(H, y1 + self.pad))) for x0, y0, x1, y1 in boxes]
        res = []
        for i in range(0, len(crops), 16): res += self.read_crops(crops[i:i + 16], beams)
        lines, meds = [], []
        for li, idx in enumerate(group_into_lines(boxes)):
            words = [dict(text=res[i][0], conf=round(res[i][1], 3), box=boxes[i]) for i in idx if res[i][0]]
            if not words: continue
            lines.append(dict(text=" ".join(w["text"] for w in words), words=words))
            p = parse_line([w["text"] for w in words], matcher, min_score); p["line"] = len(lines) - 1; p["confidence"] = round(min(w["conf"] for w in words), 3)
            if p["candidates"] or p["read"]: meds.append(p)
        return dict(ok=True, message="", lines=lines, medicines=meds)
