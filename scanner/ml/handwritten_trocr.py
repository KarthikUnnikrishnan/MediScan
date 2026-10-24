"""Offline handwritten-prescription reader for MediScan (Model B).
Put in scanner/ml/.  HandwrittenTrOCR(model_dir).read_prescription(photo, matcher) -> dict.
Model B reads ONE handwritten word (or short phrase) per image, so a text FINDER (EasyOCR detector, no recognition) first cuts the page
into word boxes, the boxes are grouped into text lines, every box is read by Model B, and each line is turned into a medicine entry."""
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
import numpy as np
import torch
from PIL import Image

FORMS = {
    "tab", "tab.", "tabs", "tablet", "tablets",
    "cap", "cap.", "caps", "capsule", "capsules",
    "syp", "syp.", "syr", "syr.", "syrup",
    "inj", "inj.", "injection",
    "susp", "susp.", "suspension",
    "drop", "drops", "oint", "cream", "gel", "lotion", "solution", "sol"
}
RE_FREQ = re.compile(r"^(\d(?:\s?[-+]\s?\d){1,3}|od|bd|tds|tid|qid|hs|sos|prn|stat)$", re.I)       # 1+0+1, 1-0-1, OD, BD, TDS ...
RE_DURATION = re.compile(r"^(\d+)\s*(day|days|d|week|weeks|wk|month|months)$", re.I)
RE_DOSE = re.compile(r"^(\d+(?:\.\d+)?)\s*(mg|mcg|gm|g|ml|iu|%)?$", re.I)
RE_DOSE_UNIT = re.compile(r"\b\d+(?:\.\d+)?\s*(mg|mcg|gm|g|ml|iu|%)\b", re.I)

# B2: Stop list for prescription mode (case-insensitive, dots/commas/slashes stripped)
RX_STOP_WORDS = {
    "dr", "doctor", "ms", "mr", "mrs", "miss", "name", "age", "sex", "gender",
    "date", "address", "sign", "signature", "rx", "hospital", "clinic", "centre", "center", "patient",
    "weight", "wt", "bp", "pulse", "pr", "temp", "spo2", "advice", "adv",
    "diagnosis", "dx", "history", "follow", "review", "reg", "no", "phone", "ph",
    "mbbs", "md", "dnb", "registration",
    # Modifiers and fillers
    "was", "the", "and", "for", "are", "not", "you", "all", "can", "has", "had", "have",
    "with", "from", "this", "that", "who", "his", "her", "she", "him", "but", "out", "day", "days",
    "old", "done", "high", "low", "more", "less", "into", "timing", "evening", "morning", "night",
    "called", "difficultency", "demonstratory", "expiration", "thankig",
    # Salt modifiers from salt_parser
    "sodium", "potassium", "hydrochloride", "dihydrochloride", "maleate", "succinate",
    "tartrate", "fumarate", "besylate", "mesylate", "phosphate", "sulfate", "sulphate",
    "citrate", "calcium", "hydrate", "anhydrous", "trihydrate", "monohydrate", "dihydrate",
    "valerate", "propionate", "dipropionate", "acetate", "gluconate", "nitrate", "lactate",
    "hcl", "ip", "bp", "usp"
}

try:
    from .medicine_matcher import PHARMA_STOPWORDS
    RX_STOP_WORDS |= {w.lower() for w in PHARMA_STOPWORDS}
except Exception:
    pass


def is_name_or_header_line(text: str) -> bool:
    """Never match words that are the doctor's or patient's name line (containing Name/Dr/Mr/Ms)."""
    return bool(re.search(r"\b(dr|doctor|mr|ms|mrs|miss|name|patient)\b", text, re.I))


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


def parse_line(words: List[str], matcher: Optional[Callable] = None, min_score=80.0):
    """words: the words of one line, left to right. matcher(text) -> list of candidates [{'name':..., 'score': 0-100, ...}] (best first).
    Enforces Precision Gate: a line is a medicine ONLY if
      (a) the matched word has >= 4 letters and score >= 85, AND
      (b) the same line has at least one supporting item: a form word (tab, cap, syp, syr, inj, drops, oint, cream),
          OR a dose with unit (mg, ml, mcg, gm, iu, %), OR a frequency (1+0+1, 1-0-1, OD, BD, TDS, HS, SOS), OR a duration (5 days, x 3d)
      ... OR score >= 95 with the word >= 5 letters and not in the stop list.
    Also never match words that are the doctor's or patient's name line (containing Name/Dr/Mr/Ms).
    Everything else is shown only as 'other text read' (no database lookup, no alternatives, no safety data)."""
    toks = [w.strip() for w in words if w and w.strip()]
    line_text = " ".join(toks)

    # B2: Never match words that are the doctor's or patient's name line
    if is_name_or_header_line(line_text):
        return dict(read="", candidates=[], form=None, dose=None, frequency=None, duration=None, text=line_text, is_medicine=False)

    form = dose = freq = duration = None
    rest = []
    UNITS = {"mg", "mcg", "gm", "g", "ml", "iu", "%"}
    PER = {"day", "days", "d", "week", "weeks", "wk", "month", "months"}
    i = 0
    while i < len(toks):
        w = toks[i]; lw = w.lower().strip(".,:;/"); nxt = toks[i + 1].lower().strip(".,:;/") if i + 1 < len(toks) else ""
        if lw in FORMS: form = form or w.strip(".,:;/")
        elif lw == "x": pass                                                    # filler in "x 5 days"
        elif RE_FREQ.match(w.replace(" ", "")) or RE_FREQ.match(lw): freq = freq or w
        elif RE_DURATION.match(lw): duration = duration or w
        elif re.fullmatch(r"\d+", lw) and nxt in PER: duration = duration or f"{w} {toks[i + 1]}"; i += 1      # "7 days" written as two words
        elif RE_DOSE.match(lw) and i > 0:
            dose = dose or (f"{w} {toks[i + 1]}" if nxt in UNITS else w); i += 1 if nxt in UNITS else 0
        else: rest.append((i, w))
        i += 1

    # Check for supporting items:
    # 1. form word: (tab, cap, syp, syr, inj, drops, oint, cream)
    has_form = bool(form or re.search(r"\b(tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syp|syr|syrup|inj|injection|drops?|oint|ointment|cream)\b", line_text, re.I))
    # 2. dose with unit: (mg, ml, mcg, gm, iu, %)
    has_dose_with_unit = bool(re.search(r"(?:\b\d+(?:\.\d+)?\s*(?:mg|mcg|gm|g|ml|iu)\b|\b\d+(?:\.\d+)?\s*%)", line_text, re.I))
    # 3. frequency: (1+0+1, 1-0-1, OD, BD, TDS, HS, SOS)
    has_freq = bool(freq or re.search(r"\b(?:\d\s*[-+]\s*\d(?:\s*[-+]\s*\d){0,2}|od|bd|tds|tid|qid|hs|sos|prn|stat)\b", line_text, re.I))
    # 4. duration: (5 days, x 3d)
    has_duration = bool(duration or re.search(r"(?:x\s*)?\b\d+\s*(?:day|days|d|week|weeks|wk|month|months)\b", line_text, re.I))

    has_supporting = bool(has_form or has_dose_with_unit or has_freq or has_duration)

    best = None
    for k, (i, w) in enumerate(rest):                        # try single words and neighbour pairs ("Napa Extra")
        for text in (w, f"{w} {rest[k+1][1]}" if k + 1 < len(rest) else None):
            if not text or len(text) < 3 or matcher is None: continue
            clean_str = re.sub(r"[\.,:;/\-\s]", "", text).lower()
            if clean_str in RX_STOP_WORDS: continue
            # If line has no supporting items, medicine can only qualify if word >= 5 letters and score >= 95
            clean_alnum = re.sub(r"[^A-Za-z0-9]", "", text)
            if not has_supporting and len(clean_alnum) < 5:
                continue
            c = matcher(text) or []
            if c and (best is None or c[0].get("score", 0) > best["cands"][0].get("score", 0)):
                best = dict(read=text, cands=c)

    is_medicine = False
    cands = []
    read = ""
    if best and best["cands"]:
        score = float(best["cands"][0].get("score", 0))
        cand_word = best["read"]
        clean_word = re.sub(r"[^A-Za-z0-9]", "", cand_word)
        word_len = len(clean_word)
        clean_lower = clean_word.lower()

        # Gate (a): matched word has >= 4 letters and score >= 85, AND line has supporting item
        cond_a = (word_len >= 4 and score >= 85.0 and has_supporting and clean_lower not in RX_STOP_WORDS)
        # Gate (b): score >= 95 with word >= 5 letters and not in stop list
        cond_b = (score >= 95.0 and word_len >= 5 and clean_lower not in RX_STOP_WORDS)

        if cond_a or cond_b:
            is_medicine = True
            cands = best["cands"][:3]
            read = cand_word

    if not is_medicine:
        read = best["read"] if best else (rest[0][1] if rest else (toks[0] if toks else ""))
        cands = []

    return dict(read=read, candidates=cands, form=form, dose=dose, frequency=freq, duration=duration, text=line_text, is_medicine=is_medicine)


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
    def read_crops(self, crops, beams=2):
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
        # B3: Skip boxes smaller than 12x12 px
        boxes = [b for b in boxes if (b[2] - b[0]) >= 12 and (b[3] - b[1]) >= 12]
        # B3: Cap at 40 boxes (sort top-to-bottom then left-to-right to preserve reading order)
        if len(boxes) > 40:
            boxes.sort(key=lambda b: (b[1], b[0]))
            boxes = boxes[:40]
        return boxes

    def read_prescription(self, photo, matcher=None, beams=2, min_score=80.0):
        """Returns {'ok', 'lines': [{'text','words':[{'text','conf','box'}]}], 'medicines': [parse_line result + 'line'], 'other_text': [...], 'timing': {...}, 'message'}."""
        from PIL import ImageOps
        t_start = time.time()
        img = photo if isinstance(photo, Image.Image) else Image.open(photo)
        img = ImageOps.exif_transpose(img).convert("RGB"); W, H = img.size

        # 1. Box detection timing
        t0_det = time.time()
        boxes = self.find_boxes(img)
        t_det = time.time() - t0_det
        if not boxes:
            return dict(
                ok=False, message="No handwriting found in the image", lines=[], medicines=[], other_text=[],
                timing=dict(detect=round(t_det, 3), read=0.0, match=0.0, total=round(time.time() - t_start, 3))
            )

        # 2. Reading crops timing (batched by 16, num_beams=beams)
        crops = [img.crop((max(0, x0 - self.pad), max(0, y0 - self.pad), min(W, x1 + self.pad), min(H, y1 + self.pad))) for x0, y0, x1, y1 in boxes]
        t0_read = time.time()
        res = []
        for i in range(0, len(crops), 16):
            res += self.read_crops(crops[i:i + 16], beams=beams)
        t_read = time.time() - t0_read

        # 3. Line grouping and matching timing
        t0_match = time.time()
        lines, meds, other_text = [], [], []
        for li, idx in enumerate(group_into_lines(boxes)):
            words = [dict(text=res[i][0], conf=round(res[i][1], 3), box=boxes[i]) for i in idx if res[i][0]]
            if not words: continue
            line_str = " ".join(w["text"] for w in words)
            lines.append(dict(text=line_str, words=words))
            p = parse_line([w["text"] for w in words], matcher=matcher, min_score=min_score)
            p["line"] = len(lines) - 1
            p["confidence"] = round(min(w["conf"] for w in words), 3)
            if p.get("is_medicine"):
                meds.append(p)
            else:
                other_text.append(line_str)
        t_match = time.time() - t0_match
        t_total = time.time() - t_start

        timing = dict(detect=round(t_det, 3), read=round(t_read, 3), match=round(t_match, 3), total=round(t_total, 3))
        return dict(ok=True, message="", lines=lines, medicines=meds, other_text=other_text, timing=timing)
