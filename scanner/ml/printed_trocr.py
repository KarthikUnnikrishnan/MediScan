"""Offline printed-label reader for MediScan (Model A).
Put in scanner/ml/ and call PrintedTrOCR(...).read_lines(photo) wherever EasyOCR / Gemini text was used.
TrOCR reads ONE line at a time, so EasyOCR's text *finder* (no recognition) cuts the photo into lines first."""
from pathlib import Path
import re
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image


class PrintedTrOCR:
    def __init__(self, model_dir, device=None, pad=4, min_conf=0.0):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = TrOCRProcessor.from_pretrained(str(model_dir), use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).eval()
        if self.dev == "cuda":
            self.model = self.model.half()        # large model: half precision keeps it well inside 4 GB
        self.model = self.model.to(self.dev)
        self.pad, self.min_conf, self._det = pad, min_conf, None

    def _detector(self):                      # text finder only (no recognition), loaded once
        if self._det is None:
            import easyocr
            self._det = easyocr.Reader(["en"], gpu=(self.dev == "cuda"), recognizer=False, verbose=False)
        return self._det

    @staticmethod
    def _tidy(t):                              # remove junk marks (quotes, @, ~, ?, ^, |) the training labels sometimes contained
        return re.sub(r"\s+", " ", re.sub(r'["@~?^|`]', " ", t)).strip().upper()

    @torch.no_grad()
    def read_crops(self, crops: List[Image.Image], beams=1) -> List[Tuple[str, float]]:
        if not crops:
            return []
        pv = self.proc.image_processor([c.convert("RGB") for c in crops], return_tensors="pt").pixel_values.to(self.dev)
        if self.dev == "cuda":
            pv = pv.half()
        tk = self.proc.tokenizer
        out = self.model.generate(pv, max_length=40, num_beams=beams, output_scores=(beams > 1), return_dict_in_generate=True,
                                  decoder_start_token_id=tk.sep_token_id, eos_token_id=tk.sep_token_id, pad_token_id=tk.pad_token_id)
        texts = [self._tidy(t) for t in self.proc.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)]
        scores = getattr(out, "sequences_scores", None)       # beam search: mean log-prob per token
        confs = torch.exp(scores).tolist() if scores is not None else [1.0] * len(texts)
        return list(zip(texts, confs))

    def find_boxes(self, img: Image.Image):
        arr = np.array(img.convert("RGB"))
        horiz, free = self._detector().detect(arr)
        boxes = []
        for x0, x1, y0, y1 in (horiz[0] if horiz else []):
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
        for pts in (free[0] if free else []):                 # tilted boxes: use their bounding rectangle
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        return sorted(boxes, key=lambda b: (b[1] // 20, b[0]))   # top to bottom, then left to right

    def read_lines(self, photo, beams=1, return_details=False):
        """photo: PIL image or path. Returns a list of text lines (UPPERCASE) for the existing parser/matcher."""
        import time
        img = photo if isinstance(photo, Image.Image) else Image.open(photo)
        img = img.convert("RGB"); W, H = img.size

        t0 = time.time()
        raw_boxes = self.find_boxes(img)
        t_boxes = time.time() - t0
        if not raw_boxes:
            if return_details:
                return [], {"box_find_sec": t_boxes, "reading_sec": 0.0, "box_count": 0}
            return []

        # Filter: drop boxes with height < 12 px or width < 20 px
        valid_boxes = [
            (x0, y0, x1, y1) for x0, y0, x1, y1 in raw_boxes
            if (y1 - y0 >= 12) and (x1 - x0 >= 20)
        ]
        if not valid_boxes and raw_boxes:
            valid_boxes = raw_boxes

        # Sort by area (largest first) and keep at most 14 boxes
        valid_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        selected_boxes = valid_boxes[:14]

        crops = [img.crop((max(0, x0 - self.pad), max(0, y0 - self.pad), min(W, x1 + self.pad), min(H, y1 + self.pad)))
                 for x0, y0, x1, y1 in selected_boxes]

        t1 = time.time()
        res = []
        for i in range(0, len(crops), 16):
            res += self.read_crops(crops[i:i + 16], beams=beams)
        t_reading = time.time() - t1

        lines = [t for t, c in res if t and c >= self.min_conf]
        if return_details:
            return lines, {"box_find_sec": t_boxes, "reading_sec": t_reading, "box_count": len(selected_boxes)}
        return lines


if __name__ == "__main__":                    # quick test:  python printed_trocr.py <model_dir> <photo>
    import sys
    for line in PrintedTrOCR(Path(sys.argv[1])).read_lines(sys.argv[2]):
        print(line)
