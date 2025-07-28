import onnxruntime
import numpy as np
import fitz  # PyMuPDF
import re
from tokenizer_for_minilm import tokenize  # lightweight tokenizer

session = onnxruntime.InferenceSession("model/model.onnx")

def encode(text):
    inputs = tokenize(text)
    outputs = session.run(None, inputs)
    return np.mean(outputs[0], axis=1)

def merge_multiline_blocks(spans, y_threshold=4.0):
    spans = sorted(spans, key=lambda s: (s["page"], s["y"]))
    merged = []
    current = spans[0]

    for span in spans[1:]:
        if (
            span["page"] == current["page"]
            and abs(span["y"] - current["y2"]) <= y_threshold
            and abs(span["size"] - current["size"]) < 1
        ):
            current["text"] += " " + span["text"]
            current["y2"] = span["y2"]
        else:
            merged.append(current)
            current = span
    merged.append(current)
    return merged

def extract_title_and_outline(pdf_path):
    doc = fitz.open(pdf_path)
    raw_spans = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or len(text.split()) < 2:
                        continue
                    raw_spans.append({
                        "text": text,
                        "size": round(span["size"], 1),
                        "page": page_num,
                        "y": span["bbox"][1],
                        "y2": span["bbox"][3],
                        "x": span["bbox"][0],
                        "x2": span["bbox"][2]
                    })

    # Merge multiline title/header lines
    merged_spans = merge_multiline_blocks(raw_spans)

    # Estimate headings only (not full text)
    heading_spans = [s for s in merged_spans if len(s["text"]) < 150]

    # Group by font size to rank levels
    size_freq = {}
    for s in heading_spans:
        size_freq[s["size"]] = size_freq.get(s["size"], 0) + 1
    sorted_sizes = sorted(size_freq.items(), key=lambda x: -x[0])
    size_to_level = {s[0]: f"H{i+1}" for i, s in enumerate(sorted_sizes[:3])}

    # Select best title (biggest + top position + earliest page + centerish)
    def score(span):
        center_x = (span["x"] + span["x2"]) / 2
        score = (
            span["size"] * 10
            - span["page"] * 2
            - span["y"]
            - abs(center_x - 300) * 0.01
        )
        return score

    title_candidates = sorted(heading_spans, key=score, reverse=True)
    best_title = title_candidates[0]["text"] if title_candidates else ""
    
    # Build clean outline (H1–H3 only, no duplicates)
    outline = []
    seen = set()
    for span in heading_spans:
        if span["text"] == best_title or span["text"] in seen:
            continue
        seen.add(span["text"])
        level = size_to_level.get(span["size"], "H3")
        if re.match(r"^\d+(\.\d+)*", span["text"]) or level in ["H1", "H2"]:
            outline.append({
                "level": level,
                "text": span["text"],
                "page": span["page"]
            })

    return {
        "title": best_title.strip(),
        "outline": outline
    }