import onnxruntime
import numpy as np
import fitz  # PyMuPDF
import re
from tokenizer_for_minilm import tokenize  # lightweight multilingual tokenizer

session = onnxruntime.InferenceSession("model/model.onnx")

def encode(text):
    inputs = tokenize(text)
    outputs = session.run(None, inputs)
    return np.mean(outputs[0], axis=1)

def fix_hyphenation(text):
    return re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

def merge_multiline_blocks(spans, y_threshold=4.0):
    spans = sorted(spans, key=lambda s: (s["page"], s["y"]))
    merged = []
    current = spans[0]

    for span in spans[1:]:
        if (
            span["page"] == current["page"]
            and abs(span["y"] - current["y2"]) <= y_threshold
            and abs(span["size"] - current["size"]) < 1
            and abs(span["x"] - current["x"]) < 40  # roughly same x-start
        ):
            current["text"] += " " + span["text"]
            current["y2"] = span["y2"]
        else:
            current["text"] = fix_hyphenation(current["text"]).strip()
            merged.append(current)
            current = span
    current["text"] = fix_hyphenation(current["text"]).strip()
    merged.append(current)
    return merged

def get_best_title_candidate(spans):
    title_blocks = [
        s for s in spans
        if s["page"] <= 2 and s["size"] >= 12
    ]

    title_blocks = sorted(title_blocks, key=lambda s: (s["page"], s["y"]))
    groups = []
    current = [title_blocks[0]] if title_blocks else []

    for span in title_blocks[1:]:
        last = current[-1]
        if (
            span["page"] == last["page"]
            and abs(span["size"] - last["size"]) <= 1
            and abs(span["y"] - last["y2"]) < 6
            and abs(span["x"] - last["x"]) < 40
        ):
            current.append(span)
        else:
            groups.append(current)
            current = [span]
    if current:
        groups.append(current)

    def group_score(g):
        cx = np.mean([(s["x"] + s["x2"]) / 2 for s in g])
        size = g[0]["size"]
        score = size * 10 - g[0]["page"] * 3 - g[0]["y"] + (len(" ".join(s["text"] for s in g)) * 0.1)
        return score - abs(cx - 300) * 0.01

    best_group = max(groups, key=group_score, default=[])
    return fix_hyphenation(" ".join(s["text"] for s in best_group)).strip()

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

    merged_spans = merge_multiline_blocks(raw_spans)
    heading_spans = [s for s in merged_spans if len(s["text"]) < 150]

    size_freq = {}
    for s in heading_spans:
        size_freq[s["size"]] = size_freq.get(s["size"], 0) + 1
    sorted_sizes = sorted(size_freq.items(), key=lambda x: -x[0])
    size_to_level = {s[0]: f"H{i+1}" for i, s in enumerate(sorted_sizes[:3])}

    best_title = get_best_title_candidate(heading_spans)

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
        "title": best_title,
        "outline": outline
    }
