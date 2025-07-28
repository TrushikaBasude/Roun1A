import os
import json
from heading_extractor import extract_title_and_outline  # ✅ FIXED NAME

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            result = extract_title_and_outline(pdf_path)

            output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✅ Processed {filename} → {output_path}")

if __name__ == "__main__":
    main()
