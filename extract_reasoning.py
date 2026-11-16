import json
import re
import argparse
from pathlib import Path

# =========================
# CONFIG DEFAULT
# =========================

DEFAULT_SPLIT = "test"  # "train" or "test"

SPLIT_CONFIG = {
    "train": {
        "input_file": "datasets/TFR/TFR_extracted/data/annotation/train_datas.json",
        "output_file": "datasets/Minh-TFR/train_datas.jsonl",
        "target_subset": "DOC_ALLS_9705",
    },
    "test": {
        "input_file": "datasets/TFR/TFR_extracted/data/annotation/test_datas.json",
        "output_file": "datasets/Minh-TFR/test_datas.jsonl",
        "target_subset": "DOC_ALLS_2518",
    },
}

# =========================
# LOAD DATA
# =========================

def load_annotations(path):
    """
    Support:
    - JSONL: one JSON object per line
    - JSON: a list of JSON objects
    - JSON: a single JSON object
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return

    try:
        data = json.loads(content)

        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data

        return

    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARNING] Invalid JSON at line {line_number}")


# =========================
# EXTRACT REASONING
# =========================

def extract_think(record):
    messages = record.get("messages", [])

    for message in messages:
        if message.get("role") != "assistant":
            continue

        content = message.get("content", "")

        match = re.search(
            r"<think>(.*?)</think>",
            content,
            flags=re.DOTALL
        )

        if match:
            return match.group(1).strip()

    return None


# =========================
# EXTRACT IMAGE
# =========================

def extract_image(record, target_subset):
    images = record.get("images", [])

    if isinstance(images, str):
        images = [images]

    for image_path in images:
        if target_subset in image_path:
            return image_path

    return None


# =========================
# MAIN
# =========================

def run(split="test", input_file=None, output_file=None, target_subset=None):
    cfg = SPLIT_CONFIG.get(split, SPLIT_CONFIG["test"])

    in_path = input_file or cfg["input_file"]
    out_path = output_file or cfg["output_file"]
    subset = target_subset or cfg["target_subset"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    total = 0
    extracted = 0
    missing_reasoning = 0
    missing_image = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for record in load_annotations(in_path):
            total += 1

            image_path = extract_image(record, subset)

            if image_path is None:
                missing_image += 1
                continue

            reasoning = extract_think(record)

            if reasoning is None:
                missing_reasoning += 1
                continue

            filename = image_path.split("/")[-1]
            local_image_path = f"datasets/Minh-TFR/{split}_datas/imgs/{filename}"

            sample = {
                "images": local_image_path,
                "content": reasoning
            }

            out.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

            extracted += 1

    print(f"\n========== SUMMARY ({split.upper()}) ==========")
    print(f"Total records:       {total}")
    print(f"Extracted:           {extracted}")
    print(f"Missing image:       {missing_image}")
    print(f"Missing reasoning:   {missing_reasoning}")
    print(f"Output:              {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract reasoning from TFR annotations")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split (train or test)")
    parser.add_argument("--input", default=None, help="Input annotation file path (optional)")
    parser.add_argument("--output", default=None, help="Output jsonl file path (optional)")
    parser.add_argument("--subset", default=None, help="Target subset name (optional)")

    args = parser.parse_args()
    run(split=args.split, input_file=args.input, output_file=args.output, target_subset=args.subset)


if __name__ == "__main__":
    main()
