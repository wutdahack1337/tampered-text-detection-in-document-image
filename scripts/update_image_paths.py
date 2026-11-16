import json
from pathlib import Path

# ============================================================
# Cấu hình các file jsonl cần cập nhật đường dẫn ảnh
# ============================================================
FILES_TO_UPDATE = [
    {
        "file_path": "datasets/Minh-TFR/train_datas.jsonl",
        # data/train_datas/DOC_ALLS_9705/imgs/9.jpg -> datasets/Minh-TFR/train_datas/imgs/9.jpg
        "old_prefix": "data/train_datas/DOC_ALLS_9705/imgs/",
        "new_prefix": "datasets/Minh-TFR/train_datas/imgs/",
    },
    {
        "file_path": "datasets/Minh-TFR/test_datas.jsonl",
        # data/test_datas/DOC_ALLS_2518/imgs/1.jpg -> datasets/Minh-TFR/test_datas/imgs/1.jpg
        "old_prefix": "data/test_datas/DOC_ALLS_2518/imgs/",
        "new_prefix": "datasets/Minh-TFR/test_datas/imgs/",
    },
]


def update_image_paths(file_path, old_prefix, new_prefix):
    path = Path(file_path)
    if not path.exists():
        print(f"[SKIP] File không tồn tại: {file_path}")
        return

    tmp_path = path.with_suffix(".tmp")
    count = 0
    total = 0

    with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            data = json.loads(line)
            img = data.get("images", "")

            # Nếu đường dẫn bắt đầu bằng old_prefix hoặc chỉ cần lấy tên file
            if img.startswith(old_prefix):
                filename = img[len(old_prefix):]
                data["images"] = new_prefix + filename
                count += 1
            else:
                # Fallback: lấy basename của ảnh
                filename = img.split("/")[-1]
                data["images"] = new_prefix + filename
                count += 1

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    tmp_path.replace(path)
    print(f"[{file_path}] Đã cập nhật: {count}/{total} dòng.")


def main():
    print("========== BẮT ĐẦU CẬP NHẬT ĐƯỜNG DẪN ẢNH ==========")
    for item in FILES_TO_UPDATE:
        update_image_paths(
            file_path=item["file_path"],
            old_prefix=item["old_prefix"],
            new_prefix=item["new_prefix"],
        )
    print("===================== HOÀN TẤT =====================")


if __name__ == "__main__":
    main()

