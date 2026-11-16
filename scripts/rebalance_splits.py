import json
import re
import random
import shutil
import time
from pathlib import Path

# ============================================================
# CẤU HÌNH & THIẾT LẬP
# ============================================================
DATA_ROOT = Path("datasets/Minh-TFR")
TRAIN_JSONL = DATA_ROOT / "train_datas.jsonl"
TEST_JSONL = DATA_ROOT / "test_datas.jsonl"

TRAIN_IMGS = DATA_ROOT / "train_datas" / "imgs"
TRAIN_MASKS = DATA_ROOT / "train_datas" / "masks"
TEST_IMGS = DATA_ROOT / "test_datas" / "imgs"
TEST_MASKS = DATA_ROOT / "test_datas" / "masks"

TARGET_TEST_SIZE = 2000
RANDOM_SEED = 42

LABELS = [
    "edge_artifacts",
    "unnatural_texture",
    "inconsistent_alignment",
    "inconsistent_font",
    "lack_of_integration",
    "text_incoherence",
]

LABEL_PATTERNS = {
    "edge_artifacts": r"Edge artifacts?:\s*([^:\n]+(?:\.[^:\n]+)*?\.)",
    "unnatural_texture": r"Unnatural texture appearance:\s*([^:\n]+(?:\.[^:\n]+)*?\.)",
    "inconsistent_alignment": r"Inconsistent alignment:\s*([^:\n]+(?:\.[^:\n]+)*?\.)",
    "inconsistent_font": r"Inconsistent font:\s*([^:\n]+(?:\.[^:\n]+)*?\.)",
    "lack_of_integration": r"Lack of integration:\s*([^:\n]+(?:\.[^:\n]+)*?\.)",
    "text_incoherence": r"Text incoherence:\s*([^:\n]+(?:\.[^:\n]+)*?\.)",
}

NEGATIVE_PHRASES = [
    "delete this item",
    "not cause",
    "no obvious",
    "contradicts",
    "no such artifacts",
    "should be deleted",
]


def extract_label_vector(content):
    vec = []
    for k in LABELS:
        m = re.search(LABEL_PATTERNS[k], content, re.I)
        if m:
            text = m.group(1).lower()
            if not any(neg in text for neg in NEGATIVE_PHRASES):
                vec.append(1)
            else:
                vec.append(0)
        else:
            vec.append(0)
    return vec


def load_dataset():
    print("--> Đang đọc file train_datas.jsonl và test_datas.jsonl...")
    train_samples = []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vec = extract_label_vector(d["content"])
            train_samples.append({"data": d, "labels": vec})

    test_samples = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vec = extract_label_vector(d["content"])
            test_samples.append({"data": d, "labels": vec})

    print(f"    Train hiện tại: {len(train_samples)} mẫu")
    print(f"    Test hiện tại:  {len(test_samples)} mẫu")
    return train_samples, test_samples


def optimize_split(train_samples, test_samples):
    N_test_orig = len(test_samples)
    N_move = N_test_orig - TARGET_TEST_SIZE  # 2518 - 2000 = 518
    N_total = len(train_samples) + N_test_orig  # 12323

    total_counts = [
        sum(s["labels"][j] for s in train_samples) + sum(s["labels"][j] for s in test_samples)
        for j in range(len(LABELS))
    ]
    test_totals = [sum(s["labels"][j] for s in test_samples) for j in range(len(LABELS))]

    # Số lượng mục tiêu trong tập test (2000 mẫu) để có phân phối chuẩn y hệt tổng thể
    target_test = [c * TARGET_TEST_SIZE / N_total for c in total_counts]
    # Số lượng mục tiêu cần chuyển từ test sang train
    target_move = [test_totals[j] - target_test[j] for j in range(len(LABELS))]

    print(f"--> Tối ưu hóa chọn {N_move} mẫu từ test chuyển sang train để phân phối bằng nhau...")

    random.seed(RANDOM_SEED)
    selected = set(random.sample(range(N_test_orig), N_move))
    unselected = list(set(range(N_test_orig)) - selected)

    curr_sums = [0] * len(LABELS)
    for idx in selected:
        v = test_samples[idx]["labels"]
        for j in range(len(LABELS)):
            curr_sums[j] += v[j]

    def calc_loss(sums):
        return sum(((sums[j] - target_move[j]) / (target_move[j] + 1e-5)) ** 2 for j in range(len(LABELS)))

    curr_loss = calc_loss(curr_sums)

    t0 = time.time()
    for _ in range(30000):
        u = random.choice(list(selected))
        v = random.choice(unselected)
        vec_u = test_samples[u]["labels"]
        vec_v = test_samples[v]["labels"]
        new_sums = [curr_sums[j] - vec_u[j] + vec_v[j] for j in range(len(LABELS))]
        new_loss = calc_loss(new_sums)
        if new_loss < curr_loss:
            curr_loss = new_loss
            curr_sums = new_sums
            selected.remove(u)
            selected.add(v)
            unselected.remove(v)
            unselected.append(u)

    print(f"    Tối ưu hoàn tất sau {time.time() - t0:.2f}s (Loss: {curr_loss:.5f})")

    # Hiển thị bảng phân phối so sánh
    remain_indices = sorted(list(set(range(N_test_orig)) - selected))
    selected_indices = sorted(list(selected))

    remain_sums = [0] * len(LABELS)
    for idx in remain_indices:
        for j in range(len(LABELS)):
            remain_sums[j] += test_samples[idx]["labels"][j]

    train_sums = [sum(s["labels"][j] for s in train_samples) for j in range(len(LABELS))]
    new_train_sums = [train_sums[j] + curr_sums[j] for j in range(len(LABELS))]

    print("\n" + "=" * 80)
    print(f"{'Artifact Label':<28} | {'Target':<9} | {'Test (2000)':<12} | {'Train (10323)':<13} | {'Diff':<6}")
    print("=" * 80)
    for j, l in enumerate(LABELS):
        test_p = remain_sums[j] / TARGET_TEST_SIZE * 100
        train_p = new_train_sums[j] / (len(train_samples) + N_move) * 100
        target_p = total_counts[j] / N_total * 100
        diff = abs(test_p - train_p)
        print(f"{l:<28} | {target_p:7.2f}% | {test_p:10.2f}% | {train_p:11.2f}% | {diff:5.2f}%")
    print("=" * 80 + "\n")

    return selected_indices, remain_indices


def apply_changes(train_samples, test_samples, move_indices, keep_test_indices):
    print("--> Bắt đầu chuyển file ảnh, mask và cập nhật jsonl...")

    # Tạo backup jsonl phòng trường hợp cần thiết
    shutil.copy(TRAIN_JSONL, TRAIN_JSONL.with_suffix(".jsonl.bak"))
    shutil.copy(TEST_JSONL, TEST_JSONL.with_suffix(".jsonl.bak"))

    new_train_records = [s["data"] for s in train_samples]
    new_test_records = []

    # Danh sách các file trong train_datas hiện tại để tránh va chạm tên
    train_existing_imgs = set(p.name for p in TRAIN_IMGS.glob("*"))

    moved_count = 0
    renamed_count = 0

    # 1. Chuyển các mẫu được chọn từ test sang train
    for idx in move_indices:
        item = test_samples[idx]["data"]
        old_img_path = Path(item["images"])
        old_mask_path = old_img_path.parent.parent / "masks" / (old_img_path.stem + ".png")

        filename = old_img_path.name
        maskname = old_mask_path.name

        # Xử lý nếu tên file bị trùng với ảnh đã có ở train
        if filename in train_existing_imgs:
            new_filename = f"test_{filename}"
            new_maskname = f"test_{maskname}"
            renamed_count += 1
        else:
            new_filename = filename
            new_maskname = maskname

        new_img_dest = TRAIN_IMGS / new_filename
        new_mask_dest = TRAIN_MASKS / new_maskname

        # Di chuyển file thực tế
        if old_img_path.exists():
            shutil.move(str(old_img_path), str(new_img_dest))
        if old_mask_path.exists():
            shutil.move(str(old_mask_path), str(new_mask_dest))

        # Cập nhật đường dẫn trong record
        train_existing_imgs.add(new_filename)
        labels_dict = {
            k: test_samples[idx]["labels"][j]
            for j, k in enumerate(LABELS)
        }
        # Đảm bảo key inconsisten_alignment giống với format có sẵn
        if "inconsistent_alignment" in labels_dict:
            labels_dict["inconsisten_alignment"] = labels_dict.pop("inconsistent_alignment")

        new_record = {
            "images": f"datasets/Minh-TFR/train_datas/imgs/{new_filename}",
            "labels": labels_dict,
            "content": item["content"],
        }
        new_train_records.append(new_record)
        moved_count += 1

    # 2. Giữ lại 2000 mẫu trong test
    for idx in keep_test_indices:
        new_test_records.append(test_samples[idx]["data"])

    # 3. Ghi lại các file JSONL
    print(f"--> Ghi file mới: {TRAIN_JSONL} ({len(new_train_records)} dòng)...")
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in new_train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"--> Ghi file mới: {TEST_JSONL} ({len(new_test_records)} dòng)...")
    with open(TEST_JSONL, "w", encoding="utf-8") as f:
        for r in new_test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n========== HOÀN THÀNH ==========")
    print(f"Đã chuyển từ test sang train: {moved_count} ảnh và masks (có {renamed_count} file được đổi tên tránh trùng).")
    print(f"Số lượng file train_datas: {len(new_train_records)} (Ảnh trên đĩa: {len(list(TRAIN_IMGS.glob('*')))})")
    print(f"Số lượng file test_datas:  {len(new_test_records)} (Ảnh trên đĩa: {len(list(TEST_IMGS.glob('*')))})")


def main():
    train_samples, test_samples = load_dataset()
    move_indices, keep_test_indices = optimize_split(train_samples, test_samples)
    apply_changes(train_samples, test_samples, move_indices, keep_test_indices)


if __name__ == "__main__":
    main()

