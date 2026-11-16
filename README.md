# Tampered Text Detection in Document Image

Nghiên cứu và phát triển giải pháp phát hiện và phân loại chữ bị làm giả trong ảnh tài liệu số bằng mô hình học sâu đa nhiệm gọn nhẹ **LightweightDTD** dựa trên **EfficientNet-B0** và đặc trưng miền tần số **BlockDCT**.

Mô hình được huấn luyện và đánh giá chi tiết trong notebook: [`multi-class-efficientb0-dtd-lightweight.ipynb`](./multi-class-efficientb0-dtd-lightweight.ipynb).

---

## 1. Kiến Trúc Mô Hình (LightweightDTD)

Mô hình được thiết kế theo hướng học đa nhiệm (Multi-task Learning) với kích thước siêu nhẹ (~4.15 triệu tham số):
* **Nhánh không gian (Spatial Branch):** Sử dụng backbone **EfficientNet-B0** trích xuất đặc trưng đa tầng ($f_0, f_1, f_2, f_3$).
* **Nhánh tần số (Frequency Branch):** Trích xuất ma trận hệ số **BlockDCT $8 \times 8$** để phát hiện các dấu vết nén JPEG bất thường ở mức vi mô, sau đó hòa trộn với tầng $f_1$ qua khối chú ý **scSE (Spatial & Channel Squeeze-and-Excitation)**.
* **Bộ giải mã LiteMID (Lightweight Multi-view Iterative Decoder):** Tích hợp lặp đa góc nhìn các đặc trưng đa tỉ lệ từ $H/4$ đến $H/32$.
* **Đầu ra đa nhiệm (Dual-head Outputs):**
  1. *Phân đoạn vị trí làm giả (Pixel-level Mask):* Dự đoán mặt nạ phân đoạn kích thước $224 \times 224$.
  2. *Phân loại 6 dạng làm giả (Multi-label Classification):* `edge_artifacts`, `unnatural_texture`, `inconsistent_alignment`, `inconsistent_font`, `lack_of_integration`, `text_incoherence`.
* **Kết quả trên tập kiểm thử (Minh-TFR TestSet):**
  * Pixel Accuracy: **99.68%**
  * Top-3 Class Hit Rate: **98.30%**
  * Precision (Micro): **85.25%** | IoU (Micro): **58.39%**

---

## 2. Chuẩn Bị Dữ Liệu (TFR Dataset)

```powershell
# Cài đặt thư viện tải dữ liệu
pip install -U huggingface_hub hf_xet

# Đăng nhập Hugging Face
hf auth login

# Tải dữ liệu TFR
python download_tfr.py

# Giải nén tập dữ liệu
cd ./datasets/TFR
& "$env:ProgramFiles\7-Zip\7z.exe" x .\TFR.zip
& "$env:ProgramFiles\7-Zip\7z.exe" x .\TFR -o".\TFR_extracted"
```

---

## 3. Hướng Dẫn Soạn Thảo & Biên Dịch Báo Cáo (LaTeX)

Thư mục `paper/` chứa toàn bộ bản thảo luận văn theo chuẩn định dạng của **Đại học Cần Thơ (Trường CNTT&TT)**.

### Bước 1: Cài Đặt Môi Trường (Chỉ cần làm 1 lần trên máy mới)

1. **Cài extension VS Code / Cursor / Antigravity IDE:**
   - Cài extension **`LaTeX Workshop`** (ID: `James-Yu.latex-workshop`).

2. **Cài thư viện Pygments (hỗ trợ hiển thị code bằng package `minted`):**
   ```bash
   pip install pygments
   ```

3. **Cài đặt bộ biên dịch TeX (Chọn 1 trong 2 cách):**
   * **Cách A (Gọn nhẹ, khuyên dùng - TinyTeX ~70MB):**
     Mở PowerShell và chạy lần lượt 2 khối lệnh sau:
     ```powershell
     # 1. Tải và giải nén TinyTeX
     Invoke-WebRequest -Uri "https://tinytex.yihui.org/install-bin-windows.ps1" -OutFile "$env:TEMP\install-tinytex.ps1"
     powershell -ExecutionPolicy Bypass -File "$env:TEMP\install-tinytex.ps1"
     ```
     ```powershell
     # 2. Cài toàn bộ các package cần thiết cho bài báo
     $env:Path = "C:\ProgramData\TinyTeX\bin\windows;" + $env:Path
     tlmgr install vntex ieeetran algorithms collection-fontsrecommended pgf adjustbox tocloft pdflscape fancyhdr emptypage ulem titlesec enumitem soul lipsum framed listings fancyvrb xpatch caption float makecell colortbl multirow minted silence biblatex biber tcolorbox environ fvextra upquote catchfile currfile csquotes ragged2e everysel tikzfill pdfcol
     ```
   * **Cách B (TeX Live Full hoặc MiKTeX):**
     Tải cài đặt [MiKTeX](https://miktex.org/) (tick chọn *"Always install missing packages on the fly"*) hoặc cài [TeX Live Full](https://tug.org/texlive/).

---

### Bước 2: Biên Dịch Xuất File PDF

#### Cách 1: Sử dụng giao diện IDE (Khuyên dùng)
1. Mở file `paper/main.tex`.
2. Bấm tổ hợp phím **`Ctrl + Alt + V`** (chọn *View in VSCode tab*) để chia đôi màn hình xem trước file PDF song song.
3. Mỗi khi chỉnh sửa nội dung, chỉ cần bấm **`Ctrl + S`**: IDE sẽ tự động biên dịch ngầm chu trình 4 bước (`pdflatex -> biber -> pdflatex*2`) và cập nhật trang PDF tức thì.
4. Giữ phím **`Ctrl` + Click chuột trái** vào bất kỳ vị trí nào trên file PDF để nhảy trực tiếp tới dòng mã nguồn `.tex` tương ứng (SyncTeX).

> **Lưu ý khi đổi máy:** Nếu IDE báo lỗi `spawn pdflatex ENOENT`, mở file `.vscode/settings.json` và cập nhật lại đường dẫn `command` của `pdflatex`, `biber`, `bibtex` cho khớp với vị trí cài TeX trên máy mới (hoặc đổi thành `"command": "pdflatex"` nếu đã thêm vào `PATH`).

#### Cách 2: Biên dịch bằng dòng lệnh (CLI - Dùng được cho mọi hệ điều hành)
Nếu không dùng VS Code, bạn có thể đứng tại thư mục `paper/` và chạy lệnh sau trên Terminal / Command Prompt:
```bash
cd paper

# 1. Biên dịch lượt đầu
pdflatex -shell-escape -interaction=nonstopmode main.tex

# 2. Xử lý danh mục trích dẫn tài liệu tham khảo
biber main

# 3. Biên dịch lại 2 lần để cập nhật Mục lục & Tọa độ khung bìa
pdflatex -shell-escape -interaction=nonstopmode main.tex
pdflatex -shell-escape -interaction=nonstopmode main.tex
```
Kết quả xuất ra file **`paper/main.pdf`** hoàn chỉnh.

---

### Cấu Trúc Thư Mục Báo Cáo (`paper/`)

```text
paper/
├── main.tex               # File điều phối chính của toàn bộ văn bản
├── main.pdf               # File PDF kết quả sau khi biên dịch
├── bib/
│   └── refs.bib           # Danh mục tài liệu tham khảo (chuẩn Biber)
├── config/                # Cấu hình font chữ, lề trang, tiêu đề, TikZ
├── images/                # Logo trường và các hình ảnh biểu đồ minh họa
├── minor-pages/           # Bìa ngoài, bìa trong, nhận xét, lời cảm ơn, mục lục, từ viết tắt
├── part-01/               # Phần I: Đặt vấn đề, mục tiêu, phương pháp, ý nghĩa
├── part-02/               # Phần II: Cơ sở lý thuyết, Thiết kế mô hình, Thực nghiệm & Đánh giá
└── part-03/               # Phần III: Kết luận và hướng nghiên cứu mở rộng
```
