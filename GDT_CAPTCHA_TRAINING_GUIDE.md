# Hướng dẫn Training Model OCR với Captcha GDT

## Tổng quan

Hệ thống này cho phép bạn training lại model OCR CAPTCHA với ảnh từ website GDT (Tổng cục Thuế). Quy trình hoàn toàn tự động từ crawl ảnh, labeling thủ công, đến training model mới.

## Cấu trúc Files

```
captcha_ocr_api-3.4.0/
├── tools/
│   ├── gdt_captcha_crawler.py      # Crawl ảnh từ GDT
│   ├── captcha_labeling_gui.py     # GUI tool để label ảnh
│   ├── training_dataset_manager.py # Quản lý dataset và training
│   └── run_complete_workflow.py    # Script chính chạy toàn bộ workflow
├── image_crawl/
│   ├── raw_captcha_images/         # Ảnh thô sau khi crawl
│   ├── train_images/               # Ảnh đã label cho training
│   ├── test_images/                # Ảnh test
│   ├── gdt_metadata.csv           # Metadata của ảnh crawl
│   └── labeling_progress.json     # Progress của việc labeling
└── ocr/models/crnn/save/
    ├── best.bin                   # Model hiện tại
    └── backups/                   # Backup các model cũ
```

## Cách sử dụng

### Option 1: Chạy toàn bộ workflow tự động

```bash
# Chạy với 1000 ảnh (mặc định)
python tools/run_complete_workflow.py

# Chạy với số lượng ảnh tùy chỉnh
python tools/run_complete_workflow.py --count 500

# Test nhanh với 20 ảnh
python tools/run_complete_workflow.py --quick-test

# Chạy với delay tùy chỉnh (để tránh rate limiting)
python tools/run_complete_workflow.py --count 1000 --delay 2.0

# Skip một số bước nếu đã thực hiện trước đó
python tools/run_complete_workflow.py --skip-crawl --skip-labeling
```

### Option 2: Chạy từng bước riêng biệt

#### Bước 1: Crawl ảnh từ GDT

```bash
# Crawl 1000 ảnh với delay 1 giây
python tools/gdt_captcha_crawler.py --count 1000 --delay 1.0

# Crawl với retry và output tùy chỉnh
python tools/gdt_captcha_crawler.py -c 500 -d 1.5 -r 5 -o image_crawl/raw_captcha_images
```

**Tính năng của Crawler:**
- Random UID generation cho mỗi request
- Retry logic với exponential backoff
- Progress tracking với tqdm
- Metadata logging vào CSV
- SSL handling và error recovery
- Rate limiting protection

#### Bước 2: Label ảnh bằng GUI

```bash
# Mở GUI tool để label ảnh
python tools/captcha_labeling_gui.py

# Với thư mục input/output tùy chỉnh
python tools/captcha_labeling_gui.py -i image_crawl/raw_captcha_images -o image_crawl/train_images
```

**Tính năng của Labeling Tool:**
- GUI thân thiện với tkinter
- Hiển thị ảnh với zoom/pan
- Validation label theo vocabulary hiện có
- Progress tracking và auto-save
- Keyboard shortcuts (←→ navigate, Enter save, Esc quit)
- Thống kê labeling real-time

**Keyboard Shortcuts:**
- `←` `→`: Navigate qua các ảnh
- `Enter`: Lưu label hiện tại
- `Ctrl+S`: Lưu label
- `Escape`: Thoát tool

#### Bước 3: Chuẩn bị dataset và training

```bash
# Chuẩn bị dataset và bắt đầu training
python tools/training_dataset_manager.py

# Chỉ chuẩn bị dataset, không training
python tools/training_dataset_manager.py --no-training

# Tùy chỉnh test ratio
python tools/training_dataset_manager.py --test-ratio 0.15

# Skip train/test split nếu đã chia trước đó
python tools/training_dataset_manager.py --skip-split
```

**Tính năng của Dataset Manager:**
- Scan và validate tất cả labeled images
- Phân tích dataset statistics (character frequency, label lengths)
- Tự động update character mapping với ký tự mới
- Backup model cũ trước khi training
- Tạo train/test split tự động
- Generate training report chi tiết

## Quy trình Training

### 1. Crawling Process
- Tạo random UID cho mỗi request đến GDT
- Download ảnh với URL: `https://www.gdt.gov.vn/TTHKApp/captcha.png?uid={random_uid}`
- Lưu ảnh với filename format: `gdt_{timestamp}_{index}.png`
- Log metadata vào CSV file
- Error handling và retry cho network issues

### 2. Labeling Process
- Load ảnh từ `raw_captcha_images/`
- Preprocess ảnh để hiển thị rõ hơn (resize, invert nếu cần)
- User nhập label thủ công
- Validate label theo vocabulary hiện có
- Lưu ảnh với filename format: `{label}.png` vào `train_images/`
- Auto-save progress để có thể resume sau

### 3. Training Process
- Scan tất cả labeled images
- Update character mapping nếu có ký tự mới
- Backup model hiện tại
- Tạo train/test split (default 80/20)
- Sử dụng training script hiện có với config từ `config.yml`
- Model được lưu vào `ocr/models/crnn/save/best.bin`

## Configuration

### Training Config (ocr/models/crnn/config.yml)
```yaml
TRAIN_PATH: "image_crawl/train_images"
TEST_PATH: "image_crawl/test_images"
BATCH_SIZE: 16
EPOCHS: 100
RNN_HIDDEN_SIZE: 256
LR: 0.001
WEIGHT_DECAY: 0.0001
CLIP_NORM: 5
ACC_THRESHOLD: 0.87
```

### Vocabulary hiện tại
Characters: `23456789abcdefghkmnopqrwxy` (25 ký tự)

Hệ thống tự động thêm ký tự mới vào vocabulary nếu phát hiện trong dataset.

## Troubleshooting

### Lỗi thường gặp

#### 1. Crawling Issues
```
❌ Failed to access the link
```
**Giải pháp:**
- Tăng delay time: `--delay 2.0`
- Giảm số lượng ảnh: `--count 500`
- Check kết nối internet
- Thử với VPN nếu bị block IP

#### 2. Labeling Issues
```
❌ Cannot load image
```
**Giải pháp:**
- Kiểm tra file ảnh có bị corrupt không
- Xóa file lỗi khỏi thư mục raw_captcha_images
- Chạy lại crawler để download ảnh mới

#### 3. Training Issues
```
❌ Training failed
```
**Giải pháp:**
- Kiểm tra có đủ labeled images không (ít nhất 100 ảnh)
- Check GPU memory nếu dùng CUDA
- Giảm batch size trong config.yml
- Kiểm tra character mapping có valid không

### Performance Tips

#### Crawling
- Sử dụng delay 1-2 giây để tránh rate limiting
- Crawl theo batch nhỏ nếu bị timeout
- Sử dụng `--retries 5` cho network không ổn định

#### Labeling
- Sử dụng keyboard shortcuts để tăng tốc
- Label theo batch (100-200 ảnh/session)
- Backup progress file thường xuyên

#### Training
- Đảm bảo có ít nhất 500-1000 ảnh labeled
- Sử dụng GPU nếu có (tự động detect)
- Monitor accuracy trong quá trình training
- Backup model cũ trước khi training

## Monitoring và Validation

### Training Metrics
- Loss function: CTC Loss
- Accuracy threshold: 87%
- Early stopping: Tự động nếu không cải thiện
- Model backup: Tự động backup trước khi overwrite

### Model Validation
- Test set accuracy
- Character-level accuracy
- Word-level accuracy
- Inference speed test

### Files được tạo
- `image_crawl/gdt_metadata.csv`: Metadata crawling
- `image_crawl/labeling_progress.json`: Progress labeling
- `image_crawl/training_preparation_report.txt`: Dataset analysis
- `image_crawl/workflow_final_report.txt`: Workflow summary
- `ocr/models/crnn/save/backups/`: Model backups

## Ví dụ Complete Workflow

```bash
# 1. Quick test với 50 ảnh
python tools/run_complete_workflow.py --count 50 --delay 1.0

# 2. Production run với 1000 ảnh
python tools/run_complete_workflow.py --count 1000 --delay 1.5

# 3. Resume từ bước labeling (đã crawl trước đó)
python tools/run_complete_workflow.py --skip-crawl

# 4. Chỉ training (đã có labeled data)
python tools/run_complete_workflow.py --skip-crawl --skip-labeling
```

## Next Steps

Sau khi training xong:

1. **Test model mới:**
   ```bash
   # Test với API endpoint
   curl -X POST "http://localhost:8000/captcha_recap_image" \
        -H "Content-Type: application/json" \
        -d '{"image_base64": "..."}'
   ```

2. **Deploy model:**
   - Model được lưu tại `ocr/models/crnn/save/best.bin`
   - Restart API service để load model mới
   - Monitor performance trên production data

3. **Continuous improvement:**
   - Collect thêm ảnh khó nhận diện
   - Retrain định kỳ với data mới
   - Fine-tune hyperparameters nếu cần

## Support

Nếu gặp vấn đề:
1. Check log files trong thư mục `image_crawl/`
2. Xem error messages chi tiết
3. Thử chạy từng bước riêng biệt để debug
4. Backup data trước khi thử fix

---

**Lưu ý:** Quy trình này có thể mất vài giờ tùy thuộc vào số lượng ảnh và tốc độ labeling thủ công. Hãy kiên nhẫn và backup progress thường xuyên!
