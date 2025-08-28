# Labeling Tool - Bản cập nhật

## Vấn đề đã sửa:
❌ **Trước:** Sau khi nhấn Enter, không tự động chuyển sang ảnh tiếp theo
✅ **Sau:** Nhấn Enter → Auto next sang ảnh tiếp theo (100ms delay)

## Cải tiến đã thêm:

### 1. Auto Next Feature
- ✅ Checkbox "Auto Next" để bật/tắt tự động chuyển ảnh
- ✅ Mặc định bật Auto Next
- ✅ Delay 100ms thay vì 1000ms để responsive hơn

### 2. Better UX
- ✅ Tự động focus vào entry field sau khi chuyển ảnh
- ✅ Load existing label nếu ảnh đã được label trước đó
- ✅ Status message hiển thị rõ ràng

### 3. Keyboard Workflow Tối ưu
```
Nhập label → Enter → Auto next → Focus entry → Sẵn sàng nhập label mới
```

## Cách sử dụng:

```bash
# Mở labeling tool
python tools/captcha_labeling_gui.py

# Hoặc với thư mục custom
python tools/captcha_labeling_gui.py -i image_crawl/raw_captcha_images -o image_crawl/train_images
```

## Keyboard Shortcuts:
- `Enter`: Lưu label và auto next (nếu bật)
- `←` `→`: Navigate ảnh manual
- `Ctrl+S`: Lưu label (không auto next)
- `Escape`: Thoát

## UI Controls:
- **Auto Next checkbox**: Bật/tắt auto chuyển ảnh sau khi save
- **Previous/Next buttons**: Navigate manual
- **Skip button**: Bỏ qua ảnh hiện tại
- **Statistics button**: Xem thống kê labeling

Workflow giờ sẽ smooth và nhanh hơn rất nhiều! 🚀
