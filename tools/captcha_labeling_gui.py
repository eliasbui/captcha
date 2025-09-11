#!/usr/bin/env python3
"""
Captcha Labeling GUI Tool
GUI tool để labeling thủ công các ảnh captcha đã crawl từ GDT
"""
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
from pathlib import Path
import tkinter as tk
import shutil
import argparse
import cv2
import os
import json

class CaptchaLabelingTool:
    def __init__(self, input_dir="image_crawl/raw_captcha_images",
                 output_dir="image_crawl/train_images",
                 progress_file="image_crawl/labeling_progress.json"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.progress_file = Path(progress_file)

        # Tạo output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing vocabulary từ mapping_char.json
        self.vocabulary = self._load_existing_vocabulary()

        # Load image list và progress
        self.image_files = self._get_image_files()
        self.progress = self._load_progress()
        self.current_index = self.progress.get('current_index', 0)

        # GUI components
        self.root = None
        self.canvas = None
        self.image_label = None
        self.label_entry = None
        self.progress_var = None
        self.status_var = None
        self.current_image = None
        self.zoom_factor = 1.0

        print(f"Loaded {len(self.image_files)} images from {self.input_dir}")
        print(f"Current vocabulary: {''.join(sorted(self.vocabulary))}")
        print(f"Progress: {self.progress.get('labeled_count', 0)}/{len(self.image_files)} images labeled")

    def _load_existing_vocabulary(self):
        """Load vocabulary từ mapping_char.json"""
        mapping_file = Path("ocr/dataset/mapping_char.json")
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                    # Extract characters (values) từ mapping
                    chars = set(mapping.values())
                    chars.discard("-")  # Loại bỏ blank character
                    return chars
            except Exception as e:
                print(f"Warning: Could not load vocabulary from {mapping_file}: {e}")

        # Default vocabulary nếu không load được
        return set("23456789abcdefghkmnopqrwxy")

    def _get_image_files(self):
        """Lấy danh sách tất cả image files trong input directory"""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        files = []
        for ext in extensions:
            files.extend(list(self.input_dir.glob(f"*{ext}")))
            files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        unique = list(set(files))
        unique.sort()
        return unique

    def _load_progress(self):
        """Load labeling progress từ JSON file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")

        return {
            'current_index': 0,
            'labeled_count': 0,
            'labeled_files': {},
            'session_start': datetime.now().isoformat()
        }

    def _save_progress(self):
        """Lưu progress vào JSON file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")

    def _validate_label(self, label):
        """Validate label text"""
        if not label or len(label.strip()) == 0:
            return False, "Label không được để trống"

        label = label.strip().lower()

        # Kiểm tra độ dài (thường captcha có 4-6 ký tự)
        if len(label) > 10:
            return False, "Label quá dài (>10 ký tự)"

        # Kiểm tra ký tự không hợp lệ
        invalid_chars = set(label) - self.vocabulary
        if invalid_chars:
            # Cho phép thêm ký tự mới, nhưng cảnh báo
            return True, f"Ký tự mới sẽ được thêm vào vocabulary: {''.join(sorted(invalid_chars))}"

        return True, "OK"

    def _preprocess_image_for_display(self, image_path):
        """Preprocess ảnh để hiển thị tốt hơn trong GUI"""
        # Đọc ảnh
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            return None

        # Nếu có alpha channel, extract nó
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:,:,3]  # Lấy alpha channel như trong main_fastapi.py
            img = 255 - img   # Invert
        elif len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize để hiển thị rõ hơn (scale up nhỏ)
        height, width = img.shape
        if height < 100 or width < 200:
            scale = max(2, min(4, 200 // width, 100 // height))
            img = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)

        # Convert sang PIL Image
        pil_img = Image.fromarray(img)
        return pil_img

    def create_gui(self):
        """Tạo GUI interface"""
        self.root = tk.Tk()
        self.root.title("GDT Captcha Labeling Tool")
        self.root.geometry("800x600")

        # Variables
        self.progress_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.auto_next_var = tk.BooleanVar(value=True)  # Auto next sau khi save

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Progress info
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(progress_frame, textvariable=self.progress_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=0, column=1, sticky=tk.E)
        progress_frame.columnconfigure(1, weight=1)

        # Image display
        image_frame = ttk.LabelFrame(main_frame, text="Captcha Image", padding="5")
        image_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # Canvas for image với scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg='white', height=200)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Label input
        input_frame = ttk.LabelFrame(main_frame, text="Label Input", padding="5")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Label:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.label_entry = ttk.Entry(input_frame, font=('Arial', 14))
        self.label_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.label_entry.bind('<Return>', lambda e: self.save_current_label())
        self.label_entry.bind('<KeyRelease>', self._on_label_change)

        # Auto next checkbox
        auto_next_check = ttk.Checkbutton(input_frame, text="Auto Next", variable=self.auto_next_var)
        auto_next_check.grid(row=0, column=2, sticky=tk.W, padx=(5, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="⬅️ Previous", command=self.previous_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="💾 Save Label", command=self.save_current_label).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(button_frame, text="⏭️ Skip", command=self.skip_current).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(button_frame, text="Next ➡️", command=self.next_image).grid(row=0, column=3, padx=(0, 5))

        button_frame.columnconfigure(4, weight=1)
        ttk.Button(button_frame, text="📊 Statistics", command=self.show_statistics).grid(row=0, column=5, padx=(5, 0))

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Control-s>', lambda e: self.save_current_label())
        self.root.bind('<Escape>', lambda e: self.root.quit())

        # Load first image
        self.update_display()

        # Focus on entry
        self.label_entry.focus()

    def _on_label_change(self, event=None):
        """Callback khi label text thay đổi"""
        label = self.label_entry.get().strip()
        if label:
            valid, message = self._validate_label(label)
            if valid:
                self.status_var.set(f"✅ {message}")
            else:
                self.status_var.set(f"❌ {message}")
        else:
            self.status_var.set("Nhập label cho ảnh này...")

    def update_display(self):
        """Cập nhật hiển thị ảnh và thông tin"""
        if not self.image_files or self.current_index >= len(self.image_files):
            self.status_var.set("Đã hoàn thành tất cả ảnh!")
            return

        current_file = self.image_files[self.current_index]

        # Update progress
        labeled_count = self.progress.get('labeled_count', 0)
        self.progress_var.set(f"Image {self.current_index + 1}/{len(self.image_files)} | "
                             f"Labeled: {labeled_count} | File: {current_file.name}")

        # Load và hiển thị ảnh
        try:
            pil_img = self._preprocess_image_for_display(current_file)
            if pil_img:
                # Convert to PhotoImage
                self.current_image = ImageTk.PhotoImage(pil_img)

                # Clear canvas và add image
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)

                # Update scroll region
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            else:
                self.canvas.delete("all")
                self.canvas.create_text(100, 50, text="Cannot load image", fill="red")
        except Exception as e:
            print(f"Error loading image {current_file}: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(100, 50, text=f"Error: {e}", fill="red")

        # Load existing label nếu có
        existing_label = self.progress.get('labeled_files', {}).get(str(current_file), '')
        self.label_entry.delete(0, tk.END)
        self.label_entry.insert(0, existing_label)

        self.status_var.set("Nhập label cho ảnh này...")

    def save_current_label(self):
        """Lưu label cho ảnh hiện tại"""
        if self.current_index >= len(self.image_files):
            return

        label = self.label_entry.get().strip().lower()
        if not label:
            messagebox.showwarning("Warning", "Vui lòng nhập label!")
            return

        valid, message = self._validate_label(label)
        if not valid:
            messagebox.showerror("Error", message)
            return

        current_file = self.image_files[self.current_index]

        try:
            # Tạo filename mới với label
            new_filename = f"{label}.png"
            new_path = self.output_dir / new_filename

            # Nếu file đã tồn tại, thêm suffix
            counter = 1
            while new_path.exists():
                new_filename = f"{label}_{counter:03d}.png"
                new_path = self.output_dir / new_filename
                counter += 1

            # Copy file với tên mới
            shutil.copy2(current_file, new_path)

            # Update progress
            if str(current_file) not in self.progress.get('labeled_files', {}):
                self.progress['labeled_count'] = self.progress.get('labeled_count', 0) + 1

            self.progress.setdefault('labeled_files', {})[str(current_file)] = label
            self.progress['last_saved'] = datetime.now().isoformat()
            self._save_progress()

            self.status_var.set(f"✅ Saved as {new_filename}")

            # Tự động chuyển sang ảnh tiếp theo nếu được bật
            if self.auto_next_var.get():
                # self.root.after(100, self.next_image)  # Delay ngắn 100ms để user thấy status message
                self.next_image()

        except Exception as e:
            messagebox.showerror("Error", f"Cannot save file: {e}")

    def next_image(self):
        """Chuyển sang ảnh tiếp theo"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.progress['current_index'] = self.current_index
            self._save_progress()
            self.update_display()
            # Focus lại entry field (update_display đã load existing label nếu có)
            self.label_entry.focus()
        else:
            messagebox.showinfo("Complete", "Đã hoàn thành tất cả ảnh!")

    def previous_image(self):
        """Quay lại ảnh trước"""
        if self.current_index > 0:
            self.current_index -= 1
            self.progress['current_index'] = self.current_index
            self._save_progress()
            self.update_display()
            # Focus lại entry field
            self.label_entry.focus()

    def skip_current(self):
        """Bỏ qua ảnh hiện tại"""
        self.next_image()

    def show_statistics(self):
        """Hiển thị thống kê labeling"""
        labeled_count = self.progress.get('labeled_count', 0)
        total_count = len(self.image_files)
        remaining = total_count - labeled_count

        # Thống kê labels
        labels = list(self.progress.get('labeled_files', {}).values())
        unique_labels = set(labels)

        stats_text = f"""Labeling Statistics:

Total Images: {total_count}
Labeled: {labeled_count}
Remaining: {remaining}
Progress: {labeled_count/total_count*100:.1f}%

Unique Labels: {len(unique_labels)}
Labels: {', '.join(sorted(unique_labels))}

Session Started: {self.progress.get('session_start', 'Unknown')}
Last Saved: {self.progress.get('last_saved', 'Never')}
"""

        messagebox.showinfo("Statistics", stats_text)

    def run(self):
        """Chạy GUI application"""
        if not self.image_files:
            print("Không tìm thấy ảnh nào trong thư mục input!")
            return

        self.create_gui()

        print("=== Captcha Labeling Tool ===")
        print("Keyboard Shortcuts:")
        print("  ← → : Navigate images")
        print("  Enter : Save label")
        print("  Ctrl+S : Save label")
        print("  Escape : Quit")
        print()

        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='GUI tool for labeling captcha images')
    parser.add_argument('-i', '--input', type=str, default='image_crawl/raw_captcha_images',
                       help='Input directory containing raw images')
    parser.add_argument('-o', '--output', type=str, default='image_crawl/train_images',
                       help='Output directory for labeled images')
    parser.add_argument('-p', '--progress', type=str, default='image_crawl/labeling_progress.json',
                       help='Progress file path')

    args = parser.parse_args()

    # Kiểm tra input directory
    if not os.path.exists(args.input):
        print(f"❌ Input directory không tồn tại: {args.input}")
        return 1

    # Tạo labeling tool
    tool = CaptchaLabelingTool(
        input_dir=args.input,
        output_dir=args.output,
        progress_file=args.progress
    )

    try:
        tool.run()
        print("\n✅ Labeling session completed.")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Labeling interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error in labeling tool: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
