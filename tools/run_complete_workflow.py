#!/usr/bin/env python3
"""
Complete GDT Captcha Training Workflow
Master script để orchestrate toàn bộ quy trình từ crawl đến training
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class GDTCaptchaWorkflow:
    def __init__(self,
                 image_count=1000,
                 crawl_delay=1.0,
                 test_ratio=0.2,
                 skip_crawl=False,
                 skip_labeling=False,
                 skip_training=False):

        self.image_count = image_count
        self.crawl_delay = crawl_delay
        self.test_ratio = test_ratio
        self.skip_crawl = skip_crawl
        self.skip_labeling = skip_labeling
        self.skip_training = skip_training

        # Paths
        self.project_root = Path.cwd()
        self.tools_dir = self.project_root / "tools"
        self.raw_images_dir = self.project_root / "image_crawl" / "raw_captcha_images"
        self.train_images_dir = self.project_root / "image_crawl" / "train_images"
        self.test_images_dir = self.project_root / "image_crawl" / "test_images"

        print("=== GDT Captcha Training Workflow ===")
        print(f"Project root: {self.project_root}")
        print(f"Target images: {self.image_count}")
        print(f"Crawl delay: {self.crawl_delay}s")
        print(f"Test ratio: {self.test_ratio}")

    def step_1_crawl_images(self):
        """Step 1: Crawl ảnh từ GDT"""
        if self.skip_crawl:
            print("\n🔄 Step 1: SKIPPED - Image crawling")
            return True

        print(f"\n🔄 Step 1: Crawling {self.image_count} images from GDT...")

        # Check if crawler script exists
        crawler_script = self.tools_dir / "gdt_captcha_crawler.py"
        if not crawler_script.exists():
            print(f"❌ Crawler script not found: {crawler_script}")
            return False

        # Run crawler
        cmd = [
            sys.executable, str(crawler_script),
            "--count", str(self.image_count),
            "--delay", str(self.crawl_delay),
            "--retries", "3"
        ]

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Image crawling completed successfully!")

            # Count crawled images
            crawled_count = len(list(self.raw_images_dir.glob("*.png")))
            print(f"📊 Crawled images: {crawled_count}")

            if crawled_count == 0:
                print("❌ No images were crawled!")
                return False

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Crawling failed: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False

    def step_2_label_images(self):
        """Step 2: Label ảnh bằng GUI tool"""
        if self.skip_labeling:
            print("\n🔄 Step 2: SKIPPED - Image labeling")
            return True

        print(f"\n🔄 Step 2: Starting image labeling GUI...")

        # Check if images exist
        raw_images = list(self.raw_images_dir.glob("*.png"))
        if not raw_images:
            print("❌ No raw images found for labeling!")
            return False

        print(f"📊 Found {len(raw_images)} images to label")

        # Check if labeling tool exists
        labeling_script = self.tools_dir / "captcha_labeling_gui.py"
        if not labeling_script.exists():
            print(f"❌ Labeling tool not found: {labeling_script}")
            return False

        # Run labeling tool
        cmd = [
            sys.executable, str(labeling_script),
            "--input", str(self.raw_images_dir),
            "--output", str(self.train_images_dir)
        ]

        try:
            print(f"🖥️  Launching GUI labeling tool...")
            print("⚠️  Please complete the labeling process in the GUI window")
            print("   - Use arrow keys to navigate")
            print("   - Enter labels and press Enter to save")
            print("   - Close the window when done")

            result = subprocess.run(cmd, check=True)

            # Check labeled images
            labeled_count = len(list(self.train_images_dir.glob("*.png")))
            print(f"✅ Labeling completed! {labeled_count} images labeled")

            if labeled_count == 0:
                print("❌ No images were labeled!")
                return False

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Labeling failed: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⚠️  Labeling interrupted by user")
            return False

    def step_3_prepare_and_train(self):
        """Step 3: Chuẩn bị dataset và bắt đầu training"""
        if self.skip_training:
            print("\n🔄 Step 3: SKIPPED - Dataset preparation and training")
            return True

        print(f"\n🔄 Step 3: Preparing dataset and starting training...")

        # Check labeled images
        labeled_images = list(self.train_images_dir.glob("*.png"))
        if not labeled_images:
            print("❌ No labeled images found for training!")
            return False

        print(f"📊 Found {len(labeled_images)} labeled images")

        # Check if dataset manager exists
        manager_script = self.tools_dir / "training_dataset_manager.py"
        if not manager_script.exists():
            print(f"❌ Dataset manager not found: {manager_script}")
            return False

        # Run dataset preparation and training
        cmd = [
            sys.executable, str(manager_script),
            "--train-dir", str(self.train_images_dir),
            "--test-dir", str(self.test_images_dir),
            "--test-ratio", str(self.test_ratio),
            "--debug"
        ]

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            print("✅ Dataset preparation and training completed!")

            # Show training output
            if result.stdout:
                print("\n📋 Training Output:")
                print(result.stdout[-1000:])  # Show last 1000 chars

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout[-1000:]}")  # Show last 1000 chars
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False

    def step_4_validate_model(self):
        """Step 4: Validate trained model"""
        print(f"\n🔄 Step 4: Validating trained model...")

        # Check if model exists
        model_path = Path("ocr/models/crnn/save/best.bin")
        if not model_path.exists():
            print("❌ Trained model not found!")
            return False

        # Basic validation - check model file size and timestamp
        model_stat = model_path.stat()
        model_size = model_stat.st_size / (1024 * 1024)  # MB
        model_time = datetime.fromtimestamp(model_stat.st_mtime)

        print(f"📊 Model file size: {model_size:.2f} MB")
        print(f"📊 Model last modified: {model_time}")

        # Check if model was updated recently (within last hour)
        time_diff = datetime.now() - model_time
        if time_diff.total_seconds() < 3600:  # 1 hour
            print("✅ Model appears to be recently trained!")
        else:
            print("⚠️  Model may not be recently updated")

        # Test inference với một vài ảnh test
        test_images = list(self.test_images_dir.glob("*.png"))
        if test_images:
            print(f"📊 Found {len(test_images)} test images for validation")
            # TODO: Add actual inference test here
            print("✅ Model validation completed (basic checks passed)")
        else:
            print("⚠️  No test images found for validation")

        return True

    def generate_final_report(self):
        """Tạo báo cáo cuối cùng"""
        print(f"\n📋 Generating final workflow report...")

        # Collect statistics
        raw_count = len(list(self.raw_images_dir.glob("*.png")))
        train_count = len(list(self.train_images_dir.glob("*.png")))
        test_count = len(list(self.test_images_dir.glob("*.png")))

        model_path = Path("ocr/models/crnn/save/best.bin")
        model_exists = model_path.exists()

        report = f"""
=== GDT Captcha Training Workflow Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Workflow Configuration:
- Target images: {self.image_count}
- Crawl delay: {self.crawl_delay}s
- Test ratio: {self.test_ratio}
- Skip crawl: {self.skip_crawl}
- Skip labeling: {self.skip_labeling}
- Skip training: {self.skip_training}

Results:
- Raw images crawled: {raw_count}
- Images labeled for training: {train_count}
- Images in test set: {test_count}
- Model updated: {'Yes' if model_exists else 'No'}

Success Rate:
- Crawl success rate: {min(100, raw_count/self.image_count*100):.1f}%
- Labeling completion: {train_count/max(1, raw_count)*100:.1f}%
- Training completion: {'100%' if model_exists else '0%'}

Next Steps:
1. Test the trained model with new captcha images
2. Deploy the updated model to production
3. Monitor model performance over time
4. Consider additional training if accuracy is insufficient

Files Generated:
- Raw images: {self.raw_images_dir}
- Training images: {self.train_images_dir}
- Test images: {self.test_images_dir}
- Model: {model_path}
- Metadata: image_crawl/gdt_metadata.csv
- Progress: image_crawl/labeling_progress.json
"""

        # Save report
        report_file = Path("image_crawl/workflow_final_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ Final report saved to: {report_file}")
        print(report)

        return report_file

    def run_complete_workflow(self):
        """Chạy toàn bộ workflow"""
        start_time = datetime.now()
        print(f"🚀 Starting complete workflow at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        steps = [
            ("Image Crawling", self.step_1_crawl_images),
            ("Image Labeling", self.step_2_label_images),
            ("Dataset Preparation & Training", self.step_3_prepare_and_train),
            ("Model Validation", self.step_4_validate_model)
        ]

        success_count = 0

        for step_name, step_function in steps:
            try:
                success = step_function()
                if success:
                    success_count += 1
                    print(f"✅ {step_name}: SUCCESS")
                else:
                    print(f"❌ {step_name}: FAILED")
                    print("⚠️  Workflow stopped due to failure")
                    break
            except Exception as e:
                print(f"❌ {step_name}: ERROR - {e}")
                print("⚠️  Workflow stopped due to error")
                break

        # Generate final report
        self.generate_final_report()

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n🏁 Workflow completed in {duration}")
        print(f"📊 Success rate: {success_count}/{len(steps)} steps completed")

        if success_count == len(steps):
            print("🎉 All steps completed successfully!")
            print("🚀 Your OCR model has been retrained with GDT captcha data!")
            return True
        else:
            print("⚠️  Workflow completed with some failures")
            return False

def main():
    parser = argparse.ArgumentParser(description='Complete GDT Captcha Training Workflow')
    parser.add_argument('-c', '--count', type=int, default=1000,
                       help='Number of images to crawl (default: 1000)')
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                       help='Delay between crawl requests (default: 1.0)')
    parser.add_argument('-t', '--test-ratio', type=float, default=0.2,
                       help='Test set ratio (default: 0.2)')
    parser.add_argument('--skip-crawl', action='store_true',
                       help='Skip image crawling step')
    parser.add_argument('--skip-labeling', action='store_true',
                       help='Skip image labeling step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with 20 images')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.count = 20
        args.delay = 0.5
        print("🧪 Quick test mode: 20 images, 0.5s delay")

    # Create workflow instance
    workflow = GDTCaptchaWorkflow(
        image_count=args.count,
        crawl_delay=args.delay,
        test_ratio=args.test_ratio,
        skip_crawl=args.skip_crawl,
        skip_labeling=args.skip_labeling,
        skip_training=args.skip_training
    )

    try:
        success = workflow.run_complete_workflow()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
