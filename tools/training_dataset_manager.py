#!/usr/bin/env python3
"""
Training Dataset Manager
Quản lý và chuẩn bị dataset cho training OCR model
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class TrainingDatasetManager:
    def __init__(self,
                 train_dir="image_crawl/train_images",
                 test_dir="image_crawl/test_images",
                 mapping_file="ocr/dataset/mapping_char.json",
                 model_backup_dir="ocr/models/crnn/save/backups"):

        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.mapping_file = Path(mapping_file)
        self.model_backup_dir = Path(model_backup_dir)

        # Tạo directories nếu chưa có
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.model_backup_dir.mkdir(parents=True, exist_ok=True)

        # Current model path
        self.current_model_path = Path("ocr/models/crnn/save/best.bin")

        print(f"Training Dataset Manager initialized")
        print(f"Train directory: {self.train_dir}")
        print(f"Test directory: {self.test_dir}")
        print(f"Mapping file: {self.mapping_file}")

    def scan_labeled_images(self):
        """Scan và validate tất cả labeled images"""
        print("Scanning labeled images...")

        # Lấy tất cả image files trong train directory
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(self.train_dir.glob(f"*{ext}")))
            image_files.extend(list(self.train_dir.glob(f"*{ext.upper()}")))

        # Parse labels từ filenames
        valid_files = []
        invalid_files = []
        labels = []

        for img_file in image_files:
            # Extract label từ filename (bỏ extension)
            label = img_file.stem.lower()

            # Remove numeric suffix nếu có (ví dụ: abc_001.png -> abc)
            if '_' in label and label.split('_')[-1].isdigit():
                label = '_'.join(label.split('_')[:-1])

            # Validate label
            if self._validate_label(label):
                valid_files.append(img_file)
                labels.append(label)
            else:
                invalid_files.append(img_file)
                print(f"⚠️  Invalid label: {img_file.name}")

        print(f"✅ Found {len(valid_files)} valid images")
        if invalid_files:
            print(f"❌ Found {len(invalid_files)} invalid images")

        return valid_files, labels, invalid_files

    def _validate_label(self, label):
        """Validate label format"""
        if not label or len(label.strip()) == 0:
            return False

        # Kiểm tra độ dài hợp lý
        if len(label) > 15:
            return False

        # Kiểm tra ký tự đặc biệt không mong muốn
        invalid_chars = set(label) & set('\\/:*?"<>|')
        if invalid_chars:
            return False

        return True

    def analyze_dataset(self, labels):
        """Phân tích dataset statistics"""
        print("\n=== Dataset Analysis ===")

        # Basic stats
        total_images = len(labels)
        unique_labels = len(set(labels))

        print(f"Total images: {total_images}")
        print(f"Unique labels: {unique_labels}")

        # Label length distribution
        label_lengths = [len(label) for label in labels]
        length_dist = Counter(label_lengths)
        print(f"\nLabel length distribution:")
        for length, count in sorted(length_dist.items()):
            print(f"  {length} chars: {count} images ({count/total_images*100:.1f}%)")

        # Character frequency
        all_chars = ''.join(labels)
        char_freq = Counter(all_chars)
        print(f"\nCharacter frequency (top 10):")
        for char, count in char_freq.most_common(10):
            print(f"  '{char}': {count} times ({count/len(all_chars)*100:.1f}%)")

        # Vocabulary
        vocabulary = set(all_chars)
        print(f"\nVocabulary size: {len(vocabulary)}")
        print(f"Characters: {''.join(sorted(vocabulary))}")

        return {
            'total_images': total_images,
            'unique_labels': unique_labels,
            'vocabulary': vocabulary,
            'character_frequency': char_freq,
            'label_lengths': label_lengths
        }

    def update_character_mapping(self, vocabulary):
        """Update character mapping với vocabulary mới"""
        print("\nUpdating character mapping...")
        vocabulary = set(vocabulary) if not isinstance(vocabulary, set) else vocabulary
        # Load existing mapping
        existing_mapping = self._load_existing_mapping()
        existing_vocab = set(existing_mapping.values()) if existing_mapping else set()
        existing_vocab.discard("-")  # Remove blank character

        # Combine vocabularies
        combined_vocab = existing_vocab | vocabulary

        if len(combined_vocab) > len(existing_vocab):
            print(f"Adding new characters: {''.join(sorted(vocabulary - existing_vocab))}")

            # Create new mapping: {index: character}
            # Start with blank character at index 0
            new_mapping = {"0": "-"}

            # Add all characters
            for i, char in enumerate(sorted(combined_vocab), start=1):
                new_mapping[str(i)] = char

            # Backup old mapping
            if self.mapping_file.exists():
                backup_file = self.mapping_file.parent / f"mapping_char_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(self.mapping_file, backup_file)
                print(f"Backed up old mapping to: {backup_file}")

            # Save new mapping
            with open(self.mapping_file, 'w') as f:
                json.dump(new_mapping, f, indent=2, ensure_ascii=False)

            print(f"✅ Updated character mapping: {len(combined_vocab)} characters")
            print(f"New vocabulary: {''.join(sorted(combined_vocab))}")

            return new_mapping
        else:
            print("✅ No new characters found, mapping unchanged")
            return existing_mapping

    def _load_existing_mapping(self):
        """Load existing character mapping"""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing mapping: {e}")
        return None

    def create_train_test_split(self, image_files, labels, test_ratio=0.2):
        """Tạo train/test split"""
        print(f"\nCreating train/test split (test ratio: {test_ratio})...")

        # Combine files và labels
        data = list(zip(image_files, labels))

        # Shuffle data
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(data)

        # Split
        split_idx = int(len(data) * (1 - test_ratio))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        print(f"Train set: {len(train_data)} images")
        print(f"Test set: {len(test_data)} images")

        # Move test images to test directory
        moved_count = 0
        for img_file, label in test_data:
            try:
                # Create unique filename in test directory
                test_filename = img_file.name
                test_path = self.test_dir / test_filename

                # Handle filename conflicts
                counter = 1
                while test_path.exists():
                    stem = test_path.stem
                    if '_' in stem and stem.split('_')[-1].isdigit():
                        base_stem = '_'.join(stem.split('_')[:-1])
                    else:
                        base_stem = stem
                    test_filename = f"{base_stem}_{counter:03d}{test_path.suffix}"
                    test_path = self.test_dir / test_filename
                    counter += 1

                # Move file
                shutil.move(str(img_file), str(test_path))
                moved_count += 1

            except Exception as e:
                print(f"Error moving {img_file}: {e}")

        print(f"✅ Moved {moved_count} images to test directory")

        return len(train_data), len(test_data)

    # def backup_current_model(self):
    #     """Backup model hiện tại trước khi train"""
    #     print("\nBacking up current model...")

    #     if not self.current_model_path.exists():
    #         print("⚠️  No existing model found to backup")
    #         return None

    #     # Create backup filename với timestamp
    #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #     backup_filename = f"best_backup_{timestamp}.bin"
    #     backup_path = self.model_backup_dir / backup_filename

    #     try:
    #         shutil.copy2(self.current_model_path, backup_path)
    #         print(f"✅ Model backed up to: {backup_path}")
    #         return backup_path
    #     except Exception as e:
    #         print(f"❌ Error backing up model: {e}")
    #         return None

    def generate_training_report(self, stats, train_count, test_count):
        """Generate training preparation report"""
        report = f"""
        === Training Dataset Preparation Report ===
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Dataset Statistics:
        - Total images: {stats['total_images']}
        - Unique labels: {stats['unique_labels']}
        - Vocabulary size: {len(stats['vocabulary'])}
        - Characters: {''.join(sorted(stats['vocabulary']))}

        Dataset Split:
        - Training images: {train_count}
        - Test images: {test_count}
        - Test ratio: {test_count/(train_count+test_count)*100:.1f}%

        Top Character Frequencies:
        """
        # Calculate total characters for percentage
        total_chars = sum(stats['character_frequency'].values())
        
        for char, count in stats['character_frequency'].most_common(10):
            # report += f"- '{char}': {count} times ({count/len(''.join([''] * stats['total_images']))*100:.1f}%)\n"
            percentage = (count/total_chars*100) if total_chars > 0 else 0
            report += f"- '{char}': {count} times ({percentage:.1f}%)\n"

        # Save report
        report_file = Path("image_crawl/training_preparation_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n✅ Training report saved to: {report_file}")
        return report_file

    def prepare_dataset_for_training(self, test_ratio=0.2, skip_split=False):
        """Main function để chuẩn bị dataset cho training"""
        print("=== Preparing Dataset for Training ===\n")

        # 1. Scan labeled images
        image_files, labels, invalid_files = self.scan_labeled_images()

        if not image_files:
            print("❌ No valid labeled images found!")
            return False

        # 2. Analyze dataset
        stats = self.analyze_dataset(labels)

        # 3. Update character mapping
        mapping = self.update_character_mapping(stats['vocabulary'])

        # # 4. Backup current model
        # backup_path = self.backup_current_model()

        # 5. Create train/test split
        if not skip_split:
            train_count, test_count = self.create_train_test_split(image_files, labels, test_ratio)
        else:
            train_count = len(image_files)
            test_count = len(list(self.test_dir.glob("*.png")))
            print(f"Skipped train/test split. Current counts - Train: {train_count}, Test: {test_count}")

        # 6. Generate report
        report_file = self.generate_training_report(stats, train_count, test_count)

        print(f"\n✅ Dataset preparation completed!")
        print(f"Ready for training with {train_count} training images and {test_count} test images")

        return True

    def start_training(self, debug=True):
        """Bắt đầu quá trình training"""
        print("\n=== Starting Training Process ===")

        try:
            # Import và chạy training function
            from ocr.models.crnn.traning import get_training

            print("Initializing training...")
            result = get_training(debug=debug)

            if result == "":
                print("✅ Training completed successfully!")
                return True
            else:
                print(f"❌ Training failed: {result}")
                return False

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset and manage training process')
    parser.add_argument('--train-dir', type=str, default='image_crawl/train_images',
                       help='Training images directory')
    parser.add_argument('--test-dir', type=str, default='image_crawl/test_images',
                       help='Test images directory')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test set ratio (default: 0.2)')
    parser.add_argument('--skip-split', action='store_true',
                       help='Skip train/test split')
    parser.add_argument('--no-training', action='store_true',
                       help='Only prepare dataset, do not start training')
    parser.add_argument('--debug', action='store_true', default=True,
                       help='Enable debug mode for training')

    args = parser.parse_args()

    # Create dataset manager
    manager = TrainingDatasetManager(
        train_dir=args.train_dir,
        test_dir=args.test_dir
    )

    try:
        # Prepare dataset
        success = manager.prepare_dataset_for_training(
            test_ratio=args.test_ratio,
            skip_split=args.skip_split
        )

        if not success:
            print("❌ Dataset preparation failed!")
            return 1

        # Start training nếu không skip
        if not args.no_training:
            training_success = manager.start_training(debug=args.debug)
            if not training_success:
                print("❌ Training process failed!")
                return 1
        else:
            print("⚠️  Training skipped as requested")

        print("\n🎉 All processes completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
