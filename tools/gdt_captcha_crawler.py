#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDT Captcha Crawler
Download captcha images from GDT website with random UIDs
"""

import os
import sys
import ssl

# Fix encoding cho Windows console
if sys.platform == "win32":
    try:
        # Set UTF-8 encoding cho stdout/stderr
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        # Fallback nếu không set được UTF-8
        pass
import csv
import time
import uuid
import requests
import argparse
from datetime import datetime
from urllib.request import Request, urlopen
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import numpy as np
import cv2

class GDTCaptchaCrawler:
    def __init__(self, output_dir="image_crawl/raw_captcha_images", metadata_file="image_crawl/gdt_metadata.csv"):
        self.base_url = "https://www.gdt.gov.vn/TTHKApp/captcha.png"
        self.output_dir = output_dir
        self.metadata_file = metadata_file
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.session = requests.Session()

        # Tạo thư mục output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)

        # Khởi tạo metadata CSV
        self._init_metadata_file()

    def _init_metadata_file(self):
        """Khởi tạo file metadata CSV với header"""
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'uid', 'url', 'timestamp', 'download_status', 'file_size', 'image_shape'])

    def _generate_random_uid(self):
        """Tạo random UID như trong yêu cầu"""
        return str(uuid.uuid4())

    def _download_single_image(self, index, total_count, delay=1.0, max_retries=3):
        """
        Download một ảnh captcha từ GDT

        Args:
            index: Index của ảnh (để đặt tên file)
            total_count: Tổng số ảnh cần download (để hiển thị progress)
            delay: Thời gian delay giữa các request (seconds)
            max_retries: Số lần retry tối đa

        Returns:
            dict: Thông tin về việc download (success, filename, metadata)
        """
        uid = self._generate_random_uid()
        url = f"{self.base_url}?uid={uid}"
        timestamp = datetime.now().isoformat()
        filename = f"gdt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:04d}.png"
        filepath = os.path.join(self.output_dir, filename)

        result = {
            'filename': filename,
            'uid': uid,
            'url': url,
            'timestamp': timestamp,
            'download_status': 'failed',
            'file_size': 0,
            'image_shape': 'unknown'
        }

        for attempt in range(max_retries):
            try:
                # Sử dụng requests thay vì urllib để dễ xử lý hơn
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                response = self.session.get(url, headers=headers, timeout=30, verify=False)
                response.raise_for_status()

                # Kiểm tra content type
                content_type = response.headers.get('content-type', '')
                if 'image' not in content_type:
                    raise ValueError(f"Invalid content type: {content_type}")

                # Lưu ảnh
                image_data = response.content
                with open(filepath, 'wb') as f:
                    f.write(image_data)

                # Verify ảnh có thể đọc được
                try:
                    img = Image.open(filepath)
                    image_shape = f"{img.width}x{img.height}"
                    img.close()

                    # Cập nhật thông tin thành công
                    result.update({
                        'download_status': 'success',
                        'file_size': len(image_data),
                        'image_shape': image_shape
                    })

                    # Delay để tránh rate limiting
                    if delay > 0:
                        time.sleep(delay)

                    return result

                except Exception as e:
                    os.remove(filepath)  # Xóa file lỗi
                    raise ValueError(f"Invalid image data: {str(e)}")

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    result['download_status'] = f'failed_after_{max_retries}_attempts: {str(e)}'
                    break

        return result

    def crawl_images(self, count=1000, delay=1.0, max_retries=3):
        """
        Crawl một số lượng ảnh captcha từ GDT

        Args:
            count: Số lượng ảnh cần crawl
            delay: Thời gian delay giữa các request (seconds)
            max_retries: Số lần retry tối đa cho mỗi ảnh

        Returns:
            dict: Thống kê kết quả crawl
        """
        print(f"Bat dau crawl {count} anh captcha tu GDT...")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata file: {self.metadata_file}")

        successful_downloads = 0
        failed_downloads = 0
        metadata_records = []

        # Progress bar
        with tqdm(total=count, desc="Crawling images") as pbar:
            for i in range(count):
                result = self._download_single_image(i + 1, count, delay, max_retries)
                metadata_records.append(result)

                if result['download_status'] == 'success':
                    successful_downloads += 1
                    pbar.set_postfix({
                        'Success': successful_downloads,
                        'Failed': failed_downloads,
                        'Rate': f"{successful_downloads/(i+1)*100:.1f}%"
                    })
                else:
                    failed_downloads += 1

                pbar.update(1)

        # Ghi metadata vào CSV
        self._save_metadata(metadata_records)

        # Thống kê kết quả
        stats = {
            'total_requested': count,
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'success_rate': successful_downloads / count * 100,
            'output_directory': self.output_dir,
            'metadata_file': self.metadata_file
        }

        print(f"\n=== Ket qua crawl ===")
        print(f"Tong so anh yeu cau: {stats['total_requested']}")
        print(f"Download thanh cong: {stats['successful_downloads']}")
        print(f"Download that bai: {stats['failed_downloads']}")
        print(f"Ty le thanh cong: {stats['success_rate']:.1f}%")
        print(f"Anh duoc luu tai: {stats['output_directory']}")
        print(f"Metadata duoc luu tai: {stats['metadata_file']}")

        return stats

    def _save_metadata(self, records):
        """Lưu metadata vào CSV file"""
        with open(self.metadata_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for record in records:
                writer.writerow([
                    record['filename'],
                    record['uid'],
                    record['url'],
                    record['timestamp'],
                    record['download_status'],
                    record['file_size'],
                    record['image_shape']
                ])

def main():
    parser = argparse.ArgumentParser(description='Crawl captcha images from GDT website')
    parser.add_argument('-c', '--count', type=int, default=1000,
                       help='Number of images to download (default: 1000)')
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('-r', '--retries', type=int, default=3,
                       help='Maximum number of retries per image (default: 3)')
    parser.add_argument('-o', '--output', type=str, default='image_crawl/raw_captcha_images',
                       help='Output directory for images')
    parser.add_argument('-m', '--metadata', type=str, default='image_crawl/gdt_metadata.csv',
                       help='Metadata CSV file path')

    args = parser.parse_args()

    # Tạo crawler instance
    crawler = GDTCaptchaCrawler(
        output_dir=args.output,
        metadata_file=args.metadata
    )

    # Bắt đầu crawl
    try:
        stats = crawler.crawl_images(
            count=args.count,
            delay=args.delay,
            max_retries=args.retries
        )

        if stats['success_rate'] < 80:
            print(f"\nWARNING: Ty le thanh cong thap ({stats['success_rate']:.1f}%)")
            print("Co the do rate limiting hoac van de ket noi. Hay thu tang delay time.")

    except KeyboardInterrupt:
        print("\n\nWARNING: Da dung crawl theo yeu cau nguoi dung")
    except Exception as e:
        print(f"\nERROR: Loi trong qua trinh crawl: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
