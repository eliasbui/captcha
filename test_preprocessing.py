#!/usr/bin/env python3
"""
Test script to verify the fixed preprocessing function handles transparent backgrounds correctly
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add the ocr module to the path
sys.path.append('ocr')
from dataset.dataset_v1 import preprocess

def create_test_image_with_transparency():
    """Create a test image with transparent background"""
    # Create a 100x100 RGBA image
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    
    # Add a white background (fully opaque)
    img[:, :, :3] = 255  # RGB channels
    img[:, :, 3] = 255   # Alpha channel (fully opaque)
    
    # Add some text-like content in the center (black text)
    img[30:70, 20:80, :3] = 0  # Black text
    img[30:70, 20:80, 3] = 255  # Fully opaque
    
    # Make the background transparent (alpha = 0)
    img[0:30, :, 3] = 0  # Top transparent
    img[70:, :, 3] = 0   # Bottom transparent
    img[:, 0:20, 3] = 0  # Left transparent
    img[:, 80:, 3] = 0   # Right transparent
    
    return img

def test_preprocessing():
    """Test the preprocessing function with different image types"""
    print("Testing preprocessing function...")
    
    # Test 1: Create and test transparent background image
    print("\n1. Testing transparent background image...")
    transparent_img = create_test_image_with_transparency()
    print(f"Original image shape: {transparent_img.shape}")
    print(f"Alpha channel unique values: {np.unique(transparent_img[:,:,3])}")
    
    # Process with white background
    processed_white = preprocess(transparent_img, background_color=255)
    print(f"Processed with white background - shape: {processed_white.shape}, min/max: {processed_white.min():.3f}/{processed_white.max():.3f}")
    
    # Process with black background
    processed_black = preprocess(transparent_img, background_color=0)
    print(f"Processed with black background - shape: {processed_black.shape}, min/max: {processed_black.min():.3f}/{processed_black.max():.3f}")
    
    # Test 2: Test with grayscale image
    print("\n2. Testing grayscale image...")
    gray_img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    processed_gray = preprocess(gray_img)
    print(f"Processed grayscale - shape: {processed_gray.shape}, min/max: {processed_gray.min():.3f}/{processed_gray.max():.3f}")
    
    # Test 3: Test with BGR image
    print("\n3. Testing BGR image...")
    bgr_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    processed_bgr = preprocess(bgr_img)
    print(f"Processed BGR - shape: {processed_bgr.shape}, min/max: {processed_bgr.min():.3f}/{processed_bgr.max():.3f}")
    
    print("\n✅ All tests completed successfully!")
    return True

def test_with_real_images():
    """Test with real images from the project if available"""
    print("\n4. Testing with real images...")
    
    # Check if there are any test images available
    test_dirs = [
        "image_crawl/ocr_images",
        "image_crawl/test_images"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"Found test directory: {test_dir}")
            files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                # Test with the first image
                test_file = os.path.join(test_dir, files[0])
                print(f"Testing with: {test_file}")
                
                try:
                    # Load image
                    img = cv2.imread(test_file, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        print(f"Loaded image shape: {img.shape}")
                        
                        # Process the image
                        processed = preprocess(img)
                        print(f"Processed image shape: {processed.shape}, min/max: {processed.min():.3f}/{processed.max():.3f}")
                        print("✅ Real image test successful!")
                        return True
                    else:
                        print(f"❌ Could not load image: {test_file}")
                except Exception as e:
                    print(f"❌ Error processing real image: {e}")
            else:
                print(f"No image files found in {test_dir}")
        else:
            print(f"Test directory not found: {test_dir}")
    
    print("⚠️ No real images available for testing")
    return False

if __name__ == "__main__":
    print("🧪 Testing Preprocessing Function")
    print("=" * 50)
    
    try:
        # Run basic tests
        test_preprocessing()
        
        # Run real image tests
        test_with_real_images()
        
        print("\n🎉 All preprocessing tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
