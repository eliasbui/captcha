#!/bin/bash

echo "Unzipping test data..."
unzip image_crawl/test.zip -d image_crawl/test_images 
echo "Done unzip test data."

echo "Unzipping train data..."
unzip image_crawl/train.zip -d image_crawl/train_images 
echo "Done unzip train data."