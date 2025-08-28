#!/bin/bash

# Specify the directory
DIRECTORY="image_crawl/ocr_images"

# Remove all .png files in the specified directory
find $DIRECTORY -name '*.png' -exec rm -f {} +