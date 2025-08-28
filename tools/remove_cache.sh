#!/bin/bash

echo "Removing __pycache__ folders..."

# Recursively remove __pycache__ folders
find . -type d -name "__pycache__" -exec rm -rf {} \;

echo "Done."