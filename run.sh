#!/usr/bin/env bash

set -e

echo "========================================"
echo "CUDA GPU Image Processing"
echo "========================================"

make clean
make build

echo ""
echo "Running GPU image processing..."
./cuda_image_processor 100

echo ""
echo "Input images:"
ls input_images | wc -l

echo "Output images:"
ls output_images | wc -l

echo ""
echo "Image data size:"
du -sh input_images output_images

echo ""
echo "Execution completed successfully."