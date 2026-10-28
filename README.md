# GPU-Accelerated Batch Image Processing using CUDA

## Project Overview

This project demonstrates GPU-accelerated batch image processing using NVIDIA CUDA.

The program generates and processes a large batch of images. Each image is transferred from CPU memory to GPU memory, processed using a CUDA kernel, and then transferred back to CPU memory.

The GPU applies a 3x3 box blur to the RGB image.

## Features

- CUDA GPU acceleration
- Batch processing of 100 images
- RGB image processing
- 3x3 blur implemented as a CUDA kernel
- OpenCV used for image input/output
- Command-line control of the number of images
- Automatic generation of sample input images
- Output images saved to a separate directory
- Makefile for compilation
- Shell script for automated execution

## Technologies

- C++
- NVIDIA CUDA
- OpenCV
- Linux
- NVIDIA GPU

## GPU

The project was tested on:

- GPU: NVIDIA L4
- CUDA compiler: NVIDIA CUDA 11.3
- OpenCV: 4.2.0

## Algorithm

Each image is processed using a CUDA kernel.

For every pixel, one CUDA thread is assigned to the pixel.

The kernel reads the surrounding 3x3 neighborhood and calculates the average RGB value:

```text

## How to Run

### Requirements
- NVIDIA CUDA Toolkit
- A CUDA-capable NVIDIA GPU
- OpenCV
- GNU Make

### Build and Run

```bash
make
make run
```

If the project uses a different run command, follow the provided `run.sh` script instead.

## Execution Evidence

The `results/` directory contains the output generated during execution. The output can be used to demonstrate the image-processing operation.

## Learning Outcomes

- Understanding CUDA kernels and parallel execution
- Allocating and transferring GPU memory
- Processing image pixels in parallel
- Using OpenCV for image input and output
Output RGB = Sum of 9 neighboring RGB pixels / 9
