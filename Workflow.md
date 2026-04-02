# Workflow

## Overview
This project demonstrates a CUDA-based real-time image processing workflow on an NVIDIA Jetson edge platform. The selected task is urban green cover estimation from a live camera feed. For each frame, the system captures the image, transfers it to GPU memory, extracts the green-channel value of every pixel, and applies a parallel reduction to compute the total green intensity.

## Code Files
The main code files used in this project are:

- `droneskyview.cu` – baseline version without optimization
- `droneskyview2.cu` – version with divergence minimization optimization
- `droneskyview3.cu` – version with minimized global memory access optimization

These files show the progression from the basic CUDA implementation to the optimized versions used for performance improvement.

## System Block Diagram
```text
USB Camera / Live Webcam
          ↓
     OpenCV Frame Capture
          ↓
 Host to Device Memory Copy
          ↓
   ExtractGreenKernel (CUDA)
          ↓
  Green Channel Array on GPU
          ↓
 ReductionKernel / ReduceSum
          ↓
   Total Green Intensity Value
          ↓
 Display Output + Performance Check
