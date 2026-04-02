# MENG3540 Parallel Programming Project Workflow

## Project Topic
Real-Time Urban Green Cover Estimation Using CUDA Reduction on Jetson Orin Nano

## Project Overview
This project presents a CUDA-based real-time image processing workflow on an NVIDIA Jetson platform. The system captures live webcam frames, transfers them to GPU memory, extracts the green-channel value from each pixel, and applies a reduction parallel pattern to compute the total green intensity of the frame. This total value is used as a simple estimate of urban green cover.

## Team Members
- Muhammad Moosa
- Adrian Anthony

## Repository Contents
- README.md
- Workflow.md
- code/
- images/

## Tools and Technologies
- NVIDIA Jetson
- CUDA
- OpenCV
- C++
- GitHub

## Objective
The objective of this project is to demonstrate a CUDA-based real-time image processing workflow for estimating urban green cover on a Jetson platform using reduction and optimization techniques.

## Code Versions
The repository includes three CUDA implementations:
- `droneskyview.cu` – baseline version
- `droneskyview2.cu` – divergence-optimized version
- `droneskyview3.cu` – global-memory-optimized version

## Repository Structure
- `README.md` – project overview
- `Workflow.md` – workflow document
- `code/` – CUDA source files
- `images/` – screenshots, diagrams, and output images
