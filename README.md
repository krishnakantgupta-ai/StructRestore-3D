# StructRestore-3D

Self-Supervised Deep Learning for FIB-SEM Microscopy Restoration

## Overview

StructRestore-3D is a research prototype for investigating
self-supervised restoration of three-dimensional FIB-SEM
microscopy images while preserving cellular ultrastructure.

## Implemented Models

1. Single-slice convolutional neural network
2. Plain 2.5D convolutional neural network
3. Reliability-gated 2.5D convolutional neural network

## Dataset

EMPIAR-10311: HeLa FIB-SEM microscopy dataset.

DOI: https://doi.org/10.6019/EMPIAR-10311

## Features

- Self-supervised model training
- Masked-pixel reconstruction
- Direct target-leakage auditing
- Patch-quality diagnostics
- Multi-slice inference
- Quantitative benchmarking using PSNR and SSIM
- Neighboring-section alignment analysis

## Preliminary Results

Six models were evaluated against noisy-input and conventional
filtering baselines. The present models have not demonstrated
superior denoising performance. Fine-structure preservation
remains an important research challenge.

## Requirements

Python 3.12, PyTorch, NumPy, SciPy, tifffile,
scikit-image and Matplotlib.

## Author

Dr. Krishna Kant Gupta

Rajiv Gandhi Institute of Information Technology and Biotechnology

Bharati Vidyapeeth (Deemed to be University), Pune, India

## Research Status

Preliminary proof of concept. Further validation is ongoing.
