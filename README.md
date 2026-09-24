# Self-Supervised Medical Image Segmentation — Representation Analysis

## Overview

This project studies self-supervised representation learning for 2D medical image segmentation.

The project is based on the research implementation **SSL-MedSeg: Self-Supervised Pretraining for 2D Medical Image Segmentation**. The original method uses BYOL-based self-supervised pretraining to learn useful visual representations from medical images.

As an extension, this project adds a **representation analysis pipeline** to examine the feature embeddings learned by the self-supervised encoder.

## Research Objective

The objective is to analyze whether self-supervised medical image representations capture meaningful differences between medical image samples.

The learned image embeddings are extracted from the pretrained encoder and projected into a 2D space using Principal Component Analysis (PCA).

## Methodology

```text
Medical Images
      ↓
Self-Supervised Encoder
      ↓
Learned Feature Embeddings
      ↓
Feature Extraction
      ↓
PCA Dimensionality Reduction
      ↓
2D Representation Visualization
