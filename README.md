# Enhancing TissueMNIST Classification via Forensic Data Analysis and ConvNeXt

## Overview

Classifying human tissue types from low-resolution medical images is a challenging task due to noise, blur, and class imbalance. The **TissueMNIST** dataset presents these exact challenges.

This project proposes a **forensic data analysis–driven pipeline** that focuses on improving **data quality before model complexity**, combined with a **ConvNeXt V2 architecture** for robust classification.

---

## Key Features

* **Forensic Preprocessing Pipeline**

  * Downsampling to **64×64** using area interpolation
  * Removes artificial upsampling noise

* **CLAHE Enhancement**

  * Improves contrast and reveals hidden micro-features
  * Handles pixel intensity skewness effectively

* **ConvNeXt V2 (Tiny)**

  * Large kernel (7×7) depthwise convolutions
  * Captures texture patterns like cellular density and structures

* **Class Imbalance Handling**

  * Weighted Cross Entropy Loss
  * Class weights computed inversely proportional to frequency

---

## Dataset

**TissueMNIST (MedMNIST v2)**

* Total Images: **236,386**
* Training: **165,466**
* Validation: **23,640**
* Test: **47,280**

### Classes (8)

* Kidney Cortex
* Kidney Medulla
* Kidney Pelvis
* Lung Adeno
* Lung Squamous
* Pancreas Ductal
* Pancreas Neuro
* Pancreas Solid

---

## Results

* **Test Accuracy:** 74.27%
* **AUC-ROC (OvR):** 0.901
* Strong improvement over baseline CNN architectures

---

## Tech Stack

* **PyTorch**
* **timm** (ConvNeXt V2)
* **Albumentations**
* **Scikit-learn**
* **Matplotlib / Seaborn**

---

## Project Structure

```
├── Main.py
├── data/
│   └── tissue_64_enhanced.npz
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── metrics.json
└── README.md
```

---

## Training Pipeline

1. **Data Loading**

   * Load preprocessed `.npz` dataset (64×64 enhanced images)

2. **Data Augmentation**

   * Horizontal & Vertical Flip
   * Random Rotation (±90°)
   * Coarse Dropout
   * Normalization

3. **Model Training**

   * Model: `convnextv2_tiny`
   * Optimizer: **AdamW**
   * Learning Rate: `2e-4`
   * Weight Decay: `0.05`
   * Scheduler: Cosine Annealing Warm Restarts
   * Mixed Precision (AMP)

4. **Evaluation**

   * Accuracy
   * Macro F1 Score
   * ROC-AUC
   * Confusion Matrix
   * ROC Curves

---

## How to Run

```bash
# Install dependencies
pip install torch timm albumentations scikit-learn matplotlib seaborn

# Run training
python Main.py
```

> Ensure dataset file is available:

```
data/tissue_64_enhanced.npz
```

---

## Authors

* **Nikunj Garg**
* **Priyanshu**
* **Gagandeep Singh**
* **Sanket**
  (Netaji Subhas University of Technology, New Delhi)

---

## Summary (TL;DR)

Instead of blindly increasing model complexity:
👉 Improve **data quality first**
👉 Use **CLAHE + proper downsampling**
👉 Then apply a strong architecture (**ConvNeXt**)

Result: **Robust performance on low-quality medical images**
