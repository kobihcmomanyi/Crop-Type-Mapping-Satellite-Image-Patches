# Crop Type Mapping from Satellite Image Patches

## Overview

This project develops and compares deep learning models for classifying crop types from satellite image patches. A custom CNN architecture is benchmarked against three transfer learning models — MobileNetV2, ResNet50, and EfficientNetB0 — using RGB satellite imagery from the EuroSAT dataset. Grad-CAM is applied to visualise and explain model predictions.

**Crop classes:** `AnnualCrop` · `Pasture` · `PermanentCrop`

---

## Requirements

### Hardware
A GPU is strongly recommended. The notebook was developed and tested on **Google Colab with a T4 GPU**.

### Python Dependencies

Install the required packages with:

```bash
pip install tensorflow scikit-learn matplotlib numpy pandas pillow requests
```

All other imports (`os`, `shutil`, `zipfile`, `hashlib`, etc.) are part of the Python standard library.

---

## How to Run

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `crop_type_mapping_from_satellite_image_patches.ipynb` via **File → Upload notebook**.
3. Enable GPU: go to **Runtime → Change runtime type → T4 GPU**.
4. Run all cells in order: **Runtime → Run all**.

The notebook will automatically download and extract the EuroSAT dataset — no manual data download is needed.

### Option B — Local Jupyter Environment

1. Clone or download this repository.

2. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy pandas pillow requests jupyter
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook crop_type_mapping_from_satellite_image_patches.ipynb
   ```

4. Run all cells from top to bottom. The dataset will be downloaded automatically to a `data/` folder in the working directory.

> **Note:** Training four models for up to 30 epochs each is computationally intensive. Expect significantly longer runtimes without a GPU.

---

## Pipeline Summary

The notebook executes the following steps in sequence:

| Step | Description |
|------|-------------|
| **1. Dataset Download** | Downloads EuroSAT.zip (~90 MB) from the DFKI server and extracts it |
| **2. Dataset Cleaning** | Removes unwanted class folders; detects corrupted, duplicate, and blurry images |
| **3. Preprocessing** | Resizes all images to 224×224, normalises pixel values to [0, 1] |
| **4. Data Augmentation** | Applies rotation, flipping, zoom, brightness, contrast, and translation |
| **5. Train/Val/Test Split** | Stratified split: 70% train / 15% validation / 15% test |
| **6. Model Building** | Builds Custom CNN, MobileNetV2, ResNet50, and EfficientNetB0 |
| **7. Model Training** | Trains all four models with Early Stopping and Learning Rate Scheduling |
| **8. Evaluation** | Reports accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and inference time |
| **9. Grad-CAM Analysis** | Generates activation heatmaps for correct predictions, false positives, and false negatives |

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Name | EuroSAT RGB |
| Source | https://madm.dfki.de/files/sentinel/EuroSAT.zip |
| Image type | RGB satellite patches (Sentinel-2) |
| Image size | 64×64 px (resized to 224×224 for training) |
| Total images used | 7,500 |
| AnnualCrop | 3,000 |
| PermanentCrop | 2,500 |
| Pasture | 2,000 |

---

## Models

- **Custom CNN** — Three convolutional blocks (32 → 64 → 128 filters), Global Average Pooling, Dense(128), Dropout(0.3), Softmax output
- **MobileNetV2** — ImageNet pretrained; frozen base + custom classification head; fine-tuned upper layers
- **ResNet50** — ImageNet pretrained; frozen base + custom classification head; fine-tuned upper layers
- **EfficientNetB0** — ImageNet pretrained; frozen base + custom classification head; fine-tuned upper layers


