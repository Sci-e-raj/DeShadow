# DeShadow: Automated Shadow Detection & Removal

> A comprehensive computer vision pipeline for automated shadow detection and illumination restoration, leveraging both classical morphological processing and deep learning segmentation.

## 📖 Overview

Shadows in images can obscure details, confuse object detection models, and degrade visual quality. This project tackles the complex problem of shadow segmentation and removal using a hybrid approach:

1. **Classical Computer Vision (`final.ipynb`)**: Mathematical image processing pipeline using OpenCV involving color space transformations (LAB/HSV), K-Means clustering, mean-shift binarization, and morphological filtering.
2. **Deep Learning (`experiments/model.ipynb`)**: A custom-built PyTorch U-Net architecture trained to generate precise semantic segmentation masks of shadow regions via Dice Loss optimization.

## ✨ Key Features

- **Multi-Method Detection**: Choose between `rgb_lab_hsv` K-Means clustering, `mean_shift_binarization`, or strict `lab_thresholding` for optimal shadow mask extraction.
- **Intelligent Illumination Restoration**: Calculates luminosity ratios between shadow cores and non-shadow borders to mathematically boost brightness without washing out details.
- **Edge Blending**: Employs targeted edge median filters strictly within HSV color bounds to seamlessly blend the transition rings around restored regions.
- **Deep Learning Alternative**: Custom PyTorch U-Net architecture built from scratch for modern AI-driven semantic segmentation.

## 🚀 Quick Start (Classical CV)

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
pip install opencv-python matplotlib numpy scikit-image jupyter torch torchvision
```

### Usage

Open `final.ipynb` and run the `ShadowRemover` pipeline on your images:

```python
# 1. Initialize the pipeline with tunable hyperparameters
remover = ShadowRemover(
    itr=3,
    method='mean_shift_binarization',
    lab_adjustment=True,
    shadow_dilation_iteration=7,
    shadow_size_threshold=20,
    plot_results=True
)

# 2. Load your image
remover.set_image("Images/your_test_image.jpg", use_blurring=False)

# 3. Detect and remove shadows!
original, masks, restored = remover.remove_shadows()
```

## 📁 Project Structure

```text
├── Images/                         # Sample datasets and input images
├── final.ipynb                     # Main classical CV pipeline (OpenCV implementation)
└── README.md
```

## 🧠 Architecture Details

### 1. Mask Calculation (Detection)

The classical pipeline extracts potential shadow pixels by mapping the image across RGB, LAB, and HSV color spaces. It uses **K-Means Clustering (K=3)** on 9-dimensional pixel vectors to group lighting variations, followed by **Otsu's Thresholding** to create a binary shadow mask.

### 2. Edge Processing (Cleanup)

Morphological transformations (elliptical dilation/opening) clean up "freckle" artifacts, guaranteeing only continuous shadow areas above the `shadow_size_threshold` are targeted.

### 3. Luminosity Restoration (Removal)

The system isolates the immediate boundary surrounding the detected shadow. We calculate the `border_average / shadow_average` LAB ratio, allowing the engine to mathematically multiply the shadow core's brightness to gracefully match the surrounding ambient lighting.
