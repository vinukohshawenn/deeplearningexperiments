# Experiment 3 — Convolutional Neural Network for CIFAR-10 Image Classification

**CS3807 – Deep Learning Laboratory**
Shiv Nadar University Chennai | B.Tech AI & Data Science, Semester V | AY 2026–27

Implementation of a Convolutional Neural Network (CNN) in TensorFlow/Keras for multi-class image classification on CIFAR-10, including convolutional feature map visualization for model interpretability.

## Overview

- Architecture: `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → Flatten → Dense(128) → Dropout(0.5) → Dense(10, Softmax)`
- All convolutions use 3×3 kernels, ReLU activation, and same padding
- Trained for 10 epochs, batch size 64, Adam optimizer, sparse categorical cross-entropy loss, 20% validation split
- Evaluated on the CIFAR-10 test set with full classification report and confusion matrix
- Convolutional feature maps visualized layer-by-layer for a sample test image

## Results Summary

| Metric | Value |
|---|---|
| Test Accuracy | 0.7345 |
| Test Loss | 0.8070 |
| Precision (macro) | 0.74 |
| Recall (macro) | 0.73 |
| F1-score (macro) | 0.74 |
| Trainable Parameters | 1,143,242 |
| Final Training Accuracy | 0.7801 |
| Final Validation Accuracy | 0.7358 |

**Best-performing class:** Automobile (F1 = 0.86)
**Worst-performing class:** Cat (F1 = 0.55) — frequently confused with Dog and Bird

Full per-class metrics, epoch-wise training log, and discussion are in [`report.pdf`](./report.pdf).

## Dataset

- **CIFAR-10**: 50,000 training images, 10,000 testing images, 10 classes, 32×32 RGB
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Loaded via `tf.keras.datasets.cifar10.load_data()` — no manual download required
- Source: [Krizhevsky, 2009 — Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)

## Repository Structure

```
.
├── source_code.py              # Full experiment script (data loading through feature map visualization)
├── report.pdf                  # Compiled report
├── figures/                    # All generated plots (referenced by report.tex)
│   ├── fig_sample_images.png
│   ├── fig_class_distribution.png
│   ├── fig_acc_loss_curves.png
│   ├── fig_confusion_matrix.png
│   ├── fig_featuremap_conv1.png
│   ├── fig_featuremap_conv2.png
│   ├── fig_featuremap_conv3.png
│   └── fig_sample_prediction.png
└── README.md
```

## Dependencies

```
tensorflow>=2.15
scikit-learn>=1.4
numpy
matplotlib
```

Install with:

```bash
pip install tensorflow scikit-learn numpy matplotlib
```

No version pinning is required for this experiment (unlike Experiment 2, this does not use SciKeras/`RandomizedSearchCV`, so there is no `scikit-learn`/`scikeras` compatibility conflict to work around).

## How to Run

1. Install dependencies (see above).
2. Run the full experiment:
   ```bash
   python source_code.py
   ```
3. The script will:
   - Download CIFAR-10 automatically (first run only, ~170 MB)
   - Build and train the CNN for 10 epochs
   - Evaluate on the test set (accuracy, classification report, confusion matrix)
   - Visualize convolutional feature maps for a sample test image
   - Display a sample prediction against its ground-truth label

> **Note:** the CIFAR-10 download can take several minutes on a slow connection. Training on CPU takes roughly 20 minutes for 10 epochs; a GPU runtime (e.g., Colab) reduces this to under 2 minutes.

## Reproducing the Report

The report (`report.tex`) is written for Overleaf/`pdflatex`. To compile:

1. Upload `report.tex`, `source_code.py`, and the `figures/` folder to your Overleaf project (or a local LaTeX install), keeping them in the same root directory.
2. Compile with `pdflatex` (two passes recommended for cross-references).

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. — Gradient-Based Learning Applied to Document Recognition, *Proceedings of the IEEE*, 1998.
2. Krizhevsky, A. — Learning Multiple Layers of Features from Tiny Images, Technical Report, University of Toronto, 2009. (CIFAR-10 Dataset)
3. Goodfellow, I., Bengio, Y., & Courville, A. — *Deep Learning*, MIT Press, 2016.
4. [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
