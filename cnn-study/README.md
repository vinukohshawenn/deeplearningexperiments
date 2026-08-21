# Experiment 4 — Transfer Learning with MobileNetV2 on CIFAR-10

**CS3807 – Deep Learning Laboratory**
Shiv Nadar University Chennai | B.Tech AI & Data Science, Semester V | AY 2026–27

Comparative study of deep CNN architectures (LeNet-5, AlexNet, VGG16, GoogleNet, ResNet) and practical implementation of transfer learning using a pretrained MobileNetV2 backbone, fine-tuned on CIFAR-10.

## Overview

- Pretrained base: **MobileNetV2** (ImageNet weights, `include_top=False`)
- CIFAR-10 images (32×32×3) resized to 96×96×3 and rescaled to MobileNetV2's expected `[-1, 1]` input range
- Classifier head: `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(10, Softmax)`
- **Phase 1 (frozen base):** 15 epochs, Adam @ lr=0.001, batch size 32
- **Phase 2 (fine-tuning):** last 20 layers of the base unfrozen, 7 more epochs, Adam @ lr=0.0001

## Results Summary

| Metric | Value |
|---|---|
| Test Accuracy (before fine-tuning) | 86.24% |
| Test Accuracy (after fine-tuning) | **87.55%** |
| Precision (weighted) | 0.8801 |
| Recall (weighted) | 0.8755 |
| F1-score (weighted) | 0.8752 |
| Total Parameters | 2,423,242 |
| Trainable Parameters (frozen-base phase) | 165,258 |
| Training Time (both phases) | ≈354 s |

**Best-performing class:** Automobile (F1 = 0.94)
**Worst-performing class:** Dog (F1 = 0.80) — recall only 0.70, frequently confused with Cat

Fine-tuning improved test accuracy by **+1.31 percentage points**. Full per-class metrics, epoch-wise training log, and discussion are in [`report.pdf`](./report.pdf).

## Dataset

- **CIFAR-10**: 50,000 training images, 10,000 testing images, 10 classes, 32×32 RGB
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Loaded via `tf.keras.datasets.cifar10.load_data()` — no manual download required
- Source: [Krizhevsky, 2009 — Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)

## Repository Structure

```
.
├── source_code.py                    # Full experiment script (data loading through evaluation)
├── report.pdf                        # Compiled report
├── figures/                          # All generated plots (referenced by report.tex)
└── README.md
```

## Dependencies

```
tensorflow>=2.15
scikit-learn>=1.4
numpy
pandas
matplotlib
seaborn
```

Install with:

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```

No version pinning is required for this experiment — it does not use SciKeras or `RandomizedSearchCV`/`GridSearchCV`, so there's no `scikit-learn`/`scikeras` compatibility conflict to work around (unlike Experiment 2).

## How to Run

1. Install dependencies (see above).
2. Run the full experiment:
   ```bash
   python cnn-cifar.py
   ```
3. The script will:
   - Download CIFAR-10 automatically (first run only, ~170 MB)
   - Download MobileNetV2 ImageNet weights automatically (first run only, ~9 MB)
   - Build the transfer learning model (frozen MobileNetV2 base + custom classifier head)
   - Train the classifier head for 15 epochs with the base frozen
   - Unfreeze the last 20 layers and fine-tune for 7 more epochs at a reduced learning rate
   - Evaluate on the test set (accuracy, precision, recall, F1, classification report, confusion matrix)
   - Generate all plots, including a before/after fine-tuning comparison and misclassified examples

> **Note:** training is fast relative to Experiments 2 and 3 because only a small classifier head (and, in phase 2, the last 20 layers of MobileNetV2) is actually being trained — roughly 354 seconds total on a GPU runtime (e.g., Colab). CPU-only training will take considerably longer.

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. — Gradient-Based Learning Applied to Document Recognition, *Proceedings of the IEEE*, 1998.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. — ImageNet Classification with Deep Convolutional Neural Networks, *NeurIPS*, 2012.
3. Simonyan, K., & Zisserman, A. — Very Deep Convolutional Networks for Large-Scale Image Recognition, *ICLR*, 2015.
4. Szegedy, C. et al. — Going Deeper with Convolutions, *CVPR*, 2015.
5. He, K., Zhang, X., Ren, S., & Sun, J. — Deep Residual Learning for Image Recognition, *CVPR*, 2016.
6. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. — MobileNetV2: Inverted Residuals and Linear Bottlenecks, *CVPR*, 2018.
7. Goodfellow, I., Bengio, Y., & Courville, A. — *Deep Learning*, MIT Press, 2016.
8. [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
