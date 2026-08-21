# CNN on CIFAR-10 — Experiment 3 (CS3807 Deep Learning Lab)

Convolutional Neural Network for multi-class image classification on CIFAR-10, built in TensorFlow/Keras.

## Architecture

```
Conv2D(32, 3x3, ReLU, same) → MaxPool(2x2)
Conv2D(64, 3x3, ReLU, same) → MaxPool(2x2)
Conv2D(128, 3x3, ReLU, same)
Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Batch size: 64
- Epochs: 10
- Trainable params: 1,143,242

## Dataset

CIFAR-10 — 50,000 train / 10,000 test images, 32×32×3 RGB, 10 classes. Loaded via `tf.keras.datasets.cifar10`, pixel values normalized to [0, 1].

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 0.7345 |
| Test Loss | 0.8070 |
| Macro F1 | 0.74 |
| Final Train Acc | 0.7801 |
| Final Val Acc | 0.7358 |

Best class: Automobile (F1 = 0.86). Worst class: Cat (F1 = 0.55) — confused mainly with Dog and Bird. Mild overfitting emerges after ~epoch 7 (val loss flattens/ticks up while train loss keeps falling).

Full classification report, confusion matrix, and per-layer feature map visualizations are in the [lab report](./Experiment_3.pdf).

## Files

```
.
├── cnn_cifar.ipynb          # dataset loading, model, training, evaluation, feature maps
├── requirements.txt
├── Experiment_3.pdf        # compiled report
└── figs/                   # generated plots (class dist, acc/loss, confusion matrix, feature maps, sample pred)
```

## Setup

```bash
pip install -r requirements.txt
python cnn_cifar.ipnyb
```

Trains the model, prints the classification report to stdout, and saves all figures to `figs/`.

## Requirements

```
tensorflow>=2.15
numpy
matplotlib
scikit-learn
seaborn
```
