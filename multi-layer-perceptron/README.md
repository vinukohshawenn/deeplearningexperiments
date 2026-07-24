# Experiment 2 — Multi-Layer Perceptron for Fashion-MNIST Classification

Implementation of a Multi-Layer Perceptron (MLP) in TensorFlow/Keras for multi-class image classification on Fashion-MNIST, including automated hyperparameter optimization via `RandomizedSearchCV` + SciKeras.

## Overview

- Baseline MLP: `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
- Trained for 20 epochs, batch size 32, Adam optimizer, categorical cross-entropy loss
- Hyperparameter search space (hidden layers, neurons, learning rate, batch size, epochs, optimizer, activation, dropout) explored using `RandomizedSearchCV` with 5-fold cross-validation (20 candidates, 100 total fits)
- Optimized model retrained on the full training set and compared against the baseline

## Results Summary

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 0.8820 | 0.8825 |
| Precision (macro) | 0.8837 | 0.8826 |
| Recall (macro) | 0.8820 | 0.8825 |
| F1-score (macro) | 0.8799 | 0.8821 |
| Training Time | 125.14 s | 66.81 s |

**Best hyperparameters found:** 1 hidden layer, 256 neurons, learning rate 0.001, batch size 128, Adam optimizer, Tanh activation, 20 epochs, dropout 0.2 (cross-validation accuracy: 0.8375).

Full results, plots, and discussion are in [`report.pdf`](./report.pdf).

## Dataset

- **Fashion-MNIST**: 60,000 training images, 10,000 testing images, 10 classes, 28×28 grayscale
- Loaded via `keras.datasets.fashion_mnist.load_data()` — no manual download required
- Source: [Zalando Research / Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## Repository Structure

```
.
├── source_code.py          # Full experiment script (data loading through final comparison)
├── report.tex               # LaTeX report source
├── report.pdf                # Compiled report
├── figures/                  # All generated plots (referenced by report.tex)
│   ├── fig_sample_images.png
│   ├── fig_class_distribution.png
│   ├── fig_train_acc.png
│   ├── fig_val_acc.png
│   ├── fig_train_loss.png
│   ├── fig_val_loss.png
│   ├── fig_confusion_matrix.png
│   ├── fig_hp_search.png
│   └── fig_model_comparison.png
└── README.md
```

## Dependencies

```
tensorflow>=2.20
scikit-learn==1.5.2
scikeras==0.13.0
numpy
pandas
matplotlib
seaborn
```

Install with:

```bash
pip install -U scikit-learn==1.5.2 scikeras==0.13.0
pip install tensorflow numpy pandas matplotlib seaborn
```

## How to Run

1. Install dependencies (see above).
2. Run the full experiment:
   ```bash
   python source_code.py
   ```
3. The script will:
   - Download Fashion-MNIST automatically (first run only)
   - Train and evaluate the baseline MLP
   - Run `RandomizedSearchCV` hyperparameter search (≈20 minutes on CPU; faster on GPU)
   - Retrain and evaluate the optimized model
   - Save all plots and CSV summaries

> **Note:** the hyperparameter search runs on a stratified 6,000-image subsample of the training set (not the full 60,000) to keep search time tractable — this is standard practice for RandomizedSearchCV on large datasets. The final optimized model is retrained on the full training set.

## Reproducing the Report

The report (`report.tex`) is written for Overleaf/`pdflatex`. To compile:

1. Upload `report.tex`, `source_code.py`, and the `figures/` folder to your Overleaf project (or a local LaTeX install), keeping them in the same root directory.
2. Compile with `pdflatex` (two passes recommended for cross-references).

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. — *Deep Learning*, MIT Press, 2016.
2. Bishop, C. M. — *Pattern Recognition and Machine Learning*, Springer, 2006.
3. Haykin, S. — *Neural Networks and Learning Machines*, Pearson, 2009.
4. Xiao, H., Rasul, K., & Vollgraf, R. — Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017.
5. [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
6. [SciKeras Documentation](https://www.adriangb.com/scikeras/)
