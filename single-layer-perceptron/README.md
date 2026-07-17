# Single Layer Perceptron from Scratch

A NumPy implementation of the **Single Layer Perceptron** for binary classification, built entirely from scratch without using machine learning libraries for the learning algorithm. The project demonstrates the complete machine learning workflow, including exploratory data analysis, feature preprocessing, model training, performance evaluation, hyperparameter analysis, and comparison with Scikit-learn.

---

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing using feature standardization
- Single Layer Perceptron implemented from scratch
- Step activation function
- Online Perceptron Learning Rule
- Training error visualization
- Weight and bias evolution tracking
- Learning rate comparison
- Performance evaluation using multiple metrics
- Benchmark against Scikit-learn's Perceptron

---

## Dataset

**Banknote Authentication Dataset**

The dataset contains extracted statistical features from genuine and forged banknotes.

| Property | Value |
|----------|------:|
| Samples | 1372 |
| Features | 4 |
| Classes | 2 |

### Features

- Variance
- Skewness
- Kurtosis
- Entropy

Target labels:

- **0** → Authentic
- **1** → Forged

Dataset Source:
https://archive.ics.uci.edu/dataset/267/banknote+authentication

---

## Project Workflow

```
Dataset
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Standardization
    │
    ▼
Train-Test Split
    │
    ▼
Perceptron Implementation
    │
    ▼
Model Training
    │
    ▼
Performance Evaluation
    │
    ▼
Learning Rate Analysis
    │
    ▼
Scikit-learn Comparison
```

---

## Exploratory Data Analysis

The notebook includes

- Histograms
- Correlation Heatmap
- Pairwise Scatter Plots
- Variance vs Skewness Scatter Plot
- Boxplots

These visualizations provide insights into feature distributions, class separability, correlations, and outliers before training the model.

---

## Data Preprocessing

Before training, the following preprocessing steps were applied:

- Missing value verification
- Feature standardization using `StandardScaler`
- Stratified 80:20 train-test split

---

## Perceptron Implementation

The model was implemented entirely using NumPy.

### Components

- Random weight initialization
- Bias initialization
- Step activation function
- Forward propagation
- Online weight updates
- Bias updates
- Epoch-wise training
- Error tracking

Unlike Scikit-learn, every step of the learning algorithm is implemented manually to better understand the Perceptron learning process.

---

## Performance

| Metric | Score |
|---------|-------:|
| Accuracy | **98.91%** |
| Precision | **98.37%** |
| Recall | **99.18%** |
| F1 Score | **98.78%** |

The custom implementation correctly classified **272 out of 275** test samples.

---

## Learning Rate Analysis

Three different learning rates were evaluated.

| Learning Rate | Test Accuracy |
|--------------:|--------------:|
| 0.001 | 98.55% |
| **0.01** | **98.91%** |
| 0.1 | 98.55% |

The experiments show that:

- **0.001** converges smoothly but slowly.
- **0.1** converges faster but exhibits larger oscillations.
- **0.01** provides the best trade-off between convergence speed and stability.

---

## Comparison with Scikit-learn

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------:|----------:|--------:|---------:|
| Custom Perceptron | **98.91%** | **98.37%** | 99.18% | **98.78%** |
| Scikit-learn Perceptron | 98.55% | 96.83% | **100.00%** | 98.39% |

The manually implemented Perceptron achieves performance comparable to Scikit-learn while providing complete transparency into the learning algorithm.

---

## Visualizations

The notebook generates:

- Feature Histograms
- Correlation Heatmap
- Pairwise Feature Scatter Plots
- Variance vs Skewness Class Distribution
- Feature Boxplots
- Training Error vs Epoch
- Weight Evolution
- Bias Evolution
- Confusion Matrix
- Learning Rate Comparison

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Repository Structure

```
.
├── data/
│   └── banknote_authentication.csv
├── perceptron_exp.ipynb
├── report.pdf
└── README.md
```

## References

- Frank Rosenblatt — *The Perceptron (1958)*
- UCI Machine Learning Repository
- Scikit-learn Documentation
