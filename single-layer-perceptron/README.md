# Single Layer Perceptron from Scratch

A complete implementation of a **Single Layer Perceptron** from scratch using Python for binary classification on the **UCI Banknote Authentication Dataset**. This project demonstrates the complete machine learning workflow—from exploratory data analysis and feature preprocessing to implementing the Perceptron learning algorithm, evaluating model performance, and comparing it with Scikit-learn's implementation.

---

## Overview

The objective of this project is to understand the fundamentals of an artificial neuron by implementing a Single Layer Perceptron without relying on machine learning libraries for the learning algorithm.

The project includes:

- Exploratory Data Analysis (EDA)
- Feature normalization
- Custom Perceptron implementation
- Model training using the Perceptron Learning Rule
- Performance evaluation
- Learning rate analysis
- Comparison with Scikit-learn's Perceptron

---

## Dataset

**Dataset:** Banknote Authentication Dataset

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/267/banknote+authentication

### Dataset Summary

| Attribute | Value |
|-----------|-------|
| Samples | 1372 |
| Features | 4 |
| Classes | 2 |
| Missing Values | None |

### Features

- Variance
- Skewness
- Kurtosis
- Entropy

Target Classes:

- **0** → Authentic Banknote
- **1** → Forged Banknote

---

## Exploratory Data Analysis

The following analyses were performed before model training:

- Feature Histograms
- Correlation Heatmap
- Pairwise Scatter Plots
- Variance vs Skewness Visualization
- Feature Boxplots

The dataset shows that the two classes are approximately linearly separable, making it suitable for a Single Layer Perceptron.

---

## Data Preprocessing

Before training:

- Missing values were checked
- Numerical features were standardized using **StandardScaler**
- Dataset split into:

  - **80% Training**
  - **20% Testing**

---

## Perceptron Implementation

The model was implemented completely from scratch using NumPy.

Implemented components include:

- Random Weight Initialization
- Bias Initialization
- Step Activation Function
- Forward Propagation
- Online Perceptron Learning Rule
- Epoch-wise Error Tracking
- Weight Evolution
- Bias Evolution

No machine learning library was used for the custom implementation.

---

## Model Evaluation

Performance was evaluated using

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Results

| Metric | Score |
|--------|-------|
| Accuracy | **98.91%** |
| Precision | **98.37%** |
| Recall | **99.18%** |
| F1-score | **98.78%** |

Only **3 test samples** were misclassified.

---

## Learning Rate Comparison

Three learning rates were evaluated.

| Learning Rate | Accuracy |
|--------------|----------|
| 0.001 | 98.55% |
| **0.01** | **98.91%** |
| 0.1 | 98.55% |

The learning rate **0.01** achieved the best balance between convergence speed and stability.

---

## Comparison with Scikit-learn

The custom implementation was compared against Scikit-learn's built-in Perceptron classifier.

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Custom Perceptron | **98.91%** | **98.37%** | 99.18% | **98.78%** |
| Scikit-learn Perceptron | 98.55% | 96.83% | **100%** | 98.39% |

The custom implementation achieved comparable performance while demonstrating the underlying Perceptron learning algorithm.

---

## Visualizations

The project includes:

- Feature Histograms
- Correlation Heatmap
- Pairwise Scatter Plots
- Variance vs Skewness Scatter Plot
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
├── notebook.ipynb
├── report.pdf
├── README.md
└── requirements.txt
```

---

## Key Learnings

- Fundamentals of Artificial Neurons
- Perceptron Learning Algorithm
- Binary Classification
- Feature Standardization
- Linear Separability
- Model Evaluation Metrics
- Hyperparameter Analysis
- Comparison between custom implementations and machine learning libraries

---

## References

- Frank Rosenblatt — *The Perceptron (1958)*
- UCI Machine Learning Repository
- Scikit-learn Documentation
- Deep Learning by Goodfellow, Bengio & Courville
