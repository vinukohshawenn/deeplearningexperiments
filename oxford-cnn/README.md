# Experiment 5 — CNN Training, Regularization, Optimization, Transfer Learning & Cross-Validation

This folder contains the completed lab report for **CS3807 – Deep Learning Laboratory,
Experiment 5**, based on a MobileNetV2 / Oxford-IIIT Pet dataset run recorded in
`cnn-oxford.ipynb`.

## Files

| File | Description |
|---|---|
| `experiment_5_1_updated.pdf` | Compiled PDF (14 pages), ready to submit as-is. |
| `images/` | 14 PNG plots extracted directly from the notebook's saved cell outputs (see table below). |


## Where the numbers came from

Every figure and every filled-in table cell is taken directly from a single run of
`cnn-oxford.ipynb` (5 epochs per configuration unless noted). Nothing was estimated or
invented — where the notebook didn't evaluate a particular stage on the CV split or the
held-out test set (e.g. the individual "Best Initialization" / "Best Optimizer" runs),
the report reports the validation accuracy that *was* measured for that stage instead,
and says so explicitly (Section 13).

### Plot-to-file mapping

| Plot # | Section |
|---|---|
| 1 | Weight Initialization — Training Loss |
| 2 | Weight Initialization — Validation Accuracy | 
| 3 | Regularization — Train/Val Accuracy |
| 4 | Regularization — Train/Val Loss |
| 5 | Batch Normalization — With vs. Without | 
| 6 | Optimizers — Training Loss | 
| 7 | Optimizers — Validation Accuracy | 
| 8 | Hyperparameters — Learning Rate |
| 9 | Hyperparameters — Batch Size | 
| 10 | Hyperparameters — Dropout Rate |
| 11 | Transfer Learning — Feature Extraction vs. Fine-Tuning | 
| 12 | Transfer Learning — Train/Val Loss | 
| 13 | 5-Fold Cross-Validation Accuracy | 
| 14 | Confusion Matrix (final model) |

### Key results summary

- **Best weight initialization:** He (16.85% best val. accuracy)
- **Best regularization:** L2 (18.12% best val. accuracy)
- **Best optimizer:** Adam (14.86% best val. accuracy)
- **Best hyperparameters:** lr = 0.001, batch size = 32, dropout = 0.0
- **Best 5-fold CV configuration:** C4 (lr = 0.001, batch = 16, dropout = 0.0) → 13.56% ± 0.74%
- **Final test accuracy:** 14.88% (Precision 13.25%, Recall 14.88%, F1 10.90%)
- **Transfer learning (pretrained MobileNetV2):** Feature extraction reached 86.41%
  val. accuracy; fine-tuning the last 4 layers improved this to 88.22%.

Note the large gap between the from-scratch CNN results (Sections 5–9 and the final
CV/test model, ~11–19%) and the transfer-learning results (Section 10, ~86–88%) — this
is expected on a 37-class fine-grained dataset with a small training set and only 5
epochs per run, and is a useful discussion point for the report's Discussion Questions
(Section 15).
