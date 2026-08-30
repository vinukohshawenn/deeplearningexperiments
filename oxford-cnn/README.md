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

| Plot # | Section | Image file |
|---|---|---|
| 1 | Weight Initialization — Training Loss | `plot1_training_loss_init.png` |
| 2 | Weight Initialization — Validation Accuracy | `plot2_val_acc_init.png` |
| 3 | Regularization — Train/Val Accuracy | `plot3_train_val_acc_reg.png` |
| 4 | Regularization — Train/Val Loss | `plot4_train_val_loss_reg.png` |
| 5 | Batch Normalization — With vs. Without | `plot5_bn_vs_no_bn.png` |
| 6 | Optimizers — Training Loss | `plot6_train_loss_opt.png` |
| 7 | Optimizers — Validation Accuracy | `plot7_val_acc_opt.png` |
| 8 | Hyperparameters — Learning Rate | `plot8_lr_vs_acc.png` |
| 9 | Hyperparameters — Batch Size | `plot9_batch_vs_acc.png` |
| 10 | Hyperparameters — Dropout Rate | `plot10_dropout_vs_acc.png` |
| 11 | Transfer Learning — Feature Extraction vs. Fine-Tuning | `plot11_feature_vs_finetune.png` |
| 12 | Transfer Learning — Train/Val Loss | `plot12_train_val_loss_ft.png` |
| 13 | 5-Fold Cross-Validation Accuracy | `plot13_cv_accuracy.png` |
| 14 | Confusion Matrix (final model) | `plot14_confusion_matrix.png` |

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

## What's still left for the student to do

The Section 14 "Required Inference for Plots" convention and the Section 15/16
discussion questions and additional exercise are intentionally left as open write-in
tasks — those call for the student's own reasoning, not just a restatement of numbers.
Short inference paragraphs have already been added under each plot/table to jump-start
that discussion, but the discussion questions themselves are unanswered.
