# Classification Threshold Choice

## The default threshold problem

The default threshold of 0.5 is rarely optimal. It is "optimal" only when the costs of false positives and false negatives are equal and the training set reflects the deployment class distribution — conditions that almost never hold simultaneously.

## How to choose a threshold

### By cost ratio

If a false negative costs C_FN and a false positive costs C_FP:

```
Optimal threshold = C_FP / (C_FP + C_FN)
```

| Scenario | FP cost | FN cost | Optimal threshold |
|----------|---------|---------|-------------------|
| Cancer screening | low | very high | ~0.1 (maximise recall) |
| Spam filter | moderate | low | ~0.8 (minimise false spam) |
| Fraud detection | low | high | ~0.2 (catch more fraud) |

### By Youden's J

Maximise TPR − FPR on the ROC curve:

```
J = sensitivity + specificity − 1 = TPR − FPR
```

This is threshold-agnostic (no cost assumption) and is a common default for medical diagnosis.

### By F1

Maximise F1 = 2 × precision × recall / (precision + recall). Good for information retrieval; does not account for true negatives.

### By precision-recall operating point

When negative class is abundant and you only care about recall and precision, plot the P-R curve and choose based on the required recall floor ("I must catch at least 80% of positives").

## Reading this report's classification stats

The stats are computed at the threshold you pass with `--threshold` (default 0.5). The ROC curve shows all possible thresholds. To explore a different operating point:

1. Open `report.json`
2. Use `roc.thresholds`, `roc.fpr`, `roc.tpr` to find the threshold you need
3. Rerun with `--threshold <new_value>`

## Warning

Never tune the threshold on the same data used to evaluate it. Use a held-out validation set or cross-validation.
