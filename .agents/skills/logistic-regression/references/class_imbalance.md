# Class Imbalance in Logistic Regression

## When to worry

| Positive rate | Status |
|---------------|--------|
| 20%–80% | Normal — proceed without special handling |
| 10%–20% or 80%–90% | Moderate imbalance — prefer balanced accuracy and AUC over accuracy |
| < 10% or > 90% | Severe imbalance — accuracy is nearly useless; consider reweighting |

## Why accuracy fails

If 5% of customers churn, a classifier that always predicts "no churn" achieves 95% accuracy. It has never correctly identified a churner. **Balanced accuracy**, **AUC**, and **F1** are much more informative.

## Remedies

### 1. Threshold tuning (recommended first step)

Default threshold is 0.5. Move it toward the minority class:
- Lower threshold → more positives predicted → higher recall, lower precision
- Use Youden's J (max TPR − FPR on ROC curve) or cost-ratio criterion

### 2. Class reweighting

Assign higher loss weight to the minority class. Equivalent to oversampling minority or undersampling majority.

```bash
# This skill: use --class-weight balanced (not yet implemented — planned)
# sklearn equivalent: LogisticRegression(class_weight='balanced')
```

### 3. Oversampling (SMOTE)

Synthesise new minority-class observations by interpolating between existing ones. Risks: unrealistic samples, information leakage if applied before CV. Use with caution.

### 4. Undersampling

Randomly drop majority-class observations. Loses information. Prefer reweighting.

## What NOT to do

- Do not upsample before train/test split — this leaks information.
- Do not report only accuracy on imbalanced data.
- Do not apply SMOTE in production (your deployment data won't be oversampled).
- Do not tune threshold on the same data used to evaluate it.

## Calibration and imbalance

Reweighting and sampling change the class prior, which affects calibration. If you need reliable probability estimates (e.g., for ranking rather than binary decision), prefer reweighting over oversampling, and recalibrate after.
