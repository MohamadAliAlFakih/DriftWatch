# DriftWatch Bank Marketing Model

## Dataset
- Name: bank-full.csv
- MD5 hash: `e6b0ca77f3f200ec5428e04dd104da53`
- Shape: 45211 rows, 17 columns
- Target: `y`, where `yes` maps to 1 and `no` maps to 0

## Training Setup
- Split strategy: stratified 70/30 train/test, random_state=42
- Cross-validation: stratified folds on the training split
- Leakage warning: `duration` is dropped because it is known only after a call ends
- `pdays` sentinel: `pdays == -1` becomes `pdays_was_minus_one`, `never_contacted_flag`,
  and `pdays_clean`
- `unknown` treatment: preserved as a real categorical value

## Model
- Class: HistGradientBoostingClassifier
- Hyperparameters:
```json
{
  "categorical_features": "from_dtype",
  "class_weight": null,
  "early_stopping": "auto",
  "interaction_cst": null,
  "l2_regularization": 0.0,
  "learning_rate": 0.06,
  "loss": "log_loss",
  "max_bins": 255,
  "max_depth": null,
  "max_features": 1.0,
  "max_iter": 200,
  "max_leaf_nodes": 31,
  "min_samples_leaf": 20,
  "monotonic_cst": null,
  "n_iter_no_change": 10,
  "random_state": 42,
  "scoring": "loss",
  "tol": 1e-07,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

## Final Test Metrics
```json
{
  "accuracy": 0.6879239162488942,
  "auc": 0.7990966091856692,
  "confusion_matrix": [
    [
      8134,
      3843
    ],
    [
      390,
      1197
    ]
  ],
  "f1": 0.3612494341330919,
  "precision": 0.2375,
  "recall": 0.7542533081285444
}
```

## Operating Threshold
```json
{
  "f1": 0.36754682019720736,
  "precision": 0.2434043299149794,
  "recall": 0.7501350621285792,
  "threshold": 0.08334068156924374
}
```

## Environment Fingerprint
```json
{
  "created_at": "2026-05-06T01:24:56.419498+00:00",
  "packages": {
    "joblib": "1.5.3",
    "mlflow": "3.12.0",
    "numpy": "2.4.4",
    "pandas": "2.3.3",
    "sklearn": "1.8.0"
  },
  "platform": "macOS-15.7.5-arm64-arm-64bit",
  "python": "3.11.15"
}
```

## Artifact Integrity
- SHA-256: `aa7801586b69c1360db2aa6181ca4aa223a30d718c865015f1e572fd23d73830`

## Intended Use
- Score bank marketing leads for subscription propensity and support drift monitoring.

## Not Intended Use
- Do not use as the only basis for customer treatment, credit decisions, or regulated actions.

## Limitations
- The model reflects historical campaign data and may drift when economic conditions change.
- Recall-focused thresholding can increase false positives.
- New categories at serving time are ignored by the one-hot encoder.
