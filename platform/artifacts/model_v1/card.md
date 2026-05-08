# DriftWatch Bank Marketing Model

## Dataset
- Name: bank-additional-full.csv
- MD5 hash: `f6cb2c1256ffe2836b36df321f46e92c`
- Shape: 41188 rows, 21 columns
- Target: `y`, where `yes` maps to 1 and `no` maps to 0

## Training Setup
- Split strategy: stratified 60/20/20 train/validation/test, random_state=42
- Cross-validation: stratified folds on the training split
- Leakage warning: `duration` is dropped because it is known only after a call ends
- `pdays` sentinel: `pdays == 999` becomes `pdays_was_999`, `never_contacted_flag`,
  and `pdays_clean`
- `unknown` treatment: preserved as a real categorical value

## Model
- Class: LogisticRegression
- Hyperparameters:
```json
{
  "C": 0.1,
  "class_weight": "balanced",
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": 0.0,
  "max_iter": 1000,
  "n_jobs": null,
  "penalty": "deprecated",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

## Final Test Metrics
```json
{
  "accuracy": 0.7194707453265355,
  "auc": 0.8007907212604368,
  "confusion_matrix": [
    [
      5238,
      2072
    ],
    [
      239,
      689
    ]
  ],
  "f1": 0.3735429655733261,
  "precision": 0.24954726548352046,
  "recall": 0.7424568965517241
}
```

## Operating Threshold
```json
{
  "f1": 0.3757085020242915,
  "precision": 0.25063017644940583,
  "recall": 0.75,
  "threshold": 0.3897478990837321
}
```

## Environment Fingerprint
```json
{
  "created_at": "2026-05-08T10:40:44.828159+00:00",
  "packages": {
    "joblib": "1.5.3",
    "mlflow": "2.16.2",
    "numpy": "2.4.4",
    "pandas": "2.3.3",
    "sklearn": "1.8.0"
  },
  "platform": "Linux-6.12.76-linuxkit-aarch64-with-glibc2.41",
  "python": "3.11.15"
}
```

## Artifact Integrity
- SHA-256: `58d03858db51c2029a3936ff0215f0994384915d3ef6b505ca47d5b07c5bc877`

## Intended Use
- Score bank marketing leads for subscription propensity and support drift monitoring.

## Not Intended Use
- Do not use as the only basis for customer treatment, credit decisions, or regulated actions.

## Limitations
- The model reflects historical campaign data and may drift when economic conditions change.
- Recall-focused thresholding can increase false positives.
- New categories at serving time are ignored by the one-hot encoder.
