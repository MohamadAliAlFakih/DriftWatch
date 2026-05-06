"""Practical model tuning helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline


def tune_selected_pipeline(
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Tune only the selected baseline model family."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    searches: dict[str, GridSearchCV | RandomizedSearchCV] = {
        "logistic_regression": GridSearchCV(
            pipeline,
            param_grid={
                "model__C": [0.1, 1.0, 3.0],
                "model__solver": ["lbfgs"],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
        "random_forest": RandomizedSearchCV(
            pipeline,
            param_distributions={
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [3, 5, 10],
            },
            n_iter=5,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": GridSearchCV(
            pipeline,
            param_grid={
                "model__learning_rate": [0.03, 0.06],
                "model__max_leaf_nodes": [15, 31],
                "model__max_iter": [100, 200],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
    }

    if model_name not in searches:
        raise ValueError({"error": "unsupported_model_for_tuning", "model_name": model_name})

    search = searches[model_name]
    search.fit(x_train, y_train)
    best_estimator = search.best_estimator_
    cv_proba = cross_val_predict(
        best_estimator,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    best_estimator.fit(x_train, y_train)
    return {
        "name": model_name,
        "pipeline": best_estimator,
        "best_params": dict(search.best_params_),
        "cv_auc": float(search.best_score_),
        "cv_proba": cv_proba,
        "tuning_search": search.__class__.__name__,
    }
