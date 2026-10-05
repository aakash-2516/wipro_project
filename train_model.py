"""
Step 1 - Train the productivity prediction model.

Run:  python train_model.py

Outputs (in ./models):
    productivity_model.joblib   trained pipeline (preprocessing + model)
    metrics.json                model comparison + final test metrics
    feature_importance.csv      permutation importance
    test_predictions.csv        actual vs predicted on the hold-out set
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_clean_data

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
SEED = 42

# ------------------------------------------------------------------ data
df = load_clean_data()
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)
print(f"Rows: {len(df)} | train: {len(X_train)} | test: {len(X_test)}")


def make_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", model)])


cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ------------------------------------------------------------------ candidates
# Tree ensembles are tuned with a small randomized search.
searches = {
    "Ridge Regression": (make_pipeline(Ridge(alpha=1.0)), None),
    "Random Forest": (
        make_pipeline(RandomForestRegressor(random_state=SEED, n_jobs=-1)),
        {
            "model__n_estimators": randint(200, 500),
            "model__max_depth": [None, 10, 15, 20],
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": [0.4, 0.6, 0.8, 1.0],
        },
    ),
    "Gradient Boosting": (
        make_pipeline(GradientBoostingRegressor(random_state=SEED)),
        {
            "model__n_estimators": randint(200, 600),
            "model__learning_rate": uniform(0.01, 0.07),
            "model__max_depth": randint(2, 6),
            "model__subsample": uniform(0.6, 0.4),
            "model__min_samples_leaf": randint(1, 10),
        },
    ),
}

results, fitted = {}, {}
for name, (pipe, grid) in searches.items():
    if grid is None:
        pipe.fit(X_train, y_train)
        best = pipe
        cv_r2 = cross_val_score(best, X_train, y_train, cv=cv, scoring="r2").mean()
    else:
        rs = RandomizedSearchCV(pipe, grid, n_iter=25, cv=cv, scoring="r2",
                                random_state=SEED, n_jobs=-1)
        rs.fit(X_train, y_train)
        best, cv_r2 = rs.best_estimator_, rs.best_score_
    pred = best.predict(X_test)
    results[name] = {
        "cv_r2": round(float(cv_r2), 4),
        "test_r2": round(float(r2_score(y_test, pred)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
    }
    fitted[name] = best
    print(f"{name:18s} CV R2={cv_r2:.3f} | Test R2={results[name]['test_r2']:.3f} | "
          f"MAE={results[name]['test_mae']:.4f} | RMSE={results[name]['test_rmse']:.4f}")

best_name = max(results, key=lambda k: results[k]["cv_r2"])
best_model = fitted[best_name]
print(f"\n--> Best model (by cross-validation): {best_name}")

# ------------------------------------------------------------------ evaluation artefacts
pred = np.clip(best_model.predict(X_test), 0, 1.2)
pd.DataFrame({"actual": y_test.values, "predicted": pred,
              "department": X_test["department"].values}
             ).to_csv("models/test_predictions.csv", index=False)

imp = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                             random_state=SEED, n_jobs=-1)
fi = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
      .sort_values("importance", ascending=False))
fi.to_csv("models/feature_importance.csv", index=False)
print("\nTop features:\n", fi.head(8).round(4).to_string(index=False))

# ------------------------------------------------------------------ final model
# Metrics above come from the hold-out set. For deployment we refit the chosen
# configuration on ALL rows so the dashboard uses every bit of data.
final_model = make_pipeline(best_model.named_steps["model"].__class__(
    **best_model.named_steps["model"].get_params()))
final_model.fit(X, y)
joblib.dump(final_model, "models/productivity_model.joblib")

metrics = {
    "best_model": best_name,
    "results": results,
    "rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features": FEATURES,
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved model + metrics to ./models/")
