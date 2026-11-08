"""Tune a Tourism XGBoost pipeline, log MLflow runs, and register the best model."""
from pathlib import Path
import json
import os
import platform
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score, average_precision_score,
                            ConfusionMatrixDisplay)
from prep import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "tourism_project/deployment"
REPORTS = ROOT / "tourism_project/reports"

def evaluate(y, pred, probability):
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc": float(average_precision_score(y, probability))}

def train():
    print("Start")
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    Xtrain = pd.read_csv(ROOT / "Xtrain.csv")[FEATURES]
    Xtest = pd.read_csv(ROOT / "Xtest.csv")[FEATURES]
    ytrain = pd.read_csv(ROOT / "ytrain.csv")["ProdTaken"]
    ytest = pd.read_csv(ROOT / "ytest.csv")["ProdTaken"]
    preprocessing = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
        (make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CATEGORICAL_FEATURES),
        remainder="drop")
    model = make_pipeline(preprocessing, xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=42, n_jobs=1))
    # Imbalance weights are derived only from the training labels.
    weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
    param_grid = {"xgbclassifier__n_estimators": [100, 200],
                  "xgbclassifier__max_depth": [3, 5],
                  "xgbclassifier__learning_rate": [0.05, 0.1],
                  "xgbclassifier__scale_pos_weight": [1.0, weight]}
    search = GridSearchCV(model, param_grid, scoring="average_precision",
                          cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                          n_jobs=2, refit=True, error_score="raise")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", public_url)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Tourism_Wellness_Purchase")
    print("MLflow tracking URI:", tracking_uri)
    with mlflow.start_run(run_name="tourism_xgboost_grid_search") as run:
        search.fit(Xtrain, ytrain)
        results = pd.DataFrame(search.cv_results_)
        results.to_csv(REPORTS / "cv_results.csv", index=False)
        # Log each grid configuration, without using the test set for selection.
        for idx, row in results.iterrows():
            with mlflow.start_run(run_name=f"candidate_{idx:02d}", nested=True):
                mlflow.log_params(row["params"])
                mlflow.log_metrics({"cv_mean_pr_auc": float(row["mean_test_score"]),
                                    "cv_std_pr_auc": float(row["std_test_score"])})
        best_model = search.best_estimator_
        probability = best_model.predict_proba(Xtest)[:, 1]
        predictions = (probability >= 0.5).astype(int)
        metrics = evaluate(ytest, predictions, probability)
        baseline = DummyClassifier(strategy="prior").fit(Xtrain, ytrain)
        baseline_metrics = evaluate(ytest, baseline.predict(Xtest), baseline.predict_proba(Xtest)[:, 1])
        # Acceptance gate is configured before test evaluation; never tune it on test results.
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Model produced non-finite evaluation metrics.")
        report = {"model": "XGBClassifier", "threshold": 0.5,
                  "selection_metric": "cross_validated_average_precision",
                  "best_cv_pr_auc": float(search.best_score_), "best_params": search.best_params_,
                  "test_metrics": metrics, "baseline_test_metrics": baseline_metrics,
                  "train_rows": len(Xtrain), "test_rows": len(Xtest),
                  "features": FEATURES, "mlflow_run_id": run.info.run_id,
                  "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                               "xgboost": xgb.__version__, "joblib": joblib.__version__},
                  "limitations": ["Historical ProdTaken is a proxy, not verified new-wellness-package demand.",
                                  "Interaction features excluded for pre-contact use.",
                                  "Sensitive attributes require fairness and governance review."]}
        (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2))
        (REPORTS / "classification_report.txt").write_text(classification_report(ytest, predictions, zero_division=0))
        ConfusionMatrixDisplay.from_predictions(ytest, predictions, display_labels=["No purchase", "Purchase"])
        plt.title("Tourism purchase prediction — held-out test set")
        plt.tight_layout()
        plt.savefig(REPORTS / "confusion_matrix.png", dpi=150)
        plt.close()
        mlflow.log_params(search.best_params_)
        mlflow.log_params({"split_seed": 42, "test_size": 0.2, "threshold": 0.5})
        mlflow.log_metric("best_cv_pr_auc", float(search.best_score_))
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_artifacts(str(REPORTS), artifact_path="evaluation")
        mlflow.sklearn.log_model(best_model, name="tourism_pipeline",
                                 input_example=Xtrain.head(3))
        joblib.dump(best_model, DEPLOYMENT / "model.joblib")
        # Only training data determines UI defaults, categories and numeric ranges.
        schema = {"features": FEATURES, "numeric": {}, "categorical": {},
                  "threshold": 0.5, "versions": report["versions"]}
        for col in NUMERIC_FEATURES:
            values = Xtrain[col].dropna()
            schema["numeric"][col] = {"min": float(values.min()), "max": float(values.max()),
                                      "default": float(values.median())}
        for col in CATEGORICAL_FEATURES:
            schema["categorical"][col] = sorted(Xtrain[col].dropna().unique().tolist())
        (DEPLOYMENT / "schema.json").write_text(json.dumps(schema, indent=2))
        # Smoke-test the persisted artifact, including unseen categorical input.
        loaded = joblib.load(DEPLOYMENT / "model.joblib")
        np.testing.assert_allclose(loaded.predict_proba(Xtest.head()), best_model.predict_proba(Xtest.head()))
        unknown = Xtest.head(1).copy()
        unknown["Occupation"] = "Previously unseen occupation"
        assert loaded.predict_proba(unknown).shape == (1, 2)
        print(json.dumps(report, indent=2))
        print(classification_report(ytest, predictions, zero_division=0))
    return report

if __name__ == "__main__":
    print("start")
    train()
