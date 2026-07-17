from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path("data/clean_data.csv")
MODEL_PATH = Path("model.pkl")
SCALER_PATH = Path("scaler.pkl")
LABEL_ENCODER_PATH = Path("label_encoder.pkl")
METRICS_PATH = Path("model_metrics.json")

FEATURES = [
    "height",
    "weight",
    "base_experience",
    "abilities_count",
    "moves_count",
    "hp",
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed",
]
TARGET = "primary_type"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing clean dataset at data/clean_data.csv. Run python notebooks/eda.py first.")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(str)

    # Target labels are stored so predictions can be mapped back to class names.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    # Fit on train only to avoid leakage into evaluation data.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=14, random_state=42),
    }

    results: dict[str, dict[str, float]] = {}
    preds_by_model: dict[str, list[int]] = {}
    confusion_by_model: dict[str, list[list[int]]] = {}
    importance_by_model: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="macro",
            zero_division=0,
        )

        results[name] = {
            "accuracy": float(acc),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
        }
        preds_by_model[name] = preds.tolist()

        # Always include all known classes in confusion matrix axes.
        cm = confusion_matrix(y_test, preds, labels=range(len(le.classes_)))
        confusion_by_model[name] = cm.tolist()

        # Provide model-specific importance for dashboard interpretability.
        if hasattr(model, "feature_importances_"):
            importance_by_model[name] = {
                feature: float(value)
                for feature, value in zip(FEATURES, model.feature_importances_)
            }
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim == 1:
                magnitude = abs(coefs)
            else:
                magnitude = abs(coefs).mean(axis=0)
            importance_by_model[name] = {
                feature: float(value)
                for feature, value in zip(FEATURES, magnitude)
            }

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = models[best_name]
    best_preds = preds_by_model[best_name]
    cm = confusion_by_model[best_name]
    importances = importance_by_model.get(best_name, {})

    best_macro_f1 = results[best_name]["macro_f1"]
    justification = (
        f"Deploy {best_name} because it achieved the highest macro-F1 ({best_macro_f1:.3f}) "
        "on the held-out stratified test set; macro-F1 was prioritized over accuracy due to class imbalance."
    )

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    # Single JSON payload keeps all evaluation artifacts app-friendly.
    payload = {
        "all_model_results": results,
        "best_model": best_name,
        "model_deployment_justification": justification,
        "confusion_matrix": cm,
        "confusion_matrices": confusion_by_model,
        "class_labels": le.classes_.tolist(),
        "feature_importance": importances,
        "feature_importance_by_model": importance_by_model,
        "features": FEATURES,
        "n_rows": int(df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }

    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
