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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=14, random_state=42),
    }

    results: dict[str, dict[str, float]] = {}
    preds_by_model: dict[str, list[int]] = {}

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

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = models[best_name]
    best_preds = preds_by_model[best_name]

    cm = confusion_matrix(y_test, best_preds, labels=range(len(le.classes_)))

    importances: dict[str, float] = {}
    if best_name == "random_forest" and hasattr(best_model, "feature_importances_"):
        importances = {
            feature: float(value)
            for feature, value in zip(FEATURES, best_model.feature_importances_)
        }

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    payload = {
        "all_model_results": results,
        "best_model": best_name,
        "confusion_matrix": cm.tolist(),
        "class_labels": le.classes_.tolist(),
        "feature_importance": importances,
        "features": FEATURES,
        "n_rows": int(df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }

    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
