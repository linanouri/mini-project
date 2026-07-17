from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pokemon Type Classification Dashboard", page_icon="🎯", layout="wide")

RAW_PATH = Path("data/raw_data.csv")
CLEAN_PATH = Path("data/clean_data.csv")
MODEL_PATH = Path("model.pkl")
SCALER_PATH = Path("scaler.pkl")
LE_PATH = Path("label_encoder.pkl")
METRICS_PATH = Path("model_metrics.json")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource
def load_artifacts() -> tuple[object | None, object | None, object | None]:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    label_encoder = joblib.load(LE_PATH) if LE_PATH.exists() else None
    return model, scaler, label_encoder


st.title("From Public API to a Deployed Classification Dashboard")
st.caption("Dataset source: PokéAPI (no authentication required)")

section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data", "EDA", "Model Evaluation", "Predict", "Deployment"],
)

raw_df = load_csv(RAW_PATH)
clean_df = load_csv(CLEAN_PATH)
metrics = load_json(METRICS_PATH)
model, scaler, label_encoder = load_artifacts()

if section == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows", int(raw_df.shape[0]) if not raw_df.empty else 0)
    c2.metric("Clean Rows", int(clean_df.shape[0]) if not clean_df.empty else 0)
    c3.metric("Classes", int(clean_df["primary_type"].nunique()) if "primary_type" in clean_df.columns else 0)
    c4.metric("Model", "Ready" if model is not None else "Missing")

    st.markdown(
        """
        This dashboard is built from an end-to-end pipeline:
        1. Fetch paginated data from PokéAPI with detail requests per Pokemon.
        2. Perform EDA and cleaning with documented decisions.
        3. Train and compare two classifiers.
        4. Persist artifacts and serve predictions interactively.
        """
    )

elif section == "Data":
    st.subheader("Raw Dataset")
    if raw_df.empty:
        st.warning("Missing data/raw_data.csv. Run: python src/fetch_and_clean.py")
    else:
        st.dataframe(raw_df.head(25), use_container_width=True)
        st.write("Columns:", list(raw_df.columns))

    st.subheader("Clean Dataset")
    if clean_df.empty:
        st.warning("Missing data/clean_data.csv. Run: python notebooks/eda.py")
    else:
        st.dataframe(clean_df.head(25), use_container_width=True)

elif section == "EDA":
    if clean_df.empty:
        st.warning("Run cleaning first to view EDA charts.")
    else:
        type_counts = clean_df["primary_type"].value_counts().reset_index()
        type_counts.columns = ["primary_type", "count"]
        st.subheader("Class Balance")
        st.plotly_chart(px.bar(type_counts, x="primary_type", y="count"), use_container_width=True)

        numeric_cols = [
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
        st.subheader("Numeric Summary")
        st.dataframe(clean_df[numeric_cols].describe().T, use_container_width=True)

        st.subheader("Relationship Plots")
        top_types = clean_df["primary_type"].value_counts().head(6).index.tolist()
        subset = clean_df[clean_df["primary_type"].isin(top_types)]
        st.plotly_chart(
            px.scatter(subset, x="speed", y="defense", color="primary_type", title="Speed vs Defense"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.scatter(subset, x="hp", y="special_attack", color="primary_type", title="HP vs Special Attack"),
            use_container_width=True,
        )

elif section == "Model Evaluation":
    if not metrics:
        st.warning("Missing model_metrics.json. Run: python src/train_model.py")
    else:
        st.subheader("Model Comparison (Macro Metrics)")
        model_results = pd.DataFrame(metrics.get("all_model_results", {})).T
        st.dataframe(model_results, use_container_width=True)

        st.write("Best model:", metrics.get("best_model", "n/a"))

        cm = metrics.get("confusion_matrix", [])
        labels = metrics.get("class_labels", [])
        if cm and labels:
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            st.subheader("Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)

        importances = metrics.get("feature_importance", {})
        if importances:
            imp_df = pd.DataFrame(
                sorted(importances.items(), key=lambda x: x[1], reverse=True),
                columns=["feature", "importance"],
            )
            st.subheader("Feature Importance")
            st.plotly_chart(px.bar(imp_df, x="feature", y="importance"), use_container_width=True)

elif section == "Predict":
    st.subheader("Predict Pokemon Primary Type")
    if model is None or scaler is None or label_encoder is None:
        st.warning("Missing model/scaler/label encoder artifacts. Run training first.")
    else:
        height = st.number_input("Height", min_value=1.0, value=10.0)
        weight = st.number_input("Weight", min_value=1.0, value=100.0)
        base_experience = st.number_input("Base Experience", min_value=1.0, value=120.0)
        abilities_count = st.number_input("Abilities Count", min_value=1.0, value=2.0)
        moves_count = st.number_input("Moves Count", min_value=1.0, value=60.0)
        hp = st.number_input("HP", min_value=1.0, value=60.0)
        attack = st.number_input("Attack", min_value=1.0, value=70.0)
        defense = st.number_input("Defense", min_value=1.0, value=65.0)
        special_attack = st.number_input("Special Attack", min_value=1.0, value=75.0)
        special_defense = st.number_input("Special Defense", min_value=1.0, value=70.0)
        speed = st.number_input("Speed", min_value=1.0, value=80.0)

        input_df = pd.DataFrame(
            [
                {
                    "height": height,
                    "weight": weight,
                    "base_experience": base_experience,
                    "abilities_count": abilities_count,
                    "moves_count": moves_count,
                    "hp": hp,
                    "attack": attack,
                    "defense": defense,
                    "special_attack": special_attack,
                    "special_defense": special_defense,
                    "speed": speed,
                }
            ]
        )

        x_scaled = scaler.transform(input_df)
        pred_encoded = model.predict(x_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"Predicted primary type: {pred_label}")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_scaled)[0]
            prob_df = pd.DataFrame(
                {
                    "type": label_encoder.classes_,
                    "probability": prob,
                }
            ).sort_values("probability", ascending=False)
            st.plotly_chart(px.bar(prob_df.head(10), x="type", y="probability"), use_container_width=True)

else:
    st.subheader("Deploy for Free")
    st.markdown(
        """
        1. Push this project to a GitHub repository.
        2. Open Streamlit Community Cloud and create a new app.
        3. Point it to app.py.
        4. Redeploy after each commit.

        If deployment fails:
        - Check requirements.txt and runtime.txt.
        - Confirm model/data artifacts are committed.
        - Read build logs for import or path errors.
        """
    )
