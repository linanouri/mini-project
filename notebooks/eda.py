from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw_data.csv"
CLEAN_PATH = DATA_DIR / "clean_data.csv"
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
NOTES_PATH = REPORTS_DIR / "cleaning_notes.md"

TARGET = "primary_type"
NUMERIC_COLS = [
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


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save_plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    counts = df[TARGET].value_counts().sort_values(ascending=False)
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Class Balance: Pokemon Primary Type")
    plt.xlabel("Primary Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png", dpi=140)
    plt.close()

    corr_df = df[NUMERIC_COLS].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, linewidths=0.4)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=140)
    plt.close()

    top_types = df[TARGET].value_counts().head(6).index.tolist()
    subset = df[df[TARGET].isin(top_types)].copy()

    plt.figure(figsize=(11, 5))
    sns.boxplot(data=subset, x=TARGET, y="attack", order=top_types)
    plt.title("Attack by Primary Type (Top 6 classes)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "attack_boxplot_by_type.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=subset, x="speed", y="defense", hue=TARGET, alpha=0.8)
    plt.title("Speed vs Defense by Type (Top 6 classes)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "speed_vs_defense_scatter.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=subset, x="hp", y="special_attack", hue=TARGET, alpha=0.8)
    plt.title("HP vs Special Attack by Type (Top 6 classes)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hp_vs_special_attack_scatter.png", dpi=140)
    plt.close()


def _write_notes(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    lines = [
        "# Cleaning Decisions",
        "",
        "1. Missing target values (`primary_type`) were dropped because supervised classification requires known labels.",
        "2. Numeric features were median-imputed to reduce sensitivity to skew and outliers.",
        "3. Exact duplicate Pokemon IDs were removed defensively.",
        "4. Outliers were reviewed but retained because extreme stats can be valid Pokemon characteristics.",
        "5. Very rare classes (<3 rows) were mapped to `other` to stabilize stratified splitting and macro metrics.",
        "",
        "## Dataset Summary",
        "",
        f"- Raw rows: {raw_df.shape[0]}",
        f"- Raw columns: {raw_df.shape[1]}",
        f"- Clean rows: {clean_df.shape[0]}",
        f"- Clean columns: {clean_df.shape[1]}",
        "",
    ]
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dirs()

    df = pd.read_csv(RAW_PATH)

    print("=== First Look ===")
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe(include="all"))
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))

    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=[TARGET])

    for col in NUMERIC_COLS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    clean_df = clean_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    rare_mask = clean_df[TARGET].map(clean_df[TARGET].value_counts()) < 3
    clean_df.loc[rare_mask, TARGET] = "other"

    _save_plots(clean_df)
    _write_notes(df, clean_df)

    clean_df.to_csv(CLEAN_PATH, index=False)

    print("\n=== Clean Output ===")
    print("Saved:", CLEAN_PATH)
    print("Shape:", clean_df.shape)
    print("Target balance:")
    print(clean_df[TARGET].value_counts())


if __name__ == "__main__":
    main()
