# Mini-Project: From Public API to a Deployed Classification Dashboard

This repository implements the full assignment pipeline end to end:

1. Pull data from a public API with no authentication
2. Explore and clean the data with justified decisions
3. Train and evaluate a classification model
4. Build a multi-section Streamlit dashboard
5. Deploy to a free public hosting platform

## Live Links

- GitHub Repository: https://github.com/linanouri/mini-project
- Streamlit App: https://mini-project-mzzhfscvhatkfr6hxztmcu.streamlit.app/

## API Choice

- API: PokéAPI
- Endpoint: https://pokeapi.co/api/v2/pokemon?limit=200
- Classification target: pokemon `primary_type`
- Why this works for the task: list endpoint is paginated and detail responses contain nested stats and types that must be flattened.

## Project Structure

project/
|- app.py
|- data/
|  |- raw_data.csv
|  |- clean_data.csv
|- notebooks/
|  |- eda.ipynb
|  |- eda.py
|- reports/
|  |- cleaning_notes.md
|  |- figures/
|- src/
|  |- fetch_and_clean.py
|  |- train_model.py
|- .streamlit/
|  |- config.toml
|- requirements.txt
|- runtime.txt
|- model.pkl
|- scaler.pkl
|- label_encoder.pkl
|- model_metrics.json
|- README.md

## Prerequisites

- GitHub account
- Git installed locally
- Python 3.11+

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Task 1: Data Acquisition

Script: `src/fetch_and_clean.py`

- Uses `requests` and handles pagination through the `next` URL from PokéAPI list responses.
- Makes a detail call per Pokemon record.
- Includes a short `time.sleep(0.2)` between detail calls.
- Flattens nested JSON fields into columns:
  - stats -> `hp`, `attack`, `defense`, `special_attack`, `special_defense`, `speed`
  - types -> `primary_type` (slot 1)
- Saves reproducible raw data to `data/raw_data.csv`.

Run:

```bash
python src/fetch_and_clean.py
```

Expected output:

- At least 200 rows collected (current pipeline run: 300 rows)
- Target column `primary_type` mostly non-null (current run: fully non-null)

## Task 2: EDA and Data Cleaning

Script: `notebooks/eda.py`

Checks performed:

- Dataset shape, `info()`, `describe()`
- Missing values by column
- Duplicate rows

Visual outputs saved in `reports/figures/`:

- class balance plot
- correlation heatmap
- relationship plots:
  - attack vs class (boxplot)
  - speed vs defense (scatter by class)
  - hp vs special_attack (scatter by class)

Cleaning decisions:

- dropped missing target rows
- median imputation for numeric features
- dropped duplicate ids
- retained outliers by domain judgment
- merged very rare classes (<3 rows) into `other` to stabilize stratified split

Justification write-up:

- `reports/cleaning_notes.md`

Clean output:

- `data/clean_data.csv`

Run:

```bash
python notebooks/eda.py
```

## Task 3: Classification Model

Script: `src/train_model.py`

- Features are numeric base stats and summary attributes.
- Target: `primary_type`
- Label encoding: `LabelEncoder`
- Split: stratified `train_test_split(test_size=0.2, random_state=42)`
- Scaling: `StandardScaler`
- Models trained:
  - logistic regression
  - random forest
- Evaluation metrics:
  - accuracy
  - macro precision
  - macro recall
  - macro F1
- Confusion matrix computed with explicit labels:
  - `labels=range(len(le.classes_))`

Artifacts saved for dashboard reuse:

- `model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `model_metrics.json`

Run:

```bash
python src/train_model.py
```

## Streamlit Dashboard

Entry point: `app.py`

Sections:

- Overview
- Data
- EDA
- Model Evaluation
- Predict
- Deployment

Run locally:

```bash
streamlit run app.py
```

## Free Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from your repository.
4. Set main file path to `app.py`.
5. Deploy.

If deployment fails:

- verify required files are committed (`data/*.csv`, `*.pkl`, `model_metrics.json`)
- verify dependency versions in `requirements.txt`
- verify `app.py` is selected as the entrypoint

## Reproducible Run Order

```bash
python src/fetch_and_clean.py
python notebooks/eda.py
python src/train_model.py
streamlit run app.py
```
