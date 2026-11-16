# ✈️ Flight Delay Prediction

Predict **before departure** whether a US domestic flight will arrive **more than 15 minutes late**
(Kaggle: *2015 Flight Delays and Cancellations*).

## Repository layout
```
├── data/
│   ├── raw/            # flights.csv, airlines.csv, airports.csv  (or put archive.zip in data/)
│   └── processed/      # created by notebooks 02-03 (parquet files)
├── model/              # feature_config.joblib, best_params.json, flight_delay_bundle.joblib
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_understanding_cleaning.ipynb
│   ├── 03_eda_feature_engineering.ipynb
│   ├── 04_model_selection.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_final_evaluation_export.ipynb
├── outputs/            # small csv reports
└── requirements.txt
```

## How to run
Run the notebooks **in order** (each one reads what the previous one wrote). Paths are resolved
relative to the repo root, so it works locally and in Colab after cloning the repo.

| # | Notebook | Reads | Writes |
|---|----------|-------|--------|
| 01 | Data collection | `data/archive.zip` or `data/raw/*.csv` | `data/raw/*.csv` |
| 02 | Understanding & cleaning | `data/raw/*.csv` | `data/processed/flights_clean.parquet`, `outputs/cleaning_log.csv` |
| 03 | EDA & feature engineering | `flights_clean.parquet` | `data/processed/train.parquet`, `test.parquet`, `model/feature_config.joblib` |
| 04 | Model selection | `train.parquet`, `feature_config.joblib` | `outputs/model_comparison.csv` |
| 05 | Hyper-parameter tuning | `train.parquet`, `feature_config.joblib` | `model/best_params.json` |
| 06 | Final evaluation & export | train/test parquet, configs, raw lookups | `model/flight_delay_bundle.joblib`, `outputs/tuned_model_comparison.csv` |
