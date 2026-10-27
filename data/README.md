# Dataset

This project uses the CIC-IDS2017 network intrusion detection dataset.

## Source

The dataset is obtained from the Kaggle source: https://www.kaggle.com/datasets/dhoogla/cicids2017.

## Storage

Raw dataset files belong in `data/raw/`.

Raw data is not committed to this GitHub repository because of
its size.

## Expected format

The initial data-loading pipeline expects Parquet files.

## Labels

The dataset contains a label identifying benign traffic or an
attack category. The exact column name and category values must
be checked before training.

## Data handling

- Keep raw data unchanged.
- Perform cleaning and preprocessing in code.
- Fit preprocessing and resampling steps on training data only.
- Do not use target labels as model input features.
