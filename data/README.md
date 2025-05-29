# Data Directory

This directory is used for storing datasets and intermediate data files.

## Default Dataset

By default, this project uses the Breast Cancer Wisconsin dataset from scikit-learn, which is loaded directly using `sklearn.datasets.load_breast_cancer()`. Therefore, no data file is required to be present in this directory.

## Directory Structure

```
data/
├── raw/          # Raw data files (if using custom datasets)
├── processed/    # Processed/transformed data
└── interim/      # Intermediate data files
```

## Using Custom Datasets

To use a custom dataset:
1. Place your raw data files in the `raw/` subdirectory
2. Update the data loading logic in `train_model.py`
3. Ensure your data follows a similar structure to the breast cancer dataset:
   - Features should be numerical
   - Target variable should be binary (0/1)
   - No missing values (or handle them in preprocessing) 