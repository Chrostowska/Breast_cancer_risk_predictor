"""
Pytest configuration file with shared fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

@pytest.fixture
def sample_data():
    """Fixture providing a small sample dataset for testing."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    y = np.array([0, 0, 1, 1])
    return X, y

@pytest.fixture
def breast_cancer_data():
    """Fixture providing the breast cancer dataset."""
    data = load_breast_cancer()
    return data.data, data.target

@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [5.0, 6.0, 7.0, 8.0],
        'feature3': [9.0, 10.0, 11.0, 12.0]
    })

@pytest.fixture
def model_parameters():
    """Fixture providing common model parameters for testing."""
    return {
        'logistic': {'C': 1.0, 'max_iter': 1000},
        'rf': {'n_estimators': 10, 'max_depth': 3},
        'svm': {'C': 1.0, 'kernel': 'rbf'},
        'knn': {'n_neighbors': 3},
        'dt': {'max_depth': 3}
    } 