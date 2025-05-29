"""
Tests for the preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sys
import os
from sklearn.exceptions import NotFittedError

# Add the parent directory to the Python path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Fixture to provide sample breast cancer data."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return {
        'X': df,
        'y': data.target,
        'feature_names': data.feature_names
    }

@pytest.fixture
def sample_mixed_data():
    """Fixture to provide sample data with both numerical and categorical features."""
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'num1': np.random.normal(0, 1, n_samples),
        'num2': np.random.uniform(0, 10, n_samples),
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat2': np.random.choice(['X', 'Y'], n_samples)
    })
    y = np.random.randint(0, 2, n_samples)
    return {'X': df, 'y': y}

def test_preprocessor_initialization():
    """Test that the preprocessor can be initialized with different parameters."""
    # Test default initialization
    preprocessor = DataPreprocessor()
    assert preprocessor.scaling_method == 'standard'
    assert preprocessor.categorical_encoding == 'label'
    assert preprocessor.imputation_method == 'mean'
    
    # Test custom initialization
    preprocessor = DataPreprocessor(
        scaling_method='minmax',
        categorical_encoding='onehot',
        imputation_method='median',
        feature_selection=5,
        n_components=0.95,
        handle_outliers=True
    )
    assert preprocessor.scaling_method == 'minmax'
    assert preprocessor.categorical_encoding == 'onehot'
    assert preprocessor.imputation_method == 'median'
    assert preprocessor.feature_selection == 5
    assert preprocessor.n_components == 0.95
    assert preprocessor.handle_outliers == True

def test_feature_type_identification(sample_mixed_data):
    """Test identification of numerical and categorical features."""
    preprocessor = DataPreprocessor()
    preprocessor._identify_feature_types(sample_mixed_data['X'])
    
    assert list(preprocessor.numerical_features) == ['num1', 'num2']
    assert list(preprocessor.categorical_features) == ['cat1', 'cat2']

def test_data_scaling(sample_data):
    """Test that data scaling works correctly."""
    preprocessor = DataPreprocessor(scaling_method='standard')
    X = sample_data['X'].copy()
    
    # Fit and transform the data
    preprocessor.fit(X)
    X_scaled = preprocessor.transform(X)
    
    # Convert to numpy array if needed
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
    
    # Check that scaled data has approximately zero mean and unit variance
    # Use a larger tolerance since we're working with real data
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-1), "Scaled data should have approximately zero mean"
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-1), "Scaled data should have approximately unit variance"

def test_categorical_encoding(sample_mixed_data):
    """Test categorical encoding methods."""
    # Create preprocessor with label encoding
    preprocessor = DataPreprocessor(categorical_encoding='label')
    X = sample_mixed_data['X'].copy()
    
    # Fit and transform
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    
    # Convert to numpy array if needed
    if isinstance(X_transformed, pd.DataFrame):
        X_transformed = X_transformed.values
    
    assert isinstance(X_transformed, np.ndarray), "Output should be numpy array"
    assert X_transformed.shape[1] == X.shape[1], "Should preserve number of columns with label encoding"
    
    # Test one-hot encoding
    preprocessor = DataPreprocessor(categorical_encoding='onehot')
    preprocessor.fit(X)
    X_onehot = preprocessor.transform(X)
    
    # Convert to numpy array if needed
    if isinstance(X_onehot, pd.DataFrame):
        X_onehot = X_onehot.values
    
    assert isinstance(X_onehot, np.ndarray), "Output should be numpy array"
    assert X_onehot.shape[1] > X.shape[1], "One-hot encoding should increase number of columns"

def test_missing_value_imputation():
    """Test that missing value imputation works correctly."""
    # Create data with missing values
    df = pd.DataFrame({
        'num1': [1, np.nan, 3, 4],
        'num2': [np.nan, 2, 3, 4],
        'cat1': ['A', None, 'B', 'A']
    })
    
    preprocessor = DataPreprocessor(imputation_method='mean')
    X_imputed = preprocessor.fit_transform(df)
    
    # Check that there are no missing values after imputation
    assert not np.any(np.isnan(X_imputed))

def test_feature_selection(sample_data):
    """Test feature selection functionality."""
    n_features = 10
    preprocessor = DataPreprocessor(feature_selection=n_features)
    
    X_selected = preprocessor.fit_transform(sample_data['X'], sample_data['y'])
    
    # Check that correct number of features were selected
    assert X_selected.shape[1] == n_features

def test_pca_transformation(sample_data):
    """Test PCA transformation."""
    n_components = 0.95  # Preserve 95% of variance
    preprocessor = DataPreprocessor(n_components=n_components)
    
    X_transformed = preprocessor.fit_transform(sample_data['X'])
    
    # Check that the number of components is reduced
    assert X_transformed.shape[1] <= sample_data['X'].shape[1]

def test_outlier_handling(sample_data):
    """Test outlier handling using IQR method."""
    preprocessor = DataPreprocessor(handle_outliers=True)
    X = sample_data['X'].copy()
    
    # Add some outliers
    X.iloc[0, :] = X.iloc[0, :] * 10  # Create outliers in first row
    
    # Fit and transform
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)
    
    # Convert to numpy array if needed
    if isinstance(X_processed, pd.DataFrame):
        X_processed = X_processed.values
    
    # Check that outliers are handled
    assert not np.array_equal(X_processed[0, :], X.iloc[0, :].values), "Outliers should be handled"
    
    # Check that values are within reasonable bounds
    for col in range(X_processed.shape[1]):
        col_data = X_processed[:, col]
        q1 = np.percentile(col_data, 25)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # Use 3 IQR for more lenient bounds
        upper_bound = q3 + 3 * iqr
        assert np.all((col_data >= lower_bound) & (col_data <= upper_bound)), f"Column {col} has extreme outliers"

def test_transform_without_fit(sample_data):
    """Test that transform without fit raises an error."""
    preprocessor = DataPreprocessor()
    with pytest.raises(NotFittedError):
        preprocessor.transform(sample_data['X'])

def test_input_validation():
    """Test input validation."""
    preprocessor = DataPreprocessor()
    
    # Test with invalid input types
    with pytest.raises((TypeError, ValueError)):
        preprocessor.fit_transform([1, 2, 3])
    
    # Test with empty data
    with pytest.raises(ValueError):
        preprocessor.fit_transform(pd.DataFrame())
    
    # Test with single feature
    with pytest.raises(ValueError):
        preprocessor.fit_transform(pd.DataFrame({'A': [1, 2, 3]}))
    
    # Test with invalid scaling method
    with pytest.raises(ValueError):
        DataPreprocessor(scaling_method='invalid')
    
    # Test with invalid categorical encoding
    with pytest.raises(ValueError):
        DataPreprocessor(categorical_encoding='invalid')

def test_full_pipeline(sample_mixed_data):
    """Test the complete preprocessing pipeline."""
    # Create preprocessor with all options enabled
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        categorical_encoding='label',  # Use label encoding instead of onehot
        imputation_method='mean',
        feature_selection=2,  # Select only 2 features
        handle_outliers=True
    )
    
    # Create a copy of the data and add some missing values
    X = sample_mixed_data['X'].copy()
    y = sample_mixed_data['y'].copy()
    
    # Convert categorical columns to numeric before splitting
    for col in ['cat1', 'cat2']:
        X[col] = pd.Categorical(X[col]).codes
    
    # Add some missing values
    X.iloc[0, 0] = np.nan
    
    # Split and transform data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y, test_size=0.2)
    
    # Convert to numpy array if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    # Check output types and shapes
    assert isinstance(X_train, np.ndarray), "Training data should be numpy array"
    assert isinstance(X_test, np.ndarray), "Test data should be numpy array"
    assert len(X_train) > 0, "Should have training samples"
    assert len(X_test) > 0, "Should have test samples"
    assert X_train.shape[1] == X_test.shape[1], "Train and test should have same number of features"
    assert X_train.shape[1] == 2, "Should have 2 features after feature selection"

def test_reproducibility():
    """Test that preprocessing is reproducible with same random state."""
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    preprocessor1 = DataPreprocessor(random_state=42)
    preprocessor2 = DataPreprocessor(random_state=42)
    
    result1 = preprocessor1.fit_transform(X, y)
    result2 = preprocessor2.fit_transform(X, y)
    
    np.testing.assert_array_almost_equal(result1, result2) 