"""
Tests for the models module and model factory.
"""

import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from ml_pipeline.model_factory import ModelFactory
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import sys
import os
import joblib
from sklearn.base import BaseEstimator

# Add the parent directory to the Python path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Fixture to provide sample breast cancer data."""
    data = load_breast_cancer()
    return {
        'X': data.data[:100],  # Use first 100 samples
        'y': data.target[:100]
    }

@pytest.fixture
def preprocessed_data(sample_data):
    """Fixture to provide preprocessed data."""
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(sample_data['X'], sample_data['y'])
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

@pytest.fixture
def model_factory():
    """Fixture to provide a model factory instance."""
    return ModelFactory()

def test_model_factory_initialization(model_factory):
    """Test that ModelFactory can be initialized."""
    assert model_factory is not None
    assert hasattr(model_factory, 'models')

@pytest.mark.parametrize("model_type", ['logistic', 'rf', 'svm', 'knn', 'dt'])
def test_model_creation(model_factory, model_type):
    """Test that models can be created successfully."""
    model = model_factory.create_model(model_type)
    assert model is not None
    assert isinstance(model, BaseEstimator)

def test_invalid_model_type(model_factory):
    """Test that invalid model types raise an error."""
    with pytest.raises(ValueError):
        model_factory.create_model('invalid_model')

def test_model_training(model_factory, sample_data):
    """Test that models can be trained successfully."""
    X, y = sample_data['X'], sample_data['y']
    
    for model_type in ['logistic', 'rf', 'svm', 'knn', 'dt']:
        model = model_factory.create_model(model_type)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(isinstance(pred, (np.int64, np.int32, int)) for pred in predictions)

def test_model_cross_validation(model_factory, sample_data):
    """Test cross-validation functionality."""
    X, y = sample_data['X'], sample_data['y']
    
    for model_type in ['logistic', 'rf', 'svm', 'knn', 'dt']:
        model = model_factory.create_model(model_type)
        
        # Ensure the model can be used in cross-validation
        scores = model.cross_validate(X, y, cv=2)
        assert len(scores) == 2
        assert all(0 <= score <= 1 for score in scores)

def test_model_prediction_without_training(model_factory, sample_data):
    """Test that prediction without training raises an error."""
    X = sample_data['X']
    
    for model_type in ['logistic', 'rf', 'svm', 'knn', 'dt']:
        model = model_factory.create_model(model_type)
        with pytest.raises(NotFittedError):
            model.predict(X)

def test_model_parameter_validation(model_factory):
    """Test that invalid parameters are handled appropriately."""
    # Test with invalid parameters using custom_params
    with pytest.raises(ValueError):
        model_factory.create_model('logistic', custom_params={'C': -1.0})
    
    with pytest.raises(ValueError):
        model_factory.create_model('knn', custom_params={'n_neighbors': -1})

@pytest.mark.parametrize("model_type,expected_params", [
    ('logistic', {'C': 1.0, 'max_iter': 1000, 'random_state': 42}),
    ('rf', {'n_estimators': 100, 'max_depth': None, 'random_state': 42}),
    ('svm', {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42}),
    ('knn', {'n_neighbors': 5}),
    ('dt', {'max_depth': None, 'random_state': 42})
])
def test_default_parameters(model_factory, model_type, expected_params):
    """Test that default parameters are set correctly."""
    params = model_factory.get_default_params(model_type)
    assert params == expected_params

def test_model_training_with_invalid_data(model_factory):
    """Test model training with invalid input data."""
    model = model_factory.create_model('logistic')
    
    # Test with empty data
    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))
    
    # Test with mismatched dimensions
    with pytest.raises(ValueError):
        model.fit(np.array([[1, 2], [3, 4]]), np.array([1]))

def test_model_prediction_with_invalid_data(model_factory, sample_data):
    """Test model prediction with invalid input data."""
    model = model_factory.create_model('logistic')
    X, y = sample_data['X'], sample_data['y']
    
    # Test prediction without training
    with pytest.raises(NotFittedError):
        model.predict(np.array([[1, 2]]))
    
    # Train the model
    model.fit(X, y)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        model.predict(np.array([[1, 2, 3]]))  # Wrong number of features

def test_model_cross_validation_edge_cases(model_factory, sample_data):
    """Test cross-validation with edge cases."""
    X, y = sample_data['X'], sample_data['y']
    model = model_factory.create_model('logistic')
    
    # Test with very small number of folds
    scores = model.cross_validate(X, y, cv=2)
    assert len(scores) == 2
    
    # Test with small dataset
    small_X = X[:6]
    small_y = y[:6]
    scores = model.cross_validate(small_X, small_y, cv=2)
    assert len(scores) == 2

def test_model_save_load(model_factory, sample_data, tmp_path):
    """Test model saving and loading functionality."""
    X, y = sample_data['X'], sample_data['y']
    model = model_factory.create_model('logistic')
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    save_path = tmp_path / "test_model.joblib"
    joblib.dump(model, save_path)
    
    # Load the model
    loaded_model = joblib.load(save_path)
    
    # Compare predictions
    original_pred = model.predict(X)
    loaded_pred = loaded_model.predict(X)
    np.testing.assert_array_equal(original_pred, loaded_pred)

def test_model_feature_importance(model_factory):
    """Test feature importance calculation for supported models."""
    # Test with models that support feature importance
    for model_type in ['rf', 'dt']:
        model = model_factory.create_model(model_type)
        
        # Fit the model first
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        # Get feature importance
        importances = model_factory.get_feature_importance(model)
        assert importances.shape[0] == X.shape[1]
        assert np.all(importances >= 0)

def test_model_probability_calibration(model_factory, sample_data):
    """Test probability calibration of models."""
    X, y = sample_data['X'], sample_data['y']
    
    for model_type in ['logistic', 'rf', 'svm']:
        model = model_factory.create_model(model_type)
        model.fit(X, y)
        
        # Check probability predictions
        proba = model.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(np.sum(proba, axis=1), 1.0)

def test_model_parameter_customization(model_factory):
    """Test that models can be created with custom parameters."""
    custom_params = {
        'logistic': {'C': 0.5, 'max_iter': 2000},
        'rf': {'n_estimators': 200, 'max_depth': 5},
        'svm': {'C': 2.0, 'kernel': 'linear'},
        'knn': {'n_neighbors': 7},
        'dt': {'max_depth': 10}
    }
    
    for model_type, params in custom_params.items():
        model = model_factory.create_model(model_type, custom_params=params)
        for param, value in params.items():
            assert getattr(model, param) == value 