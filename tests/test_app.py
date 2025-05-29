"""
Tests for the Streamlit application.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sys
import os
from unittest.mock import patch, MagicMock
import streamlit as st
from sklearn.exceptions import NotFittedError
import joblib

# Add the parent directory to the Python path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_model, predict_cancer_risk, FEATURE_INFO
from ml_pipeline.model_factory import ModelFactory
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
def model_and_preprocessor(sample_data):
    """Fixture to provide the trained model and preprocessor."""
    factory = ModelFactory()
    model = factory.create_model('logistic')
    
    # Train the model with sample data
    X_train = sample_data['X'].values
    y_train = sample_data['y']
    model.fit(X_train, y_train)
    
    # Create and fit preprocessor
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_method='mean',
        handle_missing=True
    )
    preprocessor.fit(X_train, y_train)
    
    return model, preprocessor

@pytest.fixture
def mock_streamlit():
    """Fixture to mock Streamlit components."""
    mock_st = MagicMock()
    
    # Create mock column objects
    col_mock = MagicMock()
    col_mock.metric = MagicMock()
    col_mock.write = MagicMock()
    col_mock.number_input = MagicMock(return_value=1.0)
    col_mock.warning = MagicMock()
    mock_st.columns.return_value = [col_mock] * 3
    
    # Set up button behavior
    mock_st.button = MagicMock(return_value=True)
    
    # Set up expander behavior
    expander_mock = MagicMock()
    expander_mock.write = MagicMock()
    mock_st.expander.return_value.__enter__.return_value = expander_mock
    
    # Set up other Streamlit components
    mock_st.set_page_config = MagicMock()
    mock_st.title = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.write = MagicMock()
    mock_st.error = MagicMock()
    mock_st.success = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.dataframe = MagicMock()
    
    return mock_st

@pytest.fixture
def mock_model():
    """Fixture to mock the ML model."""
    model = MagicMock()
    model.predict.return_value = np.array([1])  # Benign
    model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% benign
    return model

@pytest.fixture
def mock_preprocessor():
    """Fixture to mock the preprocessor."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.array([[1.0] * 30])
    return preprocessor

def test_model_loading():
    """Test that model and preprocessor can be loaded successfully."""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Create test model and preprocessor
    factory = ModelFactory()
    model = factory.create_model('logistic')
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Train model
    model.fit(X, y)
    
    # Create and fit preprocessor
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_method='mean',
        handle_missing=True
    )
    preprocessor.fit(X, y)
    
    # Save model and preprocessor
    model_path = os.path.join('models', 'breast_cancer_model.joblib')
    preprocessor_path = os.path.join('models', 'preprocessor.joblib')
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Load model and preprocessor
    loaded_model, loaded_preprocessor = load_model()
    
    # Check that loaded model and preprocessor have required methods
    assert hasattr(loaded_model, 'predict')
    assert hasattr(loaded_model, 'predict_proba')
    assert hasattr(loaded_preprocessor, 'transform')
    
    # Clean up
    os.remove(model_path)
    os.remove(preprocessor_path)

def test_prediction_shape(model_and_preprocessor, sample_data):
    """Test that prediction returns expected shape and type."""
    model, preprocessor = model_and_preprocessor
    features = sample_data['X'].iloc[0].to_frame().T
    
    prediction, probability = predict_cancer_risk(features, model, preprocessor)
    
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer"
    assert isinstance(probability, np.ndarray), "Probability should be a numpy array"
    assert len(probability) == 2, "Probability should have 2 values (binary classification)"
    assert 0 <= probability[0] <= 1 and 0 <= probability[1] <= 1, "Probabilities should be between 0 and 1"

def test_feature_info_completeness():
    """Test that FEATURE_INFO contains all necessary features."""
    data = load_breast_cancer()
    
    # Create a mapping of dataset feature names to our feature names
    name_mapping = {
        'mean radius': 'radius_mean',
        'mean texture': 'texture_mean',
        'mean perimeter': 'perimeter_mean',
        'mean area': 'area_mean',
        'mean smoothness': 'smoothness_mean',
        'mean compactness': 'compactness_mean',
        'mean concavity': 'concavity_mean',
        'mean concave points': 'concave_points_mean',
        'mean symmetry': 'symmetry_mean',
        'mean fractal dimension': 'fractal_dimension_mean',
        'radius error': 'radius_se',
        'texture error': 'texture_se',
        'perimeter error': 'perimeter_se',
        'area error': 'area_se',
        'smoothness error': 'smoothness_se',
        'compactness error': 'compactness_se',
        'concavity error': 'concavity_se',
        'concave points error': 'concave_points_se',
        'symmetry error': 'symmetry_se',
        'fractal dimension error': 'fractal_dimension_se',
        'worst radius': 'radius_worst',
        'worst texture': 'texture_worst',
        'worst perimeter': 'perimeter_worst',
        'worst area': 'area_worst',
        'worst smoothness': 'smoothness_worst',
        'worst compactness': 'compactness_worst',
        'worst concavity': 'concavity_worst',
        'worst concave points': 'concave_points_worst',
        'worst symmetry': 'symmetry_worst',
        'worst fractal dimension': 'fractal_dimension_worst'
    }
    
    expected_features = {name_mapping[feat] for feat in data.feature_names}
    actual_features = set(FEATURE_INFO.keys())
    
    assert len(FEATURE_INFO) == 30, "Should have exactly 30 features"
    assert expected_features == actual_features, \
        f"Feature mismatch. Missing: {expected_features - actual_features}, Extra: {actual_features - expected_features}"

def test_feature_info_validity():
    """Test that FEATURE_INFO contains valid ranges and required fields."""
    for feature, info in FEATURE_INFO.items():
        assert "description" in info, f"Feature {feature} missing description"
        assert "range" in info, f"Feature {feature} missing range"
        assert "unit" in info, f"Feature {feature} missing unit"
        assert isinstance(info["range"], tuple), f"Feature {feature} range should be a tuple"
        assert len(info["range"]) == 2, f"Feature {feature} range should have min and max values"
        assert info["range"][0] <= info["range"][1], f"Feature {feature} range min should be <= max"

def test_prediction_with_boundary_values(model_and_preprocessor):
    """Test predictions with boundary values from FEATURE_INFO."""
    model, preprocessor = model_and_preprocessor
    
    # Test with minimum values
    min_features = [info["range"][0] for info in FEATURE_INFO.values()]
    prediction_min, prob_min = predict_cancer_risk(min_features, model, preprocessor)
    assert isinstance(prediction_min, (int, np.integer)), "Prediction should work with minimum values"
    
    # Test with maximum values
    max_features = [info["range"][1] for info in FEATURE_INFO.values()]
    prediction_max, prob_max = predict_cancer_risk(max_features, model, preprocessor)
    assert isinstance(prediction_max, (int, np.integer)), "Prediction should work with maximum values"

def test_prediction_consistency(model_and_preprocessor, sample_data):
    """Test that predictions are consistent for the same input."""
    model, preprocessor = model_and_preprocessor
    features = sample_data['X'].iloc[0].values
    
    # Make multiple predictions
    prediction1, prob1 = predict_cancer_risk(features, model, preprocessor)
    prediction2, prob2 = predict_cancer_risk(features, model, preprocessor)
    
    assert prediction1 == prediction2, "Predictions should be consistent for same input"
    np.testing.assert_array_almost_equal(prob1, prob2, decimal=6, 
                                       err_msg="Probabilities should be consistent for same input")

def test_ui_initialization(mock_streamlit):
    """Test that the UI is initialized with correct title and description."""
    from app import main
    
    # Call main with mock streamlit
    main(mock_streamlit=mock_streamlit)
    
    # Verify page config was set
    mock_streamlit.set_page_config.assert_called_once_with(
        page_title="🏥 Breast Cancer Risk Predictor",
        page_icon="🏥",
        layout="wide"
    )
    
    # Verify title was set
    mock_streamlit.title.assert_called_once_with("🏥 Breast Cancer Risk Predictor")
    
    # Verify markdown description was added
    mock_streamlit.markdown.assert_called_once()

@patch('app.load_model')
def test_feature_input_creation(mock_load_model, mock_streamlit):
    """Test that input fields are created for all features."""
    # Mock model and preprocessor
    mock_model = MagicMock()
    mock_preprocessor = MagicMock()
    mock_load_model.return_value = (mock_model, mock_preprocessor)
    
    # Create mock column objects
    col_mock = MagicMock()
    col_mock.metric = MagicMock()
    col_mock.write = MagicMock()
    col_mock.warning = MagicMock()
    
    # Make number_input return appropriate values based on feature ranges
    def mock_number_input(label, **kwargs):
        for feature_name, info in FEATURE_INFO.items():
            if feature_name in label:
                min_val, max_val = info['range']
                return (min_val + max_val) / 2
        return 1.0
    
    col_mock.number_input = MagicMock(side_effect=mock_number_input)
    mock_streamlit.columns.return_value = [col_mock] * 3
    
    import app
    feature_values = app.create_feature_inputs(mock_streamlit=mock_streamlit)
    
    # Check that columns were created
    mock_streamlit.columns.assert_called_once()
    
    # Check that we got values for all features
    assert len(feature_values) == len(app.FEATURE_INFO), "Should have values for all features"
    
    # Check that each feature has a reasonable value
    for feature_name, value in feature_values.items():
        assert isinstance(value, float), f"Value for {feature_name} should be float"
        min_val, max_val = app.FEATURE_INFO[feature_name]['range']
        assert min_val * 0.5 <= value <= max_val * 2.0, f"Value for {feature_name} should be within extended range"

@patch('app.load_model')
@patch('app.predict_cancer_risk')
def test_prediction_workflow_benign(mock_predict, mock_load_model, mock_streamlit):
    """Test the prediction workflow for benign case."""
    # Mock model and preprocessor
    mock_model = MagicMock()
    mock_preprocessor = MagicMock()
    mock_load_model.return_value = (mock_model, mock_preprocessor)
    
    # Mock prediction
    mock_predict.return_value = (1, np.array([0.1, 0.9]))  # Benign with 90% probability
    
    import app
    app.main(mock_streamlit=mock_streamlit)
    
    # Check success message for benign case
    mock_streamlit.success.assert_called()
    assert "Low Risk" in mock_streamlit.success.call_args[0][0]

@patch('app.load_model')
@patch('app.predict_cancer_risk')
def test_prediction_workflow_uncertain(mock_predict, mock_load_model, mock_streamlit):
    """Test the prediction workflow for uncertain case."""
    # Mock model and preprocessor
    mock_model = MagicMock()
    mock_preprocessor = MagicMock()
    mock_load_model.return_value = (mock_model, mock_preprocessor)
    
    # Mock prediction
    mock_predict.return_value = (1, np.array([0.4, 0.6]))  # Benign but with lower confidence
    
    import app
    app.main(mock_streamlit=mock_streamlit)
    
    # Check warning message for uncertain case
    mock_streamlit.warning.assert_called()
    assert "Uncertain" in mock_streamlit.warning.call_args[0][0]

def test_input_validation():
    """Test input validation for feature values."""
    model = MagicMock()
    preprocessor = MagicMock()
    preprocessor.transform.side_effect = ValueError("Invalid input")
    
    # Test with invalid input (negative values)
    invalid_features = pd.DataFrame([-1.0] * 30).T
    prediction, probabilities = predict_cancer_risk(invalid_features, model, preprocessor)
    assert prediction is None
    assert probabilities is None

def test_prediction_with_valid_input(mock_model, mock_preprocessor):
    """Test prediction with valid input."""
    features = pd.DataFrame([1.0] * 30).T  # Valid feature values
    prediction, probabilities = predict_cancer_risk(features, mock_model, mock_preprocessor)
    
    assert prediction in [0, 1]
    assert len(probabilities) == 2
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.isclose(np.sum(probabilities), 1.0)

def test_feature_range_warnings(mock_model, mock_preprocessor, mock_streamlit):
    """Test that warnings are shown for out-of-range feature values."""
    import app
    
    # Create feature values outside normal ranges
    features = []
    for feature_name, info in FEATURE_INFO.items():
        min_val, max_val = info['range']
        features.append(max_val * 2.0)  # Well above normal range
    
    # Create a DataFrame with the features
    df = pd.DataFrame([features], columns=FEATURE_INFO.keys())
    
    # Mock the warning function
    with patch('warnings.warn') as mock_warn:
        app.create_feature_inputs(mock_streamlit=mock_streamlit)
        assert mock_warn.called, "No warning was issued for out-of-range values"

def test_model_loading_error_cases():
    """Test error handling in model loading."""
    import app
    
    # Test with missing model file
    with pytest.raises(RuntimeError) as exc_info:
        app.load_model()
    assert "Model file not found" in str(exc_info.value)
    
    # Test with invalid model file
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Create invalid model file
    model_path = os.path.join('models', 'breast_cancer_model.joblib')
    with open(model_path, 'w') as f:
        f.write('invalid model data')
    
    with pytest.raises(RuntimeError) as exc_info:
        app.load_model()
    assert "Error loading model" in str(exc_info.value)
    
    # Clean up
    os.remove(model_path)

@patch('app.load_model')
@patch('app.predict_cancer_risk')
def test_prediction_workflow_malignant(mock_predict, mock_load_model, mock_streamlit):
    """Test the prediction workflow for malignant case."""
    # Mock model and preprocessor
    mock_model = MagicMock()
    mock_preprocessor = MagicMock()
    mock_load_model.return_value = (mock_model, mock_preprocessor)
    
    # Create feature values
    features = {name: info['range'][0] for name, info in FEATURE_INFO.items()}
    
    # Mock prediction for malignant case with high confidence
    mock_predict.return_value = (0, np.array([0.95, 0.05]))
    
    import app
    app.main(mock_streamlit=mock_streamlit)
    
    # Check error message for malignant case
    mock_streamlit.error.assert_called()
    error_calls = [call[0][0] for call in mock_streamlit.error.call_args_list]
    assert any("High Risk" in str(msg) for msg in error_calls), "Should show high risk message"

@patch('app.load_model')
@patch('app.create_feature_inputs')
@patch('app.predict_cancer_risk')
def test_model_calibration_warning(mock_predict, mock_create_inputs, mock_load_model, mock_streamlit):
    """Test warning display for high confidence predictions."""
    # Mock model and preprocessor
    mock_model = MagicMock()
    mock_preprocessor = MagicMock()
    mock_load_model.return_value = (mock_model, mock_preprocessor)
    
    # Mock feature inputs
    mock_feature_values = {name: (info['range'][0] + info['range'][1]) / 2 
                          for name, info in FEATURE_INFO.items()}
    mock_create_inputs.return_value = mock_feature_values
    
    # Mock high confidence prediction
    mock_predict.return_value = (1, np.array([0.05, 0.95]))
    
    # Mock button click
    mock_streamlit.button.return_value = True
    
    import app
    app.main(mock_streamlit=mock_streamlit)
    
    # Verify warning for high confidence
    success_calls = [str(call[0][0]) for call in mock_streamlit.success.call_args_list]
    assert any("Low Risk: 95.0% probability of being benign" in msg for msg in success_calls), "Should show high confidence message"

def test_preprocessing_edge_cases(model_and_preprocessor):
    """Test preprocessing with edge cases."""
    model, preprocessor = model_and_preprocessor
    
    # Test with NaN values
    features_with_nan = pd.DataFrame({
        name: [np.nan] * 1 for name in FEATURE_INFO.keys()
    })
    prediction, probabilities = predict_cancer_risk(features_with_nan, model, preprocessor)
    assert prediction is not None, "Should handle NaN values"
    assert probabilities is not None, "Should handle NaN values"
    
    # Test with very large values instead of infinite
    features_with_large = pd.DataFrame({
        name: [1e6] * 1 for name in FEATURE_INFO.keys()
    })
    prediction, probabilities = predict_cancer_risk(features_with_large, model, preprocessor)
    assert prediction is not None, "Should handle large values"
    assert probabilities is not None, "Should handle large values"
    
    # Test with string values that can be converted to float
    features_mixed = pd.DataFrame({
        name: ['1.0'] * 1 for name in FEATURE_INFO.keys()
    })
    prediction, probabilities = predict_cancer_risk(features_mixed, model, preprocessor)
    assert prediction is not None, "Should handle string numbers"
    assert probabilities is not None, "Should handle string numbers"

def test_ui_error_handling(mock_streamlit):
    """Test UI error handling for various scenarios."""
    import app
    
    # Test model loading error
    with patch('app.load_model', side_effect=RuntimeError("Model loading failed")):
        app.main(mock_streamlit=mock_streamlit)
        mock_streamlit.error.assert_called_with("Error loading model: Model loading failed")
    
    # Test prediction error
    with patch('app.load_model', return_value=(MagicMock(), MagicMock())):
        with patch('app.predict_cancer_risk', side_effect=ValueError("Invalid input")):
            app.main(mock_streamlit=mock_streamlit)
            mock_streamlit.error.assert_called()
            assert "Error" in str(mock_streamlit.error.call_args[0][0])

def test_cleanup_after_model_loading():
    """Test proper cleanup after model loading tests."""
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'breast_cancer_model.joblib')
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    
    try:
        # Run the original test
        test_model_loading()
    finally:
        # Ensure cleanup happens even if test fails
        for path in [model_path, preprocessor_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(model_dir) and not os.listdir(model_dir):
            os.rmdir(model_dir) 