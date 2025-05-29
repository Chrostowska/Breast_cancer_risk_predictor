"""
Streamlit application for breast cancer prediction.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from ml_pipeline.model_factory import ModelFactory
from ml_pipeline.preprocessing import DataPreprocessor
from sklearn.datasets import load_breast_cancer

# Constants
MODEL_PATH = "models/model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

# Feature information dictionary
FEATURE_INFO = {
    'radius_mean': {
        'description': 'Mean of distances from center to points on the perimeter',
        'range': (6.0, 28.0),
        'unit': 'mm'
    },
    'texture_mean': {
        'description': 'Standard deviation of gray-scale values',
        'range': (9.0, 40.0),
        'unit': 'scalar'
    },
    'perimeter_mean': {
        'description': 'Mean size of the core tumor',
        'range': (43.0, 190.0),
        'unit': 'mm'
    },
    'area_mean': {
        'description': 'Mean area of the core tumor',
        'range': (140.0, 2550.0),
        'unit': 'mm²'
    },
    'smoothness_mean': {
        'description': 'Mean of local variation in radius lengths',
        'range': (0.05, 0.16),
        'unit': 'scalar'
    },
    'compactness_mean': {
        'description': 'Mean of perimeter² / area - 1.0',
        'range': (0.02, 0.35),
        'unit': 'scalar'
    },
    'concavity_mean': {
        'description': 'Mean of severity of concave portions of the contour',
        'range': (0.0, 0.5),
        'unit': 'scalar'
    },
    'concave_points_mean': {
        'description': 'Mean for number of concave portions of the contour',
        'range': (0.0, 0.2),
        'unit': 'count'
    },
    'symmetry_mean': {
        'description': 'Mean symmetry of the tumor',
        'range': (0.1, 0.3),
        'unit': 'scalar'
    },
    'fractal_dimension_mean': {
        'description': 'Mean for "coastline approximation" - 1',
        'range': (0.05, 0.1),
        'unit': 'scalar'
    },
    'radius_se': {
        'description': 'Standard error for the mean of distances from center to points on the perimeter',
        'range': (0.1, 2.9),
        'unit': 'mm'
    },
    'texture_se': {
        'description': 'Standard error for standard deviation of gray-scale values',
        'range': (0.4, 4.9),
        'unit': 'scalar'
    },
    'perimeter_se': {
        'description': 'Standard error for mean size of the core tumor',
        'range': (0.7, 22.0),
        'unit': 'mm'
    },
    'area_se': {
        'description': 'Standard error for mean area of the core tumor',
        'range': (6.0, 550.0),
        'unit': 'mm²'
    },
    'smoothness_se': {
        'description': 'Standard error for local variation in radius lengths',
        'range': (0.0, 0.03),
        'unit': 'scalar'
    },
    'compactness_se': {
        'description': 'Standard error for perimeter² / area - 1.0',
        'range': (0.0, 0.14),
        'unit': 'scalar'
    },
    'concavity_se': {
        'description': 'Standard error for severity of concave portions of the contour',
        'range': (0.0, 0.4),
        'unit': 'scalar'
    },
    'concave_points_se': {
        'description': 'Standard error for number of concave portions of the contour',
        'range': (0.0, 0.05),
        'unit': 'count'
    },
    'symmetry_se': {
        'description': 'Standard error for tumor symmetry',
        'range': (0.01, 0.07),
        'unit': 'scalar'
    },
    'fractal_dimension_se': {
        'description': 'Standard error for "coastline approximation" - 1',
        'range': (0.0, 0.03),
        'unit': 'scalar'
    },
    'radius_worst': {
        'description': 'Worst or largest mean value for distance from center to points on the perimeter',
        'range': (7.0, 37.0),
        'unit': 'mm'
    },
    'texture_worst': {
        'description': 'Worst or largest mean value for standard deviation of gray-scale values',
        'range': (12.0, 50.0),
        'unit': 'scalar'
    },
    'perimeter_worst': {
        'description': 'Worst or largest mean value for core tumor size',
        'range': (50.0, 250.0),
        'unit': 'mm'
    },
    'area_worst': {
        'description': 'Worst or largest mean value for core tumor area',
        'range': (185.0, 4250.0),
        'unit': 'mm²'
    },
    'smoothness_worst': {
        'description': 'Worst or largest mean value for local variation in radius lengths',
        'range': (0.07, 0.22),
        'unit': 'scalar'
    },
    'compactness_worst': {
        'description': 'Worst or largest mean value for perimeter² / area - 1.0',
        'range': (0.02, 1.06),
        'unit': 'scalar'
    },
    'concavity_worst': {
        'description': 'Worst or largest mean value for severity of concave portions of the contour',
        'range': (0.0, 1.25),
        'unit': 'scalar'
    },
    'concave_points_worst': {
        'description': 'Worst or largest mean value for number of concave portions of the contour',
        'range': (0.0, 0.3),
        'unit': 'count'
    },
    'symmetry_worst': {
        'description': 'Worst or largest mean value for tumor symmetry',
        'range': (0.15, 0.7),
        'unit': 'scalar'
    },
    'fractal_dimension_worst': {
        'description': 'Worst or largest mean value for "coastline approximation" - 1',
        'range': (0.06, 0.21),
        'unit': 'scalar'
    }
}

def load_model():
    """
    Load the trained model and preprocessor.
    
    Returns:
        tuple: (model, preprocessor)
    """
    try:
        # Load model
        model_path = os.path.join('models', 'breast_cancer_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join('models', 'preprocessor.joblib')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        
        # Ensure model and preprocessor are valid
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise ValueError("Invalid model: missing required methods")
            
        if not hasattr(preprocessor, 'transform'):
            raise ValueError("Invalid preprocessor: missing required methods")
            
        return model, preprocessor
    except Exception as e:
        raise RuntimeError(f"Error loading model and preprocessor: {str(e)}")

def create_feature_inputs(mock_streamlit=None):
    """
    Create input fields for all features.
    
    Args:
        mock_streamlit: Mock streamlit object for testing
        
    Returns:
        dict: Dictionary of feature values
    """
    st_obj = mock_streamlit if mock_streamlit else st
    
    feature_values = {}
    
    # Create columns for better layout
    cols = st_obj.columns(3)
    col_idx = 0
    
    for feature_name, info in FEATURE_INFO.items():
        min_val, max_val = info['range']
        default_val = (min_val + max_val) / 2
        
        # Add input field to current column
        value = cols[col_idx].number_input(
            feature_name,
            min_value=float(min_val * 0.5),  # Allow some margin below min
            max_value=float(max_val * 2.0),  # Allow some margin above max
            value=float(default_val),
            help=f"{info['description']}\nNormal range: {min_val:.2f} - {max_val:.2f} {info['unit']}",
            step=0.01,
            format="%.2f"
        )
        feature_values[feature_name] = value
        
        # Check if value is outside normal range
        if value < min_val or value > max_val:
            cols[col_idx].warning(f"⚠️ {feature_name} is outside normal range")
            warnings.warn(f"{feature_name} value {value:.2f} is outside normal range ({min_val:.2f}, {max_val:.2f})")
            
        # Move to next column
        col_idx = (col_idx + 1) % 3
        
    return feature_values

def predict_cancer_risk(features, model, preprocessor=None):
    """
    Make prediction using the model.
    
    Args:
        features: Input features (numpy array or pandas DataFrame)
        model: Trained model
        preprocessor: Optional preprocessor for raw features
        
    Returns:
        tuple: (prediction, probabilities)
    """
    try:
        # Convert list to numpy array if necessary
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        elif isinstance(features, pd.Series):
            features = features.values.reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            features = features.values
            
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Validate feature dimensions
        if features.shape[1] != 30:
            raise ValueError(f"Expected 30 features, got {features.shape[1]}")
            
        # Preprocess features if preprocessor is provided
        if preprocessor is not None:
            try:
                features = preprocessor.transform(features)
            except Exception as e:
                raise ValueError(f"Error in preprocessing: {str(e)}")
                
        # Get prediction probabilities
        try:
            probabilities = model.predict_proba(features)[0]
        except Exception as e:
            raise ValueError(f"Error in prediction: {str(e)}")
            
        # Get class prediction (0: Malignant, 1: Benign)
        prediction = model.predict(features)[0]
        
        # Convert prediction to int if necessary
        if isinstance(prediction, np.integer):
            prediction = int(prediction)
            
        return prediction, probabilities
    except Exception as e:
        print(f"Error in predict_cancer_risk: {str(e)}")
        return None, None

def display_prediction_results(prediction, probability):
    """Display prediction results with appropriate styling."""
    if prediction is None or probability is None:
        return
        
    # Display prediction
    if prediction == 0:  # Malignant
        st.error("⚠️ Prediction: Malignant")
        confidence = probability[0]
    else:  # Benign
        st.success("✅ Prediction: Benign")
        confidence = probability[1]
        
    # Display confidence with color coding
    confidence_color = "red" if confidence > 0.9 else "orange" if confidence > 0.7 else "green"
    st.markdown(
        f"<h3 style='color: {confidence_color}'>Confidence: {confidence:.1%}</h3>",
        unsafe_allow_html=True
    )
    
    # Add warning for uncertain predictions
    if 0.4 <= confidence <= 0.6:
        st.warning("⚠️ Prediction confidence is low. Please consult with healthcare professionals.")
    
    # Add warning for high confidence predictions
    if confidence > 0.9:
        st.warning(
            "⚠️ High confidence predictions should be validated by healthcare professionals. " +
            "This tool is for screening purposes only."
        )

def main(mock_streamlit=None):
    """
    Main function for the Streamlit app.
    
    Args:
        mock_streamlit: Mock streamlit object for testing
    """
    st_obj = mock_streamlit if mock_streamlit else st
    
    # Set up page configuration
    st_obj.set_page_config(
        page_title=" Breast Cancer Risk Predictor",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title and description
    st_obj.title(" Breast Cancer Risk Predictor")
    st_obj.markdown("""
    This application uses machine learning to predict breast cancer risk based on cell nucleus measurements.
    Enter the measurements below to get a prediction.
    """)
    
    try:
        # Load model and preprocessor
        model, preprocessor = load_model()
        
        # Create feature inputs
        feature_values = create_feature_inputs(mock_streamlit=st_obj)
        
        # Create prediction button
        if st_obj.button("Predict"):
            try:
                # Convert feature values to DataFrame
                features_df = pd.DataFrame([feature_values])
                
                # Make prediction
                prediction, probabilities = predict_cancer_risk(features_df, model, preprocessor)
                
                if prediction is not None and probabilities is not None:
                    # Display prediction
                    if prediction == 0:  # Malignant
                        st_obj.error(f"⚠️ High Risk: {probabilities[0]*100:.1f}% probability of malignancy")
                    else:  # Benign
                        if probabilities[1] >= 0.8:
                            st_obj.success(f"✅ Low Risk: {probabilities[1]*100:.1f}% probability of being benign")
                        else:
                            st_obj.warning(f"⚠️ Uncertain: {probabilities[1]*100:.1f}% probability of being benign")
                            
                    # Display probability details
                    col1, col2 = st_obj.columns(2)
                    col1.metric("Malignant Probability", f"{probabilities[0]*100:.1f}%")
                    col2.metric("Benign Probability", f"{probabilities[1]*100:.1f}%")
                    
                    # Add interpretation
                    with st_obj.expander("See interpretation"):
                        st_obj.write("""
                        - A high probability of being benign (>80%) suggests low risk
                        - A high probability of being malignant (>50%) suggests high risk
                        - Uncertain predictions may require additional testing
                        """)
                else:
                    st_obj.error("Error making prediction. Please check your inputs.")
            except Exception as e:
                st_obj.error(f"Error during prediction: {str(e)}")
    except Exception as e:
        st_obj.error(f"Error loading model: {str(e)}")

    # Add information about the features
    with st_obj.expander("Feature Information"):
        st_obj.write("""
        The features used in this model are measurements of cell nucleus characteristics:
        - Mean values: Average for all cells in the sample
        - SE (Standard Error): Standard error of the measurements
        - Worst values: Mean of the three largest values in the sample
        
        All measurements are taken from digitized images of fine needle aspirate (FNA) of breast mass.
        """)
        
        # Create feature description table
        feature_df = pd.DataFrame.from_dict(
            {k: {'Description': v['description'], 'Unit': v['unit']} 
             for k, v in FEATURE_INFO.items()},
            orient='index'
        )
        st_obj.dataframe(feature_df)

    # Add information about the model
    with st_obj.expander("ℹ️ About the Model"):
        st_obj.write("""
        This application uses a Logistic Regression model trained on the Wisconsin Breast Cancer dataset. 
        The model achieves:
        - 96% accuracy on test data
        - High precision and recall for both benign and malignant cases
        - Fast and reliable predictions
        
        The model takes into account various cell nucleus characteristics from the biopsy sample to make its predictions.
        """)

if __name__ == "__main__":
    main() 