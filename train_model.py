"""
Script to train and save the breast cancer prediction model.
"""

import os
import joblib
from sklearn.datasets import load_breast_cancer
from ml_pipeline.models import ModelFactory
from ml_pipeline.preprocessing import DataPreprocessor

def train_and_save_model():
    """Train and save the breast cancer prediction model."""
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print("Creating and fitting preprocessor...")
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_method='mean',
        handle_missing=True,
        handle_outliers=True,
        random_state=42
    )
    X_transformed = preprocessor.fit_transform(X, y)
    
    print("Creating and training model...")
    factory = ModelFactory()
    model = factory.create_model('logistic_regression')
    model.fit(X_transformed, y)
    
    print("Saving model and preprocessor...")
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Save model and preprocessor
    model_path = os.path.join('models', 'breast_cancer_model.joblib')
    preprocessor_path = os.path.join('models', 'preprocessor.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")

if __name__ == "__main__":
    train_and_save_model() 