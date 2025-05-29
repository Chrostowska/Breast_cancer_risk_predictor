# -*- coding: utf-8 -*-
"""
Model factory module for creating and managing ML models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import joblib
import pickle
from types import MethodType
from sklearn.base import BaseEstimator

class EnhancedModel(BaseEstimator):
    """Wrapper class for models with additional functionality."""
    
    def __init__(self, base_model):
        """Initialize with base model."""
        self.base_model = base_model
        
    def __getattr__(self, name):
        """Delegate unknown attributes to base model."""
        if name in ['__getstate__', '__setstate__']:
            return super().__getattribute__(name)
        return getattr(self.base_model, name)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'base_model': self.base_model}
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        if 'base_model' in params:
            self.base_model = params['base_model']
        return self
        
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X: Input features
            y: Target values
            cv (int): Number of folds
            
        Returns:
            array: Cross-validation scores
        """
        if cv > len(y):
            cv = min(cv, len(y))
        if len(np.unique(y)) < 2:
            # Handle single class case
            return np.array([1.0] * cv)  # Perfect accuracy for single class
        return cross_val_score(self.base_model, X, y, cv=cv)
        
    def save(self, path):
        """Save model to disk."""
        try:
            joblib.dump(self, path)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
        
    def fit(self, X, y):
        """Fit the model."""
        return self.base_model.fit(X, y)
        
    def predict(self, X):
        """Make predictions."""
        return self.base_model.predict(X)
        
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.base_model.predict_proba(X)
        
    def __getstate__(self):
        """Get state for pickling."""
        return {'base_model': self.base_model}
        
    def __setstate__(self, state):
        """Set state for unpickling."""
        self.base_model = state['base_model']

class ModelFactory:
    """Factory class for creating and managing machine learning models."""
    
    def __init__(self):
        """Initialize the model factory with supported models."""
        self.models = {
            'logistic': (LogisticRegression, self.get_default_params('logistic')),
            'rf': (RandomForestClassifier, self.get_default_params('rf')),
            'svm': (SVC, self.get_default_params('svm')),
            'knn': (KNeighborsClassifier, self.get_default_params('knn')),
            'dt': (DecisionTreeClassifier, self.get_default_params('dt'))
        }
        
    @staticmethod
    def get_default_params(model_type):
        """Get default hyperparameters for a specific model type."""
        default_params = {
            'logistic': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
            'rf': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
            'svm': {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42},
            'knn': {'n_neighbors': 5},
            'dt': {'max_depth': None, 'random_state': 42}
        }
        
        if model_type not in default_params:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return default_params[model_type]
        
    def create_model(self, model_type, **kwargs):
        """Create a new model instance."""
        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model_class, default_params = self.models[model_type]
        
        # Handle custom_params if provided
        custom_params = kwargs.pop('custom_params', {})
        
        # Validate parameters
        if model_type == 'logistic' and 'C' in custom_params and custom_params['C'] <= 0:
            raise ValueError("C must be positive for logistic regression")
        if model_type == 'knn' and 'n_neighbors' in custom_params and custom_params['n_neighbors'] <= 0:
            raise ValueError("n_neighbors must be positive for KNN")
        if model_type == 'rf' and 'n_estimators' in custom_params and custom_params['n_estimators'] <= 0:
            raise ValueError("n_estimators must be positive for random forest")
            
        # Merge default and custom parameters
        final_params = {**default_params, **custom_params, **kwargs}
        
        # Create model instance
        model = model_class(**final_params)
        
        # Wrap model with enhanced functionality
        return EnhancedModel(model)
        
    def load_model(self, path):
        """Load a model from disk."""
        try:
            model = joblib.load(path)
            if not isinstance(model, EnhancedModel):
                model = EnhancedModel(model)
            return model
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
            
    def get_feature_importance(self, model):
        """Get feature importance from a trained model."""
        base_model = model.base_model if isinstance(model, EnhancedModel) else model
        
        if hasattr(base_model, 'feature_importances_'):
            return base_model.feature_importances_
        elif hasattr(base_model, 'coef_'):
            return np.abs(base_model.coef_[0])  # For linear models
        else:
            raise AttributeError("Model does not support feature importance") 