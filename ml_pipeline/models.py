from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import numpy as np

class ModelFactory:
    """
    Factory class for creating and configuring machine learning models.
    """
    @staticmethod
    def get_model_config():
        """
        Get dictionary of model configurations with default and search parameters.
        """
        return {
            'logistic_regression': {
                'model': LogisticRegression,
                'default_params': {
                    'random_state': 42,
                    'max_iter': 1000
                },
                'search_params': {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'default_params': {
                    'random_state': 42
                },
                'search_params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'default_params': {
                    'random_state': 42,
                    'n_jobs': -1
                },
                'search_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'svm': {
                'model': SVC,
                'default_params': {
                    'random_state': 42,
                    'probability': True
                },
                'search_params': {
                    'C': np.logspace(-4, 4, 20),
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 20))
                }
            },
            'knn': {
                'model': KNeighborsClassifier,
                'default_params': {
                    'n_jobs': -1
                },
                'search_params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'default_params': {
                    'random_state': 42
                },
                'search_params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7]
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'default_params': {
                    'random_state': 42,
                    'max_iter': 1000
                },
                'search_params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            },
            'naive_bayes': {
                'model': GaussianNB,
                'default_params': {},
                'search_params': {
                    'var_smoothing': np.logspace(-11, -5, 20)
                }
            },
            'xgboost': {
                'model': XGBClassifier,
                'default_params': {
                    'random_state': 42,
                    'n_jobs': -1
                },
                'search_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }

    @staticmethod
    def create_model(model_name, params=None):
        """
        Create a model instance with specified parameters.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to create
        params : dict, optional (default=None)
            Parameters to initialize the model with
            
        Returns:
        --------
        model : estimator
            Initialized model instance
        """
        config = ModelFactory.get_model_config()
        
        if model_name not in config:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_class = config[model_name]['model']
        
        if params is None:
            params = config[model_name]['default_params']
        else:
            # Merge default params with provided params
            merged_params = config[model_name]['default_params'].copy()
            merged_params.update(params)
            params = merged_params
            
        return model_class(**params)

    @staticmethod
    def get_search_params(model_name):
        """
        Get hyperparameter search space for the specified model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        params : dict
            Dictionary of parameters to search
        """
        config = ModelFactory.get_model_config()
        
        if model_name not in config:
            raise ValueError(f"Unknown model: {model_name}")
            
        return config[model_name]['search_params'] 