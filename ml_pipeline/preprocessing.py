# -*- coding: utf-8 -*-
"""
Data preprocessing module for ML pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split as sk_train_test_split
import warnings
from sklearn.exceptions import NotFittedError

class DataPreprocessor:
    """
    A comprehensive data preprocessing class that handles various preprocessing tasks.
    """
    def __init__(self, 
                 scaling_method='standard',
                 categorical_encoding='label',
                 imputation_method='mean',
                 feature_selection=None,
                 n_components=None,
                 handle_outliers=False,
                 handle_missing=True,
                 random_state=42):
        """
        Initialize the preprocessor with specified methods.
        
        Args:
            scaling_method: Method for scaling numerical features
            categorical_encoding: Method for encoding categorical features
            imputation_method: Method for imputing missing values
            feature_selection: Number of features to select or None
            n_components: Number of PCA components or None
            handle_outliers: Whether to handle outliers
            handle_missing: Whether to handle missing values
            random_state: Random state for reproducibility
        """
        # Validate scaling method
        valid_scaling_methods = ['standard', 'minmax', 'robust', None]
        if scaling_method not in valid_scaling_methods:
            raise ValueError(f"Invalid scaling method. Must be one of {valid_scaling_methods}")
        
        # Validate categorical encoding
        valid_encodings = ['label', 'onehot']
        if categorical_encoding not in valid_encodings:
            raise ValueError(f"Invalid categorical encoding. Must be one of {valid_encodings}")
        
        # Validate imputation method
        valid_imputation_methods = ['mean', 'median', 'most_frequent', 'constant']
        if imputation_method not in valid_imputation_methods:
            raise ValueError(f"Invalid imputation method. Must be one of {valid_imputation_methods}")
        
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.imputation_method = imputation_method
        self.feature_selection = feature_selection
        self.n_components = n_components
        self.handle_outliers = handle_outliers
        self.handle_missing = handle_missing
        self.random_state = random_state
        self._is_fitted = False
        self.feature_names = None
        self.numerical_features = None
        self.categorical_features = None
        self.label_encoders = {}
        self.onehot_encoder = None
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize preprocessing components."""
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.num_scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.num_scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.num_scaler = RobustScaler()
        else:
            self.num_scaler = None
        
        # Initialize imputers for numerical and categorical data
        if self.handle_missing:
            if self.imputation_method == 'mean':
                self.num_imputer = SimpleImputer(strategy='mean')
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
            elif self.imputation_method == 'median':
                self.num_imputer = SimpleImputer(strategy='median')
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
            elif self.imputation_method == 'most_frequent':
                self.num_imputer = SimpleImputer(strategy='most_frequent')
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
            elif self.imputation_method == 'constant':
                self.num_imputer = SimpleImputer(strategy='constant', fill_value=0)
                self.cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
            else:
                self.num_imputer = SimpleImputer(strategy='mean')
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
        else:
            self.num_imputer = None
            self.cat_imputer = None
        
        # Initialize feature selector if specified
        if isinstance(self.feature_selection, int):
            k = min(self.feature_selection, 30)  # Cannot select more features than available
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif isinstance(self.feature_selection, float):
            k = max(1, int(self.feature_selection * 30))  # At least 1 feature
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        else:
            self.feature_selector = None
        
        # Initialize PCA if specified
        if self.n_components:
            self.pca = PCA(n_components=self.n_components)
        else:
            self.pca = None
            
        # Initialize categorical encoder
        if self.categorical_encoding == 'onehot':
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
    def _identify_feature_types(self, X):
        """
        Identify numerical and categorical features in the dataset.
        
        Args:
            X: Input features (pandas DataFrame)
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        numerical_features = []
        categorical_features = []
        
        for column in X.columns:
            # Check if the column contains only numeric values
            try:
                pd.to_numeric(X[column])
                numerical_features.append(column)
            except (ValueError, TypeError):
                categorical_features.append(column)
                
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        return numerical_features, categorical_features

    def fit(self, X, y=None):
        """
        Fit the preprocessor to the data.
        
        Args:
            X: Input features
            y: Target values (optional)
        """
        # Input validation
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy array or pandas DataFrame")
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        
        if X.shape[1] < 2:
            raise ValueError("X must have at least 2 features")
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Identify feature types
        self.numerical_features, self.categorical_features = self._identify_feature_types(X)
        
        # Handle categorical features first
        if self.categorical_features:
            if self.categorical_encoding == 'label':
                self.label_encoders = {}
                for col in self.categorical_features:
                    le = LabelEncoder()
                    # Convert to string to handle numeric categories
                    le.fit(X[col].astype(str))
                    self.label_encoders[col] = le
            elif self.categorical_encoding == 'onehot':
                self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.onehot_encoder.fit(X[self.categorical_features].astype(str))
        
        # Fit imputer for numerical features
        if self.handle_missing and self.numerical_features:
            self.num_imputer.fit(X[self.numerical_features])
        
        # Fit scaler for numerical features
        if self.num_scaler is not None and self.numerical_features:
            self.num_scaler.fit(X[self.numerical_features])
        
        # Fit feature selector if specified and y is provided
        if self.feature_selector is not None and y is not None:
            self.feature_selector.fit(X, y)
        
        # Fit PCA if specified
        if self.pca is not None:
            self.pca.fit(X)
        
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transform the data using fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            array-like: Transformed features
        """
        # Check if fitted
        if not self._is_fitted:
            raise NotFittedError("This DataPreprocessor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        
        # Input validation
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy array or pandas DataFrame")
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Create a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Handle outliers if specified
        if self.handle_outliers:
            X_transformed = self._handle_outliers(X_transformed)
        
        # Handle categorical features first
        if self.categorical_features:
            if self.categorical_encoding == 'label':
                for col in self.categorical_features:
                    X_transformed[col] = self.label_encoders[col].transform(X_transformed[col].astype(str))
            elif self.categorical_encoding == 'onehot':
                cat_encoded = self.onehot_encoder.transform(X_transformed[self.categorical_features].astype(str))
                cat_feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_features)
                X_transformed = pd.concat([
                    X_transformed[self.numerical_features],
                    pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X_transformed.index)
                ], axis=1)
        
        # Handle missing values for numerical features
        if self.handle_missing and self.numerical_features:
            X_transformed[self.numerical_features] = self.num_imputer.transform(X_transformed[self.numerical_features])
        
        # Scale numerical features
        if self.num_scaler is not None and self.numerical_features:
            X_transformed[self.numerical_features] = pd.DataFrame(
                self.num_scaler.transform(X_transformed[self.numerical_features]),
                columns=self.numerical_features,
                index=X_transformed.index
            )
        
        # Apply feature selection if specified
        if self.feature_selector is not None:
            selected_features = np.array(X_transformed.columns)[self.feature_selector.get_support()]
            X_transformed = pd.DataFrame(
                self.feature_selector.transform(X_transformed),
                columns=selected_features,
                index=X_transformed.index
            )
        
        # Apply PCA if specified
        if self.pca is not None:
            X_transformed = pd.DataFrame(
                self.pca.transform(X_transformed),
                columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                index=X_transformed.index
            )
        
        # Convert to numpy array if input was numpy array
        if isinstance(X, np.ndarray):
            X_transformed = X_transformed.values
        
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Input features
            y: Target values (optional)
            
        Returns:
            array-like: Transformed features
        """
        return self.fit(X, y).transform(X)

    def _handle_outliers(self, X):
        """
        Handle outliers using IQR method.
        
        Args:
            X: Input features (pandas DataFrame)
            
        Returns:
            pandas.DataFrame: Data with outliers handled
        """
        X_clean = X.copy()
        
        if self.numerical_features:
            for column in self.numerical_features:
                if pd.api.types.is_numeric_dtype(X[column]):
                    Q1 = X[column].quantile(0.25)
                    Q3 = X[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    X_clean[column] = X_clean[column].clip(lower=lower_bound, upper=upper_bound)
        
        return X_clean
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare data for training by splitting into train and test sets and applying preprocessing.
        
        Args:
            X: Input features
            y: Target values
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            tuple: (X_train_transformed, X_test_transformed, y_train, y_test)
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Split data
        X_train, X_test, y_train, y_test = sk_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit and transform training data
        X_train_transformed = self.fit_transform(X_train, y_train)
        
        # Transform test data
        X_test_transformed = self.transform(X_test)
        
        return X_train_transformed, X_test_transformed, y_train, y_test

    def get_feature_names(self):
        """Get names of features after transformation."""
        if not hasattr(self, 'feature_names_'):
            warnings.warn("Feature names are not available. Transform data first.")
            return None
        return self.feature_names_ 

    def _scale_features(self, X):
        """
        Scale features using StandardScaler.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Scaled features
        """
        if self.num_scaler is None:
            self._initialize_components()
        
        if self.num_scaler is None:
            return X  # Return unscaled features if no scaling method was specified
            
        return self.num_scaler.fit_transform(X)
        
    def _handle_missing_values(self, df):
        """
        Handle missing values in the data.
        
        Args:
            df: Input data (numpy array or pandas DataFrame)
            
        Returns:
            array-like: Data with missing values handled
        """
        # Convert numpy array to DataFrame if necessary
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        
        # For numeric columns, fill with mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # For non-numeric columns, fill with mode
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        # Convert back to numpy array if input was numpy array
        if isinstance(df, pd.DataFrame):
            return df.to_numpy()
        return df
        
    def _select_features(self, X, y, n_features=10):
        """
        Select top features based on ANOVA F-value.

        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            n_features (int): Number of features to select

        Returns:
            np.ndarray: Selected features
        """
        # Always create a new feature selector with the specified number of features
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        return self.feature_selector.fit_transform(X, y)
        
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and test sets.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def scale_features(self, X):
        """
        Scale features using StandardScaler.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Scaled features
        """
        if not self._is_fitted:
            self._initialize_components()
        
        if self.num_scaler is None:
            return X  # Return unscaled features if no scaling method was specified
            
        return self.num_scaler.transform(X)
    
    def impute_missing_values(self, X):
        """
        Impute missing values using mean strategy.
        
        Args:
            X (np.ndarray): Input features with possible missing values
            
        Returns:
            np.ndarray: Features with imputed values
        """
        return self.imputer.fit_transform(X)
    
    def select_features(self, X, y, k=10, feature_names=None):
        """
        Select top k features based on ANOVA F-value.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            k (int): Number of features to select
            feature_names (list): List of feature names
            
        Returns:
            np.ndarray: Selected features
        """
        self.feature_selector = SelectKBest(f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        if feature_names is not None:
            selected_indices = self.feature_selector.get_support()
            self.selected_feature_names = [
                name for name, selected in zip(feature_names, selected_indices)
                if selected
            ]
        
        return X_selected
    
    def transform_features(self, X):
        """
        Transform features using fitted preprocessor.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Transformed features
            
        Raises:
            NotFittedError: If preprocessor is not fitted
        """
        if not self._is_fitted:
            raise NotFittedError("Preprocessor must be fitted before transform")
        return self.scale_features(X) 