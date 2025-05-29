# Breast Cancer Prediction Project

A machine learning application for breast cancer prediction using the Wisconsin Breast Cancer dataset. This project implements a comprehensive ML pipeline with data preprocessing, model training, and a Streamlit web interface for real-time predictions.

## Project Structure

```
ml_project/
├── ml_pipeline/                # Core ML components
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing module
│   ├── models.py              # Model implementations
│   └── visualization/         # Visualization utilities
├── data/                      # Data storage
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data
│   └── interim/              # Intermediate data
├── models/                    # Saved model files
├── notebooks/                 # Jupyter notebooks
│   └── model_analysis.ipynb   # Model analysis and evaluation
├── tests/                     # Unit tests
│   ├── test_preprocessing.py
│   ├── test_app.py
│   └── test_models.py
├── app.py                     # Streamlit web application
├── train_model.py            # Model training script
├── requirements.txt          # Project dependencies
└── README.md
```

## Features

### Data Preprocessing (`preprocessing.py`)
- Feature scaling (Standard, MinMax, Robust)
- Missing value handling with multiple imputation methods
- Outlier detection and handling using IQR method
- Feature selection and dimensionality reduction
- Input validation and error handling

### Model Implementation (`models.py`)
- Model factory pattern for easy model creation
- Supported algorithms:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - XGBoost
  - Neural Networks (MLP)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Naive Bayes

### Web Application (`app.py`)
- Interactive Streamlit interface
- Real-time predictions
- Feature input validation
- Confidence score display
- Warning system for high-confidence predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml_project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate # On Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the model using the Wisconsin Breast Cancer dataset:
```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a logistic regression model
- Save the model and preprocessor to the `models` directory

### Running the Web Application

Start the Streamlit web interface:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Model Analysis

For detailed model analysis and evaluation:
1. Open Jupyter Notebook:
```bash
jupyter notebook
```
2. Navigate to `notebooks/model_analysis.ipynb`

### Running Tests

Execute the test suite:
```bash
python -m pytest tests/
```

## Dependencies

Main dependencies (see requirements.txt for complete list):
- numpy>=1.24.0
- pandas>=1.3.0
- scikit-learn>=1.3.2
- streamlit>=1.45.0
- xgboost>=1.5.0
- pytest>=7.0.0

## Development

### Setting up Development Environment

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests with coverage:
```bash
pytest --cov=ml_pipeline tests/
```

### Code Style

This project follows PEP 8 style guidelines. Key points:
- Use 4 spaces for indentation
- Maximum line length of 79 characters
- Docstrings for all public modules, functions, classes, and methods

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 