"""
Tests for the visualization module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from ml_pipeline.visualization import plot_confusion_matrix, plot_roc_curve

@pytest.fixture
def confusion_matrix_data():
    """Fixture providing sample confusion matrix data."""
    return np.array([[50, 10], [5, 35]])

@pytest.fixture
def roc_curve_data():
    """Fixture providing sample ROC curve data."""
    fpr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr = np.array([0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0])
    roc_auc = 0.85
    return fpr, tpr, roc_auc

def test_confusion_matrix_plot(confusion_matrix_data):
    """Test confusion matrix plotting functionality."""
    # Clear any existing plots
    plt.close('all')
    
    # Create the plot
    plot_confusion_matrix(confusion_matrix_data, classes=[0, 1])
    
    # Get the current figure
    fig = plt.gcf()
    
    # Verify the plot was created
    assert plt.fignum_exists(fig.number)
    
    # Check plot properties
    ax = fig.gca()
    assert ax.get_title() == 'Confusion Matrix'
    assert ax.get_xlabel() == 'Predicted Label'
    assert ax.get_ylabel() == 'True Label'
    
    plt.close('all')

def test_roc_curve_plot(roc_curve_data):
    """Test ROC curve plotting functionality."""
    # Clear any existing plots
    plt.close('all')
    
    # Create the plot
    fpr, tpr, roc_auc = roc_curve_data
    plot_roc_curve(fpr, tpr, roc_auc)
    
    # Get the current figure
    fig = plt.gcf()
    
    # Verify the plot was created
    assert plt.fignum_exists(fig.number)
    
    # Check plot properties
    ax = fig.gca()
    assert ax.get_title() == 'Receiver Operating Characteristic (ROC) Curve'
    assert ax.get_xlabel() == 'False Positive Rate'
    assert ax.get_ylabel() == 'True Positive Rate'
    
    # Verify the legend exists and contains ROC curve information
    legend = ax.get_legend()
    assert legend is not None
    assert any(f'ROC curve (AUC = {roc_auc:.2f})' in text.get_text() 
              for text in legend.get_texts())
    
    plt.close('all')

def test_invalid_confusion_matrix():
    """Test handling of invalid confusion matrix input."""
    with pytest.raises(ValueError):
        # Test with invalid shape
        invalid_cm = np.array([[1, 2, 3], [4, 5, 6]])
        plot_confusion_matrix(invalid_cm, classes=[0, 1])

def test_invalid_roc_curve_data():
    """Test handling of invalid ROC curve input."""
    with pytest.raises(ValueError):
        # Test with mismatched array lengths
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.5])  # Missing one value
        plot_roc_curve(fpr, tpr, 0.85)

@pytest.mark.parametrize("invalid_auc", [-0.5, 1.5])
def test_invalid_auc_value(invalid_auc, roc_curve_data):
    """Test handling of invalid AUC values."""
    fpr, tpr, _ = roc_curve_data
    with pytest.raises(ValueError):
        plot_roc_curve(fpr, tpr, invalid_auc) 