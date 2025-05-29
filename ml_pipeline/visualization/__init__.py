# -*- coding: utf-8 -*-
"""
Visualization module for plotting model performance metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, classes):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix to plot
        classes (list): List of class labels
    """
    if not isinstance(cm, np.ndarray) or cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be a 2x2 numpy array")
        
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve.
    
    Args:
        fpr (np.ndarray): False positive rates
        tpr (np.ndarray): True positive rates
        roc_auc (float): Area under ROC curve
    """
    if not (0 <= roc_auc <= 1):
        raise ValueError("AUC must be between 0 and 1")
    if len(fpr) != len(tpr):
        raise ValueError("FPR and TPR arrays must have the same length")
        
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', pad=20)
    plt.legend(loc="lower right")
    return fig 