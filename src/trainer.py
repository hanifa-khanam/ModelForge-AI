"""
Trainer Module - Multi-Model Training Engine

Trains multiple supervised learning models and computes performance metrics.
Provides shared metric calculation used by both training and tuning workflows.
"""
import time
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from src.config import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, DEFAULT_RANDOM_STATE
)


def calculate_metrics(y_train, train_predictions, y_test, test_predictions, 
                      problem_type: str) -> tuple:
    """
    Calculate both train and test metrics for a model.
    
    Centralized metric computation used by both train_models() and 
    tuner.evaluate_tuned_models() to ensure consistent calculations
    and identical output keys.
    
    Args:
        y_train: True training labels
        train_predictions: Model predictions on training data
        y_test: True test labels
        test_predictions: Model predictions on test data
        problem_type: "classification" or "regression"
        
    Returns:
        Tuple of (train_metrics_dict, test_metrics_dict)
    """
    if problem_type == "classification":
        train_metrics = {
            "accuracy": accuracy_score(y_train, train_predictions),
            "precision": precision_score(
                y_train, train_predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_train, train_predictions, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_train, train_predictions, average="weighted", zero_division=0
            ),
        }
        test_metrics = {
            "accuracy": accuracy_score(y_test, test_predictions),
            "precision": precision_score(
                y_test, test_predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, test_predictions, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_test, test_predictions, average="weighted", zero_division=0
            ),
        }
    else:
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        
        train_metrics = {
            "r2": r2_score(y_train, train_predictions),
            "mae": mean_absolute_error(y_train, train_predictions),
            "mse": train_mse,
            "rmse": np.sqrt(train_mse),
        }
        test_metrics = {
            "r2": r2_score(y_test, test_predictions),
            "mae": mean_absolute_error(y_test, test_predictions),
            "mse": test_mse,
            "rmse": np.sqrt(test_mse),
        }
    
    return train_metrics, test_metrics


def train_models(data: dict) -> dict:
    """
    Train multiple models and return performance results.
    
    Args:
        data: Output from preprocessing.run_preprocessing()
              Must contain: X_train, X_test, y_train, y_test, problem_type
        
    Returns:
        dict with 'problem_type' and 'results' list containing
        fitted models, metrics, and predictions
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    problem_type = data["problem_type"]
    
    # Validate inputs
    if problem_type not in ["classification", "regression"]:
        raise ValueError(
            f"Unknown problem type: '{problem_type}'. "
            "Must be 'classification' or 'regression'."
        )
    
    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty. Cannot train models.")
    
    # Select model dictionary based on problem type
    models = (
        CLASSIFICATION_MODELS if problem_type == "classification"
        else REGRESSION_MODELS
    )
    
    results = []
    
    for model_name, model in models.items():
        # Clone to avoid mutating the config's original
        current_model = clone(model)
        
        # Time the training
        start_time = time.time()
        current_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict on both train and test
        train_predictions = current_model.predict(X_train)
        test_predictions = current_model.predict(X_test)
        
        # Calculate metrics using shared function
        train_metrics, test_metrics = calculate_metrics(
            y_train, train_predictions,
            y_test, test_predictions,
            problem_type
        )
        
        results.append({
            "model_name": model_name,
            "model": current_model,
            "metrics": test_metrics,
            "train_metrics": train_metrics,
            "training_time": round(training_time, 4),
            "predictions": test_predictions,
        })
    
    return {
        "problem_type": problem_type,
        "results": results,
    }