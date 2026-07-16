"""
Tuner Module - Hyperparameter Optimization

Uses GridSearchCV to find optimal hyperparameters for each model.
Returns tuned models with cross-validated performance metrics.
Skips models without parameter grids.
"""
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from src.config import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    PARAM_GRID_CLASSIFICATION,
    PARAM_GRID_REGRESSION,
    CV_FOLDS,
    TUNING_SCORING_CLASSIFICATION,
    TUNING_SCORING_REGRESSION,
)
from src.trainer import calculate_metrics


def tune_models(preprocessed_data: dict, cv_folds: int = CV_FOLDS) -> dict:
    """
    Perform hyperparameter tuning for all applicable models.
    
    For each model with a parameter grid, runs GridSearchCV with
    cross-validation. Models without a grid are trained with defaults.
    
    Args:
        preprocessed_data: Output from preprocessing.run_preprocessing()
        cv_folds: Number of cross-validation folds
        
    Returns:
        dict with problem_type, tuned_models list, and tuning metadata
    """
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]
    problem_type = preprocessed_data["problem_type"]
    
    # Select models and parameter grids based on problem type
    if problem_type == "classification":
        models = CLASSIFICATION_MODELS
        param_grids = PARAM_GRID_CLASSIFICATION
        scoring = TUNING_SCORING_CLASSIFICATION
    else:
        models = REGRESSION_MODELS
        param_grids = PARAM_GRID_REGRESSION
        scoring = TUNING_SCORING_REGRESSION
    
    tuned_results = []
    total_combinations = 0
    
    for model_name, model in models.items():
        param_grid = param_grids.get(model_name, {})
        
        # Models without a parameter grid: train with defaults
        if not param_grid:
            current_model = clone(model)
            start_time = time.time()
            current_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            tuned_results.append({
                "model_name": model_name,
                "model": current_model,
                "best_params": {},
                "best_cv_score": None,
                "tuned": False,
                "training_time": round(training_time, 4),
                "message": "No hyperparameters to tune — using defaults.",
            })
            continue
        
        # Calculate total combinations for this model
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        total_combinations += n_combinations
        
        # Perform grid search with cross-validation
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=1,
            verbose=0,
            error_score="raise",
        )
        
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Extract best results
        best_index = grid_search.best_index_
        cv_results = grid_search.cv_results_
        
        tuned_results.append({
            "model_name": model_name,
            "model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_cv_score": round(grid_search.best_score_, 4),
            "tuned": True,
            "training_time": round(training_time, 4),
            "cv_details": {
                "mean_cv_score": round(
                    cv_results["mean_test_score"][best_index], 4
                ),
                "std_cv_score": round(
                    cv_results["std_test_score"][best_index], 4
                ),
                "n_combinations": n_combinations,
                "total_fits": n_combinations * cv_folds,
            },
        })
    
    return {
        "problem_type": problem_type,
        "tuned_models": tuned_results,
        "cv_folds": cv_folds,
        "scoring_metric": scoring,
        "total_combinations_searched": total_combinations,
        "total_fits": total_combinations * cv_folds,
    }


def evaluate_tuned_models(tuned_results: dict, preprocessed_data: dict) -> dict:
    """
    Evaluate tuned models on the test set.
    
    Uses the shared calculate_metrics() from trainer.py to ensure
    consistent metric computation with the default training workflow.
    
    Args:
        tuned_results: Output from tune_models()
        preprocessed_data: Output from preprocessing.run_preprocessing()
        
    Returns:
        dict with results list in the same format as trainer.train_models()
    """
    X_train = preprocessed_data["X_train"]
    X_test = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_test = preprocessed_data["y_test"]
    problem_type = tuned_results["problem_type"]
    
    results = []
    
    for tuned_model in tuned_results["tuned_models"]:
        model = tuned_model["model"]
        
        # Predict on both train and test sets
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Use shared metric calculation from trainer.py
        train_metrics, test_metrics = calculate_metrics(
            y_train, train_predictions,
            y_test, test_predictions,
            problem_type
        )
        
        results.append({
            "model_name": tuned_model["model_name"],
            "model": model,
            "metrics": test_metrics,
            "train_metrics": train_metrics,
            "training_time": tuned_model["training_time"],
            "predictions": test_predictions,
            "best_params": tuned_model.get("best_params", {}),
            "best_cv_score": tuned_model.get("best_cv_score"),
        })
    
    return {
        "problem_type": problem_type,
        "results": results,
    }


def get_tuning_summary(tuned_results: dict) -> dict:
    """
    Generate a human-readable summary of tuning results.
    
    Args:
        tuned_results: Output from tune_models()
        
    Returns:
        dict with summary text and per-model highlights
    """
    models_tuned = sum(1 for m in tuned_results["tuned_models"] if m["tuned"])
    models_skipped = sum(1 for m in tuned_results["tuned_models"] if not m["tuned"])
    
    # Find best model by CV score
    tuned_only = [m for m in tuned_results["tuned_models"] if m["tuned"]]
    best_tuned = max(tuned_only, key=lambda m: m["best_cv_score"]) if tuned_only else None
    
    summary_lines = [
        f"Tuned {models_tuned} models across {tuned_results['total_combinations_searched']} parameter combinations.",
        f"Total fits performed: {tuned_results['total_fits']} ({tuned_results['cv_folds']}-fold CV).",
        f"Scoring metric: {tuned_results['scoring_metric']}.",
    ]
    
    if models_skipped > 0:
        summary_lines.append(
            f"{models_skipped} models used default parameters (no grid defined)."
        )
    
    if best_tuned:
        summary_lines.append(
            f"Best CV score: {best_tuned['model_name']} "
            f"({best_tuned['best_cv_score']})."
        )
    
    return {
        "summary": " ".join(summary_lines),
        "models_tuned": models_tuned,
        "models_skipped": models_skipped,
        "best_tuned_model": best_tuned["model_name"] if best_tuned else None,
        "best_cv_score": best_tuned["best_cv_score"] if best_tuned else None,
    }