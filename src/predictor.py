"""
Loads a saved model pipeline and makes predictions on new data.
Handles all preprocessing identically to training time.
"""
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any


def load_pipeline(pipeline_path: str) -> Dict[str, Any]:
    """
    Load a saved model pipeline from disk.
    Args:
        pipeline_path: Path to .joblib file containing preprocessor, model, target_encoder, problem_type, feature_names
    Returns:
        Dictionary with all pipeline components
    Raises:
        ValueError: If required keys are missing from saved file
    """
    bundle = joblib.load(pipeline_path)
    
    required_keys = ["preprocessor", "model", "problem_type", "feature_names"]
    missing_keys = [k for k in required_keys if k not in bundle]
    
    if missing_keys:
        raise ValueError(
            f"Saved pipeline is missing required keys: {missing_keys}"
        )
    
    return bundle


def predict(pipeline: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
    """
    Make predictions on new data using a loaded pipeline.
    Preprocessing is applied automatically using the fitted preprocessor
    from training. No retraining or refitting occurs.
    Args:
        pipeline: Loaded pipeline from load_pipeline()
        data: DataFrame with feature columns for prediction 
    Returns:
        Dictionary with predictions, problem_type, and optional probabilities
    """
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    target_encoder = pipeline.get("target_encoder")
    problem_type = pipeline["problem_type"]
    feature_names = pipeline["feature_names"]
    
    # Validate that required features are present
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        raise ValueError(
            f"Input data is missing required features: {missing_features}"
        )
    
    # Select and order features as expected by the preprocessor
    X = data[feature_names].copy()
    
    # Transform using the fitted preprocessor (NOT fit_transform)
    X_processed = preprocessor.transform(X)
    
    # Make predictions
    raw_predictions = model.predict(X_processed)
    
    # Decode target labels for classification
    if problem_type == "classification" and target_encoder is not None:
        try:
            predictions = target_encoder.inverse_transform(raw_predictions)
        except Exception:
            predictions = raw_predictions
    else:
        predictions = raw_predictions
    
    # Get prediction probabilities for classification
    prediction_proba = None
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_processed)
            if target_encoder is not None:
                classes = target_encoder.classes_
            else:
                classes = model.classes_
            
            prediction_proba = {
                str(classes[i]): round(float(proba[0][i]), 4)
                for i in range(len(classes))
            }
        except Exception:
            prediction_proba = None
    
    return {
        "predictions": (
            predictions.tolist() if hasattr(predictions, "tolist") 
            else list(predictions)
        ),
        "raw_predictions": (
            raw_predictions.tolist() if hasattr(raw_predictions, "tolist") 
            else list(raw_predictions)
        ),
        "problem_type": problem_type,
        "num_samples": len(data),
        "prediction_proba": prediction_proba,
    }


def predict_single(pipeline: Dict[str, Any], 
                   input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction on a single sample provided as a dictionary.
    Args:
        pipeline: Loaded pipeline from load_pipeline()
        input_dict: Dictionary of {feature_name: value}
    Returns:
        Prediction result dictionary
    """
    input_df = pd.DataFrame([input_dict])
    return predict(pipeline, input_df)


def predict_from_file(pipeline_path: str, data_path: str) -> Dict[str, Any]:
    """
    Load pipeline and predict on a CSV file in one step.
    Convenience function for batch predictions.
    Args:
        pipeline_path: Path to .joblib pipeline file
        data_path: Path to CSV file with prediction data
    Returns:
        Prediction result dictionary
    """
    pipeline = load_pipeline(pipeline_path)
    data = pd.read_csv(data_path)
    return predict(pipeline, data)