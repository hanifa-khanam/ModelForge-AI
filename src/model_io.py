"""
Model I/O Module - Model Persistence
Handles saving and loading complete model bundles for production use.
A bundle contains everything needed to preprocess and predict:
- Fitted preprocessor (ColumnTransformer)
- Fitted model
- Target encoder (for classification)
- Problem type
- Feature names
"""
import joblib
from typing import Dict, Any


def save_bundle(
    preprocessor,
    model,
    target_encoder,
    problem_type: str,
    feature_names: list,
    save_path: str,
) -> None:
    """
    Save a complete model bundle to disk.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        model: Fitted sklearn model
        target_encoder: Fitted LabelEncoder or None
        problem_type: "classification" or "regression"
        feature_names: List of original feature column names
        save_path: File path for .joblib output
    """
    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "target_encoder": target_encoder,
        "problem_type": problem_type,
        "feature_names": feature_names,
    }
    joblib.dump(bundle, save_path)


def load_bundle(load_path: str) -> Dict[str, Any]:
    """
    Load a saved model bundle from disk.
    
    Args:
        load_path: Path to .joblib file
        
    Returns:
        Dictionary with preprocessor, model, target_encoder,
        problem_type, feature_names
    """
    return joblib.load(load_path)