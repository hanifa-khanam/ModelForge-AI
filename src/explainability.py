"""
Provides feature importance explanations for trained models.
Supports tree-based importance, linear coefficients, and aggregation
of one-hot encoded features back to original column names.
"""
import numpy as np
from typing import Dict, Any, List, Optional


def get_feature_names_after_preprocessing(preprocessor,
                                           original_features: List[str]) -> List[str]:
    """
    Extract feature names from a fitted ColumnTransformer.
    For one-hot encoded features, returns names like 'city_Lahore'.
    For numerical features, returns the original column name.
    """
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == "drop" or columns is None or len(columns) == 0:
            continue
        
        if hasattr(transformer, "named_steps"):
            encoder = transformer.named_steps.get("encoder")
            if encoder and hasattr(encoder, "get_feature_names_out"):
                try:
                    encoded_names = encoder.get_feature_names_out(columns)
                    feature_names.extend(encoded_names.tolist())
                    continue
                except Exception:
                    pass
        
        feature_names.extend(columns)
    
    return feature_names


def aggregate_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    original_columns: List[str]
) -> List[Dict[str, Any]]:
    """
    Aggregate one-hot encoded feature importances back to original columns.
    """
    aggregated = {}
    for col in original_columns:
        aggregated[col] = {"importance": 0.0, "sub_features": []}
    
    for encoded_name, importance in zip(feature_names, importance_scores):
        matched = False
        for orig_col in original_columns:
            if encoded_name == orig_col or encoded_name.startswith(orig_col + "_"):
                aggregated[orig_col]["importance"] += float(importance)
                if encoded_name != orig_col:
                    aggregated[orig_col]["sub_features"].append(encoded_name)
                matched = True
                break
        
        if not matched:
            aggregated[encoded_name] = {
                "importance": float(importance),
                "sub_features": []
            }
    
    result = [
        {
            "feature": col,
            "importance": round(info["importance"], 4),
            "sub_features": info["sub_features"]
        }
        for col, info in aggregated.items()
    ]
    
    result.sort(key=lambda x: x["importance"], reverse=True)
    return result


def explain_model(
    model,
    feature_names: List[str],
    original_columns: Optional[List[str]] = None,
    aggregate: bool = True,
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Extract and explain feature importance from a trained model.
    """
    importance = None
    method = None
    
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        method = "feature_importances"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
        method = "coefficients"
    else:
        return {
            "method": "not_available",
            "message": (
                f"Model type '{type(model).__name__}' does not provide "
                "built-in feature importance. Consider using SHAP."
            ),
            "feature_importance": [],
            "top_feature": None,
        }
    
    if len(importance) != len(feature_names):
        min_len = min(len(importance), len(feature_names))
        importance = importance[:min_len]
        feature_names = feature_names[:min_len]
    
    if aggregate and original_columns:
        feature_importance = aggregate_feature_importance(
            feature_names, importance, original_columns
        )
    else:
        feature_importance = [
            {"feature": name, "importance": round(float(imp), 4), "sub_features": []}
            for name, imp in zip(feature_names, importance)
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    feature_importance = feature_importance[:top_n]
    
    return {
        "method": method,
        "feature_importance": feature_importance,
        "top_feature": feature_importance[0]["feature"] if feature_importance else None,
        "model_type": type(model).__name__,
        "num_features": len(feature_names),
        "aggregated": aggregate and original_columns is not None,
    }


def explain_from_pipeline(
    pipeline: Dict[str, Any],
    aggregate: bool = True,
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Explain a model from a loaded pipeline dictionary.
    """
    model = pipeline["model"]
    preprocessor = pipeline.get("preprocessor")
    original_columns = pipeline.get("feature_names", [])
    
    if preprocessor and hasattr(preprocessor, "transformers_"):
        try:
            processed_names = get_feature_names_after_preprocessing(
                preprocessor, original_columns
            )
        except Exception:
            processed_names = original_columns
    else:
        processed_names = original_columns
    
    return explain_model(
        model=model,
        feature_names=processed_names,
        original_columns=original_columns if aggregate else None,
        aggregate=aggregate,
        top_n=top_n,
    )