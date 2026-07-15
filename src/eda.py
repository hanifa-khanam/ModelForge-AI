"""
Analyzes raw data and produces insights for the user before preprocessing.
All functions are read-only — they analyze but never modify the DataFrame.
"""
import pandas as pd
import numpy as np
from typing import Optional

def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive overview of the dataset.
    Returns basic dimensions, data types, memory usage,
    missing values, and duplicate information.
    """
    rows, cols = df.shape
    
    # Data type breakdown
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_summary = {str(k): v for k, v in dtype_counts.items()}
    
    # Missing value analysis
    missing_per_column = df.isnull().sum()
    missing_df = pd.DataFrame({
        "column": missing_per_column.index,
        "missing_count": missing_per_column.values,
        "missing_percentage": (missing_per_column / rows * 100).round(2).values
    })
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        "missing_percentage", ascending=False
    )
    
    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    
    return {
        "rows": rows,
        "columns": cols,
        "dtype_summary": dtype_summary,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        "total_missing": int(missing_per_column.sum()),
        "missing_percentage": round((missing_per_column.sum() / (rows * cols)) * 100, 2),
        "columns_with_missing": len(missing_df),
        "missing_details": missing_df.to_dict(orient="records")[:10],  # Top 10
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / rows) * 100, 2) if rows > 0 else 0,
    }
    
    
def analyze_target(df: pd.DataFrame, target_column: str, problem_type: str) -> dict:
    """
    Analyze the target column based on problem type.
    For classification: class distribution, balance assessment.
    For regression: distribution statistics, outlier detection.
    """
    target = df[target_column].dropna()
    
    if problem_type == "classification":
        value_counts = target.value_counts()
        
        # Class distribution
        distribution = {
            str(k): int(v) for k, v in value_counts.items()
        }
        
        # Balance assessment
        dominant_pct = (value_counts.max() / len(target)) * 100
        is_imbalanced = dominant_pct > 70
        
        imbalance_warning = None
        if is_imbalanced:
            minority_classes = [
                str(k) for k in value_counts.index 
                if value_counts[k] < len(target) * 0.15
            ]
            imbalance_warning = (
                f"Imbalanced dataset: dominant class has {dominant_pct:.1f}% "
                f"of samples. Minority classes: {', '.join(minority_classes)}"
            )
        
        return {
            "problem_type": "classification",
            "num_classes": len(value_counts),
            "total_samples": len(target),
            "distribution": distribution,
            "dominant_class_pct": round(dominant_pct, 1),
            "is_imbalanced": is_imbalanced,
            "imbalance_warning": imbalance_warning,
        }
    
    else:  # Regression
        stats = target.describe()
        
        # Outlier detection using IQR
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((target < lower_bound) | (target > upper_bound)).sum()
        
        # Skewness
        skewness = target.skew()
        skew_interpretation = (
            "approximately symmetric" if abs(skewness) < 0.5
            else "moderately skewed" if abs(skewness) < 1
            else "highly skewed"
        )
        
        return {
            "problem_type": "regression",
            "total_samples": len(target),
            "mean": round(stats["mean"], 4),
            "std": round(stats["std"], 4),
            "min": round(stats["min"], 4),
            "25%": round(stats["25%"], 4),
            "50%": round(stats["50%"], 4),
            "75%": round(stats["75%"], 4),
            "max": round(stats["max"], 4),
            "outliers": int(outliers),
            "outlier_percentage": round((outliers / len(target)) * 100, 2) if len(target) > 0 else 0,
            "skewness": round(skewness, 4),
            "skew_interpretation": skew_interpretation,
        }
    
def analyze_features(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    """
    Analyze feature columns for correlations, cardinality, and variance.
    If target_column is provided, includes correlation with target.
    """
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    result = {
        "numerical_count": len(numerical_cols),
        "categorical_count": len(categorical_cols),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }
    
    # 1. Correlation matrix (numerical only)
    if len(numerical_cols) >= 2:
        corr_matrix = df[numerical_cols].corr()
        
        # Top correlations (absolute value, excluding self-correlations)
        correlations = []
        seen = set()
        for col1 in numerical_cols:
            for col2 in numerical_cols:
                if col1 != col2 and (col2, col1) not in seen:
                    seen.add((col1, col2))
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.5:  # Only meaningful correlations
                        correlations.append({
                            "feature_1": col1,
                            "feature_2": col2,
                            "correlation": round(corr_value, 3),
                            "magnitude": "strong" if abs(corr_value) > 0.8 else "moderate"
                        })
        
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        result["correlations"] = correlations[:10]  # Top 10
        result["highly_correlated_pairs"] = len(correlations)
    else:
        result["correlations"] = []
        result["highly_correlated_pairs"] = 0
    
    # 2. Target correlations
    if target_column and target_column in numerical_cols:
        target_col = target_column
        feature_cols = [c for c in numerical_cols if c != target_col]
        
        if feature_cols:
            target_corrs = {}
            for col in feature_cols:
                target_corrs[col] = round(df[col].corr(df[target_col]), 3)
            
            # Sort by absolute correlation
            sorted_corrs = sorted(
                target_corrs.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            result["target_correlations"] = [
                {"feature": col, "correlation": corr}
                for col, corr in sorted_corrs[:10]
            ]
    
    # 3. Categorical cardinality
    if categorical_cols:
        cardinality = {}
        high_cardinality = []
        for col in categorical_cols:
            n_unique = df[col].nunique()
            cardinality[col] = n_unique
            if n_unique > 50:
                high_cardinality.append({
                    "column": col,
                    "unique_values": n_unique,
                    "warning": f"High cardinality ({n_unique} unique values)"
                })
        
        result["categorical_cardinality"] = cardinality
        result["high_cardinality_cols"] = high_cardinality
    
    # 4. Low variance features (near-constant)
    if numerical_cols:
        low_variance = []
        for col in numerical_cols:
            if df[col].nunique() <= 2:
                low_variance.append(col)
        result["low_variance_cols"] = low_variance
    
    # 5. Generate warnings
    warnings = []
    if result["highly_correlated_pairs"] > 0:
        warnings.append(
            f"Found {result['highly_correlated_pairs']} highly correlated feature pairs. "
            "Consider removing redundant features."
        )
    if categorical_cols and "high_cardinality_cols" in result:
        if result["high_cardinality_cols"]:
            warnings.append(
                f"{len(result['high_cardinality_cols'])} categorical columns have "
                "high cardinality (>50 unique values)."
            )
    if low_variance:
        warnings.append(
            f"{len(low_variance)} numerical columns have near-zero variance."
        )
    
    result["warnings"] = warnings
    return result

def run_eda(df: pd.DataFrame, target_column: Optional[str] = None, problem_type: Optional[str] = None) -> dict:
    """
    Run complete EDA pipeline.
    Can run with or without a target column.
    If target and problem_type are provided, includes target analysis.
    """
    results = {
        "dataset_overview": analyze_dataset(df),
        "feature_analysis": analyze_features(df, target_column),
    }
    
    if target_column and problem_type:
        results["target_analysis"] = analyze_target(df, target_column, problem_type)
    
    # Aggregate all warnings
    all_warnings = []
    if "warnings" in results["feature_analysis"]:
        all_warnings.extend(results["feature_analysis"]["warnings"])
    if "target_analysis" in results:
        if results["target_analysis"].get("imbalance_warning"):
            all_warnings.append(results["target_analysis"]["imbalance_warning"])
    
    results["warnings"] = all_warnings
    results["has_warnings"] = len(all_warnings) > 0
    
    return results