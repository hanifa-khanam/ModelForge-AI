"""
Validator Module - Input Validation and Quality Checks

Centralizes all validation logic:
1. Target column validation
2. Feature column validation  
3. Dataset suitability checks
4. Configuration validation
"""
import pandas as pd 

def validate_target_column(df: pd.DataFrame, target_column: str) -> dict:
    """
    Validate whether a column is suitable as a prediction target.
    """
    
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in Dataset") 
    
    target_data = df[target_column]
    unique_count = target_data.nunique() 
    null_count = target_data.isnull().sum() 
    total_rows = len(df) 
    
    # id column selection
    is_id_column = unique_count == total_rows
    
    # constant column detection
    is_constant = unique_count <= 1
    
    warnings = [] 
    
    if is_id_column:
        warnings.append("This column has unique values for every row it may be an ID column unsuitable for prediction") 
    
    if is_constant:
        warnings.append("This column has only one unique value nothing to predict") 
        
    if null_count > total_rows * 0.5:
        warnings.append(f"More than 50% of values are missing ({null_count}/{total_rows})")
        
    return {
        "column_name": target_column,
        "dtype": str(target_data.dtype),
        "unique_values": unique_count,
        "null_values": int(null_count),
        "null_percentage": round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0,
        "is_potential_id_column": is_id_column,
        "is_constant": is_constant,
        "warnings": warnings if warnings else None        
    }
    
    
def validate_feature_columns(df: pd.DataFrame, target_column: str) -> dict:
    """
    Validate feature columns (all columns except target)
    """
    
    feature_cols = [col for col in df.columns if col != target_column] 
    
    all_null_cols = [] 
    constant_cols = [] 
    
    for col in feature_cols:
        if df[col].isnull().all():
            all_null_cols.append(col) 
        elif df[col].nunique() <= 1:
            constant_cols.append(col) 
            
    return {
        "total_features" : len(feature_cols),
        "all_null_columns" : all_null_cols,
        "constant_columns" : constant_cols,
        "usable_features" : len(feature_cols) - len(all_null_cols) - len(constant_cols),
        "has_issues" : len(all_null_cols) > 0 or len(constant_cols) > 0
    }
    
def validate_dataset_for_ml(df: pd.DataFrame, target_column: str) ->dict:
    """
    Comprehensive validation: is this dataset suitable for Supervised ML?
    returns go/no-go recommendation with reasoning.
    """
    target_result = validate_target_column(df, target_column)
    feature_result = validate_feature_columns(df, target_column)
    
    # Blocking issues (can't proceed) 
    blockers = [] 
    if target_result.get("is_constant"):
        blockers.append("Target column is constant nothing to predict") 
    if feature_result["usable_features"] == 0:
        blockers.append("No usable feature columns remaining") 
        
        
    # warnings (can proceed but be careful) 
    warnings = [] 
    if target_result.get("is_potential_id_column"):
        warnings.append("Target appears to be an ID column") 
    if feature_result["all_null_columns"]:
        warnings.append(f"{len(feature_result['all_null_columns'])} features columns are entirely null") 
    if feature_result["constant_columns"]:
        warnings.append(f"{len(feature_result['constant_columns'])} features columns have zero variance") 
        
    can_proceed = len(blockers) == 0
    
    return {
        "can_proceed" : can_proceed,
        "blockers" : blockers,
        "warnings" : warnings,
        "target_validation" : target_result,
        "feature_validation": feature_result
    }
        
    