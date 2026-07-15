"""
Perprocessing Module - Automatic Data Preprocessing Pipeline
1- Problem detection      (classification, regression)
2- Feature type detection (numerical, nominal categorical, ordinal categorical)
3- Train Test Split
4- Column-Specific transformation pipelines
5- Full preprocessing pipeline assembly
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")

def detect_problem_type(y: pd.Series) -> str:
    """
    Detect whether the target column indicates a classification or regression problem.
    """
    unique_values = y.nunique() 
    dtype = y.dtype 
    
    if dtype == "object":
        return "classification" 
    
    if pd.api.types.is_integer_dtype(dtype):
        if unique_values <=20:
            return "classification"
        else:
            return "regression" 
    
    if pd.api.types.is_float_dtype(dtype):
        return "regression"
    
    return "classification"
    
    
def detect_feature_types(X: pd.DataFrame) -> tuple:
    """
    Separate features into numerical, nominal categorical, ordinal categorical columns.
    """
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist() 
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist() 
    
    # ordinal columns require domain knowledge - return empty by default
    # user can override this list by config
    ordinal_cols = [] 
    
    return numerical_cols, categorical_cols, ordinal_cols


def separate_features_target(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Split Dataframe into feature X and target vector y.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame") 
    
    X = df.drop(columns=[target_column]) 
    y = df[target_column] 
    
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, random_state: int=42) -> tuple:
    """
    Split data into training and test sets.
    """
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)


def build_preprocessing_pipeline(numerical_cols: list, nominal_cols: list, ordianl_cols: list=None, ordinal_categories: list=None) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies appropriate preprocessing to each column type.
    """
    
    if ordianl_cols is None:
        ordianl_cols = [] 
        
    transformers = [] 
    
    if numerical_cols:
        numeric_pipline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
        transformers.append(("num", numeric_pipline, numerical_cols)) 
        
    if nominal_cols:
        nominal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) 
            ]
        )
        transformers.append(("nom", nominal_pipeline, nominal_cols)) 
        
    if ordianl_cols:
        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=ordinal_categories if ordinal_categories else "auto", handle_unknown="use_encoded_value", unknown_value=-1))
            ]
        )
        transformers.append(("ord", ordinal_pipeline, ordianl_cols)) 
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    
    return preprocessor


def encode_target(y: pd.Series, problem_type: str) -> tuple:
    """
    Encode the target variable if needed. 
    Classification target (object/string) are label-encoded to integers.
    Regression targets are returned unchanged.
    """
    if problem_type == "classification" and y.dtype == "object":
        encoder = LabelEncoder() 
        y_encoded = encoder.fit_transform(y) 
        return y_encoded, encoder
    
    return y, None



def run_preprocessing(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float=0.2, 
    random_state: int=42,
    ordinal_cols: list=None,
    ordinal_categories: list=None) -> dict:
    
    """
    Execute the complete preprocessing workflow.
    """
    
    # Step 1: separate X and y
    X, y = separate_features_target(df, target_column)
    
    # Step 2: detect problem type
    problem_type = detect_problem_type(y) 
    
    # Step 3: Detect features type
    numerical_cols, nominal_cols, auto_ordinal_cols = detect_feature_types(X) 
    
    # merged auto detected ordinal with user specified
    if ordinal_cols:
        all_ordinal_cols = ordinal_cols
    else:
        all_ordinal_cols = auto_ordinal_cols
        
    # Merged ordinal columns from nominal list if they overlap
    nominal_cols = [col for col in nominal_cols if col not in all_ordinal_cols]
    
    # step 4: Split data
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size, random_state=random_state)
    
    # Step 5: Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_cols, nominal_cols, all_ordinal_cols, ordinal_categories) 
    
    # Step 6: Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train) 
    X_test_processed = preprocessor.transform(X_test)
    
    
    # Step 7: Encode target
    y_train_encoded, target_encoder = encode_target(y_train, problem_type) 
    if target_encoder is not None:
        y_test_encoded = target_encoder.transform(y_test)
    else:
        y_test_encoded = y_test
    
    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "problem_type": problem_type,
        "feature_names": numerical_cols + nominal_cols + all_ordinal_cols,
        "numerical_cols": numerical_cols,
        "nominal_cols": nominal_cols,
        "ordinal_cols": all_ordinal_cols 
    }
    
    
def save_preprocessing_artifacts(
    preprocessor: ColumnTransformer,
    target_encoder: LabelEncoder,
    problem_type: str,
    feature_names: list,
    save_path: str = "saved_models/preprocessing_pipeline.joblib"
) -> None:
    """
    Save preprocessing pipeline and metadata for production use. The saved bundle contains everything needed to preprocess new data identically to training data.
    """
    bundle = {
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "problem_type": problem_type,
        "feature_names": feature_names
    }
    joblib.dump(bundle, save_path)


def load_preprocessing_artifacts(load_path: str = "saved_models/preprocessing_pipeline.joblib") -> dict:
    """Load saved preprocessing bundle."""
    return joblib.load(load_path)