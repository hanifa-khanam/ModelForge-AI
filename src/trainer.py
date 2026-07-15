import time
import numpy as np 
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from config import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, DEFAULT_RANDOM_STATE
)

def train_models(data: dict) -> dict:
    """
    Train multiple models and return performance results
    Args:
        data: Output from preprocessing.run_preprocessing()
        Must contain: X_train, X_test, y_train, y_test, problem_type
    Returns:
        dict with 'problem_type' and 'results' list
    """
    
    # extract data from input dictionary (run_preprocessing())
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"] 
    y_test =  data["y_test"] 
    problem_type = data["problem_type"]
    
    # validate inputs
    if problem_type not in ["classification", "regression"]:
        raise ValueError(f"Unknown problem type: '{problem_type}'."
                        "Must be 'classification' or 'regression'."
        ) 
    
    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty. Cannot train models.")
    
    # select model dictionary
    if problem_type == "classification":
        models = CLASSIFICATION_MODELS
    else:
        models = REGRESSION_MODELS
        
    # train each model and collect results
    results = []
    
    for model_name, model in models.items():
        
        # Clone to avoid mutating the config's original
        current_model = clone(model) 
        
        # time the training
        start_time = time.time()
        current_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # predict
        train_prediction = current_model.predict(X_train)
        test_prediction = current_model.predict(X_test)
        
        # Calculate train metrics too (for overfitting detection)
        if problem_type == "classification":
            train_metrics = {
                "accuracy": accuracy_score(y_train, train_prediction),
                "precision": precision_score(y_train, train_prediction, average="weighted", zero_division=0),
                "recall": recall_score(y_train, train_prediction, average="weighted",zero_division=0),
                "f1": f1_score(y_train, train_prediction, average="weighted", zero_division=0)
            }
        else:
            mse = mean_squared_error(y_test, test_prediction)
            
            train_metrics = {
                "r2": r2_score(y_train, train_prediction),
                "mae": mean_absolute_error(y_test, test_prediction),
                "mse" : mse,
                "rmse": np.sqrt(mse)
            }
        
        # calculate test metrics 
        if problem_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, test_prediction),
                "precision": precision_score(y_test, test_prediction, average="weighted",     zero_division=0),
                "recall": recall_score(y_test, test_prediction, average="weighted",zero_division=0),
                "f1": f1_score(y_test, test_prediction, average="weighted", zero_division=0),
            }
            
        else:
            mse = mean_squared_error(y_test, test_prediction)
            
            metrics = {
                "r2_score": r2_score(y_test, test_prediction),
                "mae": mean_absolute_error(y_test, test_prediction),
                "mse" : mse,
                "rmse": np.sqrt(mse)
            }
        results.append({
            "model_name" : model_name,
            "model": current_model,
            "training_time" : round(training_time, 4),
            "metrics" : metrics,
            "train_metrics" : train_metrics,
            "train_prediction" : train_prediction,
            "test_prediction" : test_prediction
        })
        
    return {
        "problem_type" : problem_type,
        "results" : results
    }