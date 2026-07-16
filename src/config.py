"""
Single source of truth for all project settings.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XGBOOST_NUM_THREADS"] = "1"

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR 
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor 


#  Paths 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")

PREPROCESSING_PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")


# Preprocessing Defaults
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_SCALING_METHOD = "standard"
DEFAULT_IMPUTATION_NUMERIC = "median"
DEFAULT_IMPUTATION_CATEGORICAL = "most_frequent"
HIGH_CARDINALITY_THRESHOLD = 50
HIGH_MISSING_THRESHOLD = 0.5
MEDIUM_MISSING_THRESHOLD = 0.1


# Model Definitions
CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes" : GaussianNB(),
    "SVM" : SVC(random_state=DEFAULT_RANDOM_STATE, probability=True, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_STATE),
    "AdaBoost" : AdaBoostClassifier(random_state=DEFAULT_RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_STATE),
    "XGBoost" : XGBClassifier(random_state=DEFAULT_RANDOM_STATE, n_jobs=-1, n_estimators=100, verbosity=0, use_label_encoder=False)
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=DEFAULT_RANDOM_STATE,alpha=1.0, max_iter=1000),
    "Lasso Regression" : Lasso(random_state=DEFAULT_RANDOM_STATE,alpha= 0.1, max_iter=1000),
    "ElasticNet" : ElasticNet(random_state=DEFAULT_RANDOM_STATE, max_iter=1000),
    "Decision Tree": DecisionTreeRegressor(random_state=DEFAULT_RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=DEFAULT_RANDOM_STATE),
    "AdaBoost" : AdaBoostRegressor(random_state=DEFAULT_RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100,random_state=DEFAULT_RANDOM_STATE),
    "XGBoost" : XGBRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=-1, n_estimators=50, verbosity=0),
    "SVR" : SVR(max_iter=1000)
}


#  Metrics 
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1"]
REGRESSION_METRICS = ["r2", "mae", "mse", "rmse"]

# Evaluation Defaults
PRIMARY_METRIC = {
    "classification": "f1",
    "regression": "r2",
}

SECONDARY_METRIC = {
    "classification": "accuracy",
    "regression": "rmse",
}

# Recommendation Defaults 
# Weights for composite scoring
RECOMMENDATION_WEIGHTS = {
    "accuracy_focused": {"performance": 0.7, "speed": 0.3},
    "balanced": {"performance": 0.5, "speed": 0.5},
}

# Overfitting threshold: train_score - test_score above this triggers penalty
OVERFITTING_THRESHOLD = 0.05  # 5% gap
OVERFITTING_PENALTY = -15

# Hyperparameter Tuning 
PARAM_GRID_CLASSIFICATION = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
        "penalty": ["l2"],  # l2 works with both solvers
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}

PARAM_GRID_REGRESSION = {
    "Linear Regression": {
        # No hyperparameters to tune
    },
    "Ridge Regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky"],
    },
    "Lasso Regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000, 5000],
    },
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_iter": [1000, 5000],
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
        "loss": ["linear", "square", "exponential"],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5],
        "loss": ["squared_error", "absolute_error"],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.5],
    },
}

# Cross-Validation Settings
CV_FOLDS = 5
TUNING_SCORING_CLASSIFICATION = "f1_weighted"
TUNING_SCORING_REGRESSION = "r2"


#  UI Settings
APP_TITLE = "ModelForge AI"
APP_SUBTITLE = "Intelligent Supervised ML Benchmarking Platform"
APP_ICON = "🏗️"
APP_VERSION = "1.0.0"

PAGES = {
    "data_analysis": "1_Data_Analysis.py",
    "model_training": "2_Model_Training.py",
    "model_comparison": "3_Model_Comparison.py",
    "explainability": "4_Explainability.py",
    "predict": "5_Predict.py",
}