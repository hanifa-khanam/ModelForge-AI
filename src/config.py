"""
Single source of truth for all project settings.
"""

import os
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