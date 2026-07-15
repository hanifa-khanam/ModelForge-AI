# 🏗️ ModelForge AI

## Automatic Supervised ML Benchmarking Platform

Upload any tabular dataset and automatically build, compare, evaluate, explain, and recommend the best Supervised Machine Learning model.

## 🌐 Live Demo

[https://modelforge-ai-v1.streamlit.app](https://modelforge-ai-v1.streamlit.app)

## 🎯 Features

- **Automatic Data Analysis** - Instant dataset insights
- **Intelligent Preprocessing** - Handles missing values, encoding, scaling
- **Auto Problem Detection** - Detects regression vs classification
- **10+ Models** - Trains 10+ algorithms automatically
- **Optional Hyperparameter Tuning** - GridSearchCV with cross-validation for optimal parameters
- **Comprehensive Comparison** - Ranked leaderboard with multiple metrics
- **Smart Recommendations** - weighted scoring (performance + speed + overfitting detection)
- **Feature Importance** — Built-in model explainability with aggregation for encoded features
- **Explainability** - SHAP, feature importance, and more
- **Model Export** -  Download trained pipeline as `.joblib` bundle
- **Batch & Single Prediction** — Predict on new data via CSV upload or manual input

## 🏗️ Architecture

```text
               User Uploads CSV
                      │
                      ▼
               Data Validation
                      │
                      ▼
                     EDA
                      │
                      ▼
                Target Selection
                      │
                      ▼
              Data Preprocessing
                      │
                      ▼
         Regression or Classification?
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
  Regression Models       Classification Models
         │                         │
         └────────────┬────────────┘
                      ▼
            Performance Evaluation
                      ▼
             Model Recommendation
                      ▼
              Feature Explainability
                      ▼
                Download Model
```


## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: Built-in feature importance (tree-based + linear models)
- **Serialization**: joblib

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/hanifa-khanam/ModelForge-AI.git
cd ModelForge-AI
```

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Project Structure

```text 
       ModelForge-AI/
       ├── app.py                      # Landing page
       ├── requirements.txt            # Dependencies
       ├── README.md                   # This file
       │
       ├── src/                        # Core backend modules
       │   ├── config.py               # Central configuration & model definitions
       │   ├── data_loader.py          # CSV loading & dataset summary
       │   ├── validator.py            # Input validation & quality checks
       │   ├── eda.py                  # Exploratory data analysis
       │   ├── preprocessing.py        # Feature engineering pipeline
       │   ├── trainer.py              # Multi-model training engine
       │   ├── tuner.py                # Hyperparameter optimization (GridSearchCV)
       │   ├── evaluator.py            # Model comparison & leaderboard
       │   ├── recommender.py          # Best model selection with reasoning
       │   ├── explainability.py       # Feature importance extraction
       │   ├── predictor.py            # Production prediction engine
       │   ├── model_io.py             # Model persistence (save/load bundles)
       │   └── utils.py                # Shared utilities
       │
       ├── pages/                      # Streamlit multipage app
       │   ├── 1_Data_Analysis.py      # Upload, EDA, target selection
       │   ├── 2_Model_Training.py     # Preprocessing, training, tuning
       │   ├── 3_Model_Comparison.py   # Leaderboard, recommendation, download
       │   ├── 4_Explainability.py     # Feature importance visualization
       │   └── 5_Predict.py            # Single & batch prediction
       │
       ├── data/                       # Uploaded datasets (gitignored)
       ├── saved_models/               # Trained model bundles (gitignored)
       ├── notebooks/                  # Jupyter notebooks for experimentation
       └── assets/                     # Static assets (logo, images)
```