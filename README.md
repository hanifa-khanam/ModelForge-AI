# 🏗️ ModelForge AI

## Automatic Supervised ML Benchmarking Platform

Upload any tabular dataset and automatically build, compare, evaluate, explain, and recommend the best Supervised Machine Learning model.

## 🎯 Features

- **Automatic Data Analysis** - Instant dataset insights
- **Intelligent Preprocessing** - Handles missing values, encoding, scaling
- **Auto Problem Detection** - Detects regression vs classification
- **Multiple Models** - Trains 10+ algorithms automatically
- **Hyperparameter Tuning** - Optimizes each model
- **Comprehensive Comparison** - Visual leaderboard with metrics
- **Explainability** - SHAP, feature importance, and more
- **Smart Recommendations** - AI-powered model selection
- **Model Export** - Download trained pipeline as .pkl
- **Batch Prediction** - Predict on new data instantly

## 🏗️ Architecture

```text
               User Uploads CSV
                      │
                      ▼
               Data Inspection
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
            Performance Comparison
                      ▼
             Best Model Selection
                      ▼
             Explain Predictions
                      ▼
                Download Model
```


## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: SHAP
- **Serialization**: joblib

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ModelForge-AI.git
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
        ModelForge/
        ├── app.py                 # Main application
        ├── src/                   # Core functionality
        │   ├── data_loader.py    # Data loading & inspection
        │   ├── preprocessing.py  # Data preprocessing
        │   ├── trainer.py        # Model training
        │   ├── evaluator.py      # Model evaluation
        │   ├── recommender.py    # Model recommendation
        │   ├── explainability.py # SHAP & explanations
        │   └── predictor.py      # Predictions on new data
        ├── pages/                # Streamlit pages
        │   ├── 1_Data_Analysis.py
        │   ├── 2_Model_Training.py
        │   ├── 3_Model_Comparison.py
        │   ├── 4_Explainability.py
        │   └── 5_Predict.py
        └── models/               # Saved models
```