"""
Page 2: Model Training
Preprocess data and train multiple supervised learning models.
Supports optional hyperparameter tuning via GridSearchCV.
"""
import streamlit as st
import pandas as pd
import time
import plotly.express as px

from src.preprocessing import run_preprocessing
from src.trainer import train_models
from src.tuner import tune_models, evaluate_tuned_models, get_tuning_summary

st.set_page_config(
    page_title="Model Training - ModelForge AI",
    page_icon="🏗️",
    layout="wide",
)

# ── Initialize Session State ──
from src.session_state import init_session_state
init_session_state()

# ── Guards ──
if st.session_state.dataframe is None:
    st.warning("⚠️ No dataset found. Upload a dataset in **Data Analysis** first.")
    st.stop()

if st.session_state.target_column is None:
    st.warning("⚠️ No target column selected. Select a target in **Data Analysis** first.")
    st.stop()

df = st.session_state.dataframe
target_column = st.session_state.target_column
problem_type = st.session_state.problem_type
validation = st.session_state.validation_result

st.title("🏗️ Model Training")

# ── Current Configuration ──
col1, col2, col3 = st.columns(3)
col1.metric("Dataset", f"{df.shape[0]:,} rows × {df.shape[1]} columns")
col2.metric("Target", target_column)
col3.metric("Problem Type", problem_type.title())

if validation and validation.get("warnings"):
    with st.expander("⚠️ Validation Warnings"):
        for warning in validation["warnings"]:
            st.warning(warning)

# ── Training Configuration ──
st.header("⚙️ Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider(
        "Test Set Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Proportion of data reserved for testing."
    )

with col2:
    random_state = st.number_input(
        "Random Seed",
        value=42,
        min_value=0,
        max_value=999,
        help="Fixed seed for reproducible results."
    )

with col3:
    enable_tuning = st.checkbox(
        "Hyperparameter Tuning",
        value=False,
        help="Use GridSearchCV to find optimal parameters. Significantly slower but can improve performance."
    )

# ── Tuning Warning ──
if enable_tuning:
    st.warning(
        "⚠️ **Hyperparameter tuning enabled.** This will search hundreds of "
        "parameter combinations using 5-fold cross-validation. Training may "
        "take several minutes depending on dataset size and number of models. "
        "Disable for quick experimentation."
    )

# ── Train Button (COMPUTATION ONLY) ──
if st.button(
    "🚀 Train Models",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.is_training,
):
    st.session_state.is_training = True
    
    try:
        # ── Preprocessing ──
        with st.status("Preprocessing data...", expanded=True) as status:
            st.write("Separating features and target...")
            st.write("Detecting feature types...")
            st.write("Splitting train/test...")
            st.write("Building preprocessing pipeline...")
            
            preprocessed_data = run_preprocessing(
                df=df,
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
            )
            
            status.update(
                label="✅ Preprocessing complete!",
                state="complete",
                expanded=False,
            )
        
        # ── Training or Tuning ──
        if enable_tuning:
            with st.spinner(
                "Tuning hyperparameters... This may take several minutes."
            ):
                start_time = time.time()
                
                # Run hyperparameter tuning
                tuned_results = tune_models(preprocessed_data)
                
                # Evaluate tuned models on test set
                training_results = evaluate_tuned_models(
                    tuned_results, preprocessed_data
                )
                
                # Generate tuning summary
                tuning_summary = get_tuning_summary(tuned_results)
                
                total_time = time.time() - start_time
            
            st.session_state.training_mode = "tuned"
            st.session_state.tuning_metadata = {
                "tuned_results": tuned_results,
                "tuning_summary": tuning_summary,
            }
        
        else:
            with st.spinner("Training models... This may take a moment."):
                start_time = time.time()
                training_results = train_models(preprocessed_data)
                total_time = time.time() - start_time
            
            st.session_state.training_mode = "default"
        
        # Save to session state
        st.session_state.preprocessed_data = preprocessed_data
        st.session_state.training_results = training_results
        st.session_state.total_training_time = total_time
    
    finally:
        st.session_state.is_training = False
        st.rerun()

# ── Display Results (OUTSIDE the button block) ──
if st.session_state.training_results is not None:
    training_results = st.session_state.training_results
    preprocessed_data = st.session_state.preprocessed_data
    total_time = st.session_state.total_training_time
    training_mode = st.session_state.training_mode
    
    # ── Preprocessing Summary ──
    st.success(
        f"Training set: {preprocessed_data['X_train'].shape[0]:,} samples | "
        f"Test set: {preprocessed_data['X_test'].shape[0]:,} samples | "
        f"Features: {preprocessed_data['X_train'].shape[1]}"
    )
    
    # ── Mode Badge ──
    if training_mode == "tuned":
        st.success(
            f"✅ **Hyperparameter Tuning Complete!** "
            f"{len(training_results['results'])} models tuned and evaluated in {total_time:.1f}s"
        )
        
        # Tuning summary
        tuning_meta = st.session_state.tuning_metadata
        if tuning_meta:
            summary = tuning_meta["tuning_summary"]
            
            with st.expander("🔧 Tuning Summary", expanded=False):
                st.write(summary["summary"])
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Models Tuned", summary["models_tuned"])
                col_b.metric("Used Defaults", summary["models_skipped"])
                col_c.metric("Best CV Score", summary["best_cv_score"])
    
    else:
        st.success(
            f"✅ **Training Complete!** "
            f"{len(training_results['results'])} models trained in {total_time:.1f}s"
        )
    
    # ── Results Table ──
    st.header("📊 Training Results")
    
    summary_rows = []
    for result in training_results["results"]:
        row = {
            "Model": result["model_name"],
            "Training Time": f"{result['training_time']:.2f}s",
        }
        for metric_name, metric_value in result["metrics"].items():
            row[metric_name.upper()] = f"{metric_value:.4f}"
        summary_rows.append(row)
    
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True,
    )
    
    # ── Best Parameters (tuned mode only) ──
    if training_mode == "tuned":
        st.header("🎯 Best Hyperparameters")
        
        # Filter to only tuned models
        tuned_models = [
            r for r in training_results["results"]
            if r.get("best_params")
        ]
        
        if tuned_models:
            for model_result in tuned_models:
                with st.expander(
                    f"{model_result['model_name']} — "
                    f"CV Score: {model_result.get('best_cv_score', 'N/A')}"
                ):
                    if model_result["best_params"]:
                        st.write("**Best Parameters:**")
                        for param, value in model_result["best_params"].items():
                            st.write(f"• `{param}`: {value}")
                    else:
                        st.write("Using default parameters.")
        else:
            st.info("No models were tuned — all using default parameters.")
    
    # ── Performance Comparison Charts ──
    st.header("📈 Performance Comparison")
    
    if problem_type == "classification":
        chart_data = pd.DataFrame([
            {
                "Model": r["model_name"],
                "F1 Score": r["metrics"]["f1"],
                "Accuracy": r["metrics"]["accuracy"],
            }
            for r in training_results["results"]
        ])
        
        fig = px.bar(
            chart_data,
            x="Model",
            y=["F1 Score", "Accuracy"],
            barmode="group",
            title="Model Performance Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        chart_data = pd.DataFrame([
            {
                "Model": r["model_name"],
                "R²": r["metrics"]["r2"],
                "RMSE": r["metrics"]["rmse"],
            }
            for r in training_results["results"]
        ])
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig1 = px.bar(
                chart_data,
                x="Model",
                y="R²",
                title="R² Score by Model (higher is better)",
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_b:
            fig2 = px.bar(
                chart_data,
                x="Model",
                y="RMSE",
                title="RMSE by Model (lower is better)",
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    st.success("✅ Results saved! Proceed to **Model Comparison**.")

else:
    # No results yet — show placeholder
    st.info(
        "👆 Configure training settings above and click **Train Models** to begin. "
        "Enable hyperparameter tuning for automatic parameter optimization."
    )
    
    with st.expander("ℹ️ What models will be trained?"):
        from src.config import CLASSIFICATION_MODELS, REGRESSION_MODELS
        
        if problem_type == "classification":
            st.write("**Classification Models:**")
            for name in CLASSIFICATION_MODELS.keys():
                st.write(f"• {name}")
        else:
            st.write("**Regression Models:**")
            for name in REGRESSION_MODELS.keys():
                st.write(f"• {name}")