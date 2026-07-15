"""
Page 3: Model Comparison
Compare trained models, view leaderboard, and download models.
"""
import streamlit as st
import pandas as pd
import os

from src.evaluator import evaluate_models, compare_models
from src.recommender import recommend_model
from src.model_io import save_bundle

st.set_page_config(
    page_title="Model Comparison - ModelForge AI",
    page_icon="📊",
    layout="wide",
)

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

if st.session_state.training_results is None:
    st.warning("⚠️ No training results found. Train models in **Model Training** first.")
    st.stop()

training_results = st.session_state.training_results

if st.session_state.evaluation_results is None:
    with st.spinner("Evaluating models..."):
        st.session_state.evaluation_results = evaluate_models(training_results)
        st.session_state.recommendation = recommend_model(
            st.session_state.evaluation_results,
            training_results,
            prefer_fast=False,
        )

evaluation = st.session_state.evaluation_results
recommendation = st.session_state.recommendation

st.title("📊 Model Comparison")

# Recommendation
st.header("🏆 Recommended Model")
rec_col1, rec_col2 = st.columns([2, 1])

with rec_col1:
    st.markdown(f"### {recommendation['recommended_model']}")
    st.markdown(f"**Composite Score:** {recommendation['composite_score']}")
    for reason in recommendation["reasoning"]:
        st.write(f"• {reason}")

with rec_col2:
    st.metric("Composite Score", recommendation["composite_score"])
    st.caption(f"Mode: {recommendation['weight_mode']}")

# Leaderboard
st.header("📋 Model Leaderboard")
leaderboard = evaluation["leaderboard"]

lb_rows = []
for entry in leaderboard:
    row = {"Rank": entry["rank"], "Model": entry["model_name"]}
    for metric_name, metric_value in entry["metrics"].items():
        row[metric_name.upper()] = f"{metric_value:.4f}"
    row["Training Time"] = f"{entry['training_time']:.2f}s" if entry.get("training_time") else "N/A"
    lb_rows.append(row)

lb_df = pd.DataFrame(lb_rows)

def highlight_best(row):
    if row["Model"] == recommendation["recommended_model"]:
        return ["background-color: #825656"] * len(row)
    return [""] * len(row)

st.dataframe(lb_df.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

# Performance gap
st.header("📈 Performance Analysis")
comparison = compare_models(training_results)

col1, col2 = st.columns(2)
with col1:
    st.metric("Best Model", comparison["best_model_name"])
    st.metric("Worst Model", comparison["worst_model_name"])
with col2:
    st.metric("Performance Gap", f"{comparison['best_worst_gap']:.4f}")
    if comparison["is_dominant"]:
        st.success("Best model dominates across all metrics")
    else:
        st.info("No single model wins on every metric")

st.write(comparison["summary"])

# Detailed scores
st.header("🔍 Detailed Scoring Breakdown")
if "all_scores" in recommendation:
    score_rows = []
    for score in recommendation["all_scores"]:
        score_rows.append({
            "Model": score["model_name"],
            "Composite": score["composite"],
            "Performance": score["performance_score"],
            "Speed": score["speed_score"],
            "Overfitting": "⚠️" if score.get("is_overfitting") else "✅",
        })
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

# Download
st.header("💾 Download Model")
st.markdown("Select a model to download for production use.")

model_names = [r["model_name"] for r in training_results["results"]]
default_index = model_names.index(recommendation["recommended_model"])

selected_model_name = st.selectbox(
    "Select model to download",
    options=model_names,
    index=default_index,
)

if st.button("📥 Download Selected Model", type="primary"):
    selected_result = next(
        r for r in training_results["results"] if r["model_name"] == selected_model_name
    )
    
    preprocessed = st.session_state.preprocessed_data
    save_path = f"saved_models/{selected_model_name.replace(' ', '_').lower()}_bundle.joblib"
    os.makedirs("saved_models", exist_ok=True)
    
    save_bundle(
        preprocessor=preprocessed["preprocessor"],
        model=selected_result["model"],
        target_encoder=preprocessed.get("target_encoder"),
        problem_type=training_results["problem_type"],
        feature_names=preprocessed["feature_names"],
        save_path=save_path,
    )
    
    with open(save_path, "rb") as f:
        st.download_button(
            label="💾 Click to Download",
            data=f,
            file_name=f"{selected_model_name.replace(' ', '_').lower()}_model.joblib",
            mime="application/octet-stream",
        )
    
    st.success(f"✅ Model bundle saved as `{save_path}`")