"""
Page 4: Model Explainability
Understand feature importance and model decision-making.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from src.explainability import explain_from_pipeline

st.set_page_config(
    page_title="Explainability - ModelForge AI",
    page_icon="💡",
    layout="wide",
)
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None

if st.session_state.training_results is None:
    st.warning("⚠️ No training results found. Train models in **Model Training** first.")
    st.stop()

if st.session_state.preprocessed_data is None:
    st.warning("⚠️ No preprocessed data found. Train models in **Model Training** first.")
    st.stop()

training_results = st.session_state.training_results
preprocessed_data = st.session_state.preprocessed_data

st.title("💡 Model Explainability")
st.markdown("Understand which features drive your model's predictions.")

model_names = [r["model_name"] for r in training_results["results"]]
selected_model_name = st.selectbox("Select a model to explain", options=model_names)

selected_result = next(
    r for r in training_results["results"] if r["model_name"] == selected_model_name
)

pipeline = {
    "model": selected_result["model"],
    "preprocessor": preprocessed_data["preprocessor"],
    "feature_names": preprocessed_data["feature_names"],
}

st.header(f"📊 Feature Importance: {selected_model_name}")

with st.spinner("Calculating feature importance..."):
    explanation = explain_from_pipeline(pipeline, aggregate=True, top_n=20)

if explanation["method"] == "not_available":
    st.warning(explanation["message"])
    st.info(
        "Models with built-in importance: Random Forest, Decision Tree, "
        "Gradient Boosting, Logistic Regression, Linear Regression, Ridge."
    )
    st.stop()

st.caption(f"Method: {explanation['method']} | Aggregated: {explanation['aggregated']}")

importance_data = explanation["feature_importance"]
chart_df = pd.DataFrame(importance_data).iloc[::-1]

fig = px.bar(
    chart_df,
    x="importance", y="feature",
    orientation="h",
    title=f"Feature Importance — {selected_model_name}",
    labels={"importance": "Importance Score", "feature": "Feature"},
    color="importance",
    color_continuous_scale="Greens",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Importance Breakdown")
detail_rows = []
for item in explanation["feature_importance"]:
    detail_rows.append({
        "Feature": item["feature"],
        "Importance": item["importance"],
        "Encoded Sub-Features": ", ".join(item.get("sub_features", [])) if item.get("sub_features") else "—",
    })
st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

st.header("ℹ️ About This Model")
col1, col2 = st.columns(2)
with col1:
    st.metric("Model Type", explanation["model_type"])
    st.metric("Total Features (encoded)", explanation["num_features"])
with col2:
    st.metric("Top Feature", explanation["top_feature"])
    st.metric("Method", explanation["method"].replace("_", " ").title())

st.info(
    "**Interpretation Tip:** Feature importance shows which columns the model "
    "relies on most. A higher score means the feature has more influence on predictions."
)

# Compare across models
st.header("🔄 Compare Feature Importance Across Models")
compare_selected = st.multiselect(
    "Select models to compare",
    options=model_names,
    default=model_names[:min(3, len(model_names))],
)

if len(compare_selected) >= 2:
    compare_rows = []
    for model_name in compare_selected:
        result = next(r for r in training_results["results"] if r["model_name"] == model_name)
        pipe = {
            "model": result["model"],
            "preprocessor": preprocessed_data["preprocessor"],
            "feature_names": preprocessed_data["feature_names"],
        }
        exp = explain_from_pipeline(pipe, aggregate=True, top_n=10)
        for item in exp["feature_importance"]:
            compare_rows.append({
                "Model": model_name,
                "Feature": item["feature"],
                "Importance": item["importance"],
            })
    
    compare_df = pd.DataFrame(compare_rows)
    fig = px.bar(
        compare_df, x="Importance", y="Feature", color="Model",
        orientation="h", barmode="group",
        title="Feature Importance Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)