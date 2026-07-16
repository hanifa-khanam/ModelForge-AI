"""
Page 5: Predict
Load a saved model and make predictions on new data.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os

from src.predictor import load_pipeline, predict, predict_single

st.set_page_config(
    page_title="Predict - ModelForge AI",
    page_icon="🔮",
    layout="wide",
)

from src.session_state import init_session_state
init_session_state()


st.title("🔮 Predict with Saved Model")
st.markdown("Upload a trained model bundle and make predictions on new data.")

# Model upload
st.header("📦 Load Model")
model_file = st.file_uploader(
    "Upload a saved model bundle (.joblib)",
    type=["joblib"],
    help="Upload the model file downloaded from the Model Comparison page."
)

if model_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            tmp.write(model_file.read())
            tmp_path = tmp.name
        
        pipeline = load_pipeline(tmp_path)
        os.unlink(tmp_path)
        
        st.session_state.loaded_pipeline = pipeline
        st.session_state.pipeline_metadata = {
            "problem_type": pipeline["problem_type"],
            "feature_names": pipeline["feature_names"],
            "num_features": len(pipeline["feature_names"]),
        }
        
        st.success("✅ Model loaded successfully!")
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.session_state.loaded_pipeline = None

if st.session_state.loaded_pipeline is not None:
    pipeline = st.session_state.loaded_pipeline
    metadata = st.session_state.pipeline_metadata
    
    st.header("📋 Model Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Problem Type", metadata["problem_type"].title())
    col2.metric("Features Required", metadata["num_features"])
    col3.metric("Model Type", type(pipeline["model"]).__name__)
    
    with st.expander("🔍 View Required Features"):
        for i, feature in enumerate(metadata["feature_names"], 1):
            st.write(f"{i}. {feature}")
    
    st.header("🔮 Make Predictions")
    prediction_mode = st.radio(
        "Choose prediction mode",
        options=["Single Prediction (Manual Input)", "Batch Prediction (CSV Upload)"],
        horizontal=True,
    )
    
    if prediction_mode == "Single Prediction (Manual Input)":
        st.subheader("Enter Feature Values")
        
        with st.form("prediction_form"):
            input_values = {}
            cols = st.columns(2)
            
            for i, feature in enumerate(metadata["feature_names"]):
                with cols[i % 2]:
                    if any(kw in feature.lower() for kw in ["id", "code", "category", "type", "class"]):
                        input_values[feature] = st.text_input(feature)
                    else:
                        input_values[feature] = st.number_input(feature, value=0.0)
            
            submitted = st.form_submit_button("🔮 Predict", type="primary")
        
        if submitted:
            try:
                result = predict_single(pipeline, input_values)
                st.subheader("Prediction Result")
                prediction = result["predictions"][0]
                
                if metadata["problem_type"] == "classification":
                    st.markdown(f"### Predicted Class: **{prediction}**")
                    
                    if result.get("prediction_proba") and result["prediction_proba"]:
                        proba = result["prediction_proba"][0]
                        proba_df = pd.DataFrame({
                            "Class": list(proba.keys()),
                            "Probability": list(proba.values()),
                        })
                        fig = px.bar(proba_df, x="Class", y="Probability",
                                     title="Prediction Probabilities", color="Class")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        confidence = proba.get(str(prediction), 0) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                else:
                    st.markdown(f"### Predicted Value: **{prediction:.4f}**")
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    else:
        st.subheader("Upload CSV for Batch Prediction")
        st.info("Upload a CSV file with the same feature columns as the training data.")
        
        batch_file = st.file_uploader("Upload prediction data (CSV)", type=["csv"], key="batch_csv")
        
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.write(f"Loaded {len(batch_df)} samples for prediction.")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🔮 Run Batch Prediction", type="primary"):
                    with st.spinner("Making predictions..."):
                        result = predict(pipeline, batch_df)
                    
                    st.subheader("Prediction Results")
                    results_df = batch_df.copy()
                    results_df["Prediction"] = result["predictions"]
                    
                    if metadata["problem_type"] == "classification" and result.get("prediction_proba"):
                        confidences = []
                        for i, proba in enumerate(result["prediction_proba"]):
                            pred = result["predictions"][i]
                            confidences.append(proba.get(str(pred), 0))
                        results_df["Confidence"] = confidences
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                    
                    if metadata["problem_type"] == "classification":
                        st.subheader("Prediction Distribution")
                        dist = results_df["Prediction"].value_counts()
                        fig = px.pie(values=dist.values, names=dist.index,
                                     title="Distribution of Predictions")
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Batch prediction failed: {str(e)}")

else:
    st.info("👆 Upload a saved model bundle (.joblib) to begin.")
    st.markdown("""
    ### How to get a model bundle:
    1. **Model Training** → train models on your dataset
    2. **Model Comparison** → download the best model
    3. Return here and upload the `.joblib` file
    
    ### Prediction modes:
    - **Single Prediction:** Enter feature values manually
    - **Batch Prediction:** Upload a CSV with multiple samples
    """)