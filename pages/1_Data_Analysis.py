"""
Page 1: Data Analysis
Upload dataset, explore data quality, select target column.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data, dataset_summary
from src.validator import validate_dataset_for_ml
from src.eda import run_eda
from src.preprocessing import detect_problem_type

st.set_page_config(
    page_title="Data Analysis - ModelForge AI",
    page_icon="🔍",
    layout="wide",
)

# Initialize session state
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
if "eda_results" not in st.session_state:
    st.session_state.eda_results = None

st.title("🔍 Data Analysis")
st.markdown("Upload your dataset and explore it before training models.")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV format)",
    type=["csv"],
    help="Upload a CSV file with your training data."
)

if uploaded_file is not None:
    try:
        if st.session_state.dataframe is None:
            # Clear all downstream state
            st.session_state.target_column = None
            st.session_state.problem_type = None
            st.session_state.validation_result = None
            st.session_state.eda_results = None
            st.session_state.preprocessed_data = None
            st.session_state.training_results = None
            
            with st.spinner("Loading dataset..."):
                df = load_data(uploaded_file)
                st.session_state.dataframe = df
                st.success("Dataset loaded successfully!")
        
        df = st.session_state.dataframe
        
        # ── Dataset Overview ──
        st.header("📊 Dataset Overview")
        summary = dataset_summary(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{summary['rows']:,}")
        col2.metric("Columns", summary["columns"])
        col3.metric("Missing Values", summary["total_missing_values"])
        col4.metric("Duplicates", summary["duplicate_rows"])
        
        with st.expander("📋 Data Types & Memory"):
            st.write(f"Memory Usage: {summary['memory_usage_mb']} MB")
            st.write(f"Numeric Columns: {summary['numeric_columns']}")
            st.write(f"Categorical Columns: {summary['categorical_columns']}")
        
        if summary["columns_with_missing"] > 0:
            with st.expander(f"⚠️ Missing Values ({summary['columns_with_missing']} columns)"):
                st.write(f"Total missing: {summary['total_missing_values']} ({summary['missing_percentage']}%)")
        
        # ── Target Selection ──
        st.header("🎯 Select Target Column")
        target_column = st.selectbox(
            "Which column do you want to predict?",
            options=[""] + list(df.columns),
            index=0,
        )
        
        if target_column:
            validation = validate_dataset_for_ml(df, target_column)
            st.session_state.target_column = target_column
            st.session_state.validation_result = validation
            
            if not validation["can_proceed"]:
                st.error("⚠️ Cannot proceed:")
                for blocker in validation["blockers"]:
                    st.error(f"• {blocker}")
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    st.warning(f"• {warning}")
            
            if validation["can_proceed"]:
                problem_type = detect_problem_type(df[target_column])
                st.session_state.problem_type = problem_type
                st.info(f"Detected problem type: **{problem_type.title()}**")
                
                eda_results = run_eda(df, target_column, problem_type)
                st.session_state.eda_results = eda_results
                
                # ── Target Analysis ──
                st.header("📈 Target Analysis")
                target_analysis = eda_results["target_analysis"]
                
                if problem_type == "classification":
                    dist = target_analysis["distribution"]
                    fig = px.bar(
                        x=list(dist.keys()), y=list(dist.values()),
                        labels={"x": "Class", "y": "Count"},
                        title="Class Distribution",
                        color=list(dist.keys()),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Classes", target_analysis["num_classes"])
                    col2.metric("Dominant %", f"{target_analysis['dominant_class_pct']}%")
                    
                    if target_analysis.get("imbalance_warning"):
                        st.warning(target_analysis["imbalance_warning"])
                
                else:
                    fig = px.histogram(df, x=target_column, title=f"Distribution of {target_column}", nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    stats = target_analysis
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", stats["mean"])
                    col2.metric("Std", stats["std"])
                    col3.metric("Min", stats["min"])
                    col4.metric("Max", stats["max"])
                    
                    if stats["outliers"] > 0:
                        st.warning(f"{stats['outliers']} potential outliers ({stats['outlier_percentage']}%)")
                
                # ── Feature Exploration ──
                st.header("🔬 Feature Exploration")
                feature_analysis = eda_results["feature_analysis"]
                
                if feature_analysis.get("correlations"):
                    with st.expander(f"🔗 Correlated Features ({feature_analysis['highly_correlated_pairs']} pairs)"):
                        for corr in feature_analysis["correlations"]:
                            st.write(f"**{corr['feature_1']}** ↔ **{corr['feature_2']}**: {corr['correlation']} ({corr['magnitude']})")
                
                if feature_analysis.get("target_correlations"):
                    corr_df = pd.DataFrame(feature_analysis["target_correlations"])
                    fig = px.bar(corr_df, x="correlation", y="feature", title=f"Correlations with {target_column}", orientation="h")
                    st.plotly_chart(fig, use_container_width=True)
                
                if feature_analysis.get("high_cardinality_cols"):
                    with st.expander(f"⚠️ High Cardinality ({len(feature_analysis['high_cardinality_cols'])})"):
                        for col_info in feature_analysis["high_cardinality_cols"]:
                            st.warning(col_info["warning"])
                
                if feature_analysis.get("low_variance_cols"):
                    with st.expander(f"📉 Low Variance Features"):
                        st.write(", ".join(feature_analysis["low_variance_cols"]))
                
                if eda_results.get("warnings"):
                    st.header("⚠️ Warnings")
                    for warning in eda_results["warnings"]:
                        st.warning(warning)
                
                st.success("✅ Dataset ready! Proceed to **Model Training**.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.dataframe = None

else:
    st.info("👆 Upload a CSV file to begin.")
    st.markdown("""
    ### Workflow:
    1. **Upload** CSV dataset
    2. **Explore** data quality and distributions
    3. **Select** target column
    4. **Validate** readiness for ML
    5. **Proceed** to Model Training
    """)