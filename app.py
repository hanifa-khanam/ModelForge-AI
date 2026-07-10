import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="ModelForge AI - AutoML Advisor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .feature-card {
        background-color: #043D21;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin-bottom: 1rem;    
        }
</style>
""", unsafe_allow_html=True)

def main():
    
    st.title("🏗️ ModelForge AI") 
    st.subheader("Intelligent Supervised ML Benchmarking Platform") 

    st.markdown("---") 

    col1, col2, col3 = st.columns([1, 2, 1]) 
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1E1E2E 0%, #043D21 100%); border-radius: 15px;">
            <h2 style="color: #4ECDC4;">🚀 Start Your ML Project</h2>
            <p style="font-size: 1.1rem; color:#B8B8B8; margin: 1rem 0;">
                Upload your dataset and let ModelForge AI handle the entire ML workflow
            </p>
            <div style="margin-top: 1.5rem;">
                <a href="/pages/1_Data_Analysis.py" target="_self">
                    <button style="background-color: #FF6B6B; color: white; border: none; padding: 0.75rem 3rem; border-radius: 8px; font-size: 1.1rem; font-weight: bold; cursor: pointer;">
                        📤 Upload Dataset
                    </button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("Key Features")
    features_cols = st.columns(4) 
    
    with features_cols[0]:
        st.markdown("""
            <div class="feature-card">
                <h4 style="color: #04213D;">🔍 Auto Analysis</h4>
                <p style="color:#B8B8B8;"> Automatic datset inspection, missing values, and data type detection </p>
            </div>
        """, unsafe_allow_html=True)
    
    with features_cols[1]:
        st.markdown("""
            <div class="feature-card">
                <h4 style="color: #04213D;">🏗️ Multi-Model Training</h4>
                <p style="color:#B8B8B8;"> Train 10+ models automatically with hyperparameter tuning </p>
            </div>
        """, unsafe_allow_html=True)
        
    with features_cols[2]:
        st.markdown("""
            <div class="feature-card">
                <h4 style="color: #04213D;">💡 Explainable AI</h4>
                <p style="color:#B8B8B8;"> SHAP explanations, feature importance, and model interpretations  </p>
            </div>
        """, unsafe_allow_html=True)
        
    with features_cols[3]:
        st.markdown("""
            <div class="feature-card">
                <h4 style="color: #04213D;">🎯 Smart Recommendations </h4>
                <p style="color:#B8B8B8;"> AI-powerd model selection with detailed reasoning </p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("Workflow") 
    
    workflow_steps = [
        ("📤 **Upload Dataset**", "Upload your CSV file with tabular data"),
        ("🔍 **Data Analysis**", "Review automatic data insights and visualizations"),
        ("🎯 **Select Target**", "Choose your target column for prediction"),
        ("⚙️ **Train Models**", "Click to train all models with hyperparameter tuning"),
        ("📊 **Compare Results**", "View performance leaderboard and visualizations"),
        ("🧠 **Explain Predictions**", "Understand why each model makes predictions"),
        ("🏆 **Get Recommendation**", "Receive the best model with detailed reasoning"),
        ("💾 **Download Model**", "Export the trained pipeline for production use")   
    ]
    
    for step in workflow_steps:
        st.markdown(f"- {step[0]} - {step[1]}")
    
    
    st.markdown("---")

    # Version badge style
    st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #4ECDC4;'><strong>Version 1.0.0</strong></p>", unsafe_allow_html=True)

    # Built using section
    st.markdown("<p style='text-align: center; font-size: 0.85rem; margin-bottom: 0;'>Built using</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #888888; font-size: 0.85rem;'>Python • Streamlit • Scikit-learn • Pandas • NumPy</p>", unsafe_allow_html=True)

    # Developer credits
    st.markdown("<p style='text-align: center; font-size: 0.85rem; margin-top: 1rem; margin-bottom: 0;'>Developed by</p>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: center; margin-top: 0;'>Hanifa Khanam</h4>", unsafe_allow_html=True)

    # Social Links centered using columns
    social_col1, social_col2, social_col3 = st.columns([2, 2, 2])
    with social_col2:
        st.markdown("""
            <p style='text-align: center; font-size: 0.9rem;'>
                <a href="https://github.com/hanifa-khanam" target="_blank" style="text-decoration: none; color: #FF6B6B;">🔗 GitHub</a> &nbsp;|&nbsp; 
                <a href="https://linkedin.com/in/hanifa-khanam" target="_blank" style="text-decoration: none; color: #3498DB;">🔗 LinkedIn</a>
            </p>
        """, unsafe_allow_html=True)

    # Copyright notice at the very bottom
    st.caption("<p style='text-align: center; margin-top: 2rem;'>© 2026 ModelForge AI</p>", unsafe_allow_html=True)
    
    
    
if __name__ == "__main__":
    main()
    
