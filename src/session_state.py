import streamlit as st

DEFAULTS = {
    "dataframe": None,
    "target_column": None,
    "problem_type": None,
    "validation_result": None,
    "eda_results": None,
    "preprocessed_data": None,
    "training_results": None,
    "evaluation_results": None,
    "recommendation": None,
    "training_mode": None,
    "tuning_metadata": None,
    "total_training_time": None,
    "is_training": False,
    "loaded_pipeline": None,
    "pipeline_metadata": None,
}

def init_session_state():
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)