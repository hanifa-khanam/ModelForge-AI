import pandas as pd

def load_data(uploaded_file):
    """
    Load a CSV file into a pandas DataFrame.
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        File uploaded through Streamlit.

    Returns
    -------
    pd.DataFrame
    """
    
    return pd.read_csv(uploaded_file)


def dataset_summary(df):
    
    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Numerical Columns": len(df.select_dtypes(include="number").columns),
        "Categorical Columns": len(df.select_dtypes(exclude="number").columns),
        "Missing Values" : int(df.isnull().sum().sum()),
        "Duplicate Rows" : int(df.duplicated().sum()),
        "Memory Usage (MB)" : round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    
    return summary
