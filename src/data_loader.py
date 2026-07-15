import pandas as pd

def load_data(uploaded_file, file_type: str="csv") -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "excel":
            df = pd.read_excel(uploaded_file) 
        else:
            raise ValueError(f"Unsupported file type: {file_type}") 
    except Exception as e:
        raise ValueError(f"Failed to load file: {str(e)}")

    if df.empty:
        raise ValueError("The uploaded file contains no data") 
    
    return df 

def dataset_summary(df):
    """
    Generate a comprehensive summary of the dataset.
    """
    rows, cols = df.shape
    
    summary = {
        "Rows": rows,
        "Columns": cols,
        "Numerical Columns": len(df.select_dtypes(include=["int64", "float64"]).columns),
        "Categorical Columns": len(df.select_dtypes(exclude=["object", "category", "bool"]).columns),
        "Missing Values" : int(df.isnull().sum().sum()),
        "Missing Percentage" : round((df.isnull().sum().sum() / (rows * cols)) * 100, 2),
        "Duplicate Rows" : int(df.duplicated().sum()),
        "Duplicate Percentage" : round((df.duplicated().sum() / rows) * 100, 2) if rows > 0 else 0,
        "Memory Usage (MB)" : round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    
    return summary
