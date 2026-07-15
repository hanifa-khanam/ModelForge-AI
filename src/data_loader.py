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
    
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    missing_total = int(df.isnull().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    
    summary = {
        "rows": rows,
        "columns": cols,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "total_missing_values": missing_total,
        "columns_with_missing": int((df.isnull().sum() > 0).sum()),
        "missing_percentage": round((missing_total / (rows * cols)) * 100, 2) if rows * cols > 0 else 0,
        "duplicate_rows": duplicate_total,
        "duplicate_percentage": round((duplicate_total / rows) * 100, 2) if rows > 0 else 0,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    
    return summary
