import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any

# Default maximum dataset size to prevent OOM errors
MAX_DATASET_SIZE = 100000


def sample_large_dataset(file_path: str, max_size: int = MAX_DATASET_SIZE, random_state: int = 42) -> pd.DataFrame:
    """Smart sampling - get dataset size first, then sample appropriately with exact row count"""
    
    # Get total row count by reading just the first column
    total_rows = len(pd.read_csv(file_path, sep='\t', usecols=[0]))
    print(f"Dataset has {total_rows:,} rows")
    
    if total_rows <= max_size:
        # Small dataset - read everything
        df = pd.read_csv(file_path, sep='\t')
        print(f"Loaded all {len(df):,} rows")
        return df
    
    # Large dataset - use skiprows to get exactly max_size random rows
    print(f"Sampling exactly {max_size:,} rows from {total_rows:,} total rows")
    
    # Generate random indices to keep (excluding header row 0)
    np.random.seed(random_state)
    rows_to_keep = sorted(np.random.choice(range(1, total_rows + 1), size=max_size, replace=False))
    
    # Create skiprows list (all rows except the ones we want to keep, plus header)
    all_rows = set(range(1, total_rows + 1))  # All data rows (excluding header)
    rows_to_skip = sorted(all_rows - set(rows_to_keep))
    
    # Read with skiprows
    df = pd.read_csv(file_path, sep='\t', skiprows=rows_to_skip)
    
    print(f"Sampled exactly {len(df):,} rows from {total_rows:,} total rows")
    return df


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Clean the dataframe by removing bad columns and handling missing values.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column to preserve
        
    Returns:
        Cleaned dataframe
    """
    missing_ratio_threshold = 0.95
    df_clean = df.copy()
    
    # Identify "bad" columns to remove
    cols_to_remove = []
    
    for col in df_clean.columns:
        if col == target_col:
            continue
            
        # Remove columns with too many missing values (>50%)
        missing_ratio = df_clean[col].isnull().sum() / len(df_clean)
        if missing_ratio > missing_ratio_threshold:
            cols_to_remove.append(col)
            continue
            
        # Remove columns with single unique value (constant columns)
        if df_clean[col].nunique() <= 1:
            cols_to_remove.append(col)
            continue
            
        # Remove columns that are mostly empty strings or whitespace
        if df_clean[col].dtype == 'object':
            non_empty = df_clean[col].astype(str).str.strip()
            if (non_empty == '').sum() / len(df_clean) > missing_ratio_threshold:
                cols_to_remove.append(col)
                continue
    
    # Remove bad columns
    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} bad columns: {cols_to_remove[:5]}{'...' if len(cols_to_remove) > 5 else ''}")
        df_clean = df_clean.drop(columns=cols_to_remove)
    
    # Handle missing values
    for col in df_clean.columns:
        if col == target_col:
            continue
            
        if df_clean[col].dtype in ['int64', 'float64']:
            # For numeric columns, fill with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            # For categorical columns, fill with mode or 'unknown'
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
            else:
                df_clean[col] = df_clean[col].fillna('unknown')
    
    return df_clean


def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Automatically detect continuous and categorical columns.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column to exclude
        
    Returns:
        Tuple of (continuous_cols, category_cols)
    """
    continuous_cols = []
    category_cols = []
    
    for col in df.columns:
        if col == target_col:
            continue
            
        # Check if column is numeric
        if df[col].dtype in ['int64', 'float64']:
            # If numeric but has very few unique values, treat as categorical
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05 and df[col].nunique() < 20:
                category_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            # Non-numeric columns are categorical
            category_cols.append(col)
    
    return continuous_cols, category_cols


def load_custom_dataset(train_path: str, test_path: str, target_col: str, max_size: int = MAX_DATASET_SIZE) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, 
    List[str], List[str], int, str, Dict[str, Any]
]:
    """
    Load and preprocess custom dataset from CSV files.
    
    Args:
        train_path: Path to training CSV file (tab-separated)
        test_path: Path to test CSV file (tab-separated)
        target_col: Name of the target column
        max_size: Maximum number of rows to load (to prevent OOM)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, continuous_cols, category_cols, 
                 output_dim, metric_name, metric_hparams)
    """
    print(f"Loading training data from: {train_path}")
    train_df = sample_large_dataset(train_path, max_size=max_size)
    
    print(f"Loading test data from: {test_path}")
    test_df = sample_large_dataset(test_path, max_size=max_size)
    
    print(f"Raw data shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Verify target column exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data. Available columns: {list(train_df.columns)}")
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test data. Available columns: {list(test_df.columns)}")
    
    # Clean the dataframes
    print("Cleaning training data...")
    train_df_clean = clean_dataframe(train_df, target_col)
    
    print("Cleaning test data...")
    test_df_clean = clean_dataframe(test_df, target_col)
    
    # Ensure both datasets have the same columns (excluding target)
    train_features = set(train_df_clean.columns) - {target_col}
    test_features = set(test_df_clean.columns) - {target_col}
    
    common_features = list(train_features.intersection(test_features))
    if len(common_features) < len(train_features) or len(common_features) < len(test_features):
        print(f"Keeping {len(common_features)} common features between train and test")
    
    # Keep only common features + target
    train_df_clean = train_df_clean[common_features + [target_col]]
    test_df_clean = test_df_clean[common_features + [target_col]]
    
    print(f"Cleaned data shapes - Train: {train_df_clean.shape}, Test: {test_df_clean.shape}")
    
    # Separate features and target
    X_train = train_df_clean.drop(columns=[target_col])
    y_train = train_df_clean[target_col].copy()
    X_test = test_df_clean.drop(columns=[target_col])
    y_test = test_df_clean[target_col].copy()
    
    # Detect feature types
    continuous_cols, category_cols = detect_feature_types(train_df_clean, target_col)
    print(f"Detected {len(continuous_cols)} continuous and {len(category_cols)} categorical features")
    
    # Process target variable
    unique_targets = y_train.unique()
    output_dim = len(unique_targets)
    
    # Determine task type and metric
    if output_dim == 2:
        metric_name = "accuracy_score"
        task_type = "binary_classification"
    elif output_dim > 2 and y_train.dtype in ['int64', 'object']:
        metric_name = "balanced_accuracy_score"
        task_type = "multiclass_classification"
    else:
        metric_name = "mean_squared_error"
        task_type = "regression"
        output_dim = 1
    
    print(f"Detected task: {task_type} with {output_dim} output dimension(s)")
    
    # Encode target variable if needed
    if task_type in ["binary_classification", "multiclass_classification"]:
        le_target = LabelEncoder()
        y_train = pd.Series(le_target.fit_transform(y_train))
        y_test = pd.Series(le_target.transform(y_test))
        print(f"Target classes: {le_target.classes_}")
    
    # Process categorical features
    if category_cols:
        print(f"Encoding categorical features: {category_cols}")
        for col in category_cols:
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined_values)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Scale continuous features
    if continuous_cols:
        print(f"Scaling continuous features: {continuous_cols}")
        scaler = MinMaxScaler()
        X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
        X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    print("✓ Dataset preprocessing completed successfully!")
    
    return X_train, X_test, y_train, y_test, continuous_cols, category_cols, output_dim, metric_name, {}


def load_custom_dataset_with_split(train_path: str, target_col: str, test_size: float = 0.2, 
                                  val_size: float = 0.2, random_state: int = 42, max_size: int = MAX_DATASET_SIZE) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series,
    List[str], List[str], int, str, Dict[str, Any]
]:
    """
    Load custom dataset and split training data into train/validation sets.
    
    Args:
        train_path: Path to training CSV file (tab-separated)
        target_col: Name of the target column
        test_size: Proportion for test split (if no separate test file)
        val_size: Proportion for validation split from training data
        random_state: Random state for reproducible splits
        max_size: Maximum number of rows to load (to prevent OOM)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, continuous_cols, 
                 category_cols, output_dim, metric_name, metric_hparams)
    """
    print(f"Loading training data from: {train_path}")
    train_df = sample_large_dataset(train_path, max_size=max_size)
    
    print(f"Raw data shape: {train_df.shape}")
    
    # Verify target column exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(train_df.columns)}")
    
    # Clean the dataframe
    print("Cleaning data...")
    train_df_clean = clean_dataframe(train_df, target_col)
    print(f"Cleaned data shape: {train_df_clean.shape}")
    
    # Separate features and target
    X = train_df_clean.drop(columns=[target_col])
    y = train_df_clean[target_col].copy()
    
    # Detect feature types
    continuous_cols, category_cols = detect_feature_types(train_df_clean, target_col)
    print(f"Detected {len(continuous_cols)} continuous and {len(category_cols)} categorical features")
    
    # Process target variable
    unique_targets = y.unique()
    output_dim = len(unique_targets)
    
    # Determine task type and metric
    if output_dim == 2:
        metric_name = "accuracy_score"
        task_type = "binary_classification"
    elif output_dim > 2 and y.dtype in ['int64', 'object']:
        metric_name = "balanced_accuracy_score"
        task_type = "multiclass_classification"
    else:
        metric_name = "mean_squared_error"
        task_type = "regression"
        output_dim = 1
    
    print(f"Detected task: {task_type} with {output_dim} output dimension(s)")
    
    # Encode target variable if needed
    if task_type in ["binary_classification", "multiclass_classification"]:
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y))
        print(f"Target classes: {le_target.classes_}")
    
    # Stratified split for classification, regular split for regression
    if task_type in ["binary_classification", "multiclass_classification"]:
        stratify = y
    else:
        stratify = None
    
    # Split into train and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Split train_val into train and validation
    if task_type in ["binary_classification", "multiclass_classification"]:
        stratify_val = y_train_val
    else:
        stratify_val = None
    
    # Adjust val_size to be relative to remaining data
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, 
        random_state=random_state, stratify=stratify_val
    )
    
    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Process categorical features
    if category_cols:
        print(f"Encoding categorical features: {category_cols}")
        for col in category_cols:
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined_values = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
            le.fit(combined_values)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_val[col] = le.transform(X_val[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Scale continuous features
    if continuous_cols:
        print(f"Scaling continuous features: {continuous_cols}")
        scaler = MinMaxScaler()
        X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
        X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
        X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    print("✓ Dataset preprocessing completed successfully!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, continuous_cols, category_cols, output_dim, metric_name, {} 