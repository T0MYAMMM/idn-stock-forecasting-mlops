"""
Common configuration and utilities for BBCA Stock Forecasting Pipeline
=====================================================================

Shared constants, paths, and utility functions used across the pipeline.
Following MLflow best practices for separation of concerns.

Author: Thomas Stefen  
Date: 2025-08-04
"""

import json
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

# Project Configuration
PROJECT_NAME = "bbca_stock_forecasting"
EXPERIMENT_NAME = "BBCA_Stock_Forecasting"
MODEL_NAME = "bbca_forecasting_model"
MODEL_URI = f"models:/{MODEL_NAME}@production"

# Data Paths
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
CLEAN_DATA_DIR = f"{DATA_DIR}/clean" 
PREPROCESSED_DATA_DIR = f"{DATA_DIR}/preprocessed"

# Model Paths
MODEL_DIR = "models"
EXPERIMENT_DIR = "experiments"

# Stock Configuration
TICKER = "BBCA.JK"
START_DATE = "2022-08-01"
END_DATE = "2025-07-31"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
PERFORMANCE_THRESHOLD_STAGING = 0.7
PERFORMANCE_THRESHOLD_PRODUCTION = 0.95

# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_preprocessed_data():
    """Load preprocessed training and test data"""
    print("Loading preprocessed data...")
    
    train_path = f"{PREPROCESSED_DATA_DIR}/bbca_train.csv"
    test_path = f"{PREPROCESSED_DATA_DIR}/bbca_test.csv"
    metadata_path = f"{PREPROCESSED_DATA_DIR}/feature_metadata.json"
    
    if not all(os.path.exists(p) for p in [train_path, test_path, metadata_path]):
        raise FileNotFoundError(
            "Preprocessed data not found. Please run EDA notebook first to generate processed data."
        )
    
    # Load data
    train_data = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    test_data = pd.read_csv(test_path, index_col='Date', parse_dates=True)
    
    # Load feature metadata
    with open(metadata_path, 'r') as f:
        feature_metadata = json.load(f)
    
    print(f"✓ Training data: {train_data.shape}")
    print(f"✓ Test data: {test_data.shape}")
    print(f"✓ Date range - Train: {train_data.index.min()} to {train_data.index.max()}")
    print(f"✓ Date range - Test: {test_data.index.min()} to {test_data.index.max()}")
    
    return train_data, test_data, feature_metadata

def prepare_features_and_targets(train_data, test_data, feature_metadata):
    """Prepare features and targets for modeling"""
    # Get feature columns (exclude targets)
    target_cols = feature_metadata['target_columns']
    feature_cols = [col for col in train_data.columns if col not in target_cols]
    
    # Prepare features
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    
    # Prepare targets
    y_train = train_data['Target_Close_Next_Day']
    y_test = test_data['Target_Close_Next_Day']
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def get_original_prices(train_data, test_data):
    """Get original price data for baseline models"""
    return train_data['Close'], test_data['Close']

# ============================================================================
# MODEL EVALUATION UTILITIES  
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'rmse': float(rmse),
        'mae': float(mae), 
        'r2': float(r2),
        'mape': float(mape)
    }

def format_metrics(metrics):
    """Format metrics for display"""
    return f"RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2f}%"

# ============================================================================
# BASELINE MODEL UTILITIES
# ============================================================================

class BaselineModels:
    """Baseline models for comparison"""
    
    @staticmethod
    def naive_forecast(train_prices, n_predictions):
        """Naive forecast: next day price = current day price"""
        last_price = train_prices.iloc[-1]
        return np.full(n_predictions, last_price)
    
    @staticmethod
    def simple_moving_average(train_prices, n_predictions, window=5):
        """Simple moving average forecast"""
        sma = train_prices.rolling(window=window).mean().iloc[-1]
        return np.full(n_predictions, sma)
    
    @staticmethod
    def exponential_moving_average(train_prices, n_predictions, alpha=0.3):
        """Exponential moving average forecast"""
        ema = train_prices.ewm(alpha=alpha).mean().iloc[-1]
        return np.full(n_predictions, ema)

# ============================================================================
# MLFLOW UTILITIES
# ============================================================================

def ensure_experiment(experiment_name):
    """Ensure MLflow experiment exists"""
    import mlflow
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        print(f"Error setting up experiment: {e}")
        raise

def log_environment_info():
    """Log environment and dependency information"""
    import platform
    import sys
    
    # Try modern importlib.metadata first
    requirements = []
    try:
        import importlib.metadata as metadata
        key_packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 
                       'mlflow', 'matplotlib', 'seaborn', 'yfinance']
        
        for package_name in key_packages:
            try:
                version = metadata.version(package_name)
                requirements.append(f"{package_name}=={version}")
            except metadata.PackageNotFoundError:
                continue
                
    except ImportError:
        requirements = ["Could not determine package versions"]
    
    # Environment info
    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'packages': requirements,
        'project': PROJECT_NAME
    }
    
    return env_info

# ============================================================================
# FILE UTILITIES
# ============================================================================

def ensure_directories():
    """Ensure all required directories exist"""
    dirs = [MODEL_DIR, EXPERIMENT_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR, PREPROCESSED_DATA_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def save_json(data, file_path):
    """Save data as JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_data_quality(train_data, test_data):
    """Validate data quality before training"""
    issues = []
    
    # Check for missing values
    train_missing = train_data.isnull().sum().sum()
    test_missing = test_data.isnull().sum().sum()
    
    if train_missing > 0:
        issues.append(f"Training data has {train_missing} missing values")
    if test_missing > 0:
        issues.append(f"Test data has {test_missing} missing values")
    
    # Check for infinite values
    train_inf = np.isinf(train_data.select_dtypes(include=[np.number])).sum().sum()
    test_inf = np.isinf(test_data.select_dtypes(include=[np.number])).sum().sum()
    
    if train_inf > 0:
        issues.append(f"Training data has {train_inf} infinite values")
    if test_inf > 0:
        issues.append(f"Test data has {test_inf} infinite values")
    
    # Check data sizes
    if len(train_data) < 100:
        issues.append(f"Training data too small: {len(train_data)} samples")
    if len(test_data) < 20:
        issues.append(f"Test data too small: {len(test_data)} samples")
    
    if issues:
        print("⚠️  Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✓ Data quality validation passed")
    return True

# ============================================================================
# DISPLAY UTILITIES
# ============================================================================

def print_header(title, width=75):
    """Print formatted header"""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)

def print_section(title):
    """Print formatted section header"""
    print(f"\n{title}...")

def print_success(message):
    """Print success message"""
    print(f"✓ {message}")

def print_warning(message):
    """Print warning message"""  
    print(f"⚠️  {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")