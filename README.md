# BBCA Stock Forecasting - MLOps Pipeline

A comprehensive machine learning pipeline for forecasting BBCA (Bank Central Asia) stock prices using regression models and MLOps best practices.

## 📊 Project Overview

This project implements a complete MLOps pipeline for stock price forecasting, featuring:

- **Data Pipeline**: Automated data collection and preprocessing using yfinance
- **Feature Engineering**: Advanced technical indicators and lag features
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Model Registry**: MLflow-based model versioning and deployment
- **Batch Scoring**: Automated model evaluation and comparison
- **Prediction Service**: Real-time and batch prediction capabilities
- **Monitoring**: Comprehensive logging and metrics tracking

## 🏗️ Architecture

```
├── notebooks/          # Jupyter notebooks for EDA and analysis
│   ├── eda_bbca.ipynb  # Exploratory Data Analysis & Feature Engineering
│   └── model_bbca.ipynb # Model Development & Analysis
├── src/                # Python source code
│   ├── common.py       # Shared utilities and configuration
│   ├── data_loader.py  # Data collection utilities (yfinance)
│   ├── train.py        # Model training pipeline
│   ├── register.py     # Model registration
│   ├── batch_score.py  # Batch scoring and evaluation
│   ├── predict.py      # Prediction service
│   └── e2e_pipeline.py # End-to-end pipeline testing
├── data/               # Data storage
│   ├── raw/           # Raw data from yfinance
│   ├── clean/         # Cleaned data
│   └── preprocessed/  # Feature-engineered data
├── models/            # Local model storage
├── experiments/       # Experiment results
└── results/          # Prediction results
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

### 2. Data Collection

First, download the stock data using the data loader:

```bash
# Download BBCA stock data
python -m src.data_loader 'BBCA.JK' '2022-08-01' '2025-07-31'
```

### 3. Data Preparation

Run the EDA notebook to prepare the data:

```bash
jupyter notebook notebooks/eda_bbca.ipynb
```

This notebook will:
- Read BBCA stock data from previously downloaded files
- Perform exploratory data analysis
- Create technical indicators and features
- Save preprocessed data to `data/preprocessed/`

### 4. Model Training

```bash
# Train all models (Ridge, XGBoost, LightGBM, baselines)
python -m src.train
```

### 5. Model Registration

```bash
# Register the best model in MLflow Model Registry
python -m src.register
```

### 6. Model Evaluation

```bash
# Run batch scoring on test data
python -m src.batch_score
```

### 7. Make Predictions

```bash
# Single prediction
python -m src.predict --mode single

# Batch evaluation
python -m src.predict --mode batch

# List available models
python -m src.predict --mode list
```

### 8. End-to-End Testing

```bash
# Test the complete pipeline
python -m src.e2e_pipeline
```

## 📚 Detailed Documentation

### Notebooks

#### `notebooks/eda_bbca.ipynb`
- **Purpose**: Exploratory Data Analysis and Feature Engineering
- **Key Features**:
  - BBCA stock data reading from downloaded files (2022-2025)
  - Technical indicator calculation:
    - **Moving Averages**: SMA (5, 20, 50, 200), EMA (12, 26)
    - **Momentum Indicators**: RSI (14), MACD (12, 26, 9)
    - **Volatility Indicators**: Bollinger Bands (20, 2)
    - **Volume Indicators**: Volume vs Moving Average
  - Feature engineering and target variable creation:
    - `Target_Close_Next_Day`: Next day closing price
    - `Target_Return_Next_Day`: Next day return
    - `Target_Direction`: Price direction (1=up, 0=down)
  - Data quality validation and preprocessing
  - Train/test split with time series considerations

#### `notebooks/model_bbca.ipynb`
- **Purpose**: Model Development and Analysis
- **Key Features**:
  - **Ridge Regression**: Primary model with cross-validation
  - **XGBoost Regressor**: Gradient boosting for non-linear patterns
  - **LightGBM**: Light gradient boosting machine
  - **Linear Regression**: Baseline comparison
  - Model performance evaluation and comparison
  - Feature importance analysis
  - Prediction visualization

### Pipeline Components

#### `src/common.py`
Shared utilities and configuration:
- Project constants and paths
- Data loading and validation functions
- Model evaluation metrics (RMSE, MAE, R², MAPE)
- MLflow experiment management
- Baseline model implementations (Naive, SMA, EMA)
- Display utilities

#### `src/data_loader.py`
Data collection utilities:
- yfinance integration for Indonesian stock data
- Support for multiple tickers (BBCA.JK, UNTR.JK, ASII.JK, TLKM.JK)
- Data download and storage functions
- CSV file management

#### `src/train.py`
Model training pipeline:
- **Algorithms**: Ridge Regression, XGBoost, LightGBM, baseline models
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **MLflow Integration**: Automatic experiment tracking
- **Model Comparison**: Performance evaluation across all models
- **Baseline Models**: Naive forecast, Simple Moving Average, Exponential Moving Average

#### `src/register.py`
Model registration:
- Finds best performing model from training runs
- Registers model in MLflow Model Registry
- Sets performance-based aliases (staging/production)
- Model versioning and metadata management

#### `src/batch_score.py`
Batch scoring and evaluation:
- Loads registered models from MLflow
- Evaluates model performance on test data
- Compares different model flavors (sklearn/pyfunc)
- Saves prediction results and metrics

#### `src/predict.py`
Prediction service:
- **Single Prediction**: Real-time stock price forecasting
- **Batch Evaluation**: Model performance assessment
- **Model Loading**: Registry and local fallback support
- **Confidence Intervals**: Prediction uncertainty estimation

#### `src/e2e_pipeline.py`
End-to-end pipeline testing:
- Validates data availability
- Tests training pipeline
- Verifies model registration
- Checks scoring and prediction services
- Provides comprehensive test results

## 🔧 Configuration

### Key Parameters (`src/common.py`)

```python
# Stock Configuration
TICKER = "BBCA.JK"
START_DATE = "2022-08-01"
END_DATE = "2025-07-31"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
PERFORMANCE_THRESHOLD_STAGING = 0.7
PERFORMANCE_THRESHOLD_PRODUCTION = 0.95

# MLflow Configuration
PROJECT_NAME = "bbca_stock_forecasting"
EXPERIMENT_NAME = "BBCA_Stock_Forecasting"
MODEL_NAME = "bbca_forecasting_model"
```

## 📈 Model Performance

The pipeline trains multiple models and automatically selects the best performer:

### Model Types
1. **Baseline Models**:
   - Naive Forecast (last price)
   - Simple Moving Average (5-day)
   - Exponential Moving Average (α=0.3)

2. **Advanced Models**:
   - Ridge Regression (α: 0.1, 1.0, 10.0, 100.0, 1000.0)
   - XGBoost (various hyperparameter combinations)
   - LightGBM (various hyperparameter combinations)

### Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

### Best Model Results
Based on the analysis in `model_bbca.ipynb`:
- **Ridge Regression**: RMSE=205.07, R²=0.7976, MAPE=1.93%
- **LightGBM**: RMSE=206.69, R²=0.7944, MAPE=1.89%
- **XGBoost**: RMSE=242.31, R²=0.7174, MAPE=2.07%
- **Linear Regression**: RMSE=243.95, R²=0.7136, MAPE=2.32%

## 🚀 Deployment

### Model Registry
Models are automatically registered in MLflow Model Registry with:
- Version control
- Performance-based staging
- Production promotion criteria
- Model aliases for easy access

### Available Model URIs
```bash
# Latest version
models:/bbca_forecasting_model/latest

# Staging (R² > 0.7)
models:/bbca_forecasting_model@staging

# Production (R² > 0.8)
models:/bbca_forecasting_model@production

# Specific version
models:/bbca_forecasting_model/1
```

## 📊 Monitoring and Logging

### MLflow Tracking
- All experiments tracked in MLflow
- Model parameters and metrics logged
- Artifact storage for models and results
- Web UI available at `http://localhost:5000`

### Logging Features
- Training metrics and parameters
- Model performance comparisons
- Prediction results and confidence intervals
- Pipeline execution status

### Monitoring Metrics
1. **Model Performance**:
   - RMSE, R², MAE, MAPE
   - Prediction accuracy trends
   - Model drift indicators

2. **System Health**:
   - Pipeline success rate
   - Data availability
   - Model loading times

3. **Business Metrics**:
   - Prediction confidence intervals
   - Cost of prediction errors
   - Model response times

## 🛠️ Troubleshooting

### Debug Commands

```bash
# Check data availability
python -m src.e2e_pipeline --step data

# Test specific pipeline step
python -m src.e2e_pipeline --step train
python -m src.e2e_pipeline --step register
python -m src.e2e_pipeline --step score

# View MLflow experiments
mlflow ui --host 0.0.0.0 --port 5000
```

## 📝 Usage Examples

### Training Pipeline
```bash
# Train all models
python -m src.train

# View results in MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

### Model Registration
```bash
# Register best model
python -m src.register

# Check registered models
python -m src.predict --mode list
```

### Predictions
```bash
# Single prediction
python -m src.predict --mode single

# Batch evaluation
python -m src.predict --mode batch

# Custom model URI
python -m src.predict --mode single --model_uri models:/bbca_forecasting_model@production
```

### Batch Scoring
```bash
# Score registered model
python -m src.batch_score

# Custom model URI
python -m src.batch_score --model_uri models:/bbca_forecasting_model@staging
```

## 🔍 Key Features

### Technical Indicators
- **Moving Averages**: SMA (5, 20, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14), MACD (12, 26, 9)
- **Volatility**: Bollinger Bands (20, 2)
- **Volume**: Volume vs Moving Average ratios
- **Price Ratios**: Price vs SMA ratios

### Model Selection
- **Primary**: Ridge Regression (best performance)
- **Alternative**: XGBoost, LightGBM
- **Baseline**: Linear Regression, Naive Forecast


## 👨‍💻 Author

**Thomas Stefen**
- Date: 2025-08-04
- Project: BBCA Stock Forecasting MLOps Pipeline

## 🙏 Acknowledgments

- yfinance for stock data
- MLflow for MLOps infrastructure
- scikit-learn, XGBoost, LightGBM for machine learning algorithms
- pandas, numpy for data manipulation
