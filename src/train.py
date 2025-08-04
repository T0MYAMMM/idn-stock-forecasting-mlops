#!/usr/bin/env python3
"""
BBCA.JK Stock Price Forecasting - Training Script
=================================================

This script implements comprehensive model training with:
1. Baseline models (Naive, Simple Moving Average)
2. Advanced models (Ridge Regression, XGBoost, LightGBM)
3. MLflow experiment tracking
4. Automated model selection
5. Artifact logging for reproducibility

Author: Thomas Stefen
Date: 2025-08-03
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, Any, Tuple
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)

# MLflow configuration
EXPERIMENT_NAME = "BBCA_Stock_Forecasting"
NAME = "models"
MODEL_REGISTRY_NAME = "bbca_forecasting_model"

class BaselineModels:
    """Baseline models for comparison"""
    
    @staticmethod
    def naive_forecast(train_prices: pd.Series, test_dates: pd.DatetimeIndex) -> np.ndarray:
        """Naive forecast: next day price = current day price"""
        last_price = train_prices.iloc[-1]
        return np.full(len(test_dates), last_price)
    
    @staticmethod
    def simple_moving_average(train_prices: pd.Series, test_dates: pd.DatetimeIndex, window: int = 5) -> np.ndarray:
        """Simple moving average forecast"""
        sma = train_prices.rolling(window=window).mean().iloc[-1]
        return np.full(len(test_dates), sma)
    
    @staticmethod
    def exponential_moving_average(train_prices: pd.Series, test_dates: pd.DatetimeIndex, alpha: float = 0.3) -> np.ndarray:
        """Exponential moving average forecast"""
        ema = train_prices.ewm(alpha=alpha).mean().iloc[-1]
        return np.full(len(test_dates), ema)

class ModelEvaluator:
    """Model evaluation utilities"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        dates: pd.DatetimeIndex, model_name: str) -> plt.Figure:
        """Create prediction visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time series plot
        ax1.plot(dates, y_true, label='Actual', alpha=0.8, linewidth=2)
        ax1.plot(dates, y_pred, label='Predicted', alpha=0.8, linewidth=2)
        ax1.set_title(f'{model_name}: Actual vs Predicted')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (IDR)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.6)
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, alpha=0.8)
        ax2.set_xlabel('Actual Price (IDR)')
        ax2.set_ylabel('Predicted Price (IDR)')
        ax2.set_title(f'{model_name}: Prediction Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class StockForecaster:
    """Main forecasting class with MLflow integration"""
    
    def __init__(self):
        self.models = {}
        self.baseline_models = {}
        self.feature_metadata = None
        self.best_model_info = None
        
        # Initialize MLflow
        mlflow.set_experiment(EXPERIMENT_NAME)
        self.client = MlflowClient()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load preprocessed training and test data"""
        print("Loading preprocessed data...")
        
        train_data = pd.read_csv('data/preprocessed/bbca_train.csv', 
                                index_col='Date', parse_dates=True)
        test_data = pd.read_csv('data/preprocessed/bbca_test.csv', 
                               index_col='Date', parse_dates=True)
        
        # Load feature metadata
        with open('data/preprocessed/feature_metadata.json', 'r') as f:
            self.feature_metadata = json.load(f)
        
        print(f"Training data: {train_data.shape}")
        print(f"Test data: {test_data.shape}")
        print(f"Date range - Train: {train_data.index.min()} to {train_data.index.max()}")
        print(f"Date range - Test: {test_data.index.min()} to {test_data.index.max()}")
        
        return train_data, test_data
    
    def prepare_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple:
        """Prepare features and targets"""
        # Feature columns (exclude targets)
        target_cols = self.feature_metadata['target_columns']
        feature_cols = [col for col in train_data.columns if col not in target_cols]
        
        X_train = train_data[feature_cols]
        y_train = train_data['Target_Close_Next_Day']
        X_test = test_data[feature_cols]
        y_test = test_data['Target_Close_Next_Day']
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_baseline_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Train baseline models"""
        print("\nTraining Baseline Models...")
        
        train_prices = train_data['Close']  # Original prices (unscaled)
        test_dates = test_data.index
        y_test_actual = test_data['Target_Close_Next_Day']
        
        baseline_results = {}
        
        # 1. Naive Forecast
        with mlflow.start_run(run_name="Baseline_Naive_Forecast"):
            naive_pred = BaselineModels.naive_forecast(train_prices, test_dates)
            naive_metrics = ModelEvaluator.calculate_metrics(y_test_actual, naive_pred)
            
            # Log to MLflow
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("algorithm", "naive_forecast")
            mlflow.log_metrics(naive_metrics)
            
            # Create and log plot
            fig = ModelEvaluator.plot_predictions(y_test_actual, naive_pred, 
                                                test_dates, "Naive Forecast")
            mlflow.log_figure(fig, "naive_forecast_predictions.png")
            plt.close(fig)
            
            baseline_results['Naive Forecast'] = naive_metrics
            self.baseline_models['naive'] = {'predictions': naive_pred, 'metrics': naive_metrics}
            
            print(f"Naive Forecast - RMSE: {naive_metrics['rmse']:.2f}, R²: {naive_metrics['r2']:.4f}")
        
        # 2. Simple Moving Average (5-day)
        with mlflow.start_run(run_name="Baseline_SMA_5"):
            sma_pred = BaselineModels.simple_moving_average(train_prices, test_dates, window=5)
            sma_metrics = ModelEvaluator.calculate_metrics(y_test_actual, sma_pred)
            
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("algorithm", "simple_moving_average")
            mlflow.log_param("window", 5)
            mlflow.log_metrics(sma_metrics)
            
            fig = ModelEvaluator.plot_predictions(y_test_actual, sma_pred, 
                                                test_dates, "SMA (5-day)")
            mlflow.log_figure(fig, "sma_5_predictions.png")
            plt.close(fig)
            
            baseline_results['SMA (5-day)'] = sma_metrics
            self.baseline_models['sma_5'] = {'predictions': sma_pred, 'metrics': sma_metrics}
            
            print(f"SMA (5-day) - RMSE: {sma_metrics['rmse']:.2f}, R²: {sma_metrics['r2']:.4f}")
        
        # 3. Exponential Moving Average
        with mlflow.start_run(run_name="Baseline_EMA"):
            ema_pred = BaselineModels.exponential_moving_average(train_prices, test_dates, alpha=0.3)
            ema_metrics = ModelEvaluator.calculate_metrics(y_test_actual, ema_pred)
            
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("algorithm", "exponential_moving_average")
            mlflow.log_param("alpha", 0.3)
            mlflow.log_metrics(ema_metrics)
            
            fig = ModelEvaluator.plot_predictions(y_test_actual, ema_pred, 
                                                test_dates, "EMA (α=0.3)")
            mlflow.log_figure(fig, "ema_predictions.png")
            plt.close(fig)
            
            baseline_results['EMA (α=0.3)'] = ema_metrics
            self.baseline_models['ema'] = {'predictions': ema_pred, 'metrics': ema_metrics}
            
            print(f"EMA (α=0.3) - RMSE: {ema_metrics['rmse']:.2f}, R²: {ema_metrics['r2']:.4f}")
        
        return baseline_results
    
    def train_ridge_regression(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series):
        """Train Ridge Regression model"""
        print("\nTraining Ridge Regression (Best Model)...")
        
        with mlflow.start_run(run_name="Ridge_Regression_Optimized"):
            # Hyperparameter tuning
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
            
            ridge = Ridge(random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(ridge, param_grid, cv=tscv, 
                                     scoring='neg_root_mean_squared_error', n_jobs=-1)
            
            # Fit model
            grid_search.fit(X_train, y_train)
            best_ridge = grid_search.best_estimator_
            
            # Predictions
            train_pred = best_ridge.predict(X_train)
            test_pred = best_ridge.predict(X_test)
            
            # Metrics
            train_metrics = ModelEvaluator.calculate_metrics(y_train, train_pred)
            test_metrics = ModelEvaluator.calculate_metrics(y_test, test_pred)
            
            # Log parameters
            mlflow.log_param("model_type", "advanced")
            mlflow.log_param("algorithm", "ridge_regression")
            mlflow.log_param("best_alpha", grid_search.best_params_['alpha'])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Log metrics
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(best_ridge.coef_)
            }).sort_values('importance', ascending=False)
            
            # Log model
            mlflow.sklearn.log_model(best_ridge, NAME,
                                   input_example=X_test.iloc[:5])
            
            # Log artifacts
            feature_importance.to_csv('temp_feature_importance.csv', index=False)
            mlflow.log_artifact('temp_feature_importance.csv')
            os.remove('temp_feature_importance.csv')
            
            # Log plots
            fig = ModelEvaluator.plot_predictions(y_test, test_pred, 
                                                X_test.index, "Ridge Regression")
            mlflow.log_figure(fig, "ridge_predictions.png")
            plt.close(fig)
            
            # Feature importance plot
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(15)
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance (|Coefficient|)')
            ax.set_title('Top 15 Feature Importance (Ridge Regression)')
            plt.tight_layout()
            mlflow.log_figure(fig, "ridge_feature_importance.png")
            plt.close(fig)
            
            # Store model info
            self.models['ridge'] = {
                'model': best_ridge,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'predictions': test_pred
            }
            
            print(f"Ridge Regression - RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}")
            
            return grid_search.best_estimator_, test_metrics
    
    def compare_models(self, baseline_results: Dict, test_data: pd.DataFrame):
        """Compare all models and select the best one"""
        print("\nModel Comparison...")
        
        # Compile all results
        all_results = []
        
        # Add baseline results
        for name, metrics in baseline_results.items():
            all_results.append({
                'model_name': name,
                'model_type': 'baseline',
                **metrics
            })
        
        # Add advanced model results
        for name, model_info in self.models.items():
            all_results.append({
                'model_name': name.title(),
                'model_type': 'advanced',
                **model_info['test_metrics']
            })
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('rmse')  # Sort by RMSE (lower is better)
        
        # Log comparison results
        with mlflow.start_run(run_name="Model_Comparison"):
            mlflow.log_param("comparison_metric", "rmse")
            mlflow.log_param("best_model", results_df.iloc[0]['model_name'])
            mlflow.log_param("best_rmse", results_df.iloc[0]['rmse'])
            mlflow.log_param("best_r2", results_df.iloc[0]['r2'])
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # RMSE comparison
            axes[0,0].bar(results_df['model_name'], results_df['rmse'], alpha=0.7)
            axes[0,0].set_title('RMSE Comparison (Lower is Better)')
            axes[0,0].set_ylabel('RMSE')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # R² comparison
            axes[0,1].bar(results_df['model_name'], results_df['r2'], alpha=0.7)
            axes[0,1].set_title('R² Comparison (Higher is Better)')
            axes[0,1].set_ylabel('R²')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # MAE comparison
            axes[1,0].bar(results_df['model_name'], results_df['mae'], alpha=0.7)
            axes[1,0].set_title('MAE Comparison (Lower is Better)')
            axes[1,0].set_ylabel('MAE')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # MAPE comparison
            axes[1,1].bar(results_df['model_name'], results_df['mape'], alpha=0.7)
            axes[1,1].set_title('MAPE Comparison (Lower is Better)')
            axes[1,1].set_ylabel('MAPE (%)')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            mlflow.log_figure(fig, "model_comparison.png")
            plt.close(fig)
            
            # Save comparison results
            results_df.to_csv('temp_model_comparison.csv', index=False)
            mlflow.log_artifact('temp_model_comparison.csv')
            os.remove('temp_model_comparison.csv')
        
        # Find best model
        best_model_name = results_df.iloc[0]['model_name'].lower()
        
        if best_model_name in self.models:
            self.best_model_info = {
                'name': best_model_name,
                'model': self.models[best_model_name]['model'],
                'metrics': self.models[best_model_name]['test_metrics'],
                'type': 'advanced'
            }
        else:
            # Best model is a baseline
            self.best_model_info = {
                'name': best_model_name,
                'model': None,  # Baseline models don't have sklearn objects
                'metrics': baseline_results[results_df.iloc[0]['model_name']],
                'type': 'baseline'
            }
        
        print(f"\nBest Model: {results_df.iloc[0]['model_name']}")
        print(f"   RMSE: {results_df.iloc[0]['rmse']:.2f}")
        print(f"   R²: {results_df.iloc[0]['r2']:.4f}")
        print(f"   MAE: {results_df.iloc[0]['mae']:.2f}")
        print(f"   MAPE: {results_df.iloc[0]['mape']:.2f}%")
        
        return results_df
    
    def save_best_model(self):
        """Save the best model for deployment"""
        if self.best_model_info['type'] == 'advanced':
            print(f"\nSaving best model: {self.best_model_info['name']}")
            
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Save model
            model_path = f"models/best_model_{self.best_model_info['name']}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model_info['model'], f)
            
            # Save metadata
            metadata = {
                'model_name': self.best_model_info['name'],
                'model_type': 'advanced',
                'metrics': self.best_model_info['metrics'],
                'feature_metadata': self.feature_metadata,
                'training_date': datetime.now().isoformat(),
                'model_path': model_path
            }
            
            with open('models/model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model saved: {model_path}")
            print(f"Metadata saved: models/model_metadata.json")
        else:
            print(f"\nBest model is baseline ({self.best_model_info['name']}) - no sklearn object to save")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting BBCA Stock Forecasting Training Pipeline")
        print("=" * 75)
        
        # 1. Load data
        train_data, test_data = self.load_data()
        
        # 2. Prepare features
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_features(train_data, test_data)
        
        # 3. Train baseline models
        baseline_results = self.train_baseline_models(train_data, test_data)
        
        # 4. Train advanced models
        self.train_ridge_regression(X_train, X_test, y_train, y_test)
        
        # 5. Compare models
        comparison_results = self.compare_models(baseline_results, test_data)
        
        # 6. Save best model
        self.save_best_model()
        
        print("\nTraining pipeline completed successfully!")
        print(f"Best model: {self.best_model_info['name']}")
        print(f"MLflow experiment: {EXPERIMENT_NAME}")
        print(f"View results: http://localhost:5000")
        
        return comparison_results

def main():
    """Main execution function"""
    forecaster = StockForecaster()
    results = forecaster.run_training_pipeline()
    
    print("\n" + "="*75)
    print("TRAINING SUMMARY")
    print("="*75)
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()