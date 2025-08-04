#!/usr/bin/env python3
"""
BBCA.JK Stock Price Forecasting - Prediction Script
===================================================

This script implements model inference with:
1. Model loading from saved artifacts
2. Real-time predictions
3. Performance monitoring
4. MLflow tracking for predictions
5. Confidence intervals

Author: Thomas Stefen
Date: 2025-08-03
"""

import os
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configuration
warnings.filterwarnings('ignore')

class StockPredictor:
    """Stock price prediction class with MLflow integration"""
    
    def __init__(self, model_path: Optional[str] = None, use_registry: bool = False, registry_stage: str = "Production"):
        self.model = None
        self.metadata = None
        self.feature_metadata = None
        self.model_path = model_path or "models/best_model_ridge.pkl"
        self.use_registry = use_registry
        self.registry_stage = registry_stage
        
        # Initialize MLflow
        self.client = MlflowClient()
        
        # Load model and metadata
        self.load_model()
    
    def load_model(self):
        """Load trained model and metadata"""
        print("Loading trained model...")
        
        # Try to load from MLflow Model Registry first if requested
        if self.use_registry:
            if self.load_model_from_registry(self.registry_stage):
                return
        
        # Fall back to local file loading
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata
            with open('models/model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            # Load feature metadata
            with open('data/preprocessed/feature_metadata.json', 'r') as f:
                self.feature_metadata = json.load(f)
            
            print(f"Model loaded: {self.metadata['model_name']}")
            print(f"Model type: {self.metadata['model_type']}")
            print(f"Training date: {self.metadata['training_date']}")
            print(f"Training RMSE: {self.metadata['metrics']['rmse']:.2f}")
            print(f"Training R²: {self.metadata['metrics']['r2']:.4f}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found. Please run train.py first. Error: {e}")
    
    def load_model_from_registry(self, stage: str = "Production") -> bool:
        """Load model from MLflow Model Registry instead of local file"""
        MODEL_REGISTRY_NAME = "bbca_forecasting_model"  # Same as in train.py
        print(f"Loading model from MLflow Registry (stage/alias: {stage})...")
        
        model_version = None
        model_uri = None
        
        # Try modern alias approach first, then fallback to stages
        try:
            # Method 1: Try using aliases (recommended)
            alias_name = stage.lower()  # Convert Production -> production
            model_version = self.client.get_model_version_by_alias(MODEL_REGISTRY_NAME, alias_name)
            model_uri = f"models:/{MODEL_REGISTRY_NAME}@{alias_name}"
            print(f"Found model using alias: {alias_name}, version {model_version.version}")
            
        except Exception as alias_error:
            print(f"Alias lookup failed: {alias_error}")
            print("Trying stage-based lookup...")
            
            # Method 2: Fallback to stages (legacy)
            try:
                model_versions = self.client.get_latest_versions(
                    MODEL_REGISTRY_NAME,
                    stages=[stage]
                )
                
                if not model_versions:
                    print(f"No model found in {stage} stage")
                    return False
                    
                model_version = model_versions[0]
                model_uri = f"models:/{MODEL_REGISTRY_NAME}/{model_version.version}"
                print(f"Found model using stage: {stage}, version {model_version.version}")
                
            except Exception as stage_error:
                print(f"Stage lookup also failed: {stage_error}")
                return False
        
        if model_version and model_uri:
            try:
                # Load model from registry
                self.model = mlflow.sklearn.load_model(model_uri)
            
            # Get model version details
            model_details = self.client.get_model_version(
                MODEL_REGISTRY_NAME, 
                model_version.version
            )
            
            # Create metadata from registry info
            self.metadata = {
                'model_name': 'ridge',  # Default for registry models
                'model_type': 'advanced',
                'metrics': {
                    'rmse': 0.0,  # Will be updated if available in run
                    'r2': 0.0,
                    'mae': 0.0,
                    'mape': 0.0
                },
                'training_date': model_details.creation_timestamp,
                'model_path': model_uri,
                'registry_version': model_version.version,
                'registry_stage': stage
            }
            
            # Try to get metrics from the run that created this model version
            try:
                run_id = model_details.run_id
                run = self.client.get_run(run_id)
                if run.data.metrics:
                    # Update metadata with actual metrics
                    for metric_name in ['rmse', 'r2', 'mae', 'mape']:
                        test_metric = f"test_{metric_name}"
                        if test_metric in run.data.metrics:
                            self.metadata['metrics'][metric_name] = run.data.metrics[test_metric]
            except Exception as e:
                print(f"Could not retrieve run metrics: {e}")
            
            # Load feature metadata (still need this from local file)
            try:
                with open('data/preprocessed/feature_metadata.json', 'r') as f:
                    self.feature_metadata = json.load(f)
            except FileNotFoundError:
                print("Warning: Feature metadata not found locally")
                # Create basic feature metadata
                self.feature_metadata = {'target_columns': ['Target_Close_Next_Day', 'Target_Return_Next_Day', 'Target_Direction']}
            
            print(f"Loaded model version {model_version.version} from {stage}")
            print(f"Model description: {model_details.description or 'No description'}")
            print(f"Model URI: {model_uri}")
            if self.metadata['metrics']['rmse'] > 0:
                print(f"Model RMSE: {self.metadata['metrics']['rmse']:.2f}")
                print(f"Model R²: {self.metadata['metrics']['r2']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load from registry: {e}")
            print("Falling back to local model...")
            return False
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data for evaluation"""
        test_data = pd.read_csv('data/preprocessed/bbca_test.csv', 
                               index_col='Date', parse_dates=True)
        return test_data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Feature columns (exclude targets)
        target_cols = self.feature_metadata['target_columns']
        feature_cols = [col for col in data.columns if col not in target_cols]
        
        return data[feature_cols]
    
    def predict(self, X: pd.DataFrame, log_to_mlflow: bool = True) -> Tuple[np.ndarray, Dict]:
        """Make predictions and optionally log to MLflow"""
        print(f"Making predictions for {len(X)} samples...")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate prediction statistics
        pred_stats = {
            'mean_prediction': float(np.mean(predictions)),
            'std_prediction': float(np.std(predictions)),
            'min_prediction': float(np.min(predictions)),
            'max_prediction': float(np.max(predictions)),
            'prediction_date': datetime.now().isoformat(),
            'n_predictions': int(len(predictions))  # Convert to int for better JSON serialization
        }
        
        # Log to MLflow if requested
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters (string values)
                mlflow.log_param("model_name", self.metadata['model_name'])
                mlflow.log_param("prediction_date", pred_stats['prediction_date'])
                mlflow.log_param("n_predictions", pred_stats['n_predictions'])
                
                # Log metrics (numeric values only)
                metrics_to_log = {
                    'mean_prediction': pred_stats['mean_prediction'],
                    'std_prediction': pred_stats['std_prediction'],
                    'min_prediction': pred_stats['min_prediction'],
                    'max_prediction': pred_stats['max_prediction']
                }
                mlflow.log_metrics(metrics_to_log)
                
                # Log prediction distribution plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(predictions, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title('Prediction Distribution')
                ax.set_xlabel('Predicted Price (IDR)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                mlflow.log_figure(fig, "prediction_distribution.png")
                plt.close(fig)
        
        print(f"Predictions completed")
        print(f"   Mean prediction: {pred_stats['mean_prediction']:.2f} IDR")
        print(f"   Prediction range: {pred_stats['min_prediction']:.2f} - {pred_stats['max_prediction']:.2f} IDR")
        
        return predictions, pred_stats
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           dates: pd.DatetimeIndex, log_to_mlflow: bool = True) -> Dict:
        """Evaluate predictions against actual values"""
        print("Evaluating predictions...")
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate residuals
        residuals = y_true - y_pred
        residual_stats = {
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'residual_skewness': float(pd.Series(residuals).skew()),
            'residual_kurtosis': float(pd.Series(residuals).kurtosis())
        }
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            **residual_stats
        }
        
        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"Evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("model_name", self.metadata['model_name'])
                mlflow.log_param("evaluation_samples", len(y_true))
                mlflow.log_param("evaluation_date", datetime.now().isoformat())
                mlflow.log_metrics(metrics)
                
                # Create comprehensive evaluation plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Time series comparison
                axes[0, 0].plot(dates, y_true, label='Actual', linewidth=2, alpha=0.8)
                axes[0, 0].plot(dates, y_pred, label='Predicted', linewidth=2, alpha=0.8)
                axes[0, 0].set_title('Actual vs Predicted Over Time')
                axes[0, 0].set_ylabel('Price (IDR)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Scatter plot
                axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
                axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                               'r--', lw=2, alpha=0.8)
                axes[0, 1].set_xlabel('Actual Price (IDR)')
                axes[0, 1].set_ylabel('Predicted Price (IDR)')
                axes[0, 1].set_title('Prediction Accuracy')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Residuals over time
                axes[1, 0].plot(dates, residuals, alpha=0.7)
                axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
                axes[1, 0].set_title('Residuals Over Time')
                axes[1, 0].set_ylabel('Residuals (IDR)')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Residual distribution
                axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Residual Distribution')
                axes[1, 1].set_xlabel('Residuals (IDR)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                mlflow.log_figure(fig, "evaluation_plots.png")
                plt.close(fig)
                
                # Feature importance (if available)
                if hasattr(self.model, 'coef_'):
                    # Get all feature names
                    all_features = []
                    for category, features in self.feature_metadata.items():
                        if category != 'target_columns':
                            all_features.extend(features)
                    
                    if len(all_features) == len(self.model.coef_):
                        feature_importance = pd.DataFrame({
                            'feature': all_features,
                            'importance': np.abs(self.model.coef_)
                        }).sort_values('importance', ascending=False).head(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.barh(range(len(feature_importance)), feature_importance['importance'])
                        ax.set_yticks(range(len(feature_importance)))
                        ax.set_yticklabels(feature_importance['feature'])
                        ax.set_xlabel('Importance (|Coefficient|)')
                        ax.set_title('Top 15 Feature Importance')
                        plt.tight_layout()
                        mlflow.log_figure(fig, "feature_importance.png")
                        plt.close(fig)
        
        # Print evaluation results
        print(f"Evaluation completed:")
        print(f"   RMSE: {rmse:.2f} IDR")
        print(f"   MAE: {mae:.2f} IDR")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return metrics
    
    def predict_next_day(self, latest_data: pd.DataFrame, log_to_mlflow: bool = True) -> Dict:
        """Predict next day's closing price"""
        print("Predicting next day's closing price...")
        
        # Prepare features
        X = self.prepare_features(latest_data.tail(1))  # Use latest observation
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Calculate confidence interval (using training residuals as approximation)
        if 'residual_std' in self.metadata:
            residual_std = self.metadata['residual_std']
        else:
            # Rough approximation using RMSE
            residual_std = self.metadata['metrics']['rmse']
        
        confidence_interval = {
            'lower_95': float(prediction - 1.96 * residual_std),
            'upper_95': float(prediction + 1.96 * residual_std),
            'lower_68': float(prediction - residual_std),
            'upper_68': float(prediction + residual_std)
        }
        
        prediction_result = {
            'prediction': float(prediction),
            'prediction_date': datetime.now().isoformat(),
            'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'model_name': self.metadata['model_name'],
            **confidence_interval
        }
        
        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"NextDay_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_name", self.metadata['model_name'])
                mlflow.log_param("target_date", prediction_result['target_date'])
                mlflow.log_param("prediction_date", prediction_result['prediction_date'])
                
                # Log metrics
                mlflow.log_metric("prediction", prediction_result['prediction'])
                mlflow.log_metric("confidence_range_95", confidence_interval['upper_95'] - confidence_interval['lower_95'])
                mlflow.log_metric("lower_95", confidence_interval['lower_95'])
                mlflow.log_metric("upper_95", confidence_interval['upper_95'])
        
        print(f"Next day prediction: {prediction:.2f} IDR")
        print(f"   95% CI: [{confidence_interval['lower_95']:.2f}, {confidence_interval['upper_95']:.2f}] IDR")
        print(f"   Target date: {prediction_result['target_date']}")
        
        return prediction_result
    
    def run_batch_evaluation(self):
        """Run evaluation on test dataset"""
        print("Running batch evaluation on test dataset")
        print("=" * 75)
        
        # Load test data
        test_data = self.load_test_data()
        
        # Prepare features and targets
        X_test = self.prepare_features(test_data)
        y_test = test_data['Target_Close_Next_Day']
        
        # Make predictions
        predictions, pred_stats = self.predict(X_test, log_to_mlflow=True)
        
        # Evaluate predictions
        metrics = self.evaluate_predictions(y_test, predictions, test_data.index, log_to_mlflow=True)
        
        print("\nBatch evaluation completed!")
        return predictions, metrics
    
    def run_single_prediction(self, data_path: Optional[str] = None):
        """Run single prediction for latest data"""
        print("Running single prediction")
        print("=" * 75)
        
        if data_path:
            latest_data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        else:
            # Use latest preprocessed data
            latest_data = pd.read_csv('data/preprocessed/bbca_test.csv', 
                                    index_col='Date', parse_dates=True)
        
        # Predict next day
        prediction_result = self.predict_next_day(latest_data, log_to_mlflow=True)
        
        print("\nSingle prediction completed!")
        return prediction_result
    
    def list_available_models(self):
        """List available models in MLflow Model Registry"""
        MODEL_REGISTRY_NAME = "bbca_forecasting_model"
        
        try:
            print("\nAvailable Models in Registry:")
            print("=" * 75)
            
            # Get all model versions
            for stage in ["None", "Staging", "Production"]:
                try:
                    versions = self.client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[stage])
                    for version in versions:
                        print(f"Stage: {stage}")
                        print(f"   Version: {version.version}")
                        print(f"   Created: {version.creation_timestamp}")
                        print(f"   Description: {version.description or 'No description'}")
                        
                        # Get run metrics if available
                        try:
                            run = self.client.get_run(version.run_id)
                            if run.data.metrics:
                                rmse = run.data.metrics.get('test_rmse', 'N/A')
                                r2 = run.data.metrics.get('test_r2', 'N/A')
                                print(f"   RMSE: {rmse}")
                                print(f"   R²: {r2}")
                        except:
                            pass
                        print("-" * 30)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error listing models: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='BBCA Stock Price Prediction')
    parser.add_argument('--mode', choices=['batch', 'single', 'list'], default='batch',
                       help='Prediction mode: batch evaluation, single prediction, or list available models')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file for single prediction')
    parser.add_argument('--use_registry', action='store_true',
                       help='Load model from MLflow Model Registry instead of local file')
    parser.add_argument('--registry_stage', choices=['None', 'Staging', 'Production'], default='Production',
                       help='Model Registry stage to load from (default: Production)')
    
    args = parser.parse_args()
    
    # Handle list mode separately (doesn't need model loading)
    if args.mode == 'list':
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Create a temporary predictor just for listing
        class TempPredictor:
            def __init__(self):
                self.client = client
            def list_available_models(self):
                MODEL_REGISTRY_NAME = "bbca_forecasting_model"
                try:
                    print("\nAvailable Models in Registry:")
                    print("=" * 75)
                    
                    for stage in ["None", "Staging", "Production"]:
                        try:
                            versions = self.client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[stage])
                            for version in versions:
                                print(f"Stage: {stage}")
                                print(f"   Version: {version.version}")
                                print(f"   Created: {version.creation_timestamp}")
                                print(f"   Description: {version.description or 'No description'}")
                                
                                try:
                                    run = self.client.get_run(version.run_id)
                                    if run.data.metrics:
                                        rmse = run.data.metrics.get('test_rmse', 'N/A')
                                        r2 = run.data.metrics.get('test_r2', 'N/A')
                                        print(f"   RMSE: {rmse}")
                                        print(f"   R²: {r2}")
                                except:
                                    pass
                                print("-" * 75)
                        except:
                            pass
                except Exception as e:
                    print(f"Error listing models: {e}")
        
        temp_predictor = TempPredictor()
        temp_predictor.list_available_models()
        return
    
    # Initialize predictor for batch/single modes
    predictor = StockPredictor(
        model_path=args.model_path,
        use_registry=args.use_registry,
        registry_stage=args.registry_stage
    )
    
    if args.mode == 'batch':
        predictions, metrics = predictor.run_batch_evaluation()
        
        print("\n" + "="*75)
        print("EVALUATION SUMMARY")
        print("="*75)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
            
    elif args.mode == 'single':
        prediction_result = predictor.run_single_prediction(args.data_path)
        
        print("\n" + "="*75)
        print("PREDICTION SUMMARY")
        print("="*75)
        print(f"Predicted Price: {prediction_result['prediction']:.2f} IDR")
        print(f"Target Date: {prediction_result['target_date']}")
        print(f"95% Confidence Interval: [{prediction_result['lower_95']:.2f}, {prediction_result['upper_95']:.2f}] IDR")

if __name__ == "__main__":
    main()