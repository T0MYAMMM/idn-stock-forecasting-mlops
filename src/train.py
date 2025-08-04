import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# Local imports
from src import common

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(common.RANDOM_STATE)

class StockForecaster:
    """Stock forecasting training class with MLflow integration"""
    
    def __init__(self):
        self.client = MlflowClient()
        self.experiment_id = common.ensure_experiment(common.EXPERIMENT_NAME)
        self.best_run_info = None
        
    def train_baseline_models(self, train_prices: pd.Series, y_test: pd.Series, test_dates: pd.DatetimeIndex):
        """Train baseline models for comparison"""
        common.print_section("Training Baseline Models")
        
        baseline_results = []
        n_predictions = len(y_test)
        
        # 1. Naive Forecast
        with mlflow.start_run(run_name="Baseline_Naive"):
            predictions = common.BaselineModels.naive_forecast(train_prices, n_predictions)
            metrics = common.calculate_metrics(y_test, predictions)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("algorithm", "naive_forecast")
            mlflow.log_param("window", "N/A")
            mlflow.log_metrics(metrics)
            
            baseline_results.append({
                'run_id': mlflow.active_run().info.run_id,
                'model_name': 'Naive Forecast',
                'model_type': 'baseline',
                **metrics
            })
            
            print(f"   Naive Forecast - {common.format_metrics(metrics)}")
        
        # 2. Simple Moving Average (5-day)
        with mlflow.start_run(run_name="Baseline_SMA_5"):
            predictions = common.BaselineModels.simple_moving_average(train_prices, n_predictions, window=5)
            metrics = common.calculate_metrics(y_test, predictions)
            
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("algorithm", "simple_moving_average")
            mlflow.log_param("window", 5)
            mlflow.log_metrics(metrics)
            
            baseline_results.append({
                'run_id': mlflow.active_run().info.run_id,
                'model_name': 'SMA (5-day)',
                'model_type': 'baseline',
                **metrics
            })
            
            print(f"   SMA (5-day) - {common.format_metrics(metrics)}")
        
        # 3. Exponential Moving Average
        with mlflow.start_run(run_name="Baseline_EMA"):
            predictions = common.BaselineModels.exponential_moving_average(train_prices, n_predictions, alpha=0.3)
            metrics = common.calculate_metrics(y_test, predictions)
            
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("algorithm", "exponential_moving_average")
            mlflow.log_param("alpha", 0.3)
            mlflow.log_metrics(metrics)
            
            baseline_results.append({
                'run_id': mlflow.active_run().info.run_id,
                'model_name': 'EMA (α=0.3)',
                'model_type': 'baseline',
                **metrics
            })
            
            print(f"   EMA (α=0.3) - {common.format_metrics(metrics)}")
        
        return baseline_results
    
    def train_ridge_regression(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series, feature_cols: List[str]):
        """Train Ridge Regression with hyperparameter tuning"""
        common.print_section("Training Ridge Regression")
        
        # Hyperparameter grid
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        }
        
        results = []
        
        for alpha in param_grid['alpha']:
            try:
                with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}"):
                    # Train model
                    model = Ridge(alpha=alpha, random_state=common.RANDOM_STATE)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_metrics = common.calculate_metrics(y_train, train_pred)
                    test_metrics = common.calculate_metrics(y_test, test_pred)
                    
                    # Log parameters
                    mlflow.log_param("model_type", "advanced")
                    mlflow.log_param("algorithm", "ridge_regression")
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("n_features", len(feature_cols))
                    mlflow.log_param("random_state", common.RANDOM_STATE)
                    
                    # Log metrics
                    for key, value in train_metrics.items():
                        mlflow.log_metric(f"train_{key}", value)
                    for key, value in test_metrics.items():
                        mlflow.log_metric(f"test_{key}", value)
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        model, 
                        "model",
                        input_example=X_test.head(3)
                    )
                    
                    run_info = {
                        'run_id': mlflow.active_run().info.run_id,
                        'model_name': f'Ridge (α={alpha})',
                        'model_type': 'advanced',
                        'algorithm': 'ridge_regression',
                        'alpha': alpha,
                        **test_metrics
                    }
                    results.append(run_info)
                    
                    print(f"   Ridge (α={alpha}) - {common.format_metrics(test_metrics)}")
                    
            except Exception as e:
                print(f"   Error training Ridge (α={alpha}): {e}")
                continue
        
        return results
        return results
    
    def train_xgboost_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series, feature_cols: List[str]):
        """Train XGBoost with hyperparameter tuning"""
        common.print_section("Training XGBoost Models")
        
        # Hyperparameter grid
        param_combinations = [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
        ]
        
        results = []
        
        for params in param_combinations:
            run_name = f"XGBoost_n{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}"
            
            with mlflow.start_run(run_name=run_name):
                # Train model
                model = xgb.XGBRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    random_state=common.RANDOM_STATE,
                    verbosity=0
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_metrics = common.calculate_metrics(y_train, train_pred)
                test_metrics = common.calculate_metrics(y_test, test_pred)
                
                # Log parameters
                mlflow.log_param("model_type", "advanced")
                mlflow.log_param("algorithm", "xgboost")
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("random_state", common.RANDOM_STATE)
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value)
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)
                
                # Log model
                mlflow.xgboost.log_model(
                    model,
                    "model", 
                    input_example=X_test.head(3)
                )
                
                run_info = {
                    'run_id': mlflow.active_run().info.run_id,
                    'model_name': f'XGBoost ({params["n_estimators"]}, {params["max_depth"]}, {params["learning_rate"]})',
                    'model_type': 'advanced',
                    'algorithm': 'xgboost',
                    **params,
                    **test_metrics
                }
                results.append(run_info)
                
                print(f"   {run_info['model_name']} - {common.format_metrics(test_metrics)}")
        
        return results
    
    def train_lightgbm_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series, feature_cols: List[str]):
        """Train LightGBM with hyperparameter tuning"""
        common.print_section("Training LightGBM Models")
        
        # Hyperparameter grid
        param_combinations = [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
        ]
        
        results = []
        
        for params in param_combinations:
            run_name = f"LightGBM_n{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}"
            
            with mlflow.start_run(run_name=run_name):
                # Train model
                model = lgb.LGBMRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    random_state=common.RANDOM_STATE,
                    verbosity=-1
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_metrics = common.calculate_metrics(y_train, train_pred)
                test_metrics = common.calculate_metrics(y_test, test_pred)
                
                # Log parameters
                mlflow.log_param("model_type", "advanced")
                mlflow.log_param("algorithm", "lightgbm")
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("random_state", common.RANDOM_STATE)
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value)
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)
                
                # Log model
                mlflow.lightgbm.log_model(
                    model,
                    "model",
                    input_example=X_test.head(3)
                )
                
                run_info = {
                    'run_id': mlflow.active_run().info.run_id,
                    'model_name': f'LightGBM ({params["n_estimators"]}, {params["max_depth"]}, {params["learning_rate"]})',
                    'model_type': 'advanced',
                    'algorithm': 'lightgbm',
                    **params,
                    **test_metrics
                }
                results.append(run_info)
                
                print(f"   {run_info['model_name']} - {common.format_metrics(test_metrics)}")
        
        return results
    
    def compare_and_find_best_run(self, all_results: List[Dict]):
        """Compare all runs and identify the best performing model"""
        common.print_section("Model Comparison and Best Run Selection")
        
        # Convert to DataFrame for easy sorting
        results_df = pd.DataFrame(all_results)
        
        # Sort by RMSE (lower is better) 
        results_df = results_df.sort_values('rmse')
        
        # Display results
        print("\nTop 10 Models by RMSE:")
        print("-" * 100)
        for i, row in results_df.head(10).iterrows():
            print(f"{i+1:2d}. {row['model_name']:30s} - {common.format_metrics(row)}")
        
        # Best run
        best_run = results_df.iloc[0]
        self.best_run_info = best_run.to_dict()
        
        common.print_success(f"Best Run: {best_run['model_name']}")
        print(f"    Run ID: {best_run['run_id']}")
        print(f"    Performance: {common.format_metrics(best_run)}")
        
        # Save comparison results
        comparison_file = f"{common.EXPERIMENT_DIR}/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(comparison_file, index=False)
        common.print_success(f"Comparison results saved: {comparison_file}")
        
        return results_df, best_run
    
    def log_training_summary(self):
        """Log final training summary"""
        with mlflow.start_run(run_name="Training_Summary"):
            # Log environment info
            env_info = common.log_environment_info()
            for key, value in env_info.items():
                if isinstance(value, (str, int, float)):
                    mlflow.log_param(f"env_{key}", value)
            
            # Log best run info
            if self.best_run_info:
                mlflow.log_param("best_run_id", self.best_run_info['run_id'])
                mlflow.log_param("best_model", self.best_run_info['model_name'])
                mlflow.log_param("best_algorithm", self.best_run_info.get('algorithm', 'unknown'))
                
                # Log best performance
                for metric in ['rmse', 'mae', 'r2', 'mape']:
                    if metric in self.best_run_info:
                        mlflow.log_metric(f"best_{metric}", self.best_run_info[metric])
            
            # Log training configuration
            mlflow.log_param("experiment_name", common.EXPERIMENT_NAME)
            mlflow.log_param("random_state", common.RANDOM_STATE)
            mlflow.log_param("test_size", common.TEST_SIZE)
            mlflow.log_param("ticker", common.TICKER)
            mlflow.log_param("training_date", datetime.now().isoformat())
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        common.print_header("BBCA Stock Forecasting - Training Pipeline")
        
        try:
            # 1. Load and validate data
            common.print_section("Loading Data")
            train_data, test_data, feature_metadata = common.load_preprocessed_data()
            
            if not common.validate_data_quality(train_data, test_data):
                raise ValueError("Data quality validation failed")
            
            # 2. Prepare features and targets
            X_train, X_test, y_train, y_test, feature_cols = common.prepare_features_and_targets(
                train_data, test_data, feature_metadata
            )
            
            # 3. Get original prices for baseline models
            train_prices, test_prices = common.get_original_prices(train_data, test_data)
            
            # 4. Train all models
            all_results = []
            
            # Baseline models
            baseline_results = self.train_baseline_models(train_prices, y_test, test_data.index)
            all_results.extend(baseline_results)
            
            # Advanced models
            ridge_results = self.train_ridge_regression(X_train, X_test, y_train, y_test, feature_cols)
            all_results.extend(ridge_results)
            
            xgb_results = self.train_xgboost_models(X_train, X_test, y_train, y_test, feature_cols)
            all_results.extend(xgb_results)
            
            lgb_results = self.train_lightgbm_models(X_train, X_test, y_train, y_test, feature_cols)
            all_results.extend(lgb_results)
            
            # 5. Compare results and find best run
            results_df, best_run = self.compare_and_find_best_run(all_results)
            
            # 6. Log training summary
            self.log_training_summary()
            
            # 7. Display final summary
            common.print_header("Training Completed Successfully")
            print(f"Total Runs: {len(all_results)}")
            print(f"Best Model: {best_run['model_name']}")
            print(f"Best Performance: {common.format_metrics(best_run)}")
            print(f"Best Run ID: {best_run['run_id']}")
            print(f"Experiment: {common.EXPERIMENT_NAME}")
            print(f"MLflow UI: http://localhost:5000")
            print()
            print("Next steps:")
            print("  1. Run model registration: python -m src.register_model")
            print("  2. View results in MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")
            
            return results_df
            
        except Exception as e:
            common.print_error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    forecaster = StockForecaster()
    results = forecaster.run_training_pipeline()
    
    # Display summary table
    common.print_header("Training Summary")
    print(results[['model_name', 'model_type', 'rmse', 'r2', 'mae', 'mape']].to_string(index=False))

if __name__ == "__main__":
    main()