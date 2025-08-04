import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src import common

# Configuration
warnings.filterwarnings('ignore')

class BatchScorer:
    """Batch scoring class for model evaluation"""
    
    def __init__(self):
        self.client = MlflowClient()
        
    def load_model_by_uri(self, model_uri: str, flavor: str = "sklearn"):
        """Load model by URI with specified flavor"""
        common.print_section(f"Loading Model: {model_uri}")
        
        try:
            if flavor == "sklearn":
                model = mlflow.sklearn.load_model(model_uri)
                print(f"   ✓ Loaded sklearn model: {type(model)}")
            elif flavor == "pyfunc":
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"   ✓ Loaded pyfunc model: {type(model)}")
            else:
                raise ValueError(f"Unsupported flavor: {flavor}")
            
            return model
            
        except Exception as e:
            common.print_error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self, model_uri: str):
        """Get detailed model information"""
        try:
            # Parse model URI to extract model name and version/stage/alias
            if model_uri.startswith("models:/"):
                # Handle aliases (e.g., models:/model@production)
                if "@" in model_uri:
                    model_name = model_uri.replace("models:/", "").split("@")[0]
                    alias_name = model_uri.split("@")[1]
                    try:
                        model_version = self.client.get_model_version_by_alias(model_name, alias_name)
                    except Exception:
                        raise ValueError(f"No model found for alias: {alias_name}")
                else:
                    # Handle version or stage
                    parts = model_uri.replace("models:/", "").split("/")
                    model_name = parts[0]
                    stage_or_version = parts[1] if len(parts) > 1 else "latest"
                    
                    if stage_or_version.isdigit():
                        model_version = self.client.get_model_version(model_name, stage_or_version)
                    else:
                        # Try as stage (legacy)
                        try:
                            versions = self.client.get_latest_versions(model_name, stages=[stage_or_version])
                            if versions:
                                model_version = versions[0]
                            else:
                                raise ValueError(f"No model found for stage: {stage_or_version}")
                        except Exception:
                            raise ValueError(f"No model found for stage/alias: {stage_or_version}")
                
                # Get run information
                run_info = self.client.get_run(model_version.run_id)
                
                return {
                    'model_name': model_name,
                    'version': model_version.version,
                    'stage': model_version.current_stage,
                    'run_id': model_version.run_id,
                    'algorithm': run_info.data.params.get('algorithm', 'unknown'),
                    'metrics': run_info.data.metrics,
                    'description': model_version.description
                }
                
        except Exception as e:
            common.print_warning(f"Could not get model info: {e}")
            return {}
    
    def prepare_scoring_data(self):
        """Prepare data for scoring"""
        common.print_section("Preparing Scoring Data")
        
        # Load preprocessed data
        train_data, test_data, feature_metadata = common.load_preprocessed_data()
        
        # Prepare features
        X_train, X_test, y_train, y_test, feature_cols = common.prepare_features_and_targets(
            train_data, test_data, feature_metadata
        )
        
        common.print_success(f"Data prepared: {len(X_test)} samples, {len(feature_cols)} features")
        
        return X_train, X_test, y_train, y_test, test_data.index
    
    def score_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                   model_info: Dict, flavor: str = "sklearn"):
        """Score model and calculate metrics"""
        common.print_section(f"Scoring with {flavor.upper()} Flavor")
        
        try:
            # Make predictions
            if flavor == "sklearn":
                predictions = model.predict(X_test)
            elif flavor == "pyfunc":
                predictions = model.predict(X_test)
            else:
                raise ValueError(f"Unsupported flavor: {flavor}")
            
            # Calculate metrics
            metrics = common.calculate_metrics(y_test, predictions)
            
            # Display results
            print(f"   Model: {model_info.get('algorithm', 'unknown')}")
            print(f"   Version: {model_info.get('version', 'unknown')}")
            print(f"   Stage: {model_info.get('stage', 'unknown')}")
            print(f"   Performance: {common.format_metrics(metrics)}")
            
            # Additional statistics
            residuals = y_test - predictions
            print(f"   Prediction Range: {predictions.min():.2f} - {predictions.max():.2f}")
            print(f"   Mean Residual: {residuals.mean():.2f}")
            print(f"   Residual Std: {residuals.std():.2f}")
            
            return predictions, metrics
            
        except Exception as e:
            common.print_error(f"Scoring failed: {e}")
            raise
    
    def compare_flavors(self, model_uri: str, X_test: pd.DataFrame, y_test: pd.Series):
        """Compare sklearn and pyfunc flavors"""
        common.print_section("Comparing Model Flavors")
        
        model_info = self.get_model_info(model_uri)
        results = {}
        
        # Test sklearn flavor
        try:
            sklearn_model = self.load_model_by_uri(model_uri, "sklearn")
            sklearn_pred, sklearn_metrics = self.score_model(
                sklearn_model, X_test, y_test, model_info, "sklearn"
            )
            results['sklearn'] = {
                'predictions': sklearn_pred,
                'metrics': sklearn_metrics
            }
        except Exception as e:
            common.print_warning(f"sklearn flavor failed: {e}")
        
        # Test pyfunc flavor
        try:
            pyfunc_model = self.load_model_by_uri(model_uri, "pyfunc")
            pyfunc_pred, pyfunc_metrics = self.score_model(
                pyfunc_model, X_test, y_test, model_info, "pyfunc"
            )
            results['pyfunc'] = {
                'predictions': pyfunc_pred,
                'metrics': pyfunc_metrics
            }
        except Exception as e:
            common.print_warning(f"pyfunc flavor failed: {e}")
        
        # Compare results if both succeeded
        if 'sklearn' in results and 'pyfunc' in results:
            sklearn_pred = results['sklearn']['predictions']
            pyfunc_pred = results['pyfunc']['predictions']
            
            # Check if predictions are identical
            pred_diff = np.abs(sklearn_pred - pyfunc_pred).max()
            print(f"\n   Flavor Comparison:")
            print(f"   Max Prediction Difference: {pred_diff:.8f}")
            
            if pred_diff < 1e-6:
                common.print_success("Predictions are identical across flavors")
            else:
                common.print_warning("Predictions differ between flavors")
        
        return results
    
    def log_scoring_results(self, model_uri: str, results: Dict, model_info: Dict):
        """Log scoring results to MLflow"""
        with mlflow.start_run(run_name=f"Batch_Score_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log model info
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("model_name", model_info.get('model_name', 'unknown'))
            mlflow.log_param("model_version", model_info.get('version', 'unknown'))
            mlflow.log_param("model_stage", model_info.get('stage', 'unknown'))
            mlflow.log_param("source_run_id", model_info.get('run_id', 'unknown'))
            mlflow.log_param("scoring_date", datetime.now().isoformat())
            
            # Log metrics for each flavor
            for flavor, result in results.items():
                metrics = result['metrics']
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{flavor}_{metric_name}", value)
            
            common.print_success("Scoring results logged to MLflow")
    
    def save_predictions(self, predictions: np.ndarray, dates: pd.DatetimeIndex, 
                        y_true: pd.Series, model_info: Dict, output_dir: str = "results"):
        """Save predictions to file"""
        common.print_section("Saving Predictions")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': dates,
            'Actual': y_true.values,
            'Predicted': predictions,
            'Residual': y_true.values - predictions,
            'Abs_Error': np.abs(y_true.values - predictions),
            'Pct_Error': ((y_true.values - predictions) / y_true.values) * 100
        })
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = model_info.get('model_name', 'unknown')
        version = model_info.get('version', 'unknown')
        
        filename = f"{output_dir}/predictions_{model_name}_v{version}_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        
        common.print_success(f"Predictions saved: {filename}")
        return filename
    
    def run_batch_scoring(self, model_uri: str = None, save_results: bool = True):
        """Run complete batch scoring pipeline"""
        common.print_header("BBCA Stock Forecasting - Batch Scoring")
        
        # Use default model URI if not provided
        if model_uri is None:
            model_uri = common.MODEL_URI
        
        print(f"📍 Model URI: {model_uri}")
        
        try:
            # 1. Get model information
            model_info = self.get_model_info(model_uri)
            
            # 2. Prepare scoring data
            X_train, X_test, y_train, y_test, test_dates = self.prepare_scoring_data()
            
            # 3. Score with different flavors
            results = self.compare_flavors(model_uri, X_test, y_test)
            
            if not results:
                raise ValueError("No successful scoring results")
            
            # 4. Log results to MLflow
            self.log_scoring_results(model_uri, results, model_info)
            
            # 5. Save predictions if requested
            if save_results and 'sklearn' in results:
                predictions = results['sklearn']['predictions']
                self.save_predictions(predictions, test_dates, y_test, model_info)
            
            # 6. Display summary
            common.print_header("Batch Scoring Completed")
            print(f"📝 Model: {model_info.get('model_name', 'unknown')}")
            print(f"🔢 Version: {model_info.get('version', 'unknown')}")
            print(f"📍 Stage: {model_info.get('stage', 'unknown')}")
            print(f"🎯 Algorithm: {model_info.get('algorithm', 'unknown')}")
            print(f"📊 Samples Scored: {len(X_test)}")
            
            # Display best metrics
            best_flavor = 'sklearn' if 'sklearn' in results else list(results.keys())[0]
            best_metrics = results[best_flavor]['metrics']
            print(f"📈 Performance: {common.format_metrics(best_metrics)}")
            
            return results
            
        except Exception as e:
            common.print_error(f"Batch scoring failed: {e}")
            raise

def main():
    """Main execution function"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Batch score registered model')
    parser.add_argument('--model_uri', default="models:/bbca_forecasting_model@production",
                       help='Model URI (default: models:/bbca_forecasting_model@production)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save prediction results to file')
    
    args = parser.parse_args()
    
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    scorer = BatchScorer()
    results = scorer.run_batch_scoring(
        model_uri=args.model_uri,
        save_results=not args.no_save
    )
    
    # Display final summary
    print("\n" + "="*75)
    print("SCORING SUMMARY")
    print("="*75)
    
    for flavor, result in results.items():
        metrics = result['metrics']
        print(f"\n{flavor.upper()} Results:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()