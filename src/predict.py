import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src import common

# Configuration
warnings.filterwarnings('ignore')

class StockPredictor:
    """Clean stock price prediction class with MLflow integration"""
    
    def __init__(self, model_uri: Optional[str] = None, use_local_fallback: bool = True):
        self.model = None
        self.model_info = {}
        self.feature_metadata = None
        self.model_uri = model_uri or common.MODEL_URI
        self.use_local_fallback = use_local_fallback
        self.client = MlflowClient()
        
        # Load model and metadata
        self.load_model()
    
    def load_model(self):
        """Load model with fallback mechanism"""
        common.print_section("Loading Model")
        
        success = False
        
        # Try loading from registry first
        if self.model_uri.startswith("models:/"):
            success = self.load_from_registry()
        
        # Fallback to local model if needed
        if not success and self.use_local_fallback:
            common.print_warning("Registry loading failed, trying local fallback")
            success = self.load_from_local()
        
        if not success:
            raise RuntimeError("Failed to load model from both registry and local sources")
        
        # Load feature metadata
        self.load_feature_metadata()
    
    def load_from_registry(self) -> bool:
        """Load model from MLflow Model Registry"""
        try:
            print(f"   Loading from registry: {self.model_uri}")
            
            # Load model
            self.model = mlflow.sklearn.load_model(self.model_uri)
            
            # Get model information
            self.model_info = self.get_registry_model_info()
            
            common.print_success(f"Model loaded from registry")
            print(f"   Model: {self.model_info.get('algorithm', 'unknown')}")
            print(f"   Version: {self.model_info.get('version', 'unknown')}")
            print(f"   Stage: {self.model_info.get('stage', 'unknown')}")
            
            return True
            
        except Exception as e:
            common.print_warning(f"Registry loading failed: {e}")
            return False
    
    def load_from_local(self) -> bool:
        """Load model from local files"""
        try:
            print("   Loading from local files...")
            
            # Default local paths
            model_path = f"{common.MODEL_DIR}/best_model_ridge.pkl"
            metadata_path = f"{common.MODEL_DIR}/model_metadata.json"
            
            if not os.path.exists(model_path):
                common.print_warning(f"Local model not found: {model_path}")
                return False
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata if available
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_info = json.load(f)
            else:
                self.model_info = {
                    'model_name': 'ridge',
                    'source': 'local_file',
                    'model_path': model_path
                }
            
            common.print_success("Model loaded from local files")
            print(f"   Model: {self.model_info.get('model_name', 'unknown')}")
            print(f"   Source: local file")
            
            return True
            
        except Exception as e:
            common.print_warning(f"Local loading failed: {e}")
            return False
    
    def get_registry_model_info(self) -> Dict:
        """Get comprehensive model information from registry"""
        try:
            # Parse model URI
            parts = self.model_uri.replace("models:/", "").split("/")
            model_name = parts[0]
            stage_or_version = parts[1] if len(parts) > 1 else "latest"
            
            # Get model version
            if stage_or_version.isdigit():
                model_version = self.client.get_model_version(model_name, stage_or_version)
            else:
                versions = self.client.get_latest_versions(model_name, stages=[stage_or_version])
                model_version = versions[0] if versions else None
            
            if not model_version:
                return {}
            
            # Get run information
            run_info = self.client.get_run(model_version.run_id)
            
            return {
                'model_name': model_name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': model_version.run_id,
                'algorithm': run_info.data.params.get('algorithm', 'unknown'),
                'creation_time': model_version.creation_timestamp,
                'description': model_version.description,
                'source': 'registry',
                'metrics': dict(run_info.data.metrics)
            }
            
        except Exception as e:
            common.print_warning(f"Could not get registry model info: {e}")
            return {'source': 'registry', 'error': str(e)}
    
    def load_feature_metadata(self):
        """Load feature metadata for data preparation"""
        try:
            metadata_path = f"{common.PREPROCESSED_DATA_DIR}/feature_metadata.json"
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.feature_metadata = json.load(f)
                common.print_success("Feature metadata loaded")
            else:
                # Create basic metadata
                self.feature_metadata = {
                    'target_columns': ['Target_Close_Next_Day', 'Target_Return_Next_Day', 'Target_Direction']
                }
                common.print_warning("Using default feature metadata")
                
        except Exception as e:
            common.print_warning(f"Could not load feature metadata: {e}")
            self.feature_metadata = {
                'target_columns': ['Target_Close_Next_Day', 'Target_Return_Next_Day', 'Target_Direction']
            }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        target_cols = self.feature_metadata['target_columns']
        feature_cols = [col for col in data.columns if col not in target_cols]
        return data[feature_cols]
    
    def predict_single(self, data: pd.DataFrame, confidence_level: float = 0.95) -> Dict:
        """Make single prediction with confidence interval"""
        common.print_section("Single Prediction")
        
        try:
            # Prepare features
            X = self.prepare_features(data.tail(1))
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Estimate confidence interval using model metrics
            if 'metrics' in self.model_info and 'test_rmse' in self.model_info['metrics']:
                rmse = self.model_info['metrics']['test_rmse']
            else:
                rmse = prediction * 0.02  # 2% default uncertainty
            
            # Calculate confidence interval
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            margin = z_score * rmse
            
            result = {
                'prediction': float(prediction),
                'confidence_level': confidence_level,
                'lower_bound': float(prediction - margin),
                'upper_bound': float(prediction + margin),
                'margin_of_error': float(margin),
                'prediction_date': datetime.now().isoformat(),
                'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'model_info': self.model_info
            }
            
            print(f"   Prediction: {prediction:.2f} IDR")
            print(f"   {confidence_level*100:.0f}% Confidence Interval: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}] IDR")
            print(f"   Target Date: {result['target_date']}")
            
            return result
            
        except Exception as e:
            common.print_error(f"Single prediction failed: {e}")
            raise
    
    def predict_batch(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Make batch predictions with comprehensive metrics"""
        common.print_section("Batch Prediction")
        
        try:
            # Prepare features and targets
            X = self.prepare_features(data)
            y_true = data['Target_Close_Next_Day']
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Calculate metrics
            metrics = common.calculate_metrics(y_true, predictions)
            
            # Additional statistics
            residuals = y_true - predictions
            pred_stats = {
                'n_samples': len(predictions),
                'mean_prediction': float(np.mean(predictions)),
                'std_prediction': float(np.std(predictions)),
                'min_prediction': float(np.min(predictions)),
                'max_prediction': float(np.max(predictions)),
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'prediction_date': datetime.now().isoformat()
            }
            
            # Combine metrics
            result = {**metrics, **pred_stats}
            
            print(f"   Samples: {len(predictions)}")
            print(f"   Performance: {common.format_metrics(metrics)}")
            print(f"   Prediction Range: {result['min_prediction']:.2f} - {result['max_prediction']:.2f} IDR")
            
            return predictions, result
            
        except Exception as e:
            common.print_error(f"Batch prediction failed: {e}")
            raise
    
    def log_prediction(self, prediction_type: str, result: Dict):
        """Log prediction results to MLflow"""
        try:
            with mlflow.start_run(run_name=f"Prediction_{prediction_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log model info
                mlflow.log_param("model_uri", self.model_uri)
                mlflow.log_param("prediction_type", prediction_type)
                mlflow.log_param("model_source", self.model_info.get('source', 'unknown'))
                
                if 'model_name' in self.model_info:
                    mlflow.log_param("model_name", self.model_info['model_name'])
                if 'version' in self.model_info:
                    mlflow.log_param("model_version", self.model_info['version'])
                
                # Log prediction metrics
                for key, value in result.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        mlflow.log_metric(key, value)
                    elif isinstance(value, str) and key.endswith('_date'):
                        mlflow.log_param(key, value)
                
                common.print_success("Prediction logged to MLflow")
                
        except Exception as e:
            common.print_warning(f"Failed to log prediction: {e}")
    
    def run_single_prediction(self, data_path: Optional[str] = None, log_results: bool = True):
        """Run single prediction pipeline"""
        common.print_header("Single Stock Price Prediction")
        
        try:
            # Load data
            if data_path:
                data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
            else:
                # Use latest test data
                _, test_data, _ = common.load_preprocessed_data()
                data = test_data
            
            print(f"Using data from: {data.index[-1].strftime('%Y-%m-%d')}")
            
            # Make prediction
            result = self.predict_single(data)
            
            # Log results
            if log_results:
                self.log_prediction("single", result)
            
            return result
            
        except Exception as e:
            common.print_error(f"Single prediction pipeline failed: {e}")
            raise
    
    def run_batch_evaluation(self, log_results: bool = True):
        """Run batch evaluation pipeline"""
        common.print_header("Batch Model Evaluation")
        
        try:
            # Load test data
            _, test_data, _ = common.load_preprocessed_data()
            
            print(f"Evaluating on {len(test_data)} samples")
            print(f"Date range: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
            
            # Make predictions
            predictions, result = self.predict_batch(test_data)
            
            # Log results
            if log_results:
                self.log_prediction("batch", result)
            
            return predictions, result
            
        except Exception as e:
            common.print_error(f"Batch evaluation pipeline failed: {e}")
            raise
    
    def list_available_models(self):
        """List available models in registry"""
        common.print_header("Available Models in Registry")
        
        try:
            model_name = common.MODEL_NAME
            
            # Get registered model
            try:
                model = self.client.get_registered_model(model_name)
                print(f"Model: {model.name}")
                print(f"Description: {model.description or 'No description'}")
                print()
                
                # List versions by stage
                stages = ['Production', 'Staging', 'None']
                for stage in stages:
                    try:
                        versions = self.client.get_latest_versions(model_name, stages=[stage])
                        if versions:
                            print(f"{stage}:")
                            for version in versions:
                                print(f"   Version: {version.version}")
                                print(f"   Status: {version.status}")
                                print(f"   Created: {version.creation_timestamp}")
                                if version.description:
                                    print(f"   Description: {version.description}")
                                print()
                    except Exception:
                        continue
                        
            except Exception as e:
                common.print_warning(f"Model '{model_name}' not found in registry: {e}")
                print("Available registered models:")
                
                # List all registered models
                try:
                    models = self.client.search_registered_models()
                    if models:
                        for model in models:
                            print(f"   - {model.name}")
                    else:
                        print("   No registered models found")
                except Exception as e2:
                    common.print_error(f"Failed to list models: {e2}")
                    
        except Exception as e:
            common.print_error(f"Error listing models: {e}")

def main():
    """Main execution function"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='BBCA Stock Price Prediction')
    parser.add_argument('--mode', choices=['single', 'batch', 'list'], default='single',
                       help='Prediction mode (default: single)')
    parser.add_argument('--model_uri', default=common.MODEL_URI,
                       help=f'Model URI (default: {common.MODEL_URI})')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file for prediction')
    parser.add_argument('--no_log', action='store_true',
                       help='Do not log results to MLflow')
    parser.add_argument('--no_local_fallback', action='store_true',
                       help='Do not use local model fallback')
    
    args = parser.parse_args()
    
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Handle list mode separately
    if args.mode == 'list':
        predictor = StockPredictor(use_local_fallback=False)
        predictor.list_available_models()
        return
    
    # Initialize predictor
    predictor = StockPredictor(
        model_uri=args.model_uri,
        use_local_fallback=not args.no_local_fallback
    )
    
    # Run prediction
    if args.mode == 'single':
        result = predictor.run_single_prediction(
            data_path=args.data_path,
            log_results=not args.no_log
        )
        
        # Display summary
        common.print_header("Prediction Summary")
        print(f"Predicted Price: {result['prediction']:.2f} IDR")
        print(f"Target Date: {result['target_date']}")
        print(f"Confidence Interval ({result['confidence_level']*100:.0f}%): "
              f"[{result['lower_bound']:.2f}, {result['upper_bound']:.2f}] IDR")
        print(f"Margin of Error: ±{result['margin_of_error']:.2f} IDR")
        
    elif args.mode == 'batch':
        predictions, result = predictor.run_batch_evaluation(
            log_results=not args.no_log
        )
        
        # Display summary
        common.print_header("Evaluation Summary")
        print(f"Samples Evaluated: {result['n_samples']}")
        print(f"Performance: {common.format_metrics(result)}")
        print(f"Prediction Range: {result['min_prediction']:.2f} - {result['max_prediction']:.2f} IDR")
        print(f"Mean Residual: {result['mean_residual']:.2f} IDR")

if __name__ == "__main__":
    main()