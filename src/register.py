import mlflow
from mlflow.tracking import MlflowClient
from src import common

def register_best_model():
    """Register the best model from training runs"""
    print("Model Registration")
    print("=" * 75)
    
    client = MlflowClient()
    
    try:
        # 1. Find experiment
        experiment = mlflow.get_experiment_by_name(common.EXPERIMENT_NAME)
        if not experiment:
            print(f"Experiment '{common.EXPERIMENT_NAME}' not found")
            return None
        
        print(f"✓ Found experiment: {common.EXPERIMENT_NAME}")
        
        # 2. Get all runs and find best
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=50
        )
        
        # Filter runs with test_rmse and get best
        valid_runs = runs.dropna(subset=['metrics.test_rmse'])
        if valid_runs.empty:
            print("No runs with test_rmse found")
            return None
            
        best_run = valid_runs.loc[valid_runs['metrics.test_rmse'].idxmin()]
        run_id = best_run['run_id']
        rmse = best_run['metrics.test_rmse']
        r2 = best_run.get('metrics.test_r2', 0)
        algorithm = best_run.get('params.algorithm', 'unknown')
        
        print(f"Best run: {run_id}")
        print(f"   Algorithm: {algorithm}")
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 3. Register model
        model_uri = f"runs:/{run_id}/model"
        
        print(f"Registering model: {model_uri}")
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=common.MODEL_NAME
        )
        
        print(f"Model registered as version {model_version.version}")
        
        # 4. Set aliases based on performance
        try:
            if r2 > 0.7:  # Staging criteria
                print("Setting 'staging' alias...")
                client.set_registered_model_alias(
                    name=common.MODEL_NAME,
                    alias="staging",
                    version=model_version.version
                )
                print("'staging' alias set successfully")
                
                if r2 > 0.8:  # Production criteria
                    print("Setting 'production' alias...")
                    client.set_registered_model_alias(
                        name=common.MODEL_NAME,
                        alias="production",
                        version=model_version.version
                    )
                    print("'production' alias set successfully")
                    print("Model promoted to production!")
                else:
                    print("Model in staging (R² < 0.8 for production)")
            else:
                print(f"Model R² ({r2:.4f}) below staging threshold (0.7)")
                
        except Exception as alias_error:
            print(f"Alias setting failed: {alias_error}")
            print("Model is registered but aliases failed")
        
        # 5. Display summary
        print("\n" + "=" * 75)
        print("REGISTRATION SUMMARY")
        print("=" * 75)
        print(f"Model: {common.MODEL_NAME}")
        print(f"Version: {model_version.version}")
        print(f"Run ID: {run_id}")
        print(f"Algorithm: {algorithm}")
        print(f"Performance: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # Show available URIs
        print(f"\nAvailable Model URIs:")
        print(f"   Latest: models:/{common.MODEL_NAME}/latest")
        if r2 > 0.7:
            print(f"   Staging: models:/{common.MODEL_NAME}@staging")
        if r2 > 0.8:
            print(f"   Production: models:/{common.MODEL_NAME}@production")
        print(f"   Specific: models:/{common.MODEL_NAME}/{model_version.version}")
        
        print("\nNext steps:")
        print("  1. python -m src.batch_score")
        print("  2. python -m src.predict --mode single")
        
        return model_version
        
    except Exception as e:
        print(f"Registration failed: {e}")
        return None

def main():
    """Main execution function"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Register best model from training runs')
    parser.add_argument('--experiment_name', default=common.EXPERIMENT_NAME,
                       help=f'Experiment name (default: {common.EXPERIMENT_NAME})')
    parser.add_argument('--model_name', default=common.MODEL_NAME,
                       help=f'Model name (default: {common.MODEL_NAME})')
    
    args = parser.parse_args()
    
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    register_best_model()

if __name__ == "__main__":
    main() 