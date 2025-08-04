import os
import warnings
import subprocess

from src import common

# Configuration
warnings.filterwarnings('ignore')

def run_command(command, description):
    """Run a command and handle results"""
    common.print_section(f"Running: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            common.print_success(f"{description} completed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout)  # Show full output
            return True
        else:
            common.print_error(f"{description} failed")
            if result.stderr:
                print("Error:")
                print(result.stderr[-500:])
            return False
            
    except Exception as e:
        common.print_error(f"{description} failed with exception: {e}")
        return False

def test_data_availability():
    """Test if preprocessed data is available"""
    common.print_section("Checking Data Availability")
    
    required_files = [
        f"{common.PREPROCESSED_DATA_DIR}/bbca_train.csv",
        f"{common.PREPROCESSED_DATA_DIR}/bbca_test.csv", 
        f"{common.PREPROCESSED_DATA_DIR}/feature_metadata.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        common.print_error("Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run the EDA notebook first to generate preprocessed data.")
        return False
    
    common.print_success("All required data files found")
    return True

def test_training():
    """Test the training pipeline"""
    return run_command(
        "python -m src.train",
        "Training Pipeline"
    )

def test_registration():
    """Test model registration"""
    return run_command(
        "python -m src.register",
        "Model Registration"
    )

def test_batch_scoring():
    """Test batch scoring"""
    return run_command(
        "python -m src.batch_score",
        "Batch Scoring"
    )

def test_single_prediction():
    """Test single prediction"""
    return run_command(
        "python -m src.predict --mode single",
        "Single Prediction"
    )

def test_model_listing():
    """Test model listing"""
    return run_command(
        "python -m src.predict --mode list",
        "Model Listing"
    )

def run_full_pipeline_test():
    """Run the complete pipeline test"""
    common.print_header("BBCA Stock Forecasting - Pipeline Test")
    
    test_results = {}
    
    # Test 1: Data availability
    test_results['data'] = test_data_availability()
    if not test_results['data']:
        common.print_error("Cannot proceed without data. Exiting.")
        return test_results
    
    # Test 2: Training
    test_results['training'] = test_training()
    
    # Test 3: Registration (only if training succeeded)
    if test_results['training']:
        test_results['registration'] = test_registration()
    else:
        common.print_warning("Skipping registration due to training failure")
        test_results['registration'] = False
    
    # Test 4: Batch scoring (only if registration succeeded)
    if test_results['registration']:
        test_results['batch_scoring'] = test_batch_scoring()
    else:
        common.print_warning("Skipping batch scoring due to registration failure")
        test_results['batch_scoring'] = False
    
    # Test 5: Single prediction (can work with local fallback)
    test_results['single_prediction'] = test_single_prediction()
    
    # Test 6: Model listing
    test_results['model_listing'] = test_model_listing()
    
    # Summary
    common.print_header("Pipeline Test Summary")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    print("Detailed Results:")
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    if passed_tests == total_tests:
        common.print_success("All tests passed! Pipeline is working correctly.")
        print("\nNext steps:")
        print("  1. View experiments: mlflow ui --host 0.0.0.0 --port 5000")
        print("  2. Check registered models in MLflow UI")
        print("  3. Run production predictions")
    else:
        common.print_warning("⚠️ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("  1. Ensure preprocessed data exists (run EDA notebook)")
        print("  2. Check MLflow is running properly")
        print("  3. Verify Python environment has all dependencies")
    
    return test_results

def main():
    """Main execution function"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Test the complete MLOps pipeline')
    parser.add_argument('--step', choices=['data', 'train', 'register', 'score', 'predict', 'list', 'all'],
                       default='all', help='Test specific step or all steps')
    
    args = parser.parse_args()
    
    if args.step == 'all':
        results = run_full_pipeline_test()
    elif args.step == 'data':
        test_data_availability()
    elif args.step == 'train':
        test_training()
    elif args.step == 'register':
        test_registration()
    elif args.step == 'score':
        test_batch_scoring()
    elif args.step == 'predict':
        test_single_prediction()
    elif args.step == 'list':
        test_model_listing()

if __name__ == "__main__":
    main()