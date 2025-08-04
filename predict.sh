# Batch evaluation on test set
python predict.py --mode batch

# Single prediction for next day
python predict.py --mode single

# Use specific model
python predict.py --mode batch --model_path models/best_model_ridge.pkl