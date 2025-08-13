import sys
import os

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_preprocessing import load_and_preprocess_data
from model_architectures import build_basic_cnn
from training_utils import compile_and_train, plot_training_curves
from evaluation_metrics import evaluate_model
from prediction import predict_and_save

# Example paths for your CSV files
train_path = 'data/train.csv'
test_path = 'data/test.csv'

# Preprocess data
X_train, X_val, y_train, y_val, test_data = load_and_preprocess_data(train_path, test_path)

# Build and compile model
model = build_basic_cnn()
history = compile_and_train(model, X_train, y_train, X_val, y_val, epochs=10)

# Plot training curves
plot_training_curves(history)

# Evaluate model
evaluate_model(model, X_val, y_val)

# Make predictions and save results
output_path = 'results/predictions.csv'
predict_and_save('models/basic_cnn_model.h5', test_data, output_path)
