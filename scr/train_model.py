import sys
import os
from tensorflow.keras import models, layers
from scr.data_preprocessing import load_and_preprocess_data  # Import from 'scr' folder if it's named correctly
from scr.training_utils import compile_and_train, plot_training_curves  # Import from 'scr' folder
from scr.model_architectures import build_basic_cnn  # Import the CNN model architecture

# Ensure the root directory is added to the Python path (if needed for 'scr' folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scr')))

# Define file paths for your training and test data
train_path = r'C:\Users\chint\OneDrive\Desktop\build an ai that sees\data\train.csv'  # Update to your actual train data path
test_path = r'C:\Users\chint\OneDrive\Desktop\build an ai that sees\data\test.csv'    # Update to your actual test data path

# Load and preprocess the data
X_train, X_val, y_train, y_val, test_data = load_and_preprocess_data(train_path, test_path)

# Define and compile the CNN model
model = build_basic_cnn()  # Use the function to create a basic CNN model

# Train the model
history = compile_and_train(
    model, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    epochs=10, 
    batch_size=64, 
    model_path=r'C:\Users\chint\OneDrive\Desktop\build an ai that sees\models\model.h5', 
    history_path=r'C:\Users\chint\OneDrive\Desktop\build an ai that sees\models\history.pkl'
)

# Plot and save the training curves (accuracy and loss)
plot_training_curves(
    history, 
    save_path=r'C:\Users\chint\OneDrive\Desktop\build an ai that sees\results\training_curves.png'
)

print("Model training complete. Training curves saved.")
