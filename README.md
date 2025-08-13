# Build-an-AI-that-sees-final

## Build an AI That Sees

A simple image classification project using Convolutional Neural Networks (CNN) built with TensorFlow/Keras. It trains on image data from CSV files and generates predictions.


### 📁 Folder Structure


data/           # Contains train.csv, test.csv
models/         # Saved model and training history
results/        # Output images and predictions
scr/            # Source code files


### 🚀 How to Run

1. **Install dependencies**:
   pip install -r requirements.txt
   

2. **Train the model**:
   python train_model.py

3. **Make predictions**:
   python scr/prediction.py
   

### 📦 Output

* Trained model: `models/model.keras`
* Training curve: `results/training_curves.png`
