# Description: This script reads measured phase data from a folder structure and resizes it to a given size. 
#              It then splits the data into training and testing sets, normalizes the pixel values, and uses 
#              a CNN to predict the classes of the data. The script loops over different values of the number 
#              of runs and saves the classification report to a file in the working directory.
#              This script is the same as cnn5pytorch_cmatrix.py, but it splits the data into training, validation and test sets.
#              The validation set is used to monitor the performance of the model during training, and the test set is used to evaluate the final model.
#              The training set is used to train the model.
#              The test set is used to evaluate the final model.
#              The validation set is used to monitor the performance of the model during training.
#              The test set is used to evaluate the final model.
"""
To implement a proper train/validation/test split:
1. Data Splitting:
- Training set: 60% of the data
- Validation set: 20% of the data
- Test set: 20% of the data (remaining)
2. Training Loop Improvements:
- Added validation phase after each training epoch
- Implemented early stopping with a patience of 3 epochs
- Added tracking of the best model based on validation loss
- Added more detailed logging of both training and validation metrics
3. Model Evaluation:
- The model is now evaluated on the test set only after training is complete
- The best model state (based on validation performance) is loaded before final evaluation
The key benefits of these changes are:
- Better model selection through validation
- Prevention of overfitting through early stopping
- More reliable performance estimates through proper test set evaluation
- More detailed monitoring of model performance during training
The code:
- Train on the training set
- Validate on the validation set after each epoch
- Stop training if validation performance doesn't improve for 3 epochs
- Save the best model based on validation performance
- Finally evaluate on the test set
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from contextlib import redirect_stdout
import io
from tqdm import tqdm
import cv2
import time
import logging
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Device selection
USE_GPU = False  # Set to False to use CPU instead of GPU

# Check available devices and set up
if USE_GPU:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    else:
        device = torch.device("cpu")
        print("GPU requested but not available. Using CPU instead.")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Define the base working directory
base_working_dir = os.path.expanduser('~/Downloads/cnn_phase5_output')

# Add the current date to the working directory
current_time = time.strftime("%Y_%m_%d")
working_dir = os.path.join(base_working_dir, f'ccn_phase_5_{current_time}')
counter = 1 

# Check if the directory exists and create a new one if it does
while os.path.exists(working_dir):
    working_dir = os.path.join(base_working_dir, f'ccn_phase_5_{current_time}_{counter:02d}')
    counter += 1

os.makedirs(working_dir, exist_ok=True)
print(f"Working directory: {working_dir}")

# Create models directory
models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
print(f"Models directory: {models_dir}")

# Copy the current script file to the working directory
current_script = os.path.abspath(__file__)
shutil.copy(current_script, working_dir)

# Set up logging configuration
log_file = os.path.join(working_dir, 'cnn_phase5resizeloop.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Resize images parameters
x_resize = 128  # width
y_resize = 128  # high

# Data path
data_path = r'/Volumes/EXTERNAL_US/2024_06_12/'
# data_path = r'data3'

# Date of measurements
date=''
    
# Define the type of measure, so we can read the proper folder
data_type_measure = "SLM_Interferometer_alignment"

# Classes
classes=["input_mask_blank_screen", 
         "input_mask_horizontal_division_A", "input_mask_horizontal_division_B", 
         "input_mask_vertical_division_A", "input_mask_vertical_division_B",
         "input_mask_checkboard_1A", "input_mask_checkboard_1B", "input_mask_checkboard_1R", 
         "input_mask_checkboard_2A", "input_mask_checkboard_2B",
         "input_mask_checkboard_3A", "input_mask_checkboard_3B",
         "input_mask_checkboard_4A", "input_mask_checkboard_4B"]

# Loop over different nRUN values
for num_runs in range(230, 231, 10):
    start_time = time.time()  # Record the start time
    logging.info("\nstart_time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    logging.info(f"\nRunning with num_runs = {num_runs}")
    print(f"\nRunning with num_runs = {num_runs}")

    # Number of frames
    frame_number = num_runs * np.size(classes)

    # Log values
    logging.info("\n=================================================================")
    logging.info("Input values: ")
    logging.info(f" - Total number of RUNs: {num_runs}")
    logging.info(f" - Classes in words: {classes}")
    logging.info(f" - Classes in numbers: {np.arange(np.size(classes))}")
    logging.info(f" - Total number of classes: {np.size(classes)}")
    logging.info(f" - Total number of frames for measured data (number of RUNs * number of classes): {frame_number}")
    logging.info(f" - Data path: {data_path}")
    logging.info(f" - Date of measurements: {date}")
    logging.info(f" - Type of measure: {data_type_measure}")
    logging.info(f" - Working directory: {working_dir}")
    logging.info("=================================================================")

    # Determine high and width of the matrix reading the data
    logging.info("\n### Start of reading data ###")
    sample_measured_phase_full_path = os.path.join(data_path, date, data_type_measure, "M00001", classes[0], "files", "measured_phase.npy")
    sample_measured_phase = np.load(sample_measured_phase_full_path)
    high, width = sample_measured_phase.shape
    logging.info(f"\nLoaded sample phase with dimensions in px (high, width): {high}, {width}")

    # Load data and plot
    measured_phase = np.zeros((frame_number, high, width))
    reference_class_short = np.zeros((frame_number, 1), dtype=int)

    j = 0  # index for frame_number (varies from 0 to frame_number-1)
    for i in tqdm(range(num_runs), desc="Processing runs"):
        RUN = 'M0000' + str(i+1)
        if i+1 > 9:
            RUN = 'M000' + str(i+1)
            if i+1 > 99:
                RUN = 'M00' + str(i+1)
        k = 0  # index for classes (varies from 0 to )
        for class_name in tqdm(classes, desc=f"Processing classes for run {i+1}", leave=False):
            input_files_path = os.path.join(data_path, date, data_type_measure, RUN, class_name, "files", "measured_phase.npy")
            measured_phase[j,:,:] = np.load(input_files_path)
            reference_class_short[j] = k
            k += 1
            j += 1

    # Log arrays dimensions
    logging.info(f"- measured_phase tensor shape: {measured_phase.shape}")
    logging.info(f"- reference_class_short tensor shape: {reference_class_short.shape}")

    # Resize images
    logging.info("\n### Start of resizing ###")
    resize_measured_phase = np.zeros((frame_number, y_resize, x_resize))
    for ii in range(frame_number):
        resize_measured_phase[ii] = cv2.resize(measured_phase[ii], (y_resize, x_resize), interpolation=cv2.INTER_NEAREST)
    measured_phase = resize_measured_phase

    # Split data between Train, Validation and Test, then Normalization
    train_set_percentage = 0.6  # Set training set percentage
    val_set_percentage = 0.2    # Set validation set percentage
    # Test set will be the remaining 0.2 (1 - 0.6 - 0.2)
    
    logging.info("\n### Splitting data ###")
    # First split to get training set
    train_size = int(frame_number * train_set_percentage)
    train_measured_phase = measured_phase[:train_size]
    train_reference_class_short = reference_class_short[:train_size]
    
    # Second split to get validation and test sets
    remaining_data = measured_phase[train_size:]
    remaining_labels = reference_class_short[train_size:]
    val_size = int(len(remaining_data) * (val_set_percentage / (1 - train_set_percentage)))
    
    val_measured_phase = remaining_data[:val_size]
    val_reference_class_short = remaining_labels[:val_size]
    
    test_measured_phase = remaining_data[val_size:]
    test_reference_class_short = remaining_labels[val_size:]
    
    logging.info(f"- train_measured_phase tensor shape: {train_measured_phase.shape}")
    logging.info(f"- val_measured_phase tensor shape: {val_measured_phase.shape}")
    logging.info(f"- test_measured_phase tensor shape: {test_measured_phase.shape}")
    logging.info(f"- train_reference_class_short tensor shape: {train_reference_class_short.shape}")
    logging.info(f"- val_reference_class_short tensor shape: {val_reference_class_short.shape}")
    logging.info(f"- test_reference_class_short tensor shape: {test_reference_class_short.shape}")

    # Normalize "pixel" values to be between 0 and 1
    train_measured_phase = (train_measured_phase - np.min(train_measured_phase)) / (np.max(train_measured_phase) - np.min(train_measured_phase))
    val_measured_phase = (val_measured_phase - np.min(val_measured_phase)) / (np.max(val_measured_phase) - np.min(val_measured_phase))
    test_measured_phase = (test_measured_phase - np.min(test_measured_phase)) / (np.max(test_measured_phase) - np.min(test_measured_phase))

    # Convert numpy arrays to PyTorch tensors
    train_measured_phase = torch.tensor(train_measured_phase, dtype=torch.float32).unsqueeze(1)
    val_measured_phase = torch.tensor(val_measured_phase, dtype=torch.float32).unsqueeze(1)
    test_measured_phase = torch.tensor(test_measured_phase, dtype=torch.float32).unsqueeze(1)
    train_reference_class_short = torch.tensor(train_reference_class_short, dtype=torch.long).squeeze()
    val_reference_class_short = torch.tensor(val_reference_class_short, dtype=torch.long).squeeze()
    test_reference_class_short = torch.tensor(test_reference_class_short, dtype=torch.long).squeeze()

    # Create DataLoader for batching
    train_dataset = TensorDataset(train_measured_phase, train_reference_class_short)
    val_dataset = TensorDataset(val_measured_phase, val_reference_class_short)
    test_dataset = TensorDataset(test_measured_phase, test_reference_class_short)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the neural network architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 30 * 30, 64)  # Adjusted input size for fully connected layer
            self.fc2 = nn.Linear(64, len(classes))

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model, loss function, and optimizer
    model = CNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 10
    best_val_loss = float('inf')
    patience = 3  # Number of epochs to wait for improvement
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = running_loss/len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_acc = 100 * val_correct / val_total
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{val_acc:.2f}%'})
        
        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        logging.info(f"Epoch {epoch+1}/{n_epochs}")
        logging.info(f"Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model state based on validation loss")

    # Evaluate the model on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating on test set')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_acc = 100 * correct / total
            pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})
            
    accuracy = 100 * correct / total
    logging.info(f"Test Accuracy: {accuracy:.2f}%")

    # Save the model
    model_timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f'model_num_runs_{num_runs}_acc_{accuracy:.2f}_{model_timestamp}.pth'
    model_path = os.path.join(models_dir, model_filename)
    
    # Save model state and metadata
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'num_runs': num_runs,
        'device': str(device),
        'classes': classes,
        'timestamp': model_timestamp
    }, model_path)
    
    logging.info(f"Model saved to: {model_path}")
    print(f"Model saved to: {model_path}")

    # Print classification report
    y_true = []
    y_pred = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Generating classification report')
        for inputs, labels in pbar:
            # Move data to the appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    report = classification_report(y_true, y_pred, target_names=classes, digits=6, zero_division=0)
    logging.info("\n" + report)

    # Save classification report to file in working directory
    report_path = os.path.join(working_dir, f'classification_report_num_runs_{num_runs}.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(f'Confusion Matrix (num_runs = {num_runs})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = os.path.join(working_dir, f'confusion_matrix_num_runs_{num_runs}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Record the end time for each loop
    end_time = time.time()
    logging.info("\nend_time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    logging.info(f"\nTime elapsed for num_runs = {num_runs}: {end_time - start_time} seconds")