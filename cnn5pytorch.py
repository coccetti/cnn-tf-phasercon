# Description: This script reads measured phase data from a folder structure and resizes it to a given size. 
#              It then splits the data into training and testing sets, normalizes the pixel values, and uses 
#              a CNN to predict the classes of the data. The script loops over different values of the number 
#              of runs and saves the classification report to a file in the working directory.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from contextlib import redirect_stdout
import io
import cv2
import time
import logging
import shutil
from sklearn.metrics import classification_report

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")
    print(f"Using MPS device: {mps_device}")

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
for num_runs in range(300, 301, 10):
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
    for i in range(num_runs):
        RUN = 'M0000' + str(i+1)
        if i+1 > 9:
            RUN = 'M000' + str(i+1)
            if i+1 > 99:
                RUN = 'M00' + str(i+1)
        k = 0  # index for classes (varies from 0 to )
        for class_name in classes:
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

    # Split data between Train and Test, then Normalization
    train_set_percentage = 0.5  # Set training set percentage
    logging.info("\n### Splitting data ###")
    train_measured_phase, test_measured_phase = np.split(measured_phase, [int(frame_number * train_set_percentage)])
    train_reference_class_short, test_reference_class_short = np.split(reference_class_short, [int(frame_number * train_set_percentage)])
    logging.info(f"- train_measured_phase tensor shape: {train_measured_phase.shape}")
    logging.info(f"- test_measured_phase tensor shape: {test_measured_phase.shape}")
    logging.info(f"- train_reference_class_short tensor shape: {train_reference_class_short.shape}")
    logging.info(f"- test_reference_class_short tensor shape: {test_reference_class_short.shape}")

    # Normalize "pixel" values to be between 0 and 1
    train_measured_phase = (train_measured_phase - np.min(train_measured_phase)) / (np.max(train_measured_phase) - np.min(train_measured_phase))
    test_measured_phase = (test_measured_phase - np.min(test_measured_phase)) / (np.max(test_measured_phase) - np.min(test_measured_phase))

    # Convert numpy arrays to PyTorch tensors
    train_measured_phase = torch.tensor(train_measured_phase, dtype=torch.float32).unsqueeze(1)
    test_measured_phase = torch.tensor(test_measured_phase, dtype=torch.float32).unsqueeze(1)
    train_reference_class_short = torch.tensor(train_reference_class_short, dtype=torch.long).squeeze()
    test_reference_class_short = torch.tensor(test_reference_class_short, dtype=torch.long).squeeze()

    # Create DataLoader for batching
    train_dataset = TensorDataset(train_measured_phase, train_reference_class_short)
    test_dataset = TensorDataset(test_measured_phase, test_reference_class_short)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
    if torch.backends.mps.is_available():
        model.to(mps_device)
        logging.info(f"Using MPS device: {mps_device}")
        print(f"Using MPS device: {mps_device}")
    else:
        model.to(torch.device("cpu"))
        logging.info("Using CPU device")
        print("Using CPU device")   
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move data to the appropriate device
            inputs = inputs.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            labels = labels.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(train_loader)}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to the appropriate device
            inputs = inputs.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            labels = labels.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logging.info(f"Test Accuracy: {100 * correct / total}%")

    # Print classification report
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to the appropriate device
            inputs = inputs.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            labels = labels.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            
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

    # Record the end time for each loop
    end_time = time.time()
    logging.info("\nend_time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    logging.info(f"\nTime elapsed for num_runs = {num_runs}: {end_time - start_time} seconds")