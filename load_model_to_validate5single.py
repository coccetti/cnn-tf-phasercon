#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Validation Script

This script loads a trained CNN model and validates it on test data.
It provides functionality to load saved models, evaluate their performance,
and generate validation metrics and visualizations.
"""

# Standard library imports
import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

# Constants
USE_GPU = False  # Set to False to use CPU instead of GPU

# Define the neural network architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 30 * 30, 64)  # Adjusted input size for fully connected layer
        self.fc2 = nn.Linear(64, 14)  # 14 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path):
    """
    Load a saved model and its metadata.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        tuple: (model, metadata) containing the loaded model and its metadata
    """
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load the model
        device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize the model
        model = CNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'accuracy': checkpoint.get('accuracy'),
            'num_runs': checkpoint.get('num_runs'),
            'timestamp': checkpoint.get('timestamp')
        }
        
        return model, metadata
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def validate_model(model, test_data, device):
    """
    Validate the model on test data.
    
    Args:
        model: The trained model
        test_data: Test dataset
        device: Device to run validation on
        
    Returns:
        dict: Validation metrics
    """
    # Implementation here
    pass

def predict_file(model, file_path, device):
    """
    Predict the category of a file using the loaded model.
    
    Args:
        model: The trained model
        file_path (str): Path to the file to categorize
        device: Device to run prediction on
        
    Returns:
        str: Predicted category
    """
    try:
        # Load and preprocess the file
        input_data = load_and_preprocess_file(file_path)
        
        # Move data to device
        input_data = input_data.to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(input_data)
            prediction = torch.argmax(output, dim=1)
            
        # Get the class name from the prediction
        classes = ["input_mask_blank_screen", 
                  "input_mask_horizontal_division_A", "input_mask_horizontal_division_B", 
                  "input_mask_vertical_division_A", "input_mask_vertical_division_B",
                  "input_mask_checkboard_1A", "input_mask_checkboard_1B", "input_mask_checkboard_1R", 
                  "input_mask_checkboard_2A", "input_mask_checkboard_2B",
                  "input_mask_checkboard_3A", "input_mask_checkboard_3B",
                  "input_mask_checkboard_4A", "input_mask_checkboard_4B"]
        
        predicted_class = classes[prediction.item()]
        return predicted_class
        
    except Exception as e:
        logging.error(f"Error predicting file: {str(e)}")
        raise

def load_and_preprocess_file(file_path):
    """
    Load and preprocess a file for model prediction.
    
    Args:
        file_path (str): Path to the .npy file to process
        
    Returns:
        torch.Tensor: Preprocessed data ready for model input
    """
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Resize to 128x128 as in the training script
        data = cv2.resize(data, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Normalize values to be between 0 and 1
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Convert to PyTorch tensor and add channel dimension
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Add batch dimension
        data = data.unsqueeze(0)  # Shape: [1, 1, 128, 128]
        
        return data
        
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        raise

def main():
    """
    Main function to run model validation.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get the model path
    # model_path = os.path.join('models', 'model_num_runs_30_acc_90.95_20250507_232243.pth')
    # model_path = os.path.join('models', 'model_num_runs_250_acc_95.14_20250507_234601.pth')
    model_path = os.path.join('models', 'model_num_runs_230_acc_99.38_20250508_003116.pth')
    
    try:
        # Load the model
        logging.info(f"Loading model from {model_path}")
        model, metadata = load_model(model_path)
        
        # Print model metadata
        logging.info("Model loaded successfully!")
        logging.info("Model metadata:")
        for key, value in metadata.items():
            if value is not None:
                logging.info(f"{key}: {value}")
        
        # Prompt for file path
        while True:
            file_path = input("\nEnter the path to the file to categorize (or 'q' to quit): ")
            if file_path.lower() == 'q':
                break
                
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                continue
                
            try:
                # Get device
                device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
                
                # Make prediction
                prediction = predict_file(model, file_path, device)
                logging.info(f"Predicted category: {prediction}")
                
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
