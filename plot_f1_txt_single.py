# Description: This script reads a classification report file and plots the precision, recall, and F1-score for each class.
#

# Import the required libraries
import matplotlib.pyplot as plt
import re

# File path to the classification report
# file_path = 'data_f1/classification_report_num_runs_64_250.txt'
file_path = '/Users/fc/Downloads/cnn_phase5_1layer/ccn_phase_5_2024_12_17_03/classification_report_num_runs_64_dense128_250.txt'

# Initialize dictionaries to store metrics
precision = {}
recall = {}
f1_score = {}

# Function to extract metrics from a classification report file
def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.match(r'\s*(\S+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\d+', line)
            if match:
                class_name = match.group(1)
                precision[class_name] = float(match.group(2))
                recall[class_name] = float(match.group(3))
                f1_score[class_name] = float(match.group(4))

# Extract metrics from the file
extract_metrics(file_path)

# Plot the metrics for each class
classes = list(precision.keys())
precision_values = [precision[class_name] for class_name in classes]
recall_values = [recall[class_name] for class_name in classes]
f1_score_values = [f1_score[class_name] for class_name in classes]

plt.figure(figsize=(12, 8))
plt.plot(classes, precision_values, marker='^', linestyle='-.', color='blue', label='Precision')
plt.plot(classes, recall_values, marker='s', linestyle='--', color='green', label='Recall')
plt.plot(classes, f1_score_values, marker='o', linestyle='-', color='red', label='F1-Score')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()

# Save the plot to a file
output_file_path = file_path.replace('.txt', '.png')
plt.savefig(output_file_path)

# Show the plot
plt.show()