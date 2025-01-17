import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np

def plot_confusion_matrix(working_dir, y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f'confusion_matrix.png'))
    plt.show()

def plot_percent_confusion_matrix(working_dir, y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes, annot_kws={"size": 8})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout(pad=8.0)  # Increase padding between the plot and the labels
    plt.title(title)
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f'confusion_matrix_percent.png'))
    plt.show()