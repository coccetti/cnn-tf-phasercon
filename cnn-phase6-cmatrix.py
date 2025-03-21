# Description: This script reads measured phase data from a folder structure and resizes it to a given size. 
#              It then splits the data into training and testing sets, normalizes the pixel values, and uses 
#              a CNN to predict the classes of the data. The script loops over different values of the number 
#              of runs and saves the classification report to a file in the working directory.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os, sys
from contextlib import redirect_stdout
import io
import cv2
import time
import logging
import shutil
from torchvision import transforms
from tqdm import tqdm
import matplotlib.patches as patches
from torchviz import make_dot

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
base_working_dir = os.path.expanduser('~/Downloads/cnn_phase6_output')

# Add the current date to the working directory
current_time = time.strftime("%Y_%m_%d")
counter = 1 
working_dir = os.path.join(base_working_dir, f'ccn_phase_6_{current_time}_{counter:02d}')


# Check if the directory exists and create a new one if it does
while os.path.exists(working_dir):
    working_dir = os.path.join(base_working_dir, f'ccn_phase_6_{current_time}_{counter:02d}')
    counter += 1

os.makedirs(working_dir, exist_ok=True)
print(f"Working directory: {working_dir}")

# Copy the current script file to the working directory
current_script = os.path.abspath(__file__)
shutil.copy(current_script, working_dir)

# Set up logging configuration
log_file = os.path.join(working_dir, 'cnn_phase6resizeloop.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Resize images parameters
x_resize = 128  # width
y_resize = 128  # high

# Data path
# data_path = r'/Volumes/EXTERNAL_US/2024_06_12/'
data_path = r'data3'

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

# Add configuration dictionary
config = {
    'train_split': 0.6,
    'val_split': 0.2,
    'batch_size': 16,  # Reduced batch size
    'learning_rate': 0.001,
    'n_epochs': 50,
    'early_stopping_patience': 5,
    'resize_dims': (128, 128),
    'data_batch_size': 50
}

# Define num_runs range parameters
num_runs_start = 30
num_runs_end = 30
num_runs_step = 10

class MemoryEfficientPhaseDataset(Dataset):
    def __init__(self, data_path, date, data_type_measure, num_runs, classes, transform=None):
        self.data_path = data_path
        self.date = date
        self.data_type_measure = data_type_measure
        self.num_runs = num_runs
        self.classes = classes
        self.transform = transform
        self.length = num_runs * len(classes)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        run_idx = idx // len(self.classes)
        class_idx = idx % len(self.classes)
        
        # Format RUN string
        RUN = f"M{str(run_idx+1).zfill(5)}"
        class_name = self.classes[class_idx]
        
        # Load and process single sample
        input_files_path = os.path.join(self.data_path, self.date, self.data_type_measure, 
                                      RUN, class_name, "files", "measured_phase.npy")
        phase_data = np.load(input_files_path)
        
        # Resize
        phase_data = cv2.resize(phase_data, (x_resize, x_resize), interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        phase_data = (phase_data - np.min(phase_data)) / (np.max(phase_data) - np.min(phase_data))
        
        # Convert to tensor
        phase_data = torch.tensor(phase_data, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            phase_data = self.transform(phase_data)
            
        return phase_data, class_idx

# Define transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.5,), (0.5,))
])

def visualize_cnn_architecture(save_path, model):
    """Create an improved visual representation of the CNN architecture."""
    # Create figure with white background
    fig = plt.figure(figsize=(24, 12), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Define colors with better contrast and modern feel
    colors = {
        'conv': '#FF6B6B',      # Coral red
        'pool': '#4ECDC4',      # Turquoise
        'fc': '#45B7D1',        # Sky blue
        'input': '#96CEB4',     # Sage green
        'dropout': '#FFBE0B',   # Golden yellow
        'activation': '#9D4EDD', # Purple
        'norm': '#2EC4B6'       # Teal
    }
    
    # Layer dimensions for reference
    dimensions = {
        'input': (128, 128, 1),
        'conv1': (128, 128, 32),
        'pool1': (64, 64, 32),
        'conv2': (64, 64, 64),
        'pool2': (32, 32, 64),
        'conv3': (32, 32, 128),
        'pool3': (16, 16, 128),
        'fc1': (512,),
        'fc2': (len(classes),)
    }
    
    def add_layer_block(x, y, width, height, color, alpha=0.7):
        rect = patches.Rectangle((x, y), width, height, facecolor=color, alpha=alpha,
                               edgecolor='black', linewidth=1, zorder=1)
        ax.add_patch(rect)
        return rect
    
    def add_arrow(x1, y1, x2, y2):
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.015, head_length=0.02,
                fc='black', ec='black', zorder=0)
    
    # Starting positions
    x, base_y = 0.05, 0.5
    block_width = 0.08
    spacing = 0.09
    
    # Input layer
    input_height = 0.3
    add_layer_block(x, base_y-input_height/2, block_width, input_height, colors['input'])
    plt.text(x+block_width/2, base_y+0.2, 'Input', ha='center', fontsize=10)
    plt.text(x+block_width/2, base_y-0.25, f'{dimensions["input"][0]}×{dimensions["input"][1]}×{dimensions["input"][2]}',
             ha='center', fontsize=9)
    
    x += block_width + spacing/2
    
    # Convolutional blocks
    conv_configs = [
        ('Conv1', dimensions['conv1'], 32),
        ('Conv2', dimensions['conv2'], 64),
        ('Conv3', dimensions['conv3'], 128)
    ]
    
    for i, (name, dim, filters) in enumerate(conv_configs):
        # Convolution
        height = 0.25 - i*0.03
        add_layer_block(x, base_y-height/2, block_width, height, colors['conv'])
        plt.text(x+block_width/2, base_y+0.2, name, ha='center', fontsize=10)
        plt.text(x+block_width/2, base_y-0.25, f'{dim[0]}×{dim[1]}×{dim[2]}\n3×3, {filters}',
                ha='center', fontsize=9)
        
        # BatchNorm + ReLU
        x += block_width + spacing/3
        add_layer_block(x, base_y-height/2, block_width/2, height, colors['norm'], alpha=0.5)
        plt.text(x+block_width/4, base_y+0.15, 'BatchNorm\n+ ReLU',
                ha='center', fontsize=9)
        
        # MaxPool
        x += block_width/2 + spacing/3
        pool_height = height * 0.8
        add_layer_block(x, base_y-pool_height/2, block_width, pool_height, colors['pool'])
        plt.text(x+block_width/2, base_y+0.15, f'MaxPool\n2×2',
                ha='center', fontsize=9)
        
        if i < 2:  # Add spacing between conv blocks
            x += block_width + spacing
    
    # Dropout after last conv block
    x += block_width + spacing/2
    dropout_height = 0.15
    add_layer_block(x, base_y-dropout_height/2, block_width, dropout_height, colors['dropout'])
    plt.text(x+block_width/2, base_y+0.15, 'Dropout\n0.25',
            ha='center', fontsize=9)
    
    # Flatten
    x += block_width + spacing/2
    plt.text(x+block_width/2, base_y, 'Flatten\n32,768',
            ha='center', fontsize=9)
    
    # Fully connected layers
    x += block_width/2 + spacing/2
    fc_height = 0.12
    
    # FC1
    add_layer_block(x, base_y-fc_height/2, block_width, fc_height, colors['fc'])
    plt.text(x+block_width/2, base_y+0.15, 'FC1\n512',
            ha='center', fontsize=9)
    
    # Dropout
    x += block_width + spacing/3
    add_layer_block(x, base_y-fc_height/2, block_width/2, fc_height, colors['dropout'])
    plt.text(x+block_width/4, base_y+0.15, 'Dropout\n0.5',
            ha='center', fontsize=9)
    
    # FC2 (output)
    x += block_width/2 + spacing/2
    output_height = 0.1
    add_layer_block(x, base_y-output_height/2, block_width, output_height, colors['fc'])
    plt.text(x+block_width/2, base_y+0.15, f'FC2\n{len(classes)}',
            ha='center', fontsize=9)
    
    # Add arrows connecting all blocks
    arrow_y = base_y
    x = 0.05  # Start from input
    while x < 0.85:  # Adjust end point based on your layout
        add_arrow(x + block_width, arrow_y, x + block_width + spacing/2, arrow_y)
        x += block_width + spacing
    
    # Set plot properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    plt.title('CNN Architecture for Phase Classification', pad=20, size=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor='black', label=layer.capitalize())
        for layer, color in colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.05), ncol=len(colors),
             frameon=True, fancybox=True, shadow=True)
    
    # Calculate total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Add network details
    details = (
        f"Total Trainable Parameters: {total_params:,}\n"
        f"Input Size: {dimensions['input'][0]}×{dimensions['input'][1]}×{dimensions['input'][2]}\n"
        f"Output Classes: {len(classes)}"
    )
    plt.figtext(0.02, 0.02, details, fontsize=10, ha='left')
    
    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def visualize_cnn_torchviz(model, save_path):
    """Create a detailed visualization of the CNN using torchviz."""
    # Create a dummy input with the correct shape
    x = torch.randn(1, 1, 128, 128).to(next(model.parameters()).device)
    
    # Get the output for this input
    y = model(x)
    
    # Create the dot graph
    dot = make_dot(y, params=dict(model.named_parameters()))
    
    # Customize the graph
    dot.attr(rankdir='TB')  # Top to bottom layout
    dot.attr('node', shape='box')
    
    # Save the visualization
    dot.render(save_path, format='png', cleanup=True)
    print(f"Saved torchviz visualization to {save_path}.png")

def visualize_cnn_detailed(model, save_path):
    """Create a detailed, modern visualization of the CNN architecture."""
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    
    # Create main axis for the architecture
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
    ax_main = plt.subplot(gs[0])
    ax_info = plt.subplot(gs[1])
    
    # Modern color palette
    colors = {
        'conv': '#FF6B6B',      # Coral red
        'pool': '#4ECDC4',      # Turquoise
        'fc': '#45B7D1',        # Sky blue
        'input': '#96CEB4',     # Sage green
        'dropout': '#FFBE0B',   # Golden yellow
        'activation': '#9D4EDD', # Purple
        'norm': '#2EC4B6'       # Teal
    }
    
    # Layer dimensions and configurations
    layer_configs = [
        {'name': 'Input', 'type': 'input', 'shape': (128, 128, 1)},
        {'name': 'Conv1', 'type': 'conv', 'shape': (128, 128, 32), 'kernel': 3, 'params': 320},
        {'name': 'BN1', 'type': 'norm', 'shape': (128, 128, 32)},
        {'name': 'Pool1', 'type': 'pool', 'shape': (64, 64, 32)},
        {'name': 'Conv2', 'type': 'conv', 'shape': (64, 64, 64), 'kernel': 3, 'params': 18496},
        {'name': 'BN2', 'type': 'norm', 'shape': (64, 64, 64)},
        {'name': 'Pool2', 'type': 'pool', 'shape': (32, 32, 64)},
        {'name': 'Conv3', 'type': 'conv', 'shape': (32, 32, 128), 'kernel': 3, 'params': 73856},
        {'name': 'BN3', 'type': 'norm', 'shape': (32, 32, 128)},
        {'name': 'Pool3', 'type': 'pool', 'shape': (16, 16, 128)},
        {'name': 'Dropout1', 'type': 'dropout', 'rate': 0.25},
        {'name': 'Flatten', 'type': 'fc', 'shape': (32768,)},
        {'name': 'FC1', 'type': 'fc', 'shape': (512,), 'params': 16777728},
        {'name': 'Dropout2', 'type': 'dropout', 'rate': 0.5},
        {'name': 'FC2', 'type': 'fc', 'shape': (len(classes),), 'params': 7182}
    ]
    
    def add_layer_block(x, y, width, height, color, alpha=0.7):
        rect = patches.Rectangle((x, y), width, height, facecolor=color, alpha=alpha,
                               edgecolor='black', linewidth=1, zorder=1)
        ax_main.add_patch(rect)
        return rect
    
    # Starting positions and dimensions
    x = 0.05
    base_y = 0.5
    block_width = 0.06
    spacing = 0.04
    max_height = 0.3
    
    # Draw layers
    for i, layer in enumerate(layer_configs):
        # Adjust height based on layer type
        if layer['type'] == 'input':
            height = max_height
        elif layer['type'] == 'conv':
            height = max_height * 0.9
        elif layer['type'] == 'pool':
            height = max_height * 0.7
        elif layer['type'] == 'fc':
            height = max_height * 0.5
        else:
            height = max_height * 0.4
        
        # Add block
        add_layer_block(x, base_y - height/2, block_width, height, colors[layer['type']])
        
        # Add layer name
        ax_main.text(x + block_width/2, base_y + height/2 + 0.02, layer['name'],
                    ha='center', va='bottom', fontsize=9, rotation=0)
        
        # Add shape information
        if 'shape' in layer:
            shape_text = '×'.join(str(x) for x in layer['shape'])
            ax_main.text(x + block_width/2, base_y - height/2 - 0.02,
                        shape_text, ha='center', va='top', fontsize=8)
        
        # Add additional info (kernel size, dropout rate)
        if 'kernel' in layer:
            ax_main.text(x + block_width/2, base_y,
                        f'{layer["kernel"]}×{layer["kernel"]}', ha='center', va='center', fontsize=8)
        elif 'rate' in layer:
            ax_main.text(x + block_width/2, base_y,
                        f'p={layer["rate"]}', ha='center', va='center', fontsize=8)
        
        # Add arrow to next layer
        if i < len(layer_configs) - 1:
            ax_main.arrow(x + block_width, base_y,
                         spacing, 0, head_width=0.015, head_length=0.01,
                         fc='black', ec='black', zorder=0)
        
        x += block_width + spacing
    
    # Set main plot properties
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Add title
    ax_main.set_title('CNN Architecture for Phase Classification', pad=20, size=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor='black', label=layer_type.capitalize())
        for layer_type, color in colors.items()
    ]
    ax_main.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.1), ncol=len(colors),
                  frameon=True, fancybox=True, shadow=True)
    
    # Add network statistics in the bottom subplot
    ax_info.axis('off')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info_text = (
        f"Network Statistics:\n"
        f"Total Parameters: {total_params:,}\n"
        f"Trainable Parameters: {trainable_params:,}\n"
        f"Input Size: 128×128×1\n"
        f"Output Classes: {len(classes)}\n"
        f"Number of Convolutional Layers: 3\n"
        f"Number of Fully Connected Layers: 2"
    )
    
    ax_info.text(0.02, 0.6, info_text, fontsize=10, 
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def plot_training_metrics(epochs, train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, num_runs, save_dir):
    """Plot comprehensive training metrics."""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Accuracy Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_accuracies, 'b-', label='Training')
    ax1.plot(epochs, val_accuracies, 'r-', label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Loss Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_losses, 'b-', label='Training')
    ax2.plot(epochs, val_losses, 'r-', label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Learning Rate Plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, learning_rates, 'g-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # 4. Training Progress
    ax4 = fig.add_subplot(gs[1, 1])
    progress_data = {
        'Final Train Acc': train_accuracies[-1],
        'Final Val Acc': val_accuracies[-1],
        'Best Val Acc': max(val_accuracies),
        'Final Train Loss': train_losses[-1],
        'Final Val Loss': val_losses[-1],
        'Best Val Loss': min(val_losses)
    }
    y_pos = np.arange(len(progress_data))
    ax4.barh(y_pos, list(progress_data.values()))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(list(progress_data.keys()))
    ax4.set_title('Training Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_metrics_num_runs_{num_runs}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_maps(model, sample_input, save_path):
    """Visualize feature maps from different layers of the CNN."""
    model.eval()
    
    # Register hooks to get intermediate activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for convolutional layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(get_activation(name))
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5 * len(activations)))
    for idx, (name, activation) in enumerate(activations.items()):
        # Get the first sample and first few channels
        act = activation[0].cpu()
        n_channels = min(8, act.shape[0])
        
        for i in range(n_channels):
            plt.subplot(len(activations), 8, idx * 8 + i + 1)
            plt.imshow(act[i], cmap='viridis')
            if i == 0:
                plt.ylabel(f'Layer {name}')
            plt.xticks([])
            plt.yticks([])
    
    plt.suptitle('Feature Maps Visualization', size=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_distributions(model, save_path):
    """Plot the distribution of parameters in each layer."""
    plt.figure(figsize=(15, 10))
    
    # Collect parameters from each layer
    layer_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params[name] = param.data.cpu().numpy().flatten()
    
    # Create violin plots
    plt.violinplot(list(layer_params.values()))
    plt.xticks(range(1, len(layer_params) + 1), list(layer_params.keys()), rotation=45)
    plt.title('Parameter Distributions Across Layers')
    plt.ylabel('Parameter Value')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gradient_flow(model, save_path):
    """Plot the gradient flow through the network layers."""
    named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad and param.grad is not None]
    
    plt.figure(figsize=(15, 10))
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        layers.append(n)
        ave_grads.append(p.grad.abs().mean().cpu())
        max_grads.append(p.grad.abs().max().cpu())
    
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Loop over different nRUN values
for num_runs in range(num_runs_start, num_runs_end + 1, num_runs_step):
    start_time = time.time()
    logging.info("\nstart_time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    logging.info(f"\nRunning with num_runs = {num_runs}")
    print(f"\nRunning with num_runs = {num_runs}")

    # Create memory-efficient datasets
    full_dataset = MemoryEfficientPhaseDataset(data_path, date, data_type_measure, num_runs, classes, transform=None)
    
    # Calculate lengths for splits
    total_size = len(full_dataset)
    train_size = int(config['train_split'] * total_size)
    val_size = int(config['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders with no workers
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # Improved CNN architecture
    class CNN(nn.Module):
        def __init__(self, num_classes):
            super(CNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Dropout2d(0.25)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(128 * 16 * 16, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Add training utilities
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0

    # Modified training loop
    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
        early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
        best_val_loss = float('inf')
        
        # Lists to store metrics
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []
        learning_rates = []
        epochs = []
        
        for epoch in range(config['n_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            batch_losses = []
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["n_epochs"]} [Train]')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss = np.mean(batch_losses)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            batch_losses = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["n_epochs"]} [Val]')
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    batch_losses.append(loss.item())
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            val_loss = np.mean(batch_losses)
            val_accuracy = 100 * val_correct / val_total
            
            # Store metrics
            epochs.append(epoch + 1)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Print epoch metrics
            print(f'Epoch {epoch+1}/{config["n_epochs"]}:')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(working_dir, 'best_model.pth'))
                
                # Plot feature maps for best model
                if epoch > 0:  # Skip first epoch
                    visualize_feature_maps(model, inputs[:1], 
                                        os.path.join(working_dir, f'feature_maps_epoch_{epoch+1}.png'))
                    
                    # Plot parameter distributions
                    plot_parameter_distributions(model, 
                                              os.path.join(working_dir, f'parameter_distributions_epoch_{epoch+1}.png'))
                    
                    # Plot gradient flow
                    plot_gradient_flow(model, 
                                     os.path.join(working_dir, f'gradient_flow_epoch_{epoch+1}.png'))
        
        return epochs, train_losses, val_losses, train_accuracies, val_accuracies, learning_rates

    # Modified main training section after model creation
    model = CNN(len(classes))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Save architecture visualizations
    architecture_path = os.path.join(working_dir, f'cnn_architecture')
    visualize_cnn_architecture(architecture_path + '_matplotlib.png', model)
    visualize_cnn_torchviz(model, architecture_path + '_torchviz')
    visualize_cnn_detailed(model, architecture_path + '_detailed.png')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Train the model and get all metrics
    epochs, train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, config
    )

    # Plot comprehensive training metrics
    plot_training_metrics(epochs, train_losses, val_losses, train_accuracies, val_accuracies, 
                         learning_rates, num_runs, working_dir)

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

    # Print classification report and create confusion matrix
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            labels = labels.to(mps_device if torch.backends.mps.is_available() else torch.device("cpu"))
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=classes, digits=6, zero_division=0)
    logging.info("\n" + report)

    # Create and save confusion matrix plot with counts
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Counts (num_runs={num_runs})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save confusion matrix plot with counts
    cm_plot_path = os.path.join(working_dir, f'confusion_matrix_counts_num_runs_{num_runs}.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create and save confusion matrix plot with percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Percentages (num_runs={num_runs})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save confusion matrix plot with percentages
    cm_plot_percentage_path = os.path.join(working_dir, f'confusion_matrix_percentages_num_runs_{num_runs}.png')
    plt.savefig(cm_plot_percentage_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save classification report to file in working directory
    report_path = os.path.join(working_dir, f'classification_report_num_runs_{num_runs}.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Record the end time for each loop
    end_time = time.time()
    logging.info("\nend_time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    logging.info(f"\nTime elapsed for num_runs = {num_runs}: {end_time - start_time} seconds")