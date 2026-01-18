import torch
import os

# 1. Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Model Configuration
MODEL_CONFIGS = {
    'swin': {
        'model_name': 'swin_tiny_patch4_window7_224',
        'weights': 'models/swintransformer.pt',
        'size': 112,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'inception': {
        'model_name': 'inception_resnet_v1',
        'weights': 'models/inceptionresnetv1.pth',
        'size': 160,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }
}

# 3. Dataset Configuration
NUM_CLASSES = 100
LABELS_PATH = 'labels.txt'

def load_class_names(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        return [f"Class_{i}" for i in range(1, 101)]

CLASS_NAMES = load_class_names(LABELS_PATH)

# 4. Global Settings
BATCH_SIZE = 32 # Adjust based on your GPU memory
DATA_DIR = "config/data" # Ensure this directory contains 'train', 'val', 'test' subfolders
SAMPLE_FOLDER = "samples/" # Folder for sample images in the GUI
THRESHOLD_SWIN = 0.8 # Threshold for Swin Transformer
THRESHOLD_INCEPTION = 0.3 # Threshold for Inception Resnet V1

# 5. Training Configuration
SWIN_EPOCHS = 30
INCEPTION_EPOCHS = {
    'phase1': 10,  # Training Head Only
    'phase2': 10,  # Fine-Tuning
    'phase3': 5    # Final Tuning
}