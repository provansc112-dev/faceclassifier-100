import torch
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')

def load_class_names(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
            return sorted(classes)
    else:
        return [f"Class_{i}" for i in range(1, 101)]

CLASS_NAMES = load_class_names(LABELS_PATH)
NUM_CLASSES = len(CLASS_NAMES)

MODEL_CONFIGS = {
    'swin': {
        'model_name': 'swin_tiny_patch4_window7_224',
        'weights': os.path.join(BASE_DIR, 'models', 'swintransformer.pt'),
        'pretrained_source': os.path.join(BASE_DIR, 'models', 'swinface.pt'), # Original SwinFace weights
        'size': 112,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'inception': {
        'model_name': 'inception_resnet_v1',
        'weights': os.path.join(BASE_DIR, 'models', 'inceptionresnetv1.pth'),
        'size': 160,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }
}

BATCH_SIZE = 32 
DATA_DIR = os.path.join(BASE_DIR, "data") # Directory for train, val, test folders
SAMPLE_FOLDER = os.path.join(BASE_DIR, 'samples') # Folder for sample images
THRESHOLD_SWIN = 0.8 # Swin Transformer confidence threshold for deployment
THRESHOLD_INCEPTION = 0.3 # Inception-ResNet-V1 confidence threshold for deployment
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7] # MTCNN thresholds for P-Net, R-Net, O-Net

SWIN_EPOCHS = 30
INCEPTION_EPOCHS = {
    'phase1': 10,
    'phase2': 10,
    'phase3': 5
}