import os
import torch
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from PIL import Image
import config
import matplotlib.pyplot as plt

# INTERACTIVE AUGMENTATION FUNCTION
def apply_custom_augmentation(img, do_flip, brightness, contrast, erase_prob):
    """Custom augmentation for interactive UI"""
    if do_flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if brightness != 1.0:
        img = TF.adjust_brightness(img, brightness_factor=brightness)
    
    if contrast != 1.0:
        img = TF.adjust_contrast(img, contrast_factor=contrast)
        
    if erase_prob > 0:
        tensor_img = TF.to_tensor(img)
        eraser = transforms.RandomErasing(p=1.0, scale=(erase_prob, erase_prob), ratio=(0.3, 3.3))
        img = TF.to_pil_image(eraser(tensor_img))
    
    return img

# TRANSFORMS
def get_transforms(model_type, is_train=False):
    cfg = config.MODEL_CONFIGS[model_type]
    img_size = cfg['size']
    
    # 1. Base Resize
    tfs = [transforms.Resize((img_size, img_size))]

    # 2. Strong Augmentation (Training Only)
    if is_train:
        tfs.extend([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    # 3. ToTensor & Normalize
    tfs.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['mean'], std=cfg['std'])
    ])
    
    return transforms.Compose(tfs)

# DATALOADERS
def get_dataloaders(model_type):
    train_transform = get_transforms(model_type, is_train=True)
    val_test_transform = get_transforms(model_type, is_train=False)

    # Define paths
    train_path = os.path.join(config.DATA_DIR, 'train')
    val_path = os.path.join(config.DATA_DIR, 'val')
    test_path = os.path.join(config.DATA_DIR, 'test')

    try:
        # Initialize Datasets
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)
        
        config.NUM_CLASSES = len(train_dataset.classes)
        config.CLASS_NAMES = train_dataset.classes

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2),
            'val': DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2),
            'test': DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        }

        dataset_sizes = {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
        
        print(f"Data Loaded for {model_type.upper()}. Classes: {config.NUM_CLASSES}")
        print(f"Train: {dataset_sizes['train']} | Val: {dataset_sizes['val']} | Test: {dataset_sizes['test']}")
        
        return dataloaders, dataset_sizes

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Ensure folder structure at {config.DATA_DIR} is correct.")
        return None, None

def plot_learning_curves(history, save_path):
    """
    Plots training and validation loss/accuracy and saves to file.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot Loss
    ax1.plot(history['train_loss'], label='Training Loss', color='dodgerblue', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Validation Loss', color='orangered', marker='o', markersize=3)
    ax1.set_title('Training & Validation Loss', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', color='dodgerblue', marker='o', markersize=3)
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='orangered', marker='o', markersize=3)
    ax2.set_title('Training & Validation Accuracy', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Model Learning Curves', fontsize=20, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    print(f"Learning curves saved to: {save_path}")
    plt.close()