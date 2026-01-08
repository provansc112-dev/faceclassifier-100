import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import model  
import config
import utils

def run_evaluation(model_type='swin'):
    print(f"\n{'='*50}")
    print(f"Model: {model_type.upper()}")
    print(f"{'='*50}")

    # 1. Load Data Test
    test_transform = utils.get_transforms(model_type, is_train=False)
    
    test_dir = os.path.join(config.DATA_DIR, 'test')
    try:
        test_dataset = ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        class_names = test_dataset.classes
        print(f"Successfully loaded {len(test_dataset)} images from {len(class_names)} classes.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Load Model & Weights
    eval_model = model.load_trained_model(model_type)
    eval_model.to(config.DEVICE)
    eval_model.eval()

    # 3. Inference Loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Inference {model_type}"):
            images = images.to(config.DEVICE)
            
            outputs = eval_model(images)
            
            if isinstance(outputs, (tuple, list)):
                logits = outputs[-1] 
            else:
                logits = outputs
                
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metric Evaluation
    print(f"\nClassification Report ({model_type.upper()}):")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

    # 5. Confusion Matrix Plotting
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix Test Data - {model_type.upper()}', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.savefig(f'confusion_matrix_{model_type}.png', dpi=300)
    print(f"Confusion Matrix saved as 'cm_{model_type}.png'")
    plt.show()

if __name__ == "__main__":
    run_evaluation(model_type='inception')
    run_evaluation(model_type='swin')