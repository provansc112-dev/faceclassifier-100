import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import sys
import model as model_loader
import config
import utils

def run_evaluation(model_type='swin'):
    print(f"Starting Evaluation for Model: {model_type.upper()}")

    # 1. Load Data using utils.py
    print("Loading datasets...")
    dataloaders, dataset_sizes = utils.get_dataloaders(model_type)

    if dataloaders is None:
        print("Error: Failed to load dataloaders. Exiting.")
        return

    test_loader = dataloaders['test']
    class_names = config.CLASS_NAMES
    
    print(f"Successfully loaded data.")
    print(f"Test images: {dataset_sizes['test']}")
    print(f"Classes detected: {len(class_names)}")

    # 2. Load Model & Weights
    print("Loading trained model weights...")
    try:
        eval_model = model_loader.load_trained_model(model_type, load_best=True)
        eval_model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Inference Loop
    print("Running inference on test set...")
    all_preds = []
    all_labels = []
    device = config.DEVICE

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Progress"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = eval_model(images)
            
            if isinstance(outputs, (tuple, list)):
                logits = outputs[-1]
            else:
                logits = outputs
                
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metric Evaluation
    print(f"\nClassification Report ({model_type.upper()}):")
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Detailed report
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

    # 5. Confusion Matrix Plotting
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - {model_type.upper()}', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(config.BASE_DIR, f'confusion_matrix_{model_type}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix image saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_evaluation(model_type='inception') # 'swin' or 'inception'