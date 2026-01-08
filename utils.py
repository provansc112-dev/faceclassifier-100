import os
import torch
from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader
import config
from PIL import Image

def apply_custom_augmentation(img, do_flip, brightness, contrast, erase_prob):
    """
    This function applies custom augmentations to an image manually by user.
    """
    # 1. Flip
    if do_flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 2. Color Jitter
    jitter = T.ColorJitter(brightness=brightness, contrast=contrast)
    img = jitter(img)
    
    # 3. Random Erasing (only available in tensor form)
    img_tensor = T.ToTensor()(img)
    if erase_prob > 0:
        eraser = T.RandomErasing(p=1.0, scale=(erase_prob, erase_prob), ratio=(1, 1), value=0)
        img_tensor = eraser(img_tensor)
    
    return T.ToPILImage()(img_tensor)

def get_transforms(model_type, is_train=False):
    """
    Transforms exclusively for the specified model.
    is_train: True to train with augmentation.
              False when just doing prediction.
    """
    cfg = config.MODEL_CONFIGS[model_type]
    
    # 1. Resize
    base_transforms = [
        transforms.Resize((cfg['size'], cfg['size'])),
    ]

    # 2. Augmentation (is_train=True)
    if is_train:
        augmentation = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        base_transforms.extend(augmentation)

    # 3. ToTensor & Normalize
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['mean'], std=cfg['std'])
    ]
    
    base_transforms.extend(final_transforms)
    
    return transforms.Compose(base_transforms)

def get_dataloaders(model_type):
    """
    Creating dataloaders for training, validation, dan testing.
    Using batch_size from config.py
    """
    print(f"--- Preparing DataLoaders for {model_type.upper()} ---")
    
    train_trm = get_transforms(model_type, is_train=True)
    val_test_trm = get_transforms(model_type, is_train=False)

    # Make sure your DATA_DIR has 'train', 'val', 'test' subfolders
    train_path = os.path.join(config.DATA_DIR, 'train')
    val_path = os.path.join(config.DATA_DIR, 'val')
    test_path = os.path.join(config.DATA_DIR, 'test')

    try:
        train_dataset = datasets.ImageFolder(train_path, transform=train_trm)
        val_dataset = datasets.ImageFolder(val_path, transform=val_test_trm)
        test_dataset = datasets.ImageFolder(test_path, transform=val_test_trm)

        dataloaders = {
            'train': DataLoader(
                train_dataset, 
                batch_size=config.BATCH_SIZE, 
                shuffle=True, 
                num_workers=2
            ),
            'val': DataLoader(
                val_dataset, 
                batch_size=config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=2
            ),
            'test': DataLoader(
                test_dataset, 
                batch_size=config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=2
            )
        }

        dataset_sizes = {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
        
        print(f"Success: Found {len(train_dataset.classes)} classes.")
        print(f"Training: {dataset_sizes['train']} | Val: {dataset_sizes['val']} | Test: {dataset_sizes['test']}")
        
        return dataloaders, dataset_sizes

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Check if your folder structure is: {config.DATA_DIR}/[train, val, test]")
        return None, None
    
def get_class_names(data_dir):
    """
    Mengambil daftar nama kelas (nama folder) dari direktori train.
    Sangat berguna untuk sinkronisasi index label dengan nama orang.
    """
    train_path = os.path.join(data_dir, 'train')
    if not os.path.exists(train_path):
        return []
    
    classes = [d.name for d in os.scandir(train_path) if d.is_dir()]
    classes.sort()
    return classes

def get_device():
    return config.DEVICE