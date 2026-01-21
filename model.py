import os
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from blueprint import SwinTransformer 
import config
from collections import OrderedDict

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifier, self).__init__()
        # Base: VGGface2 pretrained
        self.base_model = InceptionResnetV1(pretrained='vggface2', classify=False)
        # Head
        embedding_size = 512
        self.head = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        embedding = self.base_model(x)
        logits = self.head(embedding)
        return logits

def load_trained_model(model_type, load_best=True):
    cfg = config.MODEL_CONFIGS[model_type]
    device = config.DEVICE
    target_weights = cfg['weights']
    
    print(f"Initializing {model_type.upper()} model...")


    if model_type == 'inception':
        model = FaceClassifier(num_classes=config.NUM_CLASSES)
        
        if load_best and os.path.exists(target_weights):
            print(f"Loading BEST fine-tuned weights from {target_weights}")
            checkpoint = torch.load(target_weights, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            new_state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print("Initializing with basic VGGFace2 (Training mode).")

    elif model_type == 'swin':
        model = SwinTransformer(
            img_size=112, patch_size=2, in_chans=3,
            num_classes=512, embed_dim=96, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.
        )
        
        if load_best and os.path.exists(target_weights):
            print(f"Loading BEST fine-tuned weights from {target_weights}")
            
            model.feature = nn.Linear(model.num_features, config.NUM_CLASSES)
            print(f"Architecture adjusted to {config.NUM_CLASSES} classes BEFORE loading.")

            checkpoint = torch.load(target_weights, map_location=device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'state_dict_backbone' in checkpoint: # Kasus sisa save lama
                state_dict = checkpoint['state_dict_backbone']
            else:
                state_dict = checkpoint
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("Weights loaded PERFECTLY.")
            except Exception as e:
                print(f"Strict load failed, trying loose load: {e}")
                model.load_state_dict(new_state_dict, strict=False)

        else:
            load_path = cfg['pretrained_source']
            print(f"Loading ORIGINAL SwinFace weights from {load_path}")
            
            if os.path.exists(load_path):
                checkpoint = torch.load(load_path, map_location=device)
                state_dict = checkpoint['state_dict_backbone'] if 'state_dict_backbone' in checkpoint else checkpoint
                
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                    
                model.load_state_dict(new_state_dict, strict=False)
                print("Original Pretrained weights loaded.")
            else:
                print(f"Warning: Pretrained source not found at {load_path}")

            model.feature = nn.Linear(model.num_features, config.NUM_CLASSES)
            print(f"Classifier head replaced for {config.NUM_CLASSES} classes (Random Init).")

    else:
        raise ValueError(f"Unknown model type {model_type}")

    model.to(device)
    return model