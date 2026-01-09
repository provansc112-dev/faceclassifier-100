import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from blueprint import SwinTransformer
import config
from collections import OrderedDict

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifier, self).__init__()
        self.base_model = InceptionResnetV1(pretrained=None, classify=False)
        embedding_size = 512
        self.head = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        embedding = self.base_model(x)
        logits = self.head(embedding)
        return logits

def load_trained_model(model_type):
    cfg = config.MODEL_CONFIGS[model_type]
    device = config.DEVICE
    weights_path = cfg['weights']
    
    print(f"Loading {model_type.upper()} model...")

    if model_type == 'inception':
        model = FaceClassifier(num_classes=config.NUM_CLASSES)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
    
    elif model_type == 'swin':
        model = SwinTransformer(num_classes=512)
        
        num_features = model.num_features
        model.feature = nn.Linear(num_features, config.NUM_CLASSES)
        
        print(f"Loading SwinFace weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        state_dict = checkpoint['state_dict_backbone'] if 'state_dict_backbone' in checkpoint else checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        
    else:
        raise ValueError(f"Unknown model type {model_type}")

    model.to(device)
    model.eval()
    return model