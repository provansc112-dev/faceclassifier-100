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
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    elif model_type == 'swin':
        model = SwinTransformer(num_classes=512)
        
        num_features = model.num_features
        model.feature = nn.Linear(num_features, config.NUM_CLASSES)
        
        print(f"Loading SwinFace weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        
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

class FaceEuclideanWrapper(nn.Module):
    def __init__(self, base_model, db_size=4991, feature_dim=512):
        super().__init__()
        self.backbone = base_model
        self.db = nn.Parameter(torch.randn(db_size, feature_dim), requires_grad=False)

    def forward(self, x):
        emb = self.backbone(x)
        
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
            
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        elif emb.dim() == 4:
            emb = emb.mean(dim=[2, 3])

        if emb.size(1) > 512:
            emb = emb[:, :512]
        elif emb.size(1) < 512:
            padding = torch.zeros(emb.size(0), 512 - emb.size(1)).to(emb.device)
            emb = torch.cat([emb, padding], dim=1)

        dist = torch.cdist(emb, self.db, p=2)
        min_dist, min_idx = torch.min(dist, dim=1)
        return min_dist, min_idx