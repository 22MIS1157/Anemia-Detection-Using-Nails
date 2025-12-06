import torch
import torch.nn as nn
import timm

class DeiTClassifier(nn.Module):
    """DeiT-based classifier for anemia detection."""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(DeiTClassifier, self).__init__()
        self.backbone = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
