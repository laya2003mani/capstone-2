"""
model.py  -  ONE shared single-image classifier.

  image (any magnification) -> ConvNeXt V2 Atto encoder (320-d) -> dropout -> Linear -> 2 logits

The same weights are used for 40X, 100X, 200X and 400X. The escalation controller
(escalate.py) simply calls this model on one magnification at a time.
"""
import timm
import torch.nn as nn

import config as C


class SharedMagClassifier(nn.Module):
    def __init__(self, backbone: str = C.BACKBONE, pretrained: bool = True,
                 num_classes: int = 2, drop: float = 0.3):
        super().__init__()
        # num_classes=0 -> timm removes the ImageNet head and returns pooled features
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.encoder.num_features, num_classes))

    def forward(self, x):
        return self.head(self.encoder(x))

    def freeze_early(self, n_stages: int):
        """Freeze the stem and the first n_stages ConvNeXt stages (0-4).
        These are the expensive high-resolution layers; freezing them skips their backward pass
        (faster) and keeps generic edge/texture features (less over-fitting on few patients)."""
        if n_stages <= 0:
            return
        for p in self.encoder.stem.parameters():
            p.requires_grad = False
        for stage in self.encoder.stages[:n_stages]:
            for p in stage.parameters():
                p.requires_grad = False
