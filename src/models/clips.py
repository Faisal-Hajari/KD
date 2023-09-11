import torch 
from torch import nn 
import timm 
import numpy as np
from models.resnet import *

class CLIPPO(nn.Module): 
    def __init__(self, net: str = "seq", projection_input: int  = 512): 
        super(CLIPPO, self).__init__()
        if net == "resnet":
            self.encoder = ResNet50(projection_input)
        elif net == "seq":
            self.encoder = nn.Sequential(*[nn.Conv2d(3, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(), nn.Flatten(), nn.AdaptiveMaxPool1d(124), nn.Linear(124, projection_input)])
        else: 
            self.encoder = timm.create_model(net, pretrained=True, num_classes=projection_input)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.proj = nn.Sequential(*[nn.Linear(projection_input, 256), nn.ReLU(), nn.Linear(256, 256)])

    def forward(self, image, text): 
        image_features = self.proj(self.encoder(image))
        text_features = self.proj (self.encoder(text))
        return image_features, text_features, self.logit_scale