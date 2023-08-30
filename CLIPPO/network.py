from typing import Union
import torch 
from torch import nn 
import timm 
import numpy as np
import torch

class ImageEncoder(nn.Module): 
    def __init__(self, timm_model:Union[str, None]=None):
        super(ImageEncoder, self).__init__()
        if timm_model: 
            self.encoder = timm.create_model(timm_model, num_classes=0)
        else:
            self.encoder = nn.Sequential(*[nn.Conv2d(3, 32, 3),
                                        nn.ReLU(), 
                                        nn.Conv2d(32, 64, 3), 
                                        nn.ReLU(), 
                                        nn.Flatten(), 
                                        nn.AdaptiveMaxPool1d(124), 
                                        nn.Linear(124, 512)])
        
        self.output_size = 512
    
    def forward(self, image): 
        return self.encoder(image)

class TextEncoder(nn.Module): 
    def __init__(self):
        super(TextEncoder, self).__init__()
    def forward(self, text): 
        raise NotImplementedError

class CLIPPO(nn.Module): 
    def __init__(self, timm_model:Union[str, None]=None): 
        super(CLIPPO, self).__init__()
        self.encoder = ImageEncoder(timm_model)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.proj = nn.Sequential(*[nn.Linear(self.encoder.output_size, 256), nn.ReLU(), nn.Linear(256, 2)])

    def forward(self, image, text): 
        image_features = self.proj(self.encoder(image))
        text_features = self.proj (self.encoder(text))
        return image_features, text_features, self.logit_scale.exp()
