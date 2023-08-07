from sentence_transformers import SentenceTransformer, models
import torch
import torch.nn.functional as F
import torch 
from torch import nn 
from torch.nn import functional as F
import timm 
import numpy as np
import disco
import torch
import torch.distributed as dist

class ImageEncoder(nn.Module): 
    def __init__(self): 
        super(ImageEncoder, self).__init__()
        self.encoder = timm.create_model('resnet18', num_classes=0)
    
    def forward(self, x):
        return self.encoder(x)

class TextEncoder(nn.Module): 
    def __init__(self) -> None:
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(10, 256)
        self.MLP = nn.Sequential(*[
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512)
        ])
        self.vocab = {str(i):i for i in range(10)}
    
    def forward(self, x): 
        # x = self.vocab[x]
        x = self.embedding(x)
        x = self.MLP(x)
        return x 

class CLIP(nn.Module): 
    def __init__(self): 
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder() 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self,text, image):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        return image_features, text_features



