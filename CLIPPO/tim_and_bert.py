from sentence_transformers import SentenceTransformer, models
import torch
import torch.nn.functional as F
import torch 
from torch import nn 
from torch.nn import functional as F
import timm 
import numpy as np
# import disco
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

        self.embedding = nn.Embedding(10, 128)
        self.MLP = nn.Sequential(*[
            nn.Linear(128, 256), 
            nn.ReLU(), 
            nn.Linear(256, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512)
        ])
        self.vocab = {str(i):i for i in range(10)}
        print("im hereeeeeee")
    
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

        return image_features/image_features.norm(dim=1, keepdim=True),\
                text_features/text_features.norm(dim=1, keepdim=True)
        all_image_feature = image_features#disco.Gather(image_features.contiguous())
        all_text_feature = text_features#image_features#disco.Gather(image_features.contiguous())
        
    
        # normalized features
        all_image_feature = all_image_feature / all_image_feature.norm(dim=1, keepdim=True)
        all_text_feature = all_text_feature / all_text_feature.norm(dim=1, keepdim=True)
        
        # cosine similarity as logits
        rank = dist.get_rank()
        bs = 90
        logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale* all_image_feature[bs*rank:bs*(rank+1)] @ all_text_feature.t()
        # logits_per_text = logit_scale * all_text_feature[bs*rank:bs*(rank+1)] @ all_image_feature.t()
        logits_per_image = logit_scale* all_image_feature @ all_text_feature.t()
        logits_per_text = logit_scale * all_text_feature @ all_image_feature.t()
        temp = 1#1e-5
        # shape = [global_batch_size, global_batch_size]
        #logits_per_image <=> I
        return logits_per_image/temp, logits_per_text/temp
    

class DINOLoss(nn.Module): 
    def __init__(self, out_dim, temp): 
        super(DINOLoss, self).__init__()
        self.temp = temp
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, text_logit, image_logit): 
        text_logit /= self.temp 
        image_logit /= self.temp 

        image_centered = F.softmax((image_logit - self.center), dim=-1)
        text_centered = F.softmax((text_logit - self.center), dim=-1)

        text_loss = torch.sum(-image_centered*F.log_softmax(text_logit, dim=-1), dim=-1)
        image_loss = torch.sum(-text_centered*F.log_softmax(image_logit, dim=-1), dim=-1) 
        return text_loss, image_loss


