import torch 
from torch import nn 
from torch.nn import functional as F
import timm 
import numpy as np
import disco
import torch
import torch.distributed as dist


class CLIPPO(nn.Module): 
    def __init__(self): 
        super(CLIPPO, self).__init__()
        self.encoder = timm.create_model('resnet18', pretrained=True, num_classes=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        projection_input = 512#768
        self.text_proj = nn.Sequential(*[nn.Linear(projection_input, 256), nn.ReLU(), nn.Linear(256, 256)])
        self.image_proj = nn.Sequential(*[nn.Linear(projection_input, 256), nn.ReLU(), nn.Linear(256, 256)])

    def forward(self, image, text): 
        image_features = self.image_proj(self.encoder(image))
        text_features = self.text_proj (self.encoder(text))
        

        torch.cuda.synchronize()
        #image_features = disco.Gather(image_features.contiguous())
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
    

# def cross_entropy(preds, targets, reduction='none'):
#     log_softmax = nn.LogSoftmax(dim=-1)
#     loss = (-targets * log_softmax(preds)).sum(1)
#     if reduction == "none":
#         return loss
#     elif reduction == "mean":
#         return loss.mean()
