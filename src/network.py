import torch 
from torch import nn 
from torch.nn import functional as F
import timm 
import numpy as np
import disco
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=768, depth=12, heads=12):
        super(ViTEncoder, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (self.image_size // self.patch_size) **2
        self.dim = dim
        
        self.patch_embedding = nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, self.dim))
        self.position_embedding = nn.Parameter(torch.randn(self.num_patches + 1, self.dim))
        
        self.transformer = nn.Transformer(self.dim, nhead=heads, num_encoder_layers=depth)
        
    def forward(self, x):
        print("$ X",x.shape)
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        print("$$ ", x.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        print("x shape ",x.shape)
        print("position embedding ", self.position_embedding.shape)
        x += self.position_embedding
        x = self.transformer(x, x)
        x = x[:, 0]
        return x
class CLIPPO(nn.Module): 
     def __init__(self): 
        super(CLIPPO, self).__init__()
        self.encoder = timm.create_model('convit_base',pretrained=True,num_classes=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        projection_input = 768  # Updating input size to match ViT's output size
        self.image_proj = nn.Sequential(*[nn.Linear(projection_input, 256), nn.ReLU(), nn.Linear(256, 512)])

     def forward(self, image, text):
        image_features = self.image_proj(self.encoder(image))
        text_features = self.image_proj(self.encoder(text))

        # torch.cuda.synchronize()
        #image_features = disco.Gather(image_features.contiguous())
        all_image_feature = image_features#disco.Gather(image_features.contiguous())
        all_text_feature = text_features#image_features#disco.Gather(image_features.contiguous())
        
    
        # normalized features
        all_image_feature = all_image_feature /all_image_feature.norm(dim=1, keepdim=True)
        all_text_feature = all_text_feature / all_text_feature.norm(dim=1, keepdim=True)
        
        # cosine similarity as logits
        # rank = dist.get_rank()
        # bs = 90
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
#     elif reduction == "mean":    # for i in mycsv[0]:
            #     text, _ = self.data[int(i)]
            #     if self.text_transfroms:
            #         text = self.image_transforms(text.convert('RGB'))
            #     texts.append(text)

            # return torch.stack(images),torch.stack(texts)
#         return loss.mean()
