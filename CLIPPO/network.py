import torch 
from torch import nn 
from torch.nn import functional as F
import timm 
import numpy as np

class CLIPPO(nn.Module): 
    def __init__(self): 
        super(CLIPPO, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', num_classes=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text): 
        image_features = self.encoder(image)
        text_features = self.encoder(text)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()