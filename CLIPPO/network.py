import torch 
from torch import nn 
from torch.nn import functional as F
import timm 

class CLIPPO(nn.Module): 
    def __init__(self, temp): 
        super(CLIPPO, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', num_classes=0)
        self.temp =  temp

    def forward(self, image, text): 
        image_features = self.encoder(image)
        text_features = self.encoder(text)
        logits = (text_features @ image_features.T) / self.temp
        images_similarity = image_features @ image_features.T
        texts_similarity = text_features @ text_features.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temp, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()