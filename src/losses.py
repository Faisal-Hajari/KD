import torch
import torch.nn as nn
from torch.functional import F

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