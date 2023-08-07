import torch 
from torch import nn 
from torch.nn import functional as F
import timm 
import numpy as np
import disco
import torch
import torch.distributed as dist

class ContrastiveLoss(nn.Module): 
    def __init__(self, temp:float=1.0 ) -> None:
        super(ContrastiveLoss, self).__init__()
        self.temp = temp 
        self.ce = nn.CrossEntropyLoss()

    def forward(self, image_logits, text_logits, logit_scale:nn.Parameter=None, norm:bool=True, ): 
        if norm :
            image_logits = image_logits / image_logits.norm(dim=1, keepdim=True)
            text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)

        if isinstance(logit_scale, None): 
            logit_scale = 1

        image_logit_mat =  logit_scale* image_logits @ text_logits.t()
        text_logit_mat = logit_scale* text_logits @ image_logits.t()

        label = torch.arange(image_logit_mat.shape[0]).long().cuda() 
        image_loss = self.ce(image_logit_mat/self.temp,label)
        text_loss = self.ce(text_logit_mat/self.temp, label)

        return (image_loss+text_loss)/2

class DINOLoss(nn.Module): 
    def __init__(self, out_dim, temp, center_momentum:float=0.9): 
        super(DINOLoss, self).__init__()
        self.temp = temp
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = center_momentum
    
    def forward(self, text_logit, image_logit, norm:bool=False): 
        if norm :
            image_logit = image_logit / image_logit.norm(dim=1, keepdim=True)
            text_logit = text_logit / text_logit.norm(dim=1, keepdim=True)

        text_logit /= self.temp 
        image_logit /= self.temp 

        image_centered = F.softmax((image_logit - self.center), dim=-1)
        text_centered = F.softmax((text_logit - self.center), dim=-1)

        text_loss = torch.sum(-image_centered*F.log_softmax(text_logit, dim=-1), dim=-1)
        image_loss = torch.sum(-text_centered*F.log_softmax(image_logit, dim=-1), dim=-1) 
        self.update_center(torch.cat([text_logit, image_logit]))
        
        return (text_loss.mean()+image_loss.mean())/2
    
    @torch.no_grad()
    def update_center(self, output): 
        batch_center = torch.sum(output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
