import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import linalg

import utils
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
        ):
        super().__init__()
        self.local_loss = local_loss
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.gather_with_grad = gather_with_grad

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        if self.world_size > 1 and self.local_loss:
            labels = labels + num_logits * self.rank
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T

        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss


class ArcCLIP(nn.Module): 
    def __init__(self):
        super().__init__()
        self.rank = utils.get_rank()
        self.world_size = utils.get_world_size()

    def _get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        if self.world_size > 1 and False:
            labels = labels + (num_logits * self.rank)
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        
        if self.world_size > 1:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            # print(all_text_features.norm(dim=1, keepdim=True))
            # exit()
            all_text_features /= all_text_features.norm(dim=1, keepdim=True)
            all_image_features /= all_image_features.norm(dim=1, keepdim=True)

            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logit_scale * all_text_features @ all_image_features.T

        else: 
            # print(text_features.norm(dim=1, keepdim=True))
            # exit()
            image_features = image_features/image_features.norm(dim=1, keepdim=True)
            text_features = text_features/text_features.norm(dim=1, keepdim=True)
            
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text
    
    def arc_face(self, image_features, text_features, logit_scale, label):
        s = 8
        m = 0.5
        sin_m = torch.sin(torch.tensor(m))
        cos_m = torch.cos(torch.tensor(m))
        w_L2 = linalg.norm(text_features.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(image_features, dim=1, keepdim=True)
        cos = (logit_scale * image_features @ text_features.T) / (x_L2 * w_L2)

        sin_m, cos_m = sin_m, cos_m
        one_hot = F.one_hot(label, num_classes=image_features.shape[0])
        sin = (cos ** 2) ** 0.5
        angle_sum = cos * cos_m - sin * sin_m
        cos = angle_sum * one_hot + cos * (1 - one_hot)
        cos = cos * s
        return cos 


    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        labels = self._get_ground_truth(device, text_features.shape[0])
        cos_image = self.arc_face(image_features, text_features, logit_scale, labels)
        cos_text = self.arc_face(text_features, image_features, logit_scale, labels)
        total_loss = (
            F.cross_entropy(cos_image, labels) +
            F.cross_entropy(cos_text, labels)
        ) / 2
        return total_loss

class DINOLoss(nn.Module): 
    def __init__(self): 
        super().__init__()
    
    def _get_labels(self, features:torch.Tensor): 
        return torch.arange(features.shape[0], device=features.device, dtype=torch.long)

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            
            all_text_features /= all_text_features.norm(dim=1, keepdim=True)
            all_image_features /= all_image_features.norm(dim=1, keepdim=True)


            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logit_scale * all_text_features @ all_image_features.T

        else: 
            image_features = image_features/image_features.norm(dim=1, keepdim=True)
            text_features = text_features/text_features.norm(dim=1, keepdim=True)

            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text


# class DiscoCLIPLoss(): 
#     def aggregate_disco(self, image_features, text_features):
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
#     bs = image_features.shape[0]

#     # We gather tensors from all gpus to get more negatives to contrast with.
#     gathered_image_features = [
#         torch.zeros_like(image_features) for _ in range(world_size)
#     ]
#     gathered_text_features = [
#         torch.zeros_like(text_features) for _ in range(world_size)
#     ]
#     dist.all_gather(gathered_image_features, image_features)
#     dist.all_gather(gathered_text_features, text_features)

#     gathered_image_features = torch.cat(gathered_image_features)
#     gathered_text_features = torch.cat(gathered_text_features)
    
#     all_image_features = gathered_image_features.requires_grad_(True)
#     all_text_features = gathered_text_features.requires_grad_(True)

#     image_features, text_features = all_image_features[bs*rank:bs*(rank+1)], all_text_features[bs*rank:bs*(rank+1)]

#     return image_features, text_features, all_image_features, all_text_features