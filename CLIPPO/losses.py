import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
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
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    all_image_features = torch.cat(
        torch.distributed.nn.all_gather(image_features),
        dim=0)
    all_text_features = torch.cat(
        torch.distributed.nn.all_gather(text_features),
        dim=0)

    return all_image_features, all_text_features

def gather(device, all_image_features, image_feature): 
    tt = torch.cat(
        torch.distributed.nn.all_gather(torch.tensor(utils.get_rank(), dtype=torch.long, device=device)[None, ])
    )
    file = open(f"heree{utils.get_rank()}.txt", 'w+')
    file.write(f"{all_image_features.grad}\n\n{image_feature.grad}")
    file.close()
    exit()


class ClipLoss(nn.Module):
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
            
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T

        else: 
            image_features = image_features/image_features.norm(dim=1, keepdim=True)
            text_features = text_features/text_features.norm(dim=1, keepdim=True)

            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device

        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale)
        labels = self._get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss

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