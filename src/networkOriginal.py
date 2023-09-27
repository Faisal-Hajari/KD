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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(510 , num_classes)

        self.gradients = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.adaptive_avg_pool2d(out, int(512/3))
        y = out.view(out.size(0), -1)
        y = F.adaptive_max_pool1d(y, 510)
        out = self.linear(y)
        if out.requires_grad:
            out.register_hook(self.activations_hook)
        return out

    def get_activations_gradient(self):
        return self.gradients


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
class CLIPPO(nn.Module): 
    def __init__(self): 
        super(CLIPPO, self).__init__()
        self.encoder = timm.create_model('resnet18', pretrained=True, num_classes=0)

        #self.encoder = nn.Sequential(*[nn.Conv2d(3, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(), nn.Flatten(), nn.AdaptiveMaxPool1d(124), nn.Linear(124, 512)])
        #self.encoder = ResNet50(512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        projection_input = 512#768
        # 4 x 64 = 256
        #self.text_proj = nn.Sequential(*[nn.Linear(projection_input, 256), nn.ReLU(), nn.Linear(256, 2)])
        self.image_proj = nn.Sequential(*[nn.Linear(projection_input, 256), nn.ReLU(), nn.Linear(256, 2)])

    def forward(self, image, text): 
        # print("this is an image ")
        # print(type(text))
        # print(type(image))
        # print(text.shape)
        # print(image.shape)
        # plt.imshow(  text  )
        # print("done ------------------------------------")
        # exit(-1)
        image_features = self.image_proj(self.encoder(image))
        print("encode============================================")
        print((self.encoder(image).shape))
        text_features = self.image_proj (self.encoder(text))
        

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
