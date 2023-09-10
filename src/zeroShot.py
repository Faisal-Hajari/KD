from collections import OrderedDict
import pandas as pd 
from dataset import * 
import torch 
from network import *
from tim_and_bert import * 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from torchvision import  transforms
device ='cuda:2'

IMAGE_SIZE=32

path_to_wieghts = "clippo_mnist_font_change_65epoch.pt"
clippo = CLIPPO()
checkpoint = torch.load(path_to_wieghts)
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    # if 'backbone' in k or 'transformer' in k:
    name = k.replace('bbox', 'point')
    new_state_dict[name.replace('module.', '')] = v
clippo.load_state_dict(new_state_dict)
clippo = clippo.to(device)
clippo.eval() 



def render_text(txt:str, image_size: int=IMAGE_SIZE, font_size: int = 16, max_chars=768,
                       background_brightness=127, text_brightness=0,
                       lower=True, monospace=False, spacing=1, min_width=4,
                       resize_method="area", max_width=28):
    if len(txt)> max_chars:
        txt = txt[:max_chars]
    if lower: 
        txt = txt.lower() 
    wrapper = textwrap.TextWrapper(width=max_width)
    lines = wrapper.wrap(txt) 
    new_txt = ""
    for line in lines: 
        new_txt+= line+'\n'
    image = Image.new("RGBA", (image_size*3,image_size*3), (background_brightness,background_brightness,background_brightness))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("unifont-15.0.06.otf", font_size*3)
    draw.text((0, 0), new_txt, (text_brightness,text_brightness,text_brightness), font=font, spacing=spacing)
    img_resized = image.resize((image_size,image_size), Image.Resampling.LANCZOS)
    return img_resized.convert("RGB")

def get_label_vectors(labels:list, clippo:nn.Module, transform, device): 
    idx_to_name = {i:k for i, k in enumerate(labels)}

    input_to_model = [render_text(str(txt), image_size=IMAGE_SIZE).convert('RGB') for txt in labels]
    input_to_model = torch.stack([transform(txt) for txt in input_to_model])
    with torch.no_grad(): 
        out = clippo.image_proj(clippo.encoder(input_to_model.to(device)))
        out /=out.norm(dim=-1, keepdim=True)
    
    return idx_to_name, out

def inferance(encoded_images:torch.Tensor, encoded_labels:torch.Tensor): 
    _, predictions = (encoded_images@encoded_labels.T).max(dim=1)
    return predictions




transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    ])

dataset = TestMnist(transform, transform)
# dataset = TestCifar(transform, transform)

valid_loader = torch.utils.data.DataLoader(dataset, 1000, shuffle=False, num_workers=1)

features = []
labels_ar = [] 
for images, _, labels in valid_loader: 
    with torch.no_grad():
        out = clippo.image_proj(clippo.encoder(images.to(device)))
        out /=out.norm(dim=1, keepdim=True)
        features.append(out)
        labels_ar.append(labels)

features = torch.concat(features)
labels_ar = torch.concat(labels_ar)

idx_to_name,  labels_lol= get_label_vectors([0,1,2
,3,4,5,6,7,8,9], clippo, transform, device)
pred = inferance(features, labels_lol.to(device)).cpu()
total = (pred == labels_ar).sum() / len(labels_ar)
print(total)

