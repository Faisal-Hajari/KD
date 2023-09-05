import pandas as pd 
from dataset import * 
from tqdm import tqdm 
from PIL import ImageFont, ImageDraw, Image 
import torch 
from network import CLIPPO
from tim_and_bert import * 
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA
import torchvision.transforms as T

clippo = CLIPPO()
clippo = clippo.cpu()
clippo.load_state_dict(torch.load("trash1.pt")) 

from torchvision import datasets, transforms
transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((32, 32))
    ])
dataset = TestCifar(transform, transform)
from PIL import ImageFont, ImageDraw, Image 

def render_text(txt:str, image_size: int=224, font_size: int = 16, max_chars=768,
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
    font = ImageFont.truetype("/home/temp/Desktop/CLIPPO/unifont-15.0.06.otf", font_size*3)
    draw.text((0, 0), new_txt, (text_brightness,text_brightness,text_brightness), font=font, spacing=spacing)
    img_resized = image.resize((image_size,image_size), Image.ANTIALIAS)
    return img_resized


import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA

features = []
labels_ar = [] 
clippo = clippo.cuda(1)
clippo.eval() 
valid_loader = torch.utils.data.DataLoader(dataset, 2000, shuffle=False, num_workers=32)
for images, _, labels in valid_loader: 
    with torch.no_grad():
        #out,_= clippo(torch.tensor(labels).cuda(1), images.cuda(1))#.squeeze()
        out = clippo.image_proj(clippo.encoder(images.cuda(1))).squeeze()
        #out = clippo.logit_scale*out/ out.norm(dim=1, keepdim=True)
        features.append(out)
        labels_ar.append(labels[None, :])

features = torch.concat(features).cpu().detach().numpy()
labels_ar = torch.concat(labels_ar, dim=1).detach().numpy()

#tsne = TSNE(n_jobs=-1).fit_transform(features)
tx, ty = features[:, 0].squeeze(), features[:, 1].squeeze()
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=pd.DataFrame({'tsne-2d-one':tx, 'tsne-2d-two':ty, 'y':labels_ar.squeeze()}),
    legend="full",
    alpha=0.4
)


#tsne = TSNE(n_jobs=-1).fit_transform(features)
tx, ty = features[:, 0].squeeze(), features[:, 1].squeeze()
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=pd.DataFrame({'tsne-2d-one':tx, 'tsne-2d-two':ty, 'y':labels_ar.squeeze()}),
    legend="full",
    alpha=0.4
)
print("here ",len(tx))
plt.xlabel('feature x')
plt.ylabel('feature y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig("figgg.png")



