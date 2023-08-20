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
#clippo.load_state_dict(torch.load("/home/cv/Desktop/KD/CLIPPO/clippo_test_10000.pt")) 
df = pd.read_csv("/home/cv/Desktop/cc3m_full_with_path.csv")
#print(df)
from torchvision import datasets, transforms
transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((32, 32))
    ])
dataset = TestMnist(transform, transform)

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
    font = ImageFont.truetype("/home/cv/Desktop/KD/CLIPPO/unifont-15.0.06.otf", font_size*3)
    draw.text((0, 0), new_txt, (text_brightness,text_brightness,text_brightness), font=font, spacing=spacing)
    img_resized = image.resize((image_size,image_size), Image.ANTIALIAS)
    return img_resized

features = []
labels_ar = [] 
clippo = clippo.cuda(1)
clippo.eval() 
valid_loader = torch.utils.data.DataLoader(dataset, 6000, shuffle=False, num_workers=32)
for images, _, labels in valid_loader: 
    with torch.no_grad():
        # out, _= clippo(torch.tensor(labels).cuda(1), images.cuda(1))#.squeeze()
        out = clippo.image_proj(clippo.encoder(images.cuda(1)))
        print((images[0].shape))
        # transform = T.ToPILImage()
        # img = transform(images[0])
        # img.show()
        # exit(-1)
        out /=out.norm(dim=1, keepdim=True)
        # out = out
        # print(out.shpae)
        features.append(out)
        labels_ar.append(labels[None, :])
        # # out, _= clippo(torch.tensor(labels).cuda(1), images.cuda(1))#.squeeze()
        # out = clippo.text_proj(clippo.encoder(images.cuda(1)))
        # out /=out.norm(dim=1, keepdim=True)
        # # out = out
        # features.append(out)
        # labels_ar.append(labels[None, :])

features = torch.concat(features).cpu().detach().numpy()
labels_ar = torch.concat(labels_ar, dim=1).detach().numpy()
print(labels_ar[0][0:10])
print(len(labels_ar[0]))
print(labels_ar[0].shape)
#tsne = TSNE().fit_transform(features)
tx, ty = features[:,0], features[:,1]


sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=pd.DataFrame({'tsne-2d-one':tx, 'tsne-2d-two':ty, 'y':labels_ar.squeeze()}),
    legend="full",
    alpha=0.2
)


features = []
labels_ar = [] 
for _, images, labels in valid_loader: 
    with torch.no_grad():
        # out, _= clippo(torch.tensor(labels).cuda(1), images.cuda(1))#.squeeze()
        out = clippo.image_proj(clippo.encoder(images.cuda(1)))
        out /=out.norm(dim=1, keepdim=True)
        # out = out
        features.append(out)
        labels_ar.append(labels[None, :])
        # # out, _= clippo(torch.tensor(labels).cuda(1), images.cuda(1))#.squeeze()
        # out = clippo.text_proj(clippo.encoder(images.cuda(1)))
        #  out /=out.norm(dim=1, keepdim=True)


features = torch.concat(features).cpu().detach().numpy()
labels_ar = torch.concat(labels_ar, dim=1).detach().numpy()
#tsne = TSNE().fit_transform(features)
tx, ty = features[:,0], features[:,1]


sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    style="y",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=pd.DataFrame({'tsne-2d-one':tx, 'tsne-2d-two':ty, 'y':labels_ar.squeeze()}),
    legend="full",
    alpha=1
)
#plt.title('Exploring Physical Attributes of Different Penguins')
plt.xlabel('feature x')
plt.ylabel('feature y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()