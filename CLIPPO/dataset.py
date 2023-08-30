from PIL import ImageFont, ImageDraw, Image 
import textwrap
from torch.utils.data import Dataset
import pandas as pd 
import torch 
import csv 
import time





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
    font = ImageFont.truetype("unifont-15.0.06.otf", font_size*3)
    draw.text((0, 0), new_txt, (text_brightness,text_brightness,text_brightness), font=font, spacing=spacing)
    img_resized = image.resize((image_size,image_size), Image.ANTIALIAS)
    return img_resized

class CC3M(Dataset):
    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        df = pd.read_csv("/home/cv/Desktop/cc3m_full_with_path.csv").iloc[:4_000]
        df['path'] = df['path'].apply(lambda x: x.replace("/home/temp/Desktop/", "/data/"))
        self.df = df
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms
    
    def __getitem__(self, index): 
        try: 
            row = self.df.iloc[index]
            image = row['path'].replace("/home/temp/Desktop", "/data")
            image = Image.open(image).convert('RGB')
            if self.image_transforms: 
                image = self.image_transforms(image) 
                if torch.isnan(image).any(): 
                    print('image in nan')
                    # assert False
            txt = row['caption']
            txt = render_text(txt).convert('RGB')
            if self.text_transfroms:
                txt = self.text_transfroms(txt)
            return image,txt #row['caption']  #txt
        
        except:  
            replace_idx = torch.randint(17_000, (1, 1)).item()
            print(f'index {index} faced an issue and was replaced with {replace_idx}')
            return self.__getitem__(replace_idx)
        
    def __len__(self): 
        return len(self.df)

from torchvision.datasets import MNIST
class Mnist(Dataset): 
    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        self.data = MNIST('mnist', download=True)
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
        image, text = self.data[index]
        if self.image_transforms: 
            image = self.image_transforms(image.convert('RGB'))  
        text = render_text(str(text), image_size=32).convert('RGB')
        if self.text_transfroms:
            text = self.text_transfroms(text)
        
        return image, text

class TestMnist(Dataset): 
    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        self.data = MNIST('mnist', download=True)
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
        image, text = self.data[index]
        if self.image_transforms: 
            image = self.image_transforms(image.convert('RGB'))  
        txtimage = render_text(str(text), image_size=32).convert('RGB')
        if self.text_transfroms:
            txtimage = self.text_transfroms(txtimage)
        
        return image, txtimage, text

class Mnist2(Dataset): 
    tmp=0
    tmp2=0
    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        self.data = MNIST('mnist', download=True)
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms 
        
    def __len__(self): 
        # return len(self.data)
        return 5421
    
    def __getitem__(self, index):
        # print('im in ----------------------------------------------')
        #print(type(index))
        
        with open('test.csv', 'r') as csv_file:
            mycsv = csv.reader(csv_file)
            mycsv = list(mycsv)
            indeex = mycsv[index]
            images, texts =[],[]
            for i in mycsv[index]:
                image, text = self.data[int(i)]
                # print("text ="+ str(text))
                if self.image_transforms: 
                    image = self.image_transforms(image.convert('RGB'))  
                text = render_text(str(text), image_size=32).convert('RGB')
                if self.text_transfroms:
                    text = self.text_transfroms(text)
                
                images.append(image)
                texts.append(text)

            return torch.stack(images),torch.stack(texts)