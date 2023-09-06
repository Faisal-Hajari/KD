from PIL import ImageFont, ImageDraw, Image 
import textwrap
from torch.utils.data import Dataset
import pandas as pd 
import torch 
import csv 
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as T



def render_text(txt:str, image_size: int=224, font_size: int = 16, max_chars=768,
                       background_brightness=127, text_brightness=0,
                       lower=True, monospace=False, spacing=1, min_width=4,
                       resize_method="area", max_width=28, font_name="unifont-15.0.06.otf", noise=False):
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
    
    font = ImageFont.truetype(font_name, font_size*3)
    draw.text((0, 0), new_txt, (text_brightness,text_brightness,text_brightness), font=font, spacing=spacing)
    img_resized = image.resize((image_size,image_size))
    
    if noise:
        output = np.copy(np.array(img_resized))

        # add salt (white grain)
        nb_salt = np.ceil(0.005 * output.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
        output[coords] = 1

        # add pepper (black grain)
        nb_pepper = np.ceil(0.005 * output.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
        output[coords] = 0

        img_resized = Image.fromarray(output)
    
    #img_resized.show()
    #exit(-1)
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

from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST

class NaiveCounter:
    number = 0

    def __add__(self, adder):
        if self.number < 3:
            self.number += adder
            return self.number
        else:
            self.number = 0
            return self.number


class Cifar(Dataset): 
    
    fonts = ["deval.otf", "unifont-15.0.06.otf", "fcb.otf", "fcl.otf"]
    airplane_font = NaiveCounter()
    automobile_font = NaiveCounter()
    bird_font = NaiveCounter()
    cat_font = NaiveCounter()
    deer_font = NaiveCounter()
    dog_font = NaiveCounter()
    frog_font = NaiveCounter()
    horse_font = NaiveCounter()
    ship_font = NaiveCounter()
    truck_font = NaiveCounter()
    

    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        self.dataset = CIFAR10(root='data/', download=True)
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms
        self.classes = self.dataset.classes
        print(self.classes)
    class_count = {}

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, index):
        image, text = self.dataset[index]

        #print("type image ", type(image))
        if self.image_transforms: 
            image = self.image_transforms(image.convert('RGB'))
        if str(text)=='0':
            text = render_text('airplane', image_size=224, font_name=self.fonts[self.airplane_font + 1]).convert('RGB')
        elif str(text)=='1':
            text = render_text('automobile', image_size=224, font_name=self.fonts[self.automobile_font + 1]).convert('RGB')
        elif str(text) =='2':
            text = render_text('bird', image_size=224, font_name=self.fonts[self.bird_font + 1]).convert('RGB')
        elif str(text) =='3':
            text = render_text('cat', image_size=224, font_name=self.fonts[self.cat_font + 1]).convert('RGB')
        elif str(text)=='4':
            text = render_text('deer', image_size=224, font_name=self.fonts[self.deer_font + 1]).convert('RGB')
        elif str(text) =='5':
            text = render_text('dog', image_size=224, font_name=self.fonts[self.dog_font + 1]).convert('RGB')
        elif str(text) =='6':
            text = render_text('frog', image_size=224, font_name=self.fonts[self.frog_font + 1]).convert('RGB')
        elif str(text) =='7':
            text = render_text('horse', image_size=224, font_name=self.fonts[self.horse_font + 1]).convert('RGB')
        elif str(text)=='8':
            text = render_text('ship', image_size=224, font_name=self.fonts[self.ship_font + 1]).convert('RGB')
        elif str(text) =='9':
            text = render_text('truck', image_size=224, font_name=self.fonts[self.truck_font + 1]).convert('RGB')
        if self.text_transfroms:
            text = self.text_transfroms(text)
        # print("text ====", label)
        # print(type(image))
        # transform = T.ToPILImage()
        # img = transform(image)
        # img.show()
        # exit(-1)
        # for _, index in self.dataset:
        #     label = self.classes[index]
        #     if label not in self.class_count:
        #         self.class_count[label] = 0
        #     self.class_count[label] += 1
        #     print(label)
        return image, text

class Mnist(Dataset): 
    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        self.data = MNIST('mnist', download=True)
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
        image, text = self.data[index]
        # print(index)
        # print("len is : "+ str(self.data[0][1]))
        # exit(-1)
        print("image type ",type(image))
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


class TestCifar(Dataset): 
    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        self.data =  CIFAR10(root='data/', train=False)
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
        image, text = self.data[index]
        #   if str(text)=='0': 
        #     text = 'airplane'
        # elif str(text)=='1': 
        #     text='automobile'
        # elif str(text)=='2': 
        #     text = 'bird'
        # elif str(text)=='3': 
        #     text='cat'
        # elif str(text)=='4': 
        #     text='deer'
        # elif str(text)=='5': 
        #     text = 'dog'
        # elif str(text)=='6': 
        #     text = 'frog'
        # elif str(text)=='7': 
        #     text = 'horse'
        # elif str(text)=='8': 
        #     text = 'ship'
        # elif str(text)=='9': 
        #     text = 'truck'
        if self.image_transforms: 
            image = self.image_transforms(image.convert('RGB'))  
        if str(text)=='0': 
            txtimage = render_text('airplane', image_size=224).convert('RGB')
        elif str(text)=='1': 
            txtimage = render_text('automobile', image_size=224).convert('RGB')
        elif str(text)=='2': 
            txtimage = render_text('bird', image_size=224).convert('RGB')
        elif str(text)=='3': 
            txtimage = render_text('cat', image_size=224).convert('RGB')
        elif str(text)=='4': 
            txtimage = render_text('deer', image_size=224).convert('RGB')
        elif str(text)=='5': 
            txtimage = render_text('dog', image_size=224).convert('RGB')
        elif str(text)=='6': 
            txtimage = render_text('frog', image_size=224).convert('RGB')
        elif str(text)=='7': 
            txtimage = render_text('horse', image_size=224).convert('RGB')
        elif str(text)=='8': 
            txtimage = render_text('ship', image_size=224).convert('RGB')
        elif str(text)=='9': 
            txtimage = render_text('truck', image_size=224).convert('RGB')
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
                print("text ="+ str(text))
                if self.image_transforms: 
                    image = self.image_transforms(image.convert('RGB'))  
                text = render_text(str(text), image_size=32).convert('RGB')
                if self.text_transfroms:
                    text = self.text_transfroms(text)
                
                images.append(image)
                texts.append(text)

            return torch.stack(images),torch.stack(texts)




            print("here index ="+str(indeex))

        # image, text = self.data[indeex]
        # print("here is the content ="+ str(text))
        # print('index'+str((index)))
        # #print(type(image))
        # print('text'+str(type(text)))
        # print('image '+str(type(image)))
        # if self.image_transforms: 
        #     image = self.image_transforms(image.convert('RGB'))  
        # text = render_text(str(text), image_size=32).convert('RGB')
        # if self.text_transfroms:
        #     text = self.text_transfroms(text)

        # image.show()
        #print("image len is ",(image))
        #print(index)
        print('im out ----------------------------------------------')
        tmp=tmp+1
        tmp2=tmp2+1
        return image, text