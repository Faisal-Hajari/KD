from PIL import ImageFont, ImageDraw, Image 
import textwrap
import pandas as pd 
from torch.utils.data import Dataset

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

class CC3M(Dataset):
    def __init__(self, text_transforms=None, image_transforms=None) -> None:
        df = pd.read_csv("/home/temp/Desktop/CLIPPO/cc3m_full_with_path.csv")
        df['path'] = df['path'].apply(lambda x: x.replace("/home/temp/Desktop/", "/data/")) 
        self.df = df
        self.text_transfroms = text_transforms
        self.image_transforms = image_transforms
    
    def __getitem__(self, index): 
        row = self.df.iloc[index]
        image = row['path']
        image = Image.open(image).convert('RGB')
        if self.image_transforms: 
            image = self.image_transforms(image) 
        txt = row['caption']
        txt = render_text(txt).convert('RGB')
        if self.text_transfroms:
            txt = self.text_transfroms(txt)
        return image, txt

    def __len__(self): 
        return len(self.df)