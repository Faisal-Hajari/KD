from torchvision import datasets, transforms
import torch 
from dataset import CC3M
transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224))
    ])
dataset = CC3M(transform, transform)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    drop_last=True,
)
from network import CLIPPO
clippo = CLIPPO(1)
clippo = clippo.cuda() 
for images, text in data_loader: 
        images = images.cuda() 
        text = text.cuda() 
        loss = clippo(image=images, text=text)
        print(loss)