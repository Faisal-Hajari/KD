import time
import os
import shutil
from typing import Callable, Any, Tuple

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from streaming import MDSWriter, StreamingDataset


# the location of our dataset
in_root = "./dataset"

# the location of the "remote" streaming dataset (`sds`).
# Upload `out_root` to your cloud storage provider of choice.
out_root = "./sds"
out_train = "./sds/train"
out_test = "./sds/test"

# the location to download the streaming dataset during training
local = './local'
local_train = './local/train'
local_test = './local/test'

# toggle shuffling in dataloader
shuffle_train = True
shuffle_test = False

# shard size limit, in bytes
size_limit = 1 << 25

# training batch size
batch_size = 32

# training hardware parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

# number of training epochs
train_epochs = 2 # increase the number of epochs for greater accuracy

# Hashing algorithm to use for dataset
hashes = ['sha1' ,'xxh64']


# Download the CIFAR10 raw dataset using torchvision
train_raw_dataset = CIFAR10(root=in_root, train=True,download=True)
test_raw_dataset = CIFAR10(root=in_root, train=False)



def write_datasets(dataset: Dataset, split_dir: str) -> None:
    fields = {
        'x': 'pil',
        'y': 'int',
    }
    indices = np.random.permutation(len(dataset))
    indices = tqdm(indices)
    with MDSWriter(out=split_dir, columns=fields, hashes=hashes, size_limit=size_limit) as out:
        for i in indices:
            x, y = dataset[i]
            out.write({
                'x': x,
                'y': y,
            })
class CIFAR10Dataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable
                ) -> None:
        super().__init__(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        x = obj['x']
        y = obj['y']
        return self.transforms(x), y

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
])



remote_test ="/home/cv/Desktop/c"
remote_train ="/home/cv/Desktop/cTest"

train_dataset = CIFAR10Dataset(remote_train, local_train, shuffle_train, batch_size=batch_size, transforms=transformation)
test_dataset  = CIFAR10Dataset(remote_test, local_test, shuffle_test, batch_size=batch_size, transforms=transformation)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
if not os.path.exists(out_train):
    write_datasets(train_raw_dataset, out_train)
    write_datasets(test_raw_dataset, out_test)
test_dataset  = CIFAR10Dataset(remote_test, local_test, shuffle_test, batch_size=batch_size, transforms=transformation)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

def fit(model: nn.Module, train_dataloader: DataLoader) -> Tuple[float, float]:
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for imgs, labels in tepoch:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            labels_hat = model(imgs)
            loss = criterion(labels_hat, labels)
            train_running_loss += loss.item()
            _, preds = torch.max(labels_hat.data, 1)
            train_running_correct += (preds == labels).sum().item()
            loss.backward()
            optimizer.step()

    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)

    return train_loss, train_accuracy



def eval(model: nn.Module, test_dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with tqdm(test_dataloader, unit="batch") as tepoch:
        for imgs, labels in tepoch:
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            val_running_correct += (preds == labels).sum().item()

    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)

    return val_loss, val_accuracy

for epoch in range(train_epochs):
    train_epoch_loss, train_epoch_accuracy = fit(model, train_dataloader)
    print(f'epoch: {epoch+1}/{train_epochs} Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}')
    val_epoch_loss, val_epoch_accuracy = eval(model, test_dataloader)
    print(f'epoch: {epoch+1}/{train_epochs} Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
