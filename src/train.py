import math
import sys 
import argparse

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torch 
from torch import nn 
#import wandb
from random import Random
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import wandb
import utils
from dataset import  Cifar,Mnist
from network import CLIPPO
from tim_and_bert import CLIP, DINOLoss

import deepspeed
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DistributedSampler, DataLoader

def train_one_epoch(clippo, data_loader, optimizer, lr_schedule, epoch, args): 

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # loss_dino = DINOLoss(512, 0.04).cuda()
    for images, text in data_loader: 
        with torch.cuda.amp.autocast(False):
            images = images.cuda() 
            text = text.cuda() 
            #TODO: add discoCLip here. 
            logits_per_image, logits_per_text = clippo(image=images.squeeze(), text=text.squeeze())
            label = torch.arange(logits_per_image.shape[0]).long().cuda() 
            loss1 = loss_img(logits_per_image, label)
            loss2 = loss_txt(logits_per_text, label)
            total_loss = (loss1+loss2)/2
            print(total_loss)
            # if utils.get_rank() == 0:
            #    wandb.log({"mini_batch_loss": total_loss.item()})
        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()), force=True)
            sys.exit(1)    
        
        clippo.backward(total_loss)
        if args.clip_grad:
            utils.clip_gradients(clippo, args.clip_grad)
        optimizer.step()

    clippo.tput_timer.update_epoch_count()
    #TODO: return logs 
    # torch.cuda.synchronize()

def main(args): 
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()


    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    torch.distributed.barrier()

    # if utils.get_rank() == 0:
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="my-awesome-project",
            
    #         # track hyperparameters and run metadata
    #         config={
    #             "learning_rate": args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
    #             "architecture": "CLIPPO",
    #             "dataset": "CC3M_test",
    #             "epochs": args.epochs,
    #         }
    #     )
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224))
    ])

    dataset = Cifar(transform, transform) #('mnist', download=True, transform=transform, target_transform=lambda x: torch.tensor(x, dtype=torch.long))#Mnist(transform, transform)

    sampler = DistributedSampler(dataset, shuffle=False)
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    clippo = CLIPPO()
    clippo = clippo.cuda() 
    if utils.has_batchnorms(clippo):
        clippo = nn.SyncBatchNorm.convert_sync_batchnorm(clippo)

    #optimizer = torch.optim.AdamW(clippo.parameters())
    optimizer = FusedAdam(clippo.parameters(), args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.)

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    print(f"model and optimizer are ready.")   
    #TODO: add option for resume training. 
    start_epoch = 0 
    print("Starting clippo training !")

    clippo, optimizer, _, lr_schedule = deepspeed.initialize(
        model=clippo,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_schedule,
        dist_init_required=True
    )

    for epoch in range(start_epoch, args.epochs):
        # data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of CLIPPO ... ============
        print("before train ")
        loss = train_one_epoch(clippo, data_loader, optimizer,
                        lr_schedule, epoch, args)
        print("losss",loss)
        # ============ writing logs ... ============
        #TODO: add logs 
        # print(loss)

        if not epoch % 10:
            if utils.get_rank() == 0:
                torch.save(clippo.state_dict(), f'clippo{epoch}.pt')

def get_args_parser():
    parser = argparse.ArgumentParser('CLIPPO', add_help=False)

    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs of training.')
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.") # with the actual 100? or 150?
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument("--checkpoint_interval", default=10, type=int, help="How often to save a model checkpoint")

    parser = deepspeed.add_config_arguments(parser)

    return parser


if __name__ == '__main__': 
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)