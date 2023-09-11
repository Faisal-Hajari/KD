import math
import sys 
import argparse

import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch 
from torch import nn 

import torch
import torchvision.transforms as T

import utils
from dataset import  Cifar, Mnist
from models.clips import CLIPPO
from losses import *
import deepspeed
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DistributedSampler, DataLoader

def get_loss(args): 
    loss = {
        'clip_loss': ClipLoss(args.local_loss, 
                              args.gather_grad),
        'dino_loss':None,
        'disco_clip':None, 
        #add losses here
    }[args.loss_name]
    return loss 

def train_one_epoch(clippo, data_loader, optimizer, lr_schedule, epoch, args): 
    criterion = get_loss(args)
    total_loss = 0 
    for images, text in data_loader: 
        images = images.cuda().half()
        text = text.cuda().half()
        image_features, text_features, logit_scale = clippo(image=images.squeeze(), text=text.squeeze())
        loss = criterion(image_features, text_features, logit_scale)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)    

        clippo.backward(loss)
        if args.clip_grad:
            utils.clip_gradients(clippo, args.clip_grad)
        optimizer.step()

    clippo.tput_timer.update_epoch_count()
    total_loss += loss.item()
    #TODO: return logs 
    return total_loss/len(data_loader)

def main(args): 
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()


    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    torch.distributed.barrier()
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((32, 32))
    ])

    #dataset = Cifar(transform, transform) #TODO: write function that return dataset from args 
    dataset = Mnist(transform, transform)
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
    # if utils.has_batchnorms(clippo):
    #     clippo = nn.SyncBatchNorm.convert_sync_batchnorm(clippo)

    if args.fused_adam: 
        optimizer = FusedAdam(clippo.parameters(), args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.)
    else: 
        optimizer = torch.optim.AdamW(clippo.parameters(), args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.)

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    print(f"model and optimizer are ready.")   
    #TODO: add option for resume training. 
    start_epoch = 0 
    print(f"Starting {args.project_name} training !")

    clippo, optimizer, _, lr_schedule = deepspeed.initialize(
        model=clippo,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_schedule,
        dist_init_required=True
    )

    for epoch in range(start_epoch, args.epochs):
        # ============ training one epoch of CLIPPO ... ============
        loss = train_one_epoch(clippo, data_loader, optimizer,
                        lr_schedule, epoch, args)
        if utils.get_rank() == 0:
            print(f"[{epoch}] losss: {loss}")
        # ============ writing logs ... ============
        #TODO: add logs 
        if not epoch % 10:
            if utils.get_rank() == 0:
                torch.save(clippo.state_dict(), f'{args.project_name}_{epoch}.pt')

def get_args_parser():
    parser = argparse.ArgumentParser('CLIPPO', add_help=False)

    parser.add_argument('--project_name', type=str, default='clippo_mnist_w_red_11k')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--checkpoint_interval", default=10, type=int, help="How often to save a model checkpoint")
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    
    parser.add_argument('--loss_name', type=str, default='clip_loss')
    parser.add_argument('--local_loss', type=utils.bool_flag, default=True)
    parser.add_argument('--gather_grad', type=utils.bool_flag, default=True)
    parser.add_argument('--fused_adam', type=utils.bool_flag, default=True)

    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")

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
    

    parser = deepspeed.add_config_arguments(parser)

    return parser


if __name__ == '__main__': 
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)