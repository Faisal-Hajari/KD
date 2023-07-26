import math
import sys 
import argparse

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torch 
from torch import nn 
import wandb

import utils
from dataset import CC3M
from network import CLIPPO

def train_one_epoch(clippo, data_loader, optimizer, lr_schedule, epoch, fp16_scaler, args): 
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    for images, text in data_loader: 
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            images = images.cuda() 
            text = text.cuda() 
            #TODO: add discoCLip here. 
            logits_per_image, logits_per_text = clippo(image=images, text=text)
            
            ground_truth = torch.arange(len(images),dtype=torch.long).cuda()

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward(retain_graph=True)
            if utils.get_rank() == 0:
                wandb.log({"mini_batch_loss": total_loss.item()})


        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()), force=True)
            sys.exit(1)    
        
        optimizer.zero_grad()
        if fp16_scaler is None:
            total_loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(clippo, args.clip_grad)
            optimizer.step()
        else: 
            fp16_scaler.scale(total_loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(clippo, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        if utils.get_rank() == 0:
            torch.save(clippo.state_dict(), 'clippo.pt')
    # print(loss)
    # print("HELLO")
    #TODO: return logs 
    # torch.cuda.synchronize()
    # return loss.item()

def main(args): 
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    if utils.get_rank() == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
                "architecture": "CLIPPO",
                "dataset": "CC3M_test",
                "epochs": args.epochs,
            }
        )
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224))
    ])
    dataset = CC3M(transform, transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")
    clippo = CLIPPO()
    clippo = clippo.cuda() 
    if utils.has_batchnorms(clippo):
        clippo = nn.SyncBatchNorm.convert_sync_batchnorm(clippo)

    optimizer = torch.optim.AdamW(clippo.parameters())
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

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
    for epoch in range(start_epoch, args.epochs):
        # data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of CLIPPO ... ============
        loss = train_one_epoch(clippo, data_loader, optimizer,
                        lr_schedule, epoch, fp16_scaler, args)
        
        # ============ writing logs ... ============
        #TODO: add logs 
        # print(loss)
    if utils.get_rank() == 0:
        torch.save(clippo.state_dict(), 'clippo.pt')
def get_args_parser():
    parser = argparse.ArgumentParser('CLIPPO', add_help=False)

    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser


if __name__ == '__main__': 
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)