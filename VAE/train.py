# train and save model
from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar as Bar
import numpy as np
from dataset import *
from model import *
from utils import *



parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='VAE_2', type=str)
parser.add_argument( '--loss', default='poisson', type=str)
parser.add_argument( '--interval', default=20, type=int)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epoch', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--original', default=0, type=int)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                    metavar='W')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')

parser.add_argument('--save_dir', default='test/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()


args.save_dir="results/"+args.save_dir
use_cuda = torch.cuda.is_available()

random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)




def main():
    state={}
    # Data TODO: load dataloader dataloader,testdataloader
    state["dataloader"]=get_dataloader(args)
    state["testdataloader"]=get_dataloader(args,shuffle=False)
    # print(len(state["testdataloader"]))
    # Model TODO: load model"""
    state["model"]=get_model(args,use_cuda)
    cudnn.benchmark = True
    state["criterion"] = get_lossfuc(args.loss)
    state["criterion_normal"] = get_lossfuc("normal")
    state["criterion_ber"] = get_lossfuc("bernoulli")
    
    optimizer = optim.Adam(state["model"].parameters(), lr=args.lr, betas=(0.9,0.999),weight_decay=args.weight_decay,amsgrad=True)

    state["optimizer"]=optimizer
    state["use_cuda"]=use_cuda
    state["args"]=args
    state["epoch"]=args.epoch
    state["total_iterations"]=state["epoch"]*len(state["dataloader"])
    print("Start training, there are total {} iterations".format(state["total_iterations"]))
    if args.dataset=="VAE":
        train(state)
    elif args.dataset=="VAE_2":
        train_2(state)
    Z=test(state)
    print(Z.shape)
    if args.dataset=="VAE":
        torch.save(Z,"VAE_poi.pkl")
    elif args.dataset=="VAE_2":
        torch.save(Z,"VAE_2.pkl")
    


def train(state):
    dataloader=state["dataloader"]
    model=state["model"]
    criterion=state["criterion"]
    optimizer=state["optimizer"]
    # writer=state["writer"]
    use_cuda=state["use_cuda"]
    args=state["args"]
    epoch=state["epoch"]
    iteration=0
    print(args)


    # switch to train mode
    model.train()
    test_batch=None

    for epoch in range(state["epoch"]):
        print("This is epoch {}".format(epoch))
        # continue

        batch_time = AverageMeter()
        data_time = AverageMeter()
        recon_losses = AverageMeter()
        KLD_losses = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=int(len(dataloader)/args.interval)+1 )

        
        for batch_idx, inputs in enumerate(dataloader):
            inputs=torch.tensor(inputs[0])
            iteration+=1
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs= inputs.cuda()
            
            if test_batch is None:
                test_batch=inputs.clone()

            # compute output
            outputs, mean, logvar = model(inputs)
            if args.loss=="poisson":
                recon_loss = criterion(outputs, inputs ) #,reduction='sum')#*1e-5
            else:
                recon_loss = criterion(outputs, inputs,reduction='sum')#*1e-5
            KLD_loss,KLD_loss_dim=model.KL_divergence(mean,logvar)


            loss=recon_loss+args.gamma*KLD_loss


            recon_losses.update(recon_loss.data.item(), inputs.size(0))
            KLD_losses.update(KLD_loss.data.item(), inputs.size(0))
            losses.update(loss.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # tensorboard
    #         if iteration%args.interval==1:
            if iteration%(args.interval)==0:

                bar.suffix  = 'Epoch {epoch} ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | recon_loss: {recon_loss: .4f} | KLD_loss: {KLD_loss: .4f} '.format(
                            epoch=epoch,
                            batch=batch_idx + 1,
                            size=len(state["dataloader"]),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            recon_loss=recon_losses.avg,
                            KLD_loss=KLD_losses.avg,
                            )
                bar.next()
        bar.finish()
    return (recon_losses.avg, KLD_losses.avg,losses.avg)

def train_2(state):
    dataloader=state["dataloader"]
    model=state["model"]
    criterion_normal=state["criterion_normal"]
    criterion_ber=state["criterion_ber"]
    optimizer=state["optimizer"]
    use_cuda=state["use_cuda"]
    args=state["args"]
    epoch=state["epoch"]
    iteration=0
    print(args)
    # C=get_C(iteration,state,args)

    # switch to train mode
    model.train()
    test_batch=None

    for epoch in range(state["epoch"]):
        print("This is epoch {}".format(epoch))
        # continue

        batch_time = AverageMeter()
        data_time = AverageMeter()
        recon_losses = AverageMeter()
        KLD_losses = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=int(len(dataloader)/args.interval)+1 )

        
        for batch_idx, inputs in enumerate(dataloader):
            inputs=torch.tensor(inputs[0])

            iteration+=1
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs= inputs.cuda()
            
            if test_batch is None:
                test_batch=inputs.clone()

            # compute output
            outputs, mean, logvar = model(inputs)
            index_arr=inputs>1e-6
            # print(index_arr.float())
            recon_loss_ber = criterion_ber(outputs[1], 1-index_arr.float(),reduction='mean')*0.1
            outputs[0][1-index_arr]=0.0


            # inputs[index_arr]=0.0
            recon_loss_normal = criterion_normal(outputs[0], inputs,reduction='sum')*1.
            recon_loss=recon_loss_ber+recon_loss_normal

            KLD_loss,KLD_loss_dim=model.KL_divergence(mean,logvar)


            loss=recon_loss+args.gamma*KLD_loss


            recon_losses.update(recon_loss.data.item(), inputs.size(0))
            KLD_losses.update(KLD_loss.data.item(), inputs.size(0))
            losses.update(loss.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # tensorboard
    #         if iteration%args.interval==1:
            if iteration%(args.interval)==0:
                print()
                print(recon_loss_normal.item(),recon_loss_ber.item())

                bar.suffix  = 'Epoch {epoch} ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | recon_loss: {recon_loss: .4f} | KLD_loss: {KLD_loss: .4f} '.format(
                            epoch=epoch,
                            batch=batch_idx + 1,
                            size=len(state["dataloader"]),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            recon_loss=recon_losses.avg,
                            KLD_loss=KLD_losses.avg,
                            )
                bar.next()
        bar.finish()
    return (recon_losses.avg, KLD_losses.avg,losses.avg)

def test(state):
    dataloader=state["testdataloader"]
    model=state["model"]
    criterion=state["criterion"]
    optimizer=state["optimizer"]
    # writer=state["writer"]
    use_cuda=state["use_cuda"]
    args=state["args"]
    epoch=state["epoch"]
    iteration=0
    print(args)

    model.eval()
    test_batch=None

    print("This is epoch {}".format(epoch))
    # continue

    batch_time = AverageMeter()
    data_time = AverageMeter()
    recon_losses = AverageMeter()
    KLD_losses = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=int(len(dataloader)/args.interval)+1 )
    Z=[]

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            inputs=torch.tensor(inputs[0])
            # if args.dataset!="dsprites":
            #     inputs=inputs.float()
            iteration+=1
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs= inputs.cuda()
            
            if test_batch is None:
                test_batch=inputs.clone()
            # compute output
            outputs, mean, logvar = model(inputs)
            if use_cuda:
                Z.append(mean.cpu().numpy())
            else:
                Z.append(mean.numpy())
    return np.vstack(Z)




if __name__ == '__main__':
    main()
