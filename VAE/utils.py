import torch.nn.functional as F
import torch 
import os
import numpy as np
from PIL import Image

from torch.nn import PoissonNLLLoss


def get_lossfuc(loss):
    if loss=="normal":
        return F.mse_loss
        pass
    elif loss=="bernoulli":
        return F.binary_cross_entropy_with_logits
        pass
    elif loss=="poisson":
        return PoissonNLLLoss(log_input =True,reduction="sum")
        pass
    pass
def save_checkpoint(state,path,epoch):
    filepath = os.path.join(path, "cp{}.t7".format(epoch))
    torch.save(state, filepath)

def get_C(iteration,state,args):
    if iteration>args.C_stop_iter:
        C=args.C_max
    else:
        C=iteration/args.C_stop_iter*(args.C_max-args.C_start)+args.C_start
    C=torch.tensor(C)
    if state["use_cuda"]:
        C=C.cuda()
    return C

# def get_C(iteration,state,args):
#     C=iteration/state["total_iterations"]*args.C_max+1.0
#     C=torch.log(torch.tensor(C))
#     if state["use_cuda"]:
#         C=C.cuda()
#     return C
    
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def get_image(data_list):
    images=[]
    for data in data_list:
        data=data.transpose(0,2,3,1)
        N,H,W,C=data.shape
        image=np.ones((int(H+1),int((W+1)*N),int(C)))
        for n in range(N):
            image[0:H,n*(W+1):n*(W+1)+W,:]=data[n,:,:,:]
        images.append(image)
    images=np.vstack(images)
    images=np.clip(images,0,1)
    if C==3:
        images=(images*255).round().astype(np.uint8)
    # print(images.max())
    # print(images.min())
    images=torch.tensor(images)
    return images