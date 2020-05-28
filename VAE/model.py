# deep learning models
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_ as kaiming
import torch.nn.functional as F

# class Reshape(nn.Module):
#     def __init__(self, shape):
#         super(Reshape, self).__init__()
#         self.shape = shape

#     def forward(self, x):
#         return x.view(self.shape)

class VAE(nn.Module):
    def __init__(self,D=558, z_size=10):
        super(VAE, self).__init__()
        self.z_size = z_size
        self.D=D

        self.encoder = nn.Sequential(
            nn.Linear(self.D, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.z_size*2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.D),
        )


    def reparametrize(self, mu, logvar): # https://zhuanlan.zhihu.com/p/27549418
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            z = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            z = torch.FloatTensor(std.size()).normal_()
        return z*std+mu
        # return z.mul(std).add_(mu)
    
    def KL_divergence(self,mean,logvar):# TODO: why the kld make mean goes to zero and logvar goes to -100?
        kl_divergence = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kl_divergence = (kl_divergence).mul_(-0.5)
        kl_divergence_each_sample=torch.sum(kl_divergence.sum(1))/mean.shape[0]
        kl_divergence_each_dim=kl_divergence.mean(0)
        return kl_divergence_each_sample, kl_divergence_each_dim
        # kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())/mean.shape[0]
        # return kl_divergence
        pass


    def forward(self, x):
        mean_var = self.encoder(x)
        std_normal=torch.randn((x.shape[0],self.z_size))
        mean = mean_var[:, :self.z_size]
        logvar = mean_var[:, self.z_size:] # predict log varance because it is more stable and make sure var should always be positive https://stats.stackexchange.com/questions/353220/why-in-variational-auto-encoder-gaussian-variational-family-we-model-log-sig
        z = self.reparametrize(mean,logvar)
        output = self.decoder(z).view(x.size())

        return output, mean, logvar
    
    # def forward_with_mean(self,x,loss="normal"):
    #     mean_var = self.encoder(x)
    #     mean = mean_var[:, :self.z_size]
    #     output = self.decoder(mean).view(x.size())
    #     if loss=="bernoulli":
    #         output=torch.sigmoid(output)

    #     return output, mean

    # def forward_explore_var(self,x,axis=0,interval=0.3,max_range=3,loss="normal"):
    #     assert x.shape[0]==1
    #     var_range=torch.arange(-max_range, max_range+0.01, interval)
    #     num_image=len(var_range)
    #     mean_var = self.encoder(x)
    #     mean = mean_var[:, :self.z_size]
    #     means=mean.repeat(num_image,1)
    #     # print(means.shape)
    #     for index in range(num_image):
    #         means[index,axis]=var_range[index]
    #     output = self.decoder(means).view((len(var_range),x.shape[1],x.shape[2],x.shape[3]))
    #     if loss=="bernoulli":
    #         output=torch.sigmoid(output)
    #     return output

class VAE_2(nn.Module):
    def __init__(self,D=558, z_size=10):
        super(VAE_2, self).__init__()
        self.z_size = z_size
        self.D=D

        self.encoder = nn.Sequential(
            nn.Linear(self.D, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.z_size*2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.D),
        )
        self.decoder_2 = nn.Sequential(
            nn.Linear(self.z_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.D),
        )


    def reparametrize(self, mu, logvar): # https://zhuanlan.zhihu.com/p/27549418
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            z = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            z = torch.FloatTensor(std.size()).normal_()
        return z*std+mu
        # return z.mul(std).add_(mu)
    
    def KL_divergence(self,mean,logvar):# TODO: why the kld make mean goes to zero and logvar goes to -100?
        kl_divergence = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kl_divergence = (kl_divergence).mul_(-0.5)
        kl_divergence_each_sample=torch.sum(kl_divergence.sum(1))/mean.shape[0]
        kl_divergence_each_dim=kl_divergence.mean(0)
        return kl_divergence_each_sample, kl_divergence_each_dim
        # kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())/mean.shape[0]
        # return kl_divergence
        pass


    def forward(self, x):
        mean_var = self.encoder(x)
        std_normal=torch.randn((x.shape[0],self.z_size))
        mean = mean_var[:, :self.z_size]
        logvar = mean_var[:, self.z_size:] # predict log varance because it is more stable and make sure var should always be positive https://stats.stackexchange.com/questions/353220/why-in-variational-auto-encoder-gaussian-variational-family-we-model-log-sig
        z = self.reparametrize(mean,logvar)
        output = self.decoder(z).view(x.size())
        output_2 = self.decoder(z).view(x.size())

        return (output,output_2), mean, logvar

def get_model(args,use_cuda):
    if args.dataset=="VAE":
        model=VAE()
    elif args.dataset=="VAE_2":
        model=VAE_2()
    else:
        raise ValueError("data set not found! ")
    if use_cuda:
        return model.cuda()
    else:
        return model
