# generate dataset
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import torch
import copy
import pandas as pd
def get_dataloader(args,shuffle=True):
    data=pd.read_csv("gene_expression.txt",sep=" ",header=None).values
    # print(data)
    dataset=TensorDataset(torch.tensor(data).float())
    dataloader=DataLoader(dataset,shuffle=shuffle,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    return dataloader

