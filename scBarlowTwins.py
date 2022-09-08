# std libraries
import os
import math as m
import random
import copy
import argparse

# datascience/single cell libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
# plt.style.use('seaborn')

# # sklearn clustering
# from sklearn.cluster import KMeans
# from sklearn import mixture

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from anndata.experimental.pytorch import AnnLoader
from BarlowTwins import scBarlowTwins


sc.settings.set_figure_params(dpi=320, facecolor='white')

# device for torch 
if torch.cuda.is_available():
    device = "cuda";
    print('==> Using GPU (CUDA)')

else :
    device = "cpu"
    print('==> Using CPU')
    print('    -> Warning: Using CPUs will yield to slower training time than GPUs')


# creating the parser
parser = argparse.ArgumentParser()

parser.add_argument('--in_path', type = str, required = True, help = 'Path to adata object')
parser.add_argument('--out_path', type = str, required = True, help = 'Path for output anndata object')
parser.add_argument('--train_split', default = True, action='store_true', help='Looks for split and pulls out `Train` labeled cells for training, default = True')
parser.add_argument('--out_name', type = str, default = '', required = False, help = 'Name for output h5ad file. Default name will be scBT.h5ad')
# parser.add_argument('--min_filtering', default = True, action='store_true', help='Performs a minimum filtering using `filter_genes` & `filter_cells`, default = True')

args = parser.parse_args()

# reading in dataset
adata = sc.read_h5ad(args.in_path)

datasplit = args.train_split

if datasplit:
        print("    -> Extracting the training data.") 
        
        adata_test = adata[adata.obs['split'] == 'test'] # test data
        adata = adata[adata.obs['split'] == 'train'] # train data
        
        
else:
    print("    -> Data is ready as is.")

dataloader = AnnLoader(adata, batch_size=128, shuffle=True, use_cuda=device)
# dataloader_test = AnnLoader(adata_test, batch_size=128, shuffle=True, use_cuda=device)

def train_AE(model, train_loader, batch_size=128, lr=0.0001, epochs=50):

        optimizer = torch.optim.Adam(params=model.parameters(), 
                                lr=lr, 
                                betas=(0.9, 0.999), 
                                eps=1e-08, 
                                weight_decay=0.005, 
                                amsgrad=False)
        
        
        print("==>Training")
#         print(f"Total number of *trainable parameters* : {count_parameters(model)}")
        
        for epoch in range(epochs):
            for batch in train_loader:
                
                ################# Forward #################
                z1, z2, recon1, recon2, barlow_loss = model(batch.X.float(), batch.X.float())
                
                # MSE Loss
                mse_loss1 = F.mse_loss(recon1, batch.X.float())
                mse_loss2 = F.mse_loss(recon2, batch.X.float())
                
                # Combined Loss 
                loss = mse_loss1 + mse_loss2 + barlow_loss
                
                ################# Backward #################
                optimizer.zero_grad()
                
                loss.backward()
    
                optimizer.step()

            ################# Logs #################
            print('Epoch [{}/{}], Joint loss:{:.4f}'.format(epoch+1, epochs, loss.item()))
            
            # find best model
            state = loss.item()
            is_best = False
        
        return model

bt_model = scBarlowTwins(input_size=adata.n_vars, projection_sizes=[1024, 1024, 1024]).to(device)


# train model
bt_trained = train_AE(bt_model, 
                       dataloader, 
                       batch_size=128, 
                       lr=0.0001, 
                       epochs=200)


# new method for extracting latent space
# full_data = dataloader.dataset[:]
# test_data = dataloader_test.dataset[:]

test_data_tensor = torch.tensor(adata_test.X).to(torch.float32)
lspace = bt_model.encoder(test_data_tensor.to(device))[:, :50] # get latent space

# does not work 
# z = bt_trained(torch.Tensor(adata.X), torch.Tensor(adata.X))[0].detach().numpy()
# z2 = bt_trained(torch.Tensor(adata.X), torch.Tensor(adata.X))[1].detach().numpy()


# add autoencoder latent data to scanpy object
adata.obsm['X_AE_1'] = lspace.data.cpu().numpy()
# adata.obsm['X_AE_2'] = z2
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=None, use_rep='X_AE_1', random_state=2022, key_added='ae_cord')
sc.tl.leiden(adata, resolution=0.4,random_state=2022, restrict_to=None, key_added='leiden_ae', 
                  obsp='ae_cord_connectivities')

sc.tl.umap(adata, neighbors_key='ae_cord', n_components=2)

sc.pl.umap(adata, color=['leiden_ae','seurat_annotations', 'stim'], use_raw=True, save='_scBT_latent.pdf')




# save scanpy object                                                                                                    
if opt.out_name:
    adata_test.write_h5ad(''.join(map(str,[(args.out_path), (args.out_name),'scBT.h5ad'])))
else:
    adata_test.write_h5ad(''.join(map(str,[(args.out_path),'scBT.h5ad'])))
    
    
