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

# sklearn clustering
from sklearn.cluster import KMeans
from sklearn import mixture

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

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

# parser.add_argument('--in_path', type = str, required = True, help = 'Path to adata object')
# parser.add_argument('--train_split', default = True, action='store_true', help='Looks for split and pulls out `Train` labeled cells for training, default = True')
parser.add_argument('--min_filtering', default = True, action='store_true', help='Performs a minimum filtering using `filter_genes` & `filter_cells`, default = True')

# args = parser.parse_args()

# # reading in dataset

# adata = sc.read_h5ad(args.in_path)

# datasplit = args.train_split

# if datasplit:
#         print("    -> Extracting the training data.") 
        
#         adata = adata[adata.obs['split'] == 'train']
        
# else:
#     print("    -> Data is ready as is.")

# # read in testing data
# adata = sc.read_h5ad('~/JupyterNBs/Misc_Projects/pbmc_3k_dca.h5ad')
pbmc3k = sc.datasets.pbmc3k_processed()
adata = sc.datasets.pbmc3k()
celltypes_series = pbmc3k.obs['louvain']
filt_barcodes = list(celltypes_series.index)
adata.obs['barcodes'] = adata.obs.index

# filter certain barcodes
adata = adata[adata.obs.index.isin(filt_barcodes)]

adata.obs['louvain'] = celltypes_series

# ### std processing ###
# # filter low expressed genes
# if args.min_filtering:
#     print("    -> Filtering lowly expressed genes.")
#     sc.pp.filter_genes(adata, min_counts=3)
#     sc.pp.filter_cells(adata, min_counts=3)

# else:
#     print("    -> Filtering skipped.")

# filter low expressed genes
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.filter_cells(adata, min_counts=3)




# # keep raw data object
# adata.raw = adata.copy()

sc.pp.normalize_per_cell(adata)
# calculate size factors from library sizes (n_counts)
adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

# log trans
sc.pp.log1p(adata)

# # save raw object
# adata.raw = adata

# compute variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print("Highly variable genes: %d"%sum(adata.var.highly_variable))

#plot variable genes
sc.pl.highly_variable_genes(adata)

# subset for variable genes in the dataset
adata = adata[:, adata.var['highly_variable']]


# save raw object
adata.raw = adata

# scale data
sc.pp.scale(adata)


def train_AE(model, x, batch_size=128, lr=0.0001, epochs=50):

        optimizer = torch.optim.Adam(params=model.parameters(), 
                                lr=lr, 
                                betas=(0.9, 0.999), 
                                eps=1e-08, 
                                weight_decay=0.005, 
                                amsgrad=False)
    
        dataset = TensorDataset(torch.Tensor(x))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("==>Training")

#         print(f"Total number of *trainable parameters* : {count_parameters(model)}")
        
        for epoch in range(epochs):
            for batch_idx, (x_batch) in enumerate(dataloader):
                
                x_tensor = Variable(x_batch).to(device)
                
                ################# Forward #################
                z1, z2, recon1, recon2, barlow_loss = model(x_tensor, x_tensor, device=device)
                
                # MSE Loss
                mse_loss1 = F.mse_loss(recon1, x_tensor)
                mse_loss2 = F.mse_loss(recon2, x_tensor)
                
                # Combined Loss 
                loss = mse_loss1 + mse_loss2 + barlow_loss
                
                ################# Backward #################
                optimizer.zero_grad()
                
                loss.backward()
    
                optimizer.step()

            ################# Logs #################
            print('Epoch [{}/{}], Combined loss:{:.4f}'.format(epoch+1, epochs, loss.item()))
            
            # find best model
            state = loss.item()
            is_best = False
        
        return model

bt_model = scBarlowTwins(input_size=adata.n_vars).to(device)


# train model
bt_trained = train_AE(bt_model, 
                       x=adata.X, 
                       batch_size=128, 
                       lr=0.0001, 
                       epochs=500)

##### SHOULD BE IN A DIFFERENT SCRIPT

# dca_model.eval()

# with torch.no_grad():
#     # Get generated latent space
#     z = dca_model.encoder(adata.X);
#     z_numpy = z.cpu().detach().numpy()


z = dca_trained(torch.Tensor(adata.X), torch.Tensor(adata.X))[0].detach().numpy()
z2 = dca_trained(torch.Tensor(adata.X), torch.Tensor(adata.X))[1].detach().numpy()


# add autoencoder latent data to scanpy object
adata.obsm['X_AE_1'] = z
adata.obsm['X_AE_2'] = z2
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=None, use_rep='X_AE_1', random_state=2022, key_added='ae_cord')
sc.tl.leiden(adata, resolution=0.4,random_state=2022, restrict_to=None, key_added='leiden_ae', 
                  obsp='ae_cord_connectivities')

sc.tl.umap(adata, neighbors_key='ae_cord', n_components=2)
# sc.tl.draw_graph(adata, neighbors_key='ae_cord')

# sc.pl.draw_graph(adata, color=['leiden_ae','CD8A', 'NKG7', 'CD4'], use_raw=True)

sc.pl.umap(adata, color=['leiden_ae','CD8A', 'NKG7', 'CD4'], use_raw=True, save='_twindca_latent.pdf')


# save scanpy object
adata.write_h5ad('../../scSimCLR.h5ad')