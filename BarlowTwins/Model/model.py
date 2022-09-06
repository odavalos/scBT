import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .projector import Projector
# from .autoencoder import Autoencoder
# from scBT.utils import 


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# 13766
class scBarlowTwins(nn.Module):
    '''
    BarlowTwins projector and loss adapted from https://github.com/facebookresearch/barlowtwins
    '''
    
    def __init__(self, latent_dim = 50, input_size = 1723, projection_sizes=[8192, 8192, 8192],
                 lambd=3.9e-3, scale_factor=1):

        if input_size == None or latent_dim == None:
            raise ValueError('Please provide a value for each input_size, and latent_dim')
          
        super(scBarlowTwins, self).__init__()
        self.in_dim = input_size;
        self.zdim = latent_dim;
        # self.backbone = backbone;
        self.lambd = lambd;
        self.scale_factor = scale_factor;
        # self.last_layer = final_layer;
        
        # autoencoder
        self.encoder = Encoder(input_size = self.in_dim,  latent_dim = self.zdim); # encoder
        self.decoder = Decoder(input_size = self.in_dim, latent_dim = self.zdim); # decoder
        
        # projector
        self.projector = Projector(latent_dim = 50); # Projector 
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(1024, affine=False) # change eventually



    def forward(self, x1, x2):
        """
        Forward pass of the autoencoder
        """
        
        # autoencoder
        z1 = self.encoder(x1); 
        z2 = self.encoder(x2);
        recon1 = self.decoder(z1); 
        recon2 = self.decoder(z2);
        
        # projector
        p1 = self.projector(z1);
        p2 = self.projector(z2)

        # empirical cross-correlation matrix
        c = torch.mm(self.bn(p1).T, self.bn(p2))
        c.div_(z1.shape[0])


        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor*(on_diag + self.lambd * off_diag)
        
        return z1, z2, recon1, recon2, loss


        # return z1, z2, recon1, recon2
