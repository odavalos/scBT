import torch.nn as nn

# 13766
class Encoder(nn.Module):
    def __init__(self, latent_dim = 50, input_size = 1723, dropout = 0.1):
        """
        
        The Encoder class
          
        """
        if latent_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension')
            
        super(Encoder, self).__init__();
        self.in_dim = input_size;
        self.zdim = latent_dim;
        self.drop = dropout;

        self.enc = nn.Sequential(
                                nn.Dropout(p=self.drop),
                                nn.Linear(self.in_dim, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
            
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
            
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.BatchNorm1d(128),
            
                                nn.Linear(128, self.zdim)
                                           )
        
    def forward(self, x):        
        """
        
        Forward pass of the encoder
        
        """

        z = self.enc(x)
        
        
        return z