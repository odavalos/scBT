import torch.nn as nn

class Projector(nn.Module):
    """

    The Projector class

    """

    def __init__(self, latent_dim = 50, projection_sizes=[8192, 8192, 8192],lambd=3.9e-3, scale_factor=1):

        super(Projector, self).__init__();
        self.zdim = latent_dim;

        self.proj = nn.Sequential(
                            nn.Linear(self.zdim, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),

                            nn.Linear(1024, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),

                            nn.Linear(1024, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),

                            nn.Linear(1024, 1024)
                                       )



    def forward(self, x):        
        """

        Forward pass of the projector

        """

        p = self.proj(x)


        return p
    

        
        
