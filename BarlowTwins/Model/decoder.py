import torch.nn as nn

# 13766
class Decoder(nn.Module):
    def __init__(self, latent_dim = 50, input_size = 1723):
        """
        The Decoder class
        """
        super(Decoder, self).__init__();
        self.in_dim = input_size;
        self.zdim = latent_dim;
        # self.last_layer = final_layer;

        # decoder
        self.dec = nn.Sequential(
                                nn.Linear(self.zdim, 128),
                                nn.ReLU(),
                                nn.BatchNorm1d(128),
                                
                                nn.Linear(128, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
            
                                nn.Linear(256, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                
                                nn.Linear(512, self.in_dim)
        )
        
            
            
    def forward(self, z):
            
            
        d = self.dec(z)



        return d