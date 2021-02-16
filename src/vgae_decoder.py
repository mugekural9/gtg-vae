import torch
from torch.nn import Linear

# GraphVAE (ref: Simonovsky et al)
class VGAE_decoder(torch.nn.Module):
    def __init__(self, in_channels, nmax, nodeclass_num):
       super(VGAE_decoder, self).__init__()
       self.in_channels = in_channels
       self.nmax = nmax 
       self.nodefeatures_linear = Linear(in_channels, in_channels*nmax)

       # self.nodeclass_num = nodeclass_num
       # self.nodeclass_linear = Linear(in_channels, nodeclass_num*nmax)
      

    def forward(self, z):
        """z: (B,Hdim) """
        
        # F_hat = self.nodeclass_linear(z)
        # F_hat = torch.reshape(F_hat, (-1, self.nmax, self.nodeclass_num))     # GinBatch, nmax, nodeclass_num
        feat_hat = self.nodefeatures_linear(z)
        feat_hat = torch.reshape(feat_hat, (-1, self.nmax, self.in_channels)) # Ginbatch, nmax, feature_dim
        return feat_hat

