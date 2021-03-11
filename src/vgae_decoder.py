import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import torch.nn as nn
# GraphVAE (ref: Simonovsky et al)
class VGAE_decoder(torch.nn.Module):
    def __init__(self, in_channels, nmax, nodelabel_num, phase):
       super(VGAE_decoder, self).__init__()
       self.in_channels = in_channels
       self.phase = phase
       self.nmax = nmax
       self.nodelabel_num = nodelabel_num

       self.adj_linear_1 = Linear(in_channels, 128)
       self.adj_linear_2 = Linear(128, 256)
       self.adj_linear_3 = Linear(256, in_channels)
       self.adj_linear = Linear(in_channels, nmax*nmax) 
       # self.nodelabel_linear = Linear(in_channels, nmax*nodelabel_num)
       self.dropout = Dropout(0.1)
       
      

    def forward(self, z): #, text_z):
        """z: (B,Hdim) or (B*maxnode, Hdim) """

        z = self.adj_linear_1(z)
        z = self.adj_linear_2(z)
        z = self.adj_linear_3(z)
        z = F.relu(z)

        #breakpoint() 
        adj_matrix = self.adj_linear(z)
        adj_matrix = torch.reshape(adj_matrix, (z.size(0), self.nmax, self.nmax))
        adj_matrix = torch.sigmoid(adj_matrix)

        # z = self.dropout(z)
        # nodelabel_scores = F.relu(self.nodelabel_linear(z))
        # nodelabel_scores = torch.reshape(nodelabel_scores, (z.size(0), self.nmax, self.nodelabel_num))
               
        return adj_matrix, None # nodelabel_scores # adj_matrix
    
