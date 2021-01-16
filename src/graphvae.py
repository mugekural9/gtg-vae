import torch
from torch_geometric.nn import GCNConv, GAE, VGAE
import torch.nn.functional as F
from torch.nn import Linear

class VGAE_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
       super(VGAE_encoder, self).__init__()
       self.out_channels = out_channels
       self.conv1 = GCNConv(in_channels,2 * out_channels)
       self.conv2_mu = GCNConv(2 * out_channels, out_channels)
       self.conv2_sig = GCNConv(2 * out_channels, out_channels)


    def forward(self, x, edge_index):
       x = F.relu(self.conv1(x, edge_index))
       x_mu = self.conv2_mu(x,edge_index)
       x_sig = self.conv2_sig(x,edge_index)
       return x_mu,x_sig



# GraphVAE (ref: Simonovsky et al)
class VGAE_decoder(torch.nn.Module):
    def __init__(self, in_channels, nmax, edgeclass_num, nodeclass_num):
       super(VGAE_decoder, self).__init__()
       self.in_channels = in_channels
       self.nmax = nmax
       self.edgeclass_num = edgeclass_num
       self.nodeclass_num = nodeclass_num
       
       self.adj_linear = Linear(in_channels, nmax*nmax)
       self.edgeclass_linear = Linear(in_channels, edgeclass_num*nmax*nmax)
       self.nodeclass_linear = Linear(in_channels, nodeclass_num*nmax*nmax)
            


    def forward(self, z):
       A_hat = self.adj_linear(z)
       E_hat = self.edgeclass_linear(z)
       F_hat = self.nodeclass_linear(z)

       A_hat = torch.reshape(A_hat, (-1, self.nmax, self.nmax))
       E_hat = torch.reshape(E_hat, (-1, self.edgeclass_num, self.nmax, self.nmax))
       F_hat = torch.reshape(F_hat, (-1, self.nodeclass_num, self.nmax, self.nmax))
    
         
       print("A_hat: ", A_hat.shape)
       print("E_hat: ", E_hat.shape)
       print("F_hat: ", F_hat.shape)




   
