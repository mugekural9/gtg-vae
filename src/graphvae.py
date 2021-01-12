import torch
from torch_geometric.nn import GCNConv, GAE,VGAE
import torch.nn.functional as F

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
       print("x_mu: ", x_mu)
       print("x_sig: ", x_sig)
       return x_mu,x_sig



#GraphVAE (ref: Simonovsky et al)
class VGAE_decoder(torch.nn.Module):
    def __init__(self, in_channels, nmax, edgeclass_num, nodeclass_num):
       super(VGAE_encoder, self).__init__()
       self.out_channels = out_channels
       self.adj_linear = nn.Linear(nmax, nmax, in_channels)
       self.edgeclass_linear = nn.Linear(edgeclass_num, nmax, nmax, in_channels)
       self.nodeclass_linear = nn.Linear(nodeclass_num, nmax, nmax, in_channels)
            


    def forward(self, z):
       A_hat = self.adj_linear(z) 
       E_hat = self.edgeclass_linear(z)
       F_hat = self.nodeclass_linear(z)
       
       print("A_hat: ", A_hat)
       print("E_hat: ", E_hat)
       print("F_hat: ", F_hat)




   
