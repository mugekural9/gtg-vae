import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Dropout

class VGAE_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
       super(VGAE_encoder, self).__init__()
       self.hidden_channels = 2*out_channels
       self.conv1 = GCNConv(in_channels, self.hidden_channels)
       self.conv2_mu  = GCNConv( self.hidden_channels, out_channels)
       self.conv2_sig = GCNConv( self.hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        d = Dropout(p=0.33)
        x = d(x)
        

        x_mu = self.conv2_mu(x,edge_index)
        x_sig = self.conv2_sig(x,edge_index)
        return x_mu,x_sig

