import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

MAX_LOGSTD = 10

class GraphVAE(torch.nn.Module):
    """
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    
    def __init__(self, input_dim, hidden_dim, nmax, edgeclass_num, nodeclass_num):
       super(GraphVAE, self).__init__()
       self.encoder = VGAE_encoder(input_dim, out_channels=hidden_dim)
       self.decoder = VGAE_decoder(hidden_dim, nmax, edgeclass_num, nodeclass_num)
   
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)

        # print("self.__mu__:", self.__mu__.shape)
        # print("self.__logstd__:", self.__logstd__.shape)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    
    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


    def recon_loss(self, a_hat, f_hat, x, pos_edge_index, batch, nodetoken_ids, src_dict, node_dict):
        # pos_edge_index: (2, num_of_edges)
        # f_hat: (B, maxnode, nodelabelsize)
        # a_hat: (B, maxnode, maxnode)
        # x: (maxnode, nodefeatures)
        # nodetoken_ids: (B, maxnode)

        batch_size = f_hat.size(0)
        maxnode = f_hat.size(1)

        # F_hat loss
        # Create indexing matrix for batch: [batch, 1]
        batch_index = torch.arange(0, batch_size).view(batch_size, 1)
        node_index  = torch.arange(0, maxnode).view(1, maxnode).repeat(batch_size,1)
        m =  nn.Softmax(dim=2)
        f_hat = m(f_hat)

        
        nodelabel_probs = f_hat[batch_index, node_index, nodetoken_ids.data]
        nodelabel_loss = -torch.log(nodelabel_probs + 1e-15).mean()

        # See predictions...
        predicted_node_tokens = torch.argmax(f_hat,2) # (B, maxnode)
        correct_predicted_node_tokens = (predicted_node_tokens == nodetoken_ids).sum()
        # for i in range(len(predicted_node_tokens)):
        #     tokens = []; gold_tokens = []
        #     #i=0
        #     #print("i:..",i)
        #     for tok in predicted_node_tokens[i]:
        #         tok = tok.item()
        #         if tok < len(src_dict):
        #             tokens.append(src_dict.itos[tok].replace("~", "_"))          
        #     print("tokens:", tokens)
        #     for tok in nodetoken_ids[i]:
        #         tok = tok.item()
        #         if tok < len(src_dict):
        #             gold_tokens.append(src_dict.itos[tok].replace("~", "_"))          
        #     print("gold_tokens:", gold_tokens)
       
        # A_hat loss
        graphs = batch[pos_edge_index[0]].unsqueeze(0) # 1, num of edges
        coordinates = torch.cat([graphs, pos_edge_index], 0)  # 3, num_of_edges
        graph_coords = coordinates[0] # num_of_edges
        edgesource_coords = coordinates[1] % maxnode
        edgetarget_coords = coordinates[2] % maxnode 
        pos_probs = a_hat[graph_coords.data, edgesource_coords.data, edgetarget_coords.data]
        pos_loss = -torch.log(torch.sigmoid(pos_probs + 1e-15).mean())

        # print("pos_edge_index:", pos_edge_index)
        # print("a_hat:", torch.sigmoid(a_hat))
        # pos_probs = []
        # for c in zip(graph_coords, edgesource_coords, edgetarget_coords):
        #     pos_probs.append(a_hat[c[0], c[1], c[2]])
        # pos_probs = torch.tensor(pos_probs)
        
        
        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(pos_edge_index)
        # all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp) #, maxnode) #, pos_edge_index.size(1)) 
        graphs = batch[neg_edge_index[0]].unsqueeze(0) # 1, num of edges
        coordinates = torch.cat([graphs, neg_edge_index], 0)  # 3, num_of_edges
        graph_coords = coordinates[0] # num_of_edges
        edgesource_coords = coordinates[1] % maxnode
        edgetarget_coords = coordinates[2] % maxnode
        neg_probs = a_hat[graph_coords.data, edgesource_coords.data, edgetarget_coords.data]
        neg_loss = -torch.log(1 - torch.sigmoid(neg_probs + 1e-15).mean())

                
        kl_loss = 1 / x.size(0) * self.kl_loss()
        # print("kl_loss:", kl_loss)
        # print("nodelabel_loss:", nodelabel_loss)
        # print("pos_loss:", pos_loss)
        # print("neg_loss:", neg_loss)
        
        return kl_loss+ nodelabel_loss+ pos_loss+ neg_loss, correct_predicted_node_tokens
 



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
       #self.edgeclass_num = edgeclass_num
       self.nodeclass_num = nodeclass_num
       
       self.adj_linear = Linear(in_channels, nmax*nmax)
       #self.edgeclass_linear = Linear(in_channels, edgeclass_num*nmax*nmax)
       self.nodeclass_linear = Linear(in_channels, nodeclass_num*nmax)
       self.nodefeatures_linear = Linear(in_channels, in_channels*nmax)
            

    def forward(self, z):
        A_hat = self.adj_linear(z)
        # A_hat = F.relu(A_hat)
        # E_hat = self.edgeclass_linear(z)
        F_hat = self.nodeclass_linear(z)
        F_hat = F.relu(F_hat)
        A_hat = torch.reshape(A_hat, (-1, self.nmax, self.nmax)) # GinBatch, nmax, nmax 
        # E_hat = torch.reshape(E_hat, (-1, self.edgeclass_num, self.nmax, self.nmax)) # GinBatch, edgeclass_num, nmax, nmax
        F_hat = torch.reshape(F_hat, (-1, self.nmax, self.nodeclass_num)) # GinBatch, nmax, nodeclass_num

        Feat_hat = self.nodefeatures_linear(z)
        Feat_hat = torch.reshape(Feat_hat, (-1, self.nmax, self.in_channels)) # Ginbatch, nmax, feature_dim
        return A_hat, F_hat, Feat_hat # E_hat 

