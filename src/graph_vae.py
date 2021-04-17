import torch
import torch.nn as nn
import pdb, math
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import global_max_pool, GCNConv
from maxspantree import mst_graph

MAX_LOGSTD = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

class GraphVAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, mlp_hid_dim, nmax, phase):
       super(GraphVAE, self).__init__()
       self.phase = phase
       self.nmax = nmax
       output_dim = nmax * (nmax + 1) // 2
       self.encoder = VGAE_encoder(input_dim, out_channels=latent_dim)
       self.decoder = MLP_VAE_plain(latent_dim, mlp_hid_dim, output_dim, nmax)
       
    def encode(self, batch, *args, **kwargs):
        self.__h__ = self.encoder(*args, **kwargs)
        #self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        self.__h__ = global_max_pool(self.__h__, batch)          # (B,Zdim)
        #self.__logstd__ = global_max_pool(self.__logstd__, batch)  # (B,Zdim)
        #z = self.reparametrize(self.__mu__, self.__logstd__)
        return self.__h__ #self.__logstd__

    def decode(self, z):
        out = self.decoder(z)
        return out
    
    def loss(self, out, adj, gold_edges, report):
        adj = torch.tensor(adj).float()
        B, maxnode, _ = adj.size()
        out_tensor = out.cpu().data       
        graph_recon_loss = 0
        total_edge_recall    = torch.tensor(0).float() 
        total_edge_precision = torch.tensor(0).float() 
        # Graph reconstruction...
        for b in range(B):
            _out_tensor = out_tensor[b,:].unsqueeze(0)
            recon_adj_lower = self.recover_adj_lower(_out_tensor) 
            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower) # make symmetric
            adj_data = adj[b].cpu().data #[0]
            adj_permuted = adj_data
            adj_vectorized = adj_permuted[torch.triu(torch.ones(self.nmax,self.nmax))== 1].squeeze_()
            adj_vectorized_var = adj_vectorized.cuda()
            adj_recon_loss = self.adj_recon_loss(out[b], adj_vectorized_var)
            graph_recon_loss += adj_recon_loss
            if report:
                edge_recall, edge_precision = self.graph_statistics(recon_adj_tensor, list(gold_edges[b]), report)
                total_edge_recall += edge_recall
                total_edge_precision += edge_precision
        graph_recon_loss /= B
        total_edge_recall /= B
        total_edge_precision /= B
        return graph_recon_loss, total_edge_recall, total_edge_precision

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.nmax, self.nmax)
        adj[torch.triu(torch.ones(self.nmax, self.nmax)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def adj_recon_loss(self, adj_pred, adj_truth):
        # F.binary_cross_entropy(adj_truth, adj_pred)
        return F.binary_cross_entropy(adj_pred, adj_truth)

    def graph_statistics(self, adj_matrix, gold_edges, report):
        pred_edges = mst_graph(adj_matrix)
        count = 0; gold_edgelength = 0; pred_edgelength = 0                
        if gold_edges is not None:
            gold_edgelength += len(gold_edges)
        if pred_edges is not None:
            pred_edgelength += len(pred_edges)
        for j in pred_edges:
            if (j[0],j[1]) in gold_edges or (j[1], j[0]) in gold_edges:
                count += 1
        if gold_edgelength == 0 or pred_edgelength == 0:
            return torch.tensor(1), torch.tensor(1) ## check this!
        edge_recall = count / gold_edgelength
        edge_precision = count/ pred_edgelength
        if False: #report:
            print("\ngold_edges:", gold_edges)
            print("pred_edges:", pred_edges)        
        return edge_recall, edge_precision

    
class VGAE_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
       super(VGAE_encoder, self).__init__()
       self.hidden_channels = out_channels * 2
       self.conv1 = GCNConv(in_channels, self.hidden_channels)
       self.relu = nn.ReLU()
       self.conv2 = GCNConv(self.hidden_channels, out_channels)
       #self.conv2_mu  = GCNConv( self.hidden_channels, out_channels)
       #self.conv2_sig = GCNConv( self.hidden_channels, out_channels)
       
       
    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        #x_mu = self.conv2_mu(x,edge_index)
        #x_sig = self.conv2_sig(x,edge_index)
        return self.conv2(x, edge_index)  #x_mu,x_sig

    
class MLP_VAE_plain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nmax):
        super(MLP_VAE_plain, self).__init__()
        self.decode_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decode_2 = nn.Linear(hidden_dim, output_dim) 
        self.sigmoid = nn.Sigmoid()

        self.nmax = nmax
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                
    def forward(self, h):
        y = self.decode_1(h)
        y = self.relu(y)
        y = self.decode_2(y)
        out = self.sigmoid(y)
        return out
