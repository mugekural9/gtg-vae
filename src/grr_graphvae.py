import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from maxspantree import mst_graph
from graphvae import GraphVAEM


class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(GraphVAE, self).__init__()
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.graph_vae = GraphVAEM(input_dim, hidden_dim, max_num_nodes, 38926, 1)    
        self.vae = MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        self.max_num_nodes = max_num_nodes
        
    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    
    def forward(self, x, edge_index, batch, adj, gold_edges, report):

        graph_z = self.graph_vae.encode(batch, x, edge_index)
        #graph_z = torch.load('tensor.pt')
       
        h_decode = self.vae(graph_z) 
        out = F.sigmoid(h_decode)
        out_tensor = out.cpu().data
        #out_tensor = torch.load('out_tensor.pt')
       
        B, maxnode, _ = adj.size() #out_tensor.size()
    
        #loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        #loss_kl /= B* self.max_num_nodes * self.max_num_nodes # normalize

        #torch.save(out_tensor, 'out_tensor.pt')
        #print(out_tensor)
            
        loss_kl = self.graph_vae.kl_loss() / x.size(0)
        total_loss = 0; total_edge_recall = torch.tensor(0).float(); total_edge_precision = torch.tensor(0).float();
            
        for b in range(B):
            
            _out_tensor = out_tensor[b,:].unsqueeze(0)
            recon_adj_lower = self.recover_adj_lower(_out_tensor) 
            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower) # make symmetric
            adj_data = adj[b].cpu().data #[0]
            adj_permuted = adj_data
            adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()
            adj_vectorized_var = adj_vectorized.cuda()
            adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[b])
            total_loss += adj_recon_loss
            if report:
                edge_recall, edge_precision = self.graph_statistics(recon_adj_tensor, list(gold_edges[b]), report)
                total_edge_recall += edge_recall.float()
                total_edge_precision += edge_precision.float()
        
        total_loss /= B
        total_loss += loss_kl
    
        return total_loss, total_edge_recall.float(), total_edge_precision.float()

    def adj_recon_loss(self, adj_truth, adj_pred):
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
        edge_recall = torch.tensor(count / gold_edgelength).float() 
        edge_precision = torch.tensor(count/ pred_edgelength).float()

        if False: #report:
            print("\ngold_edges:", gold_edges)
            print("pred_edges:", pred_edges)        
        return edge_recall, edge_precision


class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.decode_1 = nn.Linear(h_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size) 
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.decode_1(h)
        y = self.relu(y)
        y = self.decode_2(y)
        return y 

