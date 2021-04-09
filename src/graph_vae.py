import torch
import torch.nn as nn
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import global_max_pool
from vgae_encoder import VGAE_encoder

import pdb, math
from maxspantree import mst_graph
MAX_LOGSTD = 10
EPS = 1e-15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


class GraphVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, nmax, nodelabel_num, phase):
       super(GraphVAE, self).__init__()
       self.phase = phase
       self.encoder = VGAE_encoder(input_dim, out_channels=hidden_dim)
       #self.decoder = VGAE_decoder(hidden_dim, nmax, nodelabel_num, phase)
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, batch, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        self.__mu__ = global_max_pool(self.__mu__, batch)          # (B,Zdim)
        self.__logstd__ = global_max_pool(self.__logstd__, batch)  # (B,Zdim)
        #z = self.reparametrize(self.__mu__, self.__logstd__)
        return self.__mu__, self.__logstd__

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def kl_loss(self):
        return -0.5 * torch.mean(torch.sum(1 + 2 * self.logstd - self.mu**2 - self.logstd.exp()**2, dim=1))

    '''
    def recon_loss(self, adj_matrix, nodelabel_scores, x, pos_edge_index, batch, nodetoken_ids, src_dict, phase, gold_edges, neg_edge_index, adj_input):
        """
        pos_edge_index: (2, num_of_edges)
        adj_matrix: (B, maxnode, maxnode)
        nodelabel_scores: (B, maxnode, nodelabel_num)
        x: (maxnode, nodefeatures)
        nodetoken_ids: (B, maxnode)
        """
        B, maxnode = nodetoken_ids.size()
        
        if False: # self.phase == 2:
            # Nodelabel loss...
            # Create indexing matrix for batch: [batch, 1]
            batch_index = torch.arange(0, B).view(B, 1)
            node_index  = torch.arange(0, maxnode).view(1, maxnode).repeat(B, 1)
            m =  nn.Softmax(dim=2)
            _nodelabel_probs = m(nodelabel_scores)
            nodelabel_probs = _nodelabel_probs[batch_index, node_index, nodetoken_ids.data]
            nodelabel_loss = -torch.log(nodelabel_probs + 1e-15).mean()
            predicted_node_tokens = torch.argmax(nodelabel_scores,2) # (B, maxnode)
            nodelabel_acc = (predicted_node_tokens == nodetoken_ids).sum() / (B*maxnode)
            
            if False:
                # See predictions...
                for i in range(len(predicted_node_tokens)):
                    tokens = []; gold_tokens = []
                    for tok in nodetoken_ids[i]:
                        tok = tok.item()
                        if tok < len(src_dict):
                            gold_tokens.append(src_dict.itos[tok].replace("~", "_"))          
                    print("\ngold_tokens:", gold_tokens)
                    for tok in predicted_node_tokens[i]:
                        tok = tok.item()
                        if tok < len(src_dict):
                            tokens.append(src_dict.itos[tok].replace("~", "_"))          
                    print("tokens:", tokens)

        # Kl loss
        if phase == 1:
            kl_loss =  self.kl_loss() / x.size(0)
        elif phase == 2:
            kl_loss =  torch.tensor(0)

        # Adjacency matrix loss...                 
        _src   = torch.remainder(pos_edge_index[0], maxnode)
        _tgt   = torch.remainder(pos_edge_index[1], maxnode)
        _batch = torch.floor(pos_edge_index[0] / maxnode)
        pos_prob = adj_matrix[_batch.detach().cpu().numpy(), _src.detach().cpu().numpy(), _tgt.detach().cpu().numpy()].squeeze(0)
        pos_loss = -torch.log(pos_prob + EPS).mean()

        # Do not include self-loops in negative samples
        _neg_edge_index = negative_sampling(pos_edge_index, num_neg_samples=pos_edge_index.size(1)) 
        neg_edge_index = torch.cat((_neg_edge_index, neg_edge_index), 1)
        _src   = torch.remainder(neg_edge_index[0], maxnode)
        _tgt   = torch.remainder(neg_edge_index[1], maxnode)
        _batch = torch.floor(neg_edge_index[0] / maxnode)
        neg_prob = adj_matrix[_batch.detach().cpu().numpy(), _src.detach().cpu().numpy(), _tgt.detach().cpu().numpy()].squeeze(0)
        neg_loss = -torch.log(1-neg_prob + EPS).mean()
        
        edge_recall, edge_precision = self.graph_statistics(adj_matrix, pos_edge_index, gold_edges, adj_input)
        # roc_auc_score, avg_precision_score = self.test(pos_prob, neg_prob, pos_edge_index, neg_edge_index)
        roc_auc_score, avg_precision_score = torch.tensor(0), torch.tensor(0)
        nodelabel_loss, nodelabel_acc = torch.tensor(0), torch.tensor(0)
        return kl_loss, pos_loss, neg_loss, nodelabel_loss, roc_auc_score, avg_precision_score, nodelabel_acc, edge_recall, edge_precision

    
    def test(self, pos_pred, neg_pred, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        
        pos_y = pos_pred.new_ones(pos_edge_index.size(1)) #276
        neg_y = pos_pred.new_zeros(neg_edge_index.size(1)) #276 
        y = torch.cat([pos_y, neg_y], dim=0) #552 

        pred = torch.cat([pos_pred, neg_pred], dim=0) #552
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def graph_statistics(self, adj_matrix, pos_edge_index, gold_edges, adj_input):
        #print(adj_input)
        #breakpoint()
        pred_edges = mst_graph(adj_matrix)
        count = 0
        gold_edgelength = 0
        pred_edgelength = 0
        for b in pred_edges.keys():
            if gold_edges is not None:
                gold_edgelength += len(gold_edges[b])
            if pred_edges is not None:
                pred_edgelength += len(pred_edges[b])
            #print("\npred_edges[b]:", pred_edges[b])
            #print("gold_edges[b]:", gold_edges[b])
            for j in pred_edges[b]:
                if (j[0],j[1]) in gold_edges[b] or (j[1], j[0]) in gold_edges[b]:
                    count += 1

        if gold_edgelength == 0 or pred_edgelength == 0:
            return torch.tensor(0), torch.tensor(0) ## check this!
        edge_recall = torch.tensor(count / gold_edgelength) 
        edge_precision = torch.tensor(count/ pred_edgelength)
        # print("\ngold_edges:", gold_edges)
        # print("pred_edges:", pred_edges)
        # print("edge_recall:", edge_recall)
        # print("edge_precision:", edge_precision)
        return edge_recall, edge_precision
    '''
