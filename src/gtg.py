import torch
import torch.nn as nn
from graphvae import GraphVAEM
from torch_geometric.nn import global_max_pool

class GTG(nn.Module):
    def __init__(self, input_dim, hidden_dim, nmax, text_encoder, nodelabel_num, phase=1): #text_encoder 
       super(GTG, self).__init__()
       self.phase = phase
       self.graph_vae = GraphVAEM(input_dim, hidden_dim, nmax, nodelabel_num, phase)
       self.text_encoder = text_encoder 


    def forward(self, x, edge_index, batch, dec_seq, nodetoken_ids, src_dict, gold_edges, neg_edge_index, adj_input):
        txt_kl_loss =  torch.tensor(0)

        # work with transformer_encoder...
        # Encode text...
        # src_seq = dec_seq.t()          
        # src_enc = self.text_encoder(src_seq)
        # text_z = src_enc[:, :1, :].squeeze(dim=1)
                  
        # Encode graph...
        #breakpoint()
        graph_z = self.graph_vae.encode(batch, x, edge_index)
           
        if self.phase == 1:
            z = graph_z
        elif self.phase == 2:
            z = text_z
            
        #feat_hat, f_hat = self.graph_vae.decode(z)
        adj_matrix, nodelabel_scores = self.graph_vae.decode(z)
        
        # Calculate P(g'|z) loss
        kl_loss, pos_loss, neg_loss, nodelabel_loss, roc_auc_score, avg_precision_score, nodelabel_acc, edge_recall, edge_precision  = self.graph_vae.recon_loss(adj_matrix, nodelabel_scores, x, edge_index, batch, nodetoken_ids, src_dict, self.phase, gold_edges, neg_edge_index, adj_input)

        # Calculate KL divergence between graph_z and text_z
        if self.phase == 2:
            criterion = nn.KLDivLoss(reduction='batchmean')
            softmax = nn.Softmax(dim=1)
            p = torch.log(softmax(text_z))
            q = softmax(graph_z)
            txt_kl_loss = criterion(p,q)        
        
        losses = dict()
        losses["kl_loss"] = kl_loss
        losses["pos_loss"] = pos_loss
        losses["neg_loss"] = neg_loss
        losses["nodelabel_loss"] = nodelabel_loss
        losses["txt_kl_loss"] = txt_kl_loss
        losses["loss"] = sum(losses.values())

        # print(losses)
        metrics = dict()
        metrics["roc_auc_score"] = roc_auc_score
        metrics["avg_precision_score"] = avg_precision_score
        metrics["nodelabel_acc"] = nodelabel_acc
        metrics["edge_recall"] = edge_recall
        metrics["edge_precision"] = edge_precision
        
        return losses, metrics
