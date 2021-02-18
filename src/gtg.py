import torch
import torch.nn as nn
from graphvae import GraphVAE
from torch_geometric.nn import global_max_pool

class GTG(nn.Module):
    def __init__(self, input_dim, hidden_dim, nmax, text_encoder, phase=1): #text_encoder 
       super(GTG, self).__init__()
       self.phase = phase
       self.graph_vae = GraphVAE(input_dim, hidden_dim, nmax, phase)
       self.text_encoder = text_encoder 


    def forward(self, x, edge_index, batch, dec_seq, nodetoken_ids, src_dict):
        txt_kl_loss =  torch.tensor(0)

        # work with transformer_encoder...
        # Encode text...
        src_seq = dec_seq.t()          
        src_enc = self.text_encoder(src_seq)
        text_z = src_enc[:, :1, :].squeeze(dim=1)
           
        
        # Encode graph...
        if self.phase == 1:
            graph_z = self.graph_vae.encode(batch, x, edge_index)
            z = graph_z
        elif self.phase == 2:
            z = text_z
            
        feat_hat = self.graph_vae.decode(z)
        d = nn.Dropout(p=0.33)
        feat_hat = d(feat_hat)
        
        # Calculate P(g'|z) loss
        kl_loss, pos_loss, neg_loss, roc_auc_score, avg_precision_score  = self.graph_vae.recon_loss(feat_hat, x, edge_index, batch, nodetoken_ids, src_dict)

        # Calculate KL divergence between graph_z and text_z
        if self.phase == 1:
            criterion = nn.KLDivLoss(reduction='batchmean')
            softmax = nn.Softmax(dim=1)
            p = torch.log(softmax(text_z))
            q = softmax(graph_z)
            txt_kl_loss = criterion(p,q)        
        
        losses = dict()
        losses["kl_loss"] = kl_loss
        losses["pos_loss"] = pos_loss
        losses["neg_loss"] = neg_loss
        losses["txt_kl_loss"] = txt_kl_loss
        losses["loss"] = sum(losses.values())
        
        metrics = dict()
        metrics["roc_auc_score"] = roc_auc_score
        metrics["avg_precision_score"] = avg_precision_score
        
        return losses, metrics
