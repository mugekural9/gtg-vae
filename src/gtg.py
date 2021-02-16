import torch.nn as nn
from graphvae import GraphVAE
from torch_geometric.nn import global_max_pool

class GTG(nn.Module):
    def __init__(self, input_dim, hidden_dim, nmax, nodeclass_num, text_encoder): 
       super(GTG, self).__init__()
       self.graph_vae = GraphVAE(input_dim, hidden_dim, nmax, nodeclass_num)
       self.text_encoder = text_encoder 


    def forward(self, x, edge_index, batch, dec_seq, nodetoken_ids, src_dict):

        # Encode graph...
        graph_z = self.graph_vae.encode(batch, x, edge_index)
        feat_hat = self.graph_vae.decode(graph_z)
        
        # Calculate P(g'|z) loss
        kl_loss, pos_loss, neg_loss, roc_auc_score, avg_precision_score  = self.graph_vae.recon_loss(feat_hat, x, edge_index, batch, nodetoken_ids, src_dict)
        
        # Encode text...
        src_seq = nodetoken_ids.t()          
        #src_enc = feat_hat.transpose(0,1)    

        # work with transformer_decoder...
        # self.text_encoder.decoder.init_state(src_seq, src_enc) # src_seq should be maxnode,B, src_enc maxnode,B,Zdim
        # dec_output, *_ = self.text_encoder.decoder(dec_seq, step=None) # src_seq as dec_seq, but any need for shifting? dec_output: (Tdec, B, Zdim)         
        # text_z = dec_output[:1, :,:].squeeze(0)

        # work with transformer_encoder...
        src_emb, src_enc, _ = self.text_encoder.encoder(src_seq)
        text_z = src_enc[:1, :,:].squeeze(0)
        # import pdb; pdb.set_trace()
        
        # Calculate z-z' loss
        #loss = nn.MSELoss()
        #mse_loss = loss(graph_z, text_z) 

        
        losses = dict()
        losses["kl_loss"] = kl_loss
        losses["pos_loss"] = pos_loss
        losses["neg_loss"] = neg_loss
        #losses["mse_loss"] = mse_loss
        losses["loss"] = sum(losses.values())
        
        metrics = dict()
        metrics["roc_auc_score"] = roc_auc_score
        metrics["avg_precision_score"] = avg_precision_score
        
        return losses, metrics  #nodelabel_loss correct_predicted_node_tokens
