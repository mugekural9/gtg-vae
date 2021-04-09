import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from maxspantree import mst_graph
from graph_vae import GraphVAE
from transformers import BertModel
from torch.nn import Linear
from transformers import BertModel, BertConfig
from text_vae import DAE

# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()

class GTVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, mlp_hid_dim, max_num_nodes, text_vocab, phase=1):
        super(GTVAE, self).__init__()
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.graph_vae_encoder = GraphVAE(input_dim, latent_dim, max_num_nodes, 38926, 1)    
        self.graph_vae_decoder = MLP_VAE_plain(latent_dim, mlp_hid_dim, output_dim, max_num_nodes, 38926)
        self.text_vae = DAE(latent_dim, text_vocab)
        self.joint_mu_linear_1  = nn.Linear(latent_dim, 512)
        self.joint_var_linear_1 = nn.Linear(latent_dim, 512)
        self.joint_mu_linear_2  = nn.Linear(512, latent_dim)
        self.joint_var_linear_2  = nn.Linear(512, latent_dim)

        self.max_num_nodes = max_num_nodes
        self.phase= phase
        self.zg_emb = nn.Linear(latent_dim, latent_dim)
        self.drop = nn.Dropout(0.5)

        
    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def loss_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

    def forward(self, x, edge_index, batch, adj, gold_edges, report, dec_seq, encoding, nodetoken_ids, kl_anneal_w):
        B, maxnode, _ = adj.size()
        total_loss           = 0;
        graph_recon_loss     = 0; 
        text_recon_loss      = 0 
        total_edge_recall    = torch.tensor(0).float() 
        total_edge_precision = torch.tensor(0).float() 
        total_edge_recall_m = torch.tensor(0).float(); total_edge_precision_m = torch.tensor(0).float()

        # Encode graph...
        graph_mu, graph_var = self.graph_vae_encoder.encode(batch, x, edge_index)

        # Encode text...
        inputs = dec_seq.t()
        targets= torch.cat((dec_seq[:,1:],torch.ones(B,1).to("cuda")), 1).long().t() #B,T shift targets
        text_mu, text_var = self.text_vae.autoenc(inputs, targets, is_train=True)

        # Encode graph and text together...     
        # exp1
        #joint_mu  = self.joint_mu_linear_2(F.relu(self.joint_mu_linear_1(torch.cat((graph_mu, text_mu), dim=1))))
        #joint_var = self.joint_var_linear_2(F.relu(self.joint_var_linear_1(torch.cat((graph_var, text_var), dim=1))))
        # exp2
        joint_mu   = self.joint_mu_linear_2(F.relu(self.joint_mu_linear_1((graph_mu + text_mu) /2)))
        joint_var  = self.joint_var_linear_2(F.relu(self.joint_var_linear_1((graph_var + text_var) /2)))

        #ver1
        joint_kl_loss = self.loss_kl(joint_mu, joint_var)
        joint_z = self.reparameterize(joint_mu, joint_var)
        input_z = self.drop(graph_mu) + self.zg_emb(joint_z)

        #ver2
        #joint_kl_loss = self.loss_kl(text_mu, text_var)
        #joint_z = self.reparameterize(text_mu, text_var)

        # Decode text...
        logits, _ = self.text_vae.decode(joint_z, inputs)
        text_recon_loss = self.text_vae.loss_rec(logits, targets).mean()
        text_acc = self.text_vae.accuracy(logits, targets)

        #if src_dict is not None and report:
        #     self.text_vae.generate(joint_z, 20, 'greedy', src_dict)
       
        # Decode graph...
        h_decode = self.graph_vae_decoder(input_z) 
        out = F.sigmoid(h_decode)
        out_tensor = out.cpu().data       
        # Graph reconstruction...
        for b in range(B):
            _out_tensor = out_tensor[b,:].unsqueeze(0)
            recon_adj_lower = self.recover_adj_lower(_out_tensor) 
            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower) # make symmetric
            adj_data = adj[b].cpu().data #[0]
            adj_permuted = adj_data
            adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()
            adj_vectorized_var = adj_vectorized.cuda()
            adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[b])
            graph_recon_loss += adj_recon_loss
            if report:
                edge_recall, edge_precision = self.graph_statistics(recon_adj_tensor, list(gold_edges[b]), report)
                total_edge_recall += edge_recall
                total_edge_precision += edge_precision
        
        graph_recon_loss /= B
        total_edge_recall /= B
        total_edge_precision /= B
        total_loss =  graph_recon_loss + text_recon_loss + (kl_anneal_w * joint_kl_loss) 
        return total_loss, total_edge_recall, total_edge_precision, text_acc, total_edge_recall_m, total_edge_precision_m, joint_kl_loss , graph_recon_loss, text_recon_loss

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
        edge_recall = count / gold_edgelength
        edge_precision = count/ pred_edgelength
        if False: #report:
            print("\ngold_edges:", gold_edges)
            print("pred_edges:", pred_edges)        
        return edge_recall, edge_precision


class MLP_VAE_plain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nmax, vocab_size):
        super(MLP_VAE_plain, self).__init__()
        self.decode_1 = nn.Linear(input_dim, hidden_dim)
        self.decode_2 = nn.Linear(hidden_dim, output_dim) 
        self.relu = nn.ReLU()
        self.nmax = nmax
        self.vocab_size = vocab_size
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                
    def forward(self, h):
        y = self.decode_1(h)
        y = self.relu(y)
        y = self.decode_2(y)
        return y

