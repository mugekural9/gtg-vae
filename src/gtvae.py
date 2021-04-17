import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from maxspantree import mst_graph
from transformers import BertModel, BertConfig
from graph_vae import GraphVAE
from text_vae import DAE, AAE

# Initializing a BERT bert-base-uncased style configuration
# configuration = BertConfig()

class GTVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, mlp_hid_dim, max_num_nodes, vocabsize, padidx, phase=1):
        super(GTVAE, self).__init__()
        self.graph_vae = GraphVAE(input_dim, latent_dim, mlp_hid_dim, max_num_nodes, 1)    
        self.text_vae = DAE(latent_dim, vocabsize, padidx)
        #self.joint_mu_linear_1   = nn.Linear(latent_dim, 512)
        #self.joint_mu_linear_2   = nn.Linear(512, latent_dim)
        #self.joint_var_linear_1  = nn.Linear(latent_dim, 512)
        #self.joint_var_linear_2  = nn.Linear(512, latent_dim)
        #self.zg_emb = nn.Linear(latent_dim, latent_dim)        
        self.zg_mu = nn.Linear(latent_dim, latent_dim)
        self.zg_var = nn.Linear(latent_dim, latent_dim)
        #self.relu = nn.ReLU()
        #self.bn = nn.BatchNorm1d(latent_dim)
        #self.drop = nn.Dropout(0.2)
        self.max_num_nodes = max_num_nodes
        self.phase= phase
       
    def reparameterize(self, mu, logvar):
        if True: #self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def loss_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

    # def forward(self, x, edge_index, batch, adj, gold_edges, dec_seq, kl_anneal_w, at, flip, train_decoder=False, train_decoder_2=False, is_dev=False):#, targets): # report, dec_seq, encoding, kl_anneal_w):
    #     B, _ = dec_seq.size()
    #     text_acc, _text_acc, _edge_recall, _edge_precision, t_graph_recon_loss, joint_kl_loss, edge_recall, edge_precision, graph_recon_loss, loss, l1loss = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    #     # Encode graph...
    #     graph_h = self.graph_vae.encode(batch, x, edge_index)
    #     # Encode text...
    #     inputs  = dec_seq.t() 
    #     targets = torch.cat((dec_seq[:,1:],torch.ones(B,1).to("cuda")), 1).long().t() #B,T shift targets
    #     text_h = self.text_vae.autoenc(inputs, targets) # B,Lat        

    #     # Decode graph...
    #     input_g = self.zg_emb(graph_h)
    #     out_tensor = self.graph_vae.decode(input_g) 
    #     graph_recon_loss, edge_recall, edge_precision = self.graph_vae.loss(out_tensor, adj, gold_edges, True) 

    #     if train_decoder:
    #         loss += graph_recon_loss
    #     else:
    #         l1 = nn.L1Loss()
    #         l1loss = l1(text_h, graph_h)
    #         loss += l1loss
            
    #     # Decode graph from text...
    #     input_t = self.zg_emb(text_h)
    #     out_tensor = self.graph_vae.decode(input_t) 
    #     t_graph_recon_loss, _edge_recall, _edge_precision = self.graph_vae.loss(out_tensor, adj, gold_edges, True)
        
    #     return loss, text_acc, joint_kl_loss, graph_recon_loss, l1loss, edge_recall, edge_precision, _text_acc, _edge_recall, _edge_precision




    def forward_1(self, inputs, targets):
        loss = 0 
        # Encode text...
        text_h = self.text_vae.autoenc(inputs, targets) # B,Lat        
        text_mu = self.zg_mu(text_h)
        text_var = self.zg_var(text_h)
        text_z = self.reparameterize(text_mu, text_var)
        text_kl_loss = self.loss_kl(text_mu, text_var)
        loss += text_kl_loss
        # Decode text...
        logits, _ = self.text_vae.decode(text_z, inputs)
        text_recon_loss = self.text_vae.loss_rec(logits, targets).mean()
        text_acc = self.text_vae.accuracy(logits, targets, True)
        loss += text_recon_loss
        return loss, text_acc


    
    def forward(self, x, edge_index, batch, adj, gold_edges, dec_seq, kl_anneal_w, is_dev=False):#, targets): # report, dec_seq, encoding, kl_anneal_w):

        B, _ = dec_seq.size()
        text_acc, _text_acc, _edge_recall, _edge_precision, t_graph_recon_loss, joint_kl_loss, edge_recall, edge_precision, graph_recon_loss, loss, l1loss = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # Encode graph...
        graph_h = self.graph_vae.encode(batch, x, edge_index)
        # Encode text...
        pad_tokens = torch.full((B, 1), 0).to('cuda') #102:tokenizer.sep_token_id
        inputs  = dec_seq.t()
        targets = torch.cat((dec_seq[:,1:], pad_tokens), 1).long().t() #B,T shift targets 
        text_h = self.text_vae.autoenc(inputs, targets) # B,Lat        

        joint_h = (text_h + graph_h)/2
        joint_mu = self.zg_mu(joint_h)
        joint_var = self.zg_var(joint_h)

        joint_z = self.reparameterize(joint_mu, joint_var)
        joint_kl_loss = self.loss_kl(joint_mu, joint_var)
        loss += joint_kl_loss * kl_anneal_w
    
        # Decode text...
        logits, _ = self.text_vae.decode(joint_z, inputs)
        text_recon_loss = self.text_vae.loss_rec(logits, targets).mean()
        text_acc = self.text_vae.accuracy(logits, targets, is_dev)
        loss += text_recon_loss

        # Decode graph...
        out_tensor = self.graph_vae.decode(joint_z) 
        graph_recon_loss, edge_recall, edge_precision = self.graph_vae.loss(out_tensor, adj, gold_edges, False) 
        #loss += graph_recon_loss

        return loss, text_acc, joint_kl_loss, graph_recon_loss, text_recon_loss, edge_recall, edge_precision, _text_acc, _edge_recall, _edge_precision

     
         ########################################
         # text_mu  = self.joint_mu_linear_2(F.relu(self.joint_mu_linear_1(text_mu)))
         # text_var = self.joint_var_linear_2(F.relu(self.joint_var_linear_1(text_var)))
         # text_z = self.reparameterize(text_mu, text_var)
         # input_z = self.zg_emb(text_z)
         # out_tensor = self.graph_vae.decode(input_z)
         # _, _edge_recall, _edge_precision = self.graph_vae.loss(out_tensor, adj, gold_edges, True)#is_dev)
         # # # Calculate KL divergence between text_z and graph_z
         # criterion = nn.KLDivLoss(reduction='batchmean')
         # softmax = nn.Softmax(dim=1)
         # p = torch.log(softmax(graph_z))
         # q = softmax(text_z)
         # text_kl_loss = criterion(p,q)        
         # Decode text...
         # input_t = self.zg_emb(graph_h) #+ self.drop(graph_h) 
         # logits, _ = self.text_vae.decode(input_t, inputs)
         # text_recon_loss = self.text_vae.loss_rec(logits, targets).mean()
         # _text_acc = self.text_vae.accuracy(logits, targets, is_dev)
         # DAAE
         # d_loss, adv = self.text_vae.loss_adv(joint_z)
         # text_recon_loss = self.text_vae.loss_rec(logits, targets).mean()
         # adv_loss = adv
         # lvar_loss = text_var.abs().sum(dim=1).mean()
         # loss = text_recon_loss + (10 * adv) + (0.01 * lvar_loss)
         #if is_dev:
         #    self.text_vae.generate(joint_z, 20, 'greedy')

        

    





        ###################################
        # Encode graph and text together...     
        # exp1
        # joint_mu  = self.joint_mu_linear_2(F.relu(self.joint_mu_linear_1(torch.cat((graph_mu, text_mu), dim=1))))
        # joint_var = self.joint_var_linear_2(F.relu(self.joint_var_linear_1(torch.cat((graph_var, text_var), dim=1))))
        # exp2
        # joint_mu  = self.joint_mu_linear_2(F.relu(self.joint_mu_linear_1((graph_mu + text_mu) /2)))
        # joint_var = self.joint_var_linear_2(F.relu(self.joint_var_linear_1((graph_var + text_var) /2)))        
        # v1
        #joint_kl_loss = self.loss_kl(joint_mu, joint_var)
        #joint_z = self.reparameterize(joint_mu, joint_var)
        #input_z = self.drop(graph_mu) + self.zg_emb(joint_z)
        # v2
        #joint_z = self.reparameterize(text_mu, text_var)
        #joint_kl_loss = self.loss_kl(text_mu, text_var)
        
        # Decode text...
        #logits, _ = self.text_vae.decode(joint_z, inputs)
        #text_recon_loss = self.text_vae.loss_rec(logits, targets).mean()
        #text_acc = self.text_vae.accuracy(logits, targets, is_dev)
        # DAAE
        # d_loss, adv = self.text_vae.loss_adv(joint_z)
        # text_recon_loss = self.text_vae.loss_rec(logits, targets).mean()
        # adv_loss = adv
        # lvar_loss = text_var.abs().sum(dim=1).mean()
        # loss = text_recon_loss + (10 * adv) + (0.01 * lvar_loss)
        #if is_dev:
        #    self.text_vae.generate(joint_z, 20, 'greedy')

        # # Graph ONLY
        # _graph_mu  = self.joint_mu_linear_2(F.relu(self.joint_mu_linear_1(graph_mu)))
        # _graph_var = self.joint_var_linear_2(F.relu(self.joint_var_linear_1(graph_var)))
        # _graph_z = self.reparameterize(_graph_mu, _graph_var)
        # # Calculate KL divergence between graph_z and joint_z
        # criterion = nn.KLDivLoss(reduction='batchmean')
        # softmax = nn.Softmax(dim=1)
        # p = torch.log(softmax(joint_z))
        # q = softmax(_graph_z)
        # graph_joint_kl_loss = criterion(p,q)        
        # input = self.text_vae.generate(joint_z, targets.size(0)+1, 'greedy')
        # pred_tokens = torch.stack(input).squeeze(1)[1:, :]
        # _text_acc = (((pred_tokens == targets) * (targets != self.text_vae.vocab.pad)).sum() / (targets != self.text_vae.vocab.pad).sum()).item()
        # # Text ONLY
        # _text_mu  = self.joint_mu_linear_2(F.relu(self.joint_mu_linear_1(text_mu)))
        # _text_var = self.joint_var_linear_2(F.relu(self.joint_var_linear_1(text_var)))
        # _text_z = self.reparameterize(_text_mu, _text_var)
        # # # Calculate KL divergence between text_z and joint_z
        # criterion = nn.KLDivLoss(reduction='batchmean')
        # softmax = nn.Softmax(dim=1)
        # p = torch.log(softmax(joint_z))
        # q = softmax(_text_z)
        # text_joint_kl_loss = criterion(p,q)        
        # out_tensor = self.graph_vae.decode(_text_z) 
        # _, _edge_recall, _edge_precision = self.graph_vae.loss(out_tensor, adj, gold_edges, is_dev) #report)
        # loss =  text_recon_loss + (joint_kl_loss * kl_anneal_w) + graph_recon_loss + (0.1 * (graph_joint_kl_loss + text_joint_kl_loss))


