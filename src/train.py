import os
import argparse
import yaml
import torch
import json
import time
import logging
import numpy as np
import networkx as nx
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from amr_parsing.io import AMRIO
from vocab import Vocab
from gtvae import GTVAE

#from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
#from datautil import *
#import pickle
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

# Logging..
timestr = time.strftime("%Y%m%d-%H%M%S")
# writer = SummaryWriter()
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# def setup_logger(name, log_file, level=logging.INFO):
#     """To setup as many loggers as you want"""
#     handler = logging.FileHandler(log_file)        
#     handler.setFormatter(formatter)
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)
#     return logger
# train_logger = setup_logger('train_logger', timestr+'_train.log')
# dev_logger   = setup_logger('dev_logger',   timestr+'_dev.log')
# test_logger  = setup_logger('test_logger',  timestr+'_test.log')
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)    
# logging.basicConfig(filename=timestr+'_app.log', format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')


# def loaddata(src_dict, maxnode):
#     def prep_data_list(amrs, bpe_data_sents, bpe_data_nodes):
#         data_list = []
#         graphs = []
#         sents = []
#         for amr in amrs:
#             src_tokens = amr.tokens
#             graph = amr.graph
#             G = graph._G
#             graphs.append(G)
#             mytoks, myvars = graph.get_tgt_tokens()
#             if G.number_of_nodes() > maxnode:
#                  continue
             
#             # if '_' in ' '.join(amr.tokens):
#             #     continue

#             node_UNIQ_IDS = dict()
#             for idx, nd in enumerate(myvars):
#                 node_UNIQ_IDS[nd] = idx
#             nodes = dict()
#             edges = []

#             node_ids = []; node_ids_local = []; snttoken_ids = []

#             fileprefix = amr.id.split(" ::")[0]
#             bpe_tokens =  bpe_data_sents[fileprefix]
#             node_bpe_tokens = bpe_data_nodes[fileprefix] 

#             for token in bpe_tokens:
#                 snttokenid = src_dict[token]
#                 snttoken_ids.append(snttokenid)

#             # Add BOS (<s>) and EOS (</s>) to the sentence...
#             snttoken_ids = [2] + snttoken_ids 
#             snttoken_ids.append(4)

#             adj = np.zeros((G.number_of_nodes(), G.number_of_nodes()))                
#             gold_edges = []
#             for edge in graph.edges():
#                 #breakpoint()
#                 if edge.source in node_UNIQ_IDS and edge.target in node_UNIQ_IDS:
#                     s_id = node_UNIQ_IDS[edge.source]
#                     t_id = node_UNIQ_IDS[edge.target]
#                     edges.append([s_id, t_id])
#                     edges.append([t_id, s_id]) ##TODO:For now assuming nondirected AMR graphs; both edge and reverse included
#                     edges.append([s_id, s_id])
#                     edges.append([t_id, t_id])

#                     if s_id != t_id:
#                         gold_edges.append((s_id, t_id))
#                         adj[s_id, t_id] = 1
#                         adj[t_id, s_id] = 1

#             gold_edges = set(gold_edges)
#             if len(gold_edges) == 0:
#                 continue
#             for nodetoken in node_bpe_tokens[:-1]: #last one is trivial
#                 nodelabel = nodetoken.split(" ")[0]
#                 nodevocabid = src_dict[nodelabel]
#                 node_ids.append(nodevocabid)

#             adj = np.asarray(adj) + np.identity(G.number_of_nodes())
#             orig_node_ids = list(node_ids)
#             # Pad node ids to have max number of nodes
#             missing_counts = maxnode -len(node_ids)
#             if missing_counts > 0:
#                 node_ids.extend([1] * missing_counts)
#             elif missing_counts < 0:
#                 node_ids = node_ids[:missing_counts]

#             neg_edge_index= []
#             for i in range(maxnode-missing_counts, maxnode):
#                 neg_edge_index.append((i,i))

#             num_nodes = len(orig_node_ids)
#             adj_padded = np.zeros((maxnode, maxnode))
#             adj_padded[:min(num_nodes,maxnode), :min(num_nodes,maxnode)] = adj[:min(num_nodes, maxnode), :min(num_nodes,maxnode)] # delete exceeds

#             nodeids = torch.LongTensor(node_ids).to(device).transpose(-1,0)
#             nodeids = torch.unsqueeze(nodeids, 1)
#             edge_index = torch.tensor(edges, dtype=torch.long)
#             neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)

#             x = np.identity(maxnode)
#             x = torch.tensor(x, dtype=torch.float) 

#             sents.append(amr.tokens)
#             data = Data(x=x, edge_index=edge_index.t().contiguous())
#             data.__setitem__("snttoken_ids", snttoken_ids)
#             data.__setitem__("nodetoken_ids", nodeids)
#             data.__setitem__("orig_node_ids", orig_node_ids)
#             data.__setitem__("gold_edges", gold_edges)
#             data.__setitem__("neg_edge_index", neg_edge_index.t().contiguous())
#             data.__setitem__("adj_padded", adj_padded)
#             data.__setitem__("snts", amr.sentence)
#             data.__setitem__("prefixes", fileprefix)
#             data_list.append(data)
#         ##end of one AMR
#         return data_list, graphs, sents
        
#     # Load data...
#     trn_path = '../data/AMR/amr_2.0/train.txt.features.preproc'
#     dev_path = '../data/AMR/amr_2.0/dev.txt.features.preproc'
#     tst_path = '../data/AMR/amr_2.0/test.txt.features.preproc'
#     def read_amrs(path):
#         amrs = []
#         for i,amr in enumerate(AMRIO.read(path)):
#             #if i>5000:
#             #    break
#             amrs.append(amr)
#         return amrs

#     trn_amrs = read_amrs(trn_path)
#     dev_amrs = read_amrs(dev_path)
#     tst_amrs = read_amrs(tst_path)
        
#     # Load nearest bpe tokens for node labels and src words...
#     bpe_data_sents_trn = json.loads(open("../data/sent_bpes_trn.json", "r").read())
#     bpe_data_sents_dev = json.loads(open("../data/sent_bpes_dev.json", "r").read())
#     bpe_data_sents_tst = json.loads(open("../data/sent_bpes_tst.json", "r").read())
#     bpe_data_nodes_trn = json.loads(open("../data/node_bpes_trn.json", "r").read())
#     bpe_data_nodes_dev = json.loads(open("../data/node_bpes_dev.json", "r").read())
#     bpe_data_nodes_tst = json.loads(open("../data/node_bpes_tst.json", "r").read())
#     trn_data_list, trn_graphs, trn_snts = prep_data_list(trn_amrs, bpe_data_sents_trn, bpe_data_nodes_trn)
#     dev_data_list, dev_graphs, dev_snts = prep_data_list(dev_amrs, bpe_data_sents_dev, bpe_data_nodes_dev)
#     tst_data_list, tst_graphs, tst_snts = prep_data_list(tst_amrs, bpe_data_sents_tst, bpe_data_nodes_tst)
#     return trn_data_list, dev_data_list, tst_data_list, trn_graphs, dev_graphs, tst_graphs, trn_snts, dev_snts, tst_snts

def loaddata(maxnode):
    def prep_data_list(graphs):
        data_list = []
        for graph in graphs:
            num_of_nodes = len(graph['nodes'])
            if num_of_nodes > maxnode or 'edges' not in graph:
                continue
            edges = []
            gold_edges = []
            adj = np.zeros((num_of_nodes, num_of_nodes))                
            for edge in graph['edges']:
                s_id = edge['source']
                t_id = edge['target']
                edges.append((s_id, t_id))
                edges.append((t_id, s_id)) ##TODO:For now assuming nondirected AMR graphs; both edge and reverse included
                if (s_id, s_id) not in edges:
                    edges.append((s_id, s_id)) # self-loops
                if (t_id, t_id) not in edges:
                    edges.append((t_id, t_id)) # self-loops
                if s_id != t_id:
                    gold_edges.append((s_id, t_id))
                    adj[s_id, t_id] = 1
                    adj[t_id, s_id] = 1
            gold_edges = set(gold_edges)
            # Create padded adj matrix
            adj = np.asarray(adj) + np.identity(num_of_nodes)
            adj_padded = np.zeros((maxnode, maxnode))
            adj_padded[:num_of_nodes, :num_of_nodes] = adj
            # Add to data object
            x = np.identity(maxnode)
            x = torch.tensor(x, dtype=torch.float) 
            #adj_padded = torch.tensor(adj_padded, dtype= torch.float)
            edge_index = torch.tensor(edges, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index.t().contiguous())
            data.__setitem__("gold_edges", gold_edges)
            data.__setitem__("adj_padded", adj_padded)
            data.__setitem__("snts", graph['input'])
            data_list.append(data)
        ##end of one AMR
        return data_list
        
    # Load data...
    trn_amr_data_path = '/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/data/MRP2020/cf/training/amr.mrp'
    val_amr_data_path = '/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/data/MRP2020/cf/validation/amr.mrp'
    f = open(trn_amr_data_path, "r+")
    trn_amrs = [json.loads(l) for l in f.readlines()]
    f = open(val_amr_data_path, "r+")
    val_amrs = [json.loads(l) for l in f.readlines()]
    trn_data_list = prep_data_list(trn_amrs)
    val_data_list = prep_data_list(val_amrs)
    return trn_data_list, val_data_list



def dev(loader, model, epoch, kl_anneal_w):
    epoch_loss = 0; epoch_graph_recon_loss = 0; epoch_t_graph_recon_loss = 0; epoch_joint_kl_loss, epoch_text_recon_loss = 0, 0 
    epoch_edge_recall = 0; epoch_edge_precision = 0; epoch_text_acc = 0; epoch_edge_recall_m = 0; epoch_edge_precision_m = 0; epoch_g_text_acc = 0 
    epoch_t_edge_recall = 0; epoch_t_edge_precision = 0 

    model.eval()
    for (step, data) in enumerate(loader):
        with torch.no_grad():
            data = data.to(device)
            dec_seq = tokenizer(data.__getitem__("snts"), return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            x = data.x # 640,10
            edge_index = data.edge_index # 2,numofedges
            batch = data.batch # 640
            gold_edges = data.gold_edges # 64
            adj_padded = data.adj_padded # 64

            loss, text_acc, joint_kl_loss, graph_recon_loss, text_recon_loss, edge_recall, edge_precision, _g_text_acc, _t_edge_recall, _t_edge_precision = model(x, edge_index, batch, adj_padded, gold_edges, dec_seq, kl_anneal_w)
            batchSize = len(data.snts)
            epoch_loss += loss.item() * batchSize
            epoch_text_acc += text_acc * batchSize
            epoch_joint_kl_loss += joint_kl_loss * batchSize
            epoch_graph_recon_loss += graph_recon_loss * batchSize
            epoch_text_recon_loss += text_recon_loss * batchSize
            epoch_edge_recall += edge_recall * batchSize
            epoch_edge_precision += edge_precision * batchSize
            epoch_g_text_acc += _g_text_acc * batchSize
            epoch_t_edge_recall += _t_edge_recall * batchSize
            epoch_t_edge_precision += _t_edge_precision * batchSize

            
    num_instances = len(loader.dataset)
    # print('DEV Epoch: {}, Loss: {:.3f}, Graph recon loss: {:.3f}, Text recon loss: {:.3f}, Joint kl loss: {:.3f}, Recall: {:.3f}, Precision: {:.3f}, Text acc: {:.3f}, Gr Text acc: {:.3f}, T edge_recall:  {:.3f}, T edge_precision:  {:.3f}, kl_anneal_w: {}'
    #               .format(epoch,
    #                       epoch_loss/ num_instances,
    #                       epoch_graph_recon_loss/num_instances,
    #                       epoch_text_recon_loss/num_instances,
    #                       epoch_joint_kl_loss/num_instances,
    #                       epoch_edge_recall/ num_instances,
    #                       epoch_edge_precision/ num_instances,
    #                       epoch_text_acc/ num_instances,
    #                       epoch_g_text_acc / num_instances,
    #                       epoch_t_edge_recall / num_instances,
    #                       epoch_t_edge_precision / num_instances,
    #                       kl_anneal_w))
   
    print('DEV')
    print('Epoch: {}'.format(epoch))
    print('Epoch loss: {:.3f}'.format(epoch_loss / num_instances))
    print('Epoch graph_recon loss: {:.3f}'.format(epoch_graph_recon_loss/num_instances), 'Epoch text_recon_loss: {:.3f}'.format(epoch_text_recon_loss / num_instances))
    print('Epoch joint_kl_loss: {:.3f}'.format(epoch_joint_kl_loss/num_instances))
    print('Epoch edge_recall: {:.3f}'.format(epoch_edge_recall/num_instances), 'Epoch edge_precision: {:.3f}'.format(epoch_edge_precision/num_instances))
    print('Epoch text_acc: {:.3f}'.format(epoch_text_acc/num_instances))
    #print('Epoch _t_edge_recall: {:.3f}'.format(epoch_t_edge_recall/num_instances),'Epoch _t_edge_precision: {:.3f}'.format(epoch_t_edge_precision/ num_instances))
    #print('Epoch _g_text_acc: {:.3f}'.format(epoch_g_text_acc/num_instances))
    #print('kl_anneal_w:', kl_anneal_w)
    
def train(train_loader, dev_loader, model, epochs, model_name):
    #logging.info("trnsize: {}, devsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(dev_loader.dataset), epochs)) 
    Nkl= 500
    at = 0.3
    lr = 0.001
    print("trnsize: {}, devsize: {}, num_epochs: {}, nkl: {}, at: {}, lr: {}".format(len(train_loader.dataset), len(dev_loader.dataset), epochs, Nkl, at, lr)) 
    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        model.train()
        kl_anneal_w =  min(epoch/Nkl, 1)
        report = False
        if epoch % 1 == 0:
            report = True
        epoch_loss = 0; epoch_graph_recon_loss = 0; epoch_t_graph_recon_loss = 0;  epoch_joint_kl_loss, epoch_text_recon_loss = 0, 0
        epoch_edge_recall = 0; epoch_edge_precision = 0; epoch_text_acc = 0; epoch_edge_recall_m = 0; epoch_edge_precision_m = 0; epoch_g_text_acc = 0
        epoch_t_edge_recall = 0; epoch_t_edge_precision = 0 

        for step, data in enumerate(train_loader):
            data = data.to(device)
            opt.zero_grad()
            dec_seq = tokenizer(data.__getitem__("snts"), return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            x = data.x # 640,10
            edge_index = data.edge_index # 2,numofedges
            batch = data.batch # 640
            gold_edges = data.gold_edges # 64
            adj_padded = data.adj_padded # 64

            loss, text_acc, joint_kl_loss, graph_recon_loss, text_recon_loss, edge_recall, edge_precision, _g_text_acc, _t_edge_recall, _t_edge_precision = model(x, edge_index, batch, adj_padded, gold_edges, dec_seq, kl_anneal_w)
            loss.backward()
            opt.step() 
            batchSize = len(data.gold_edges)
            epoch_loss += loss.item() * batchSize
            epoch_text_acc += text_acc * batchSize
            epoch_joint_kl_loss += joint_kl_loss * batchSize
            epoch_graph_recon_loss += graph_recon_loss * batchSize
            epoch_text_recon_loss += text_recon_loss * batchSize
            epoch_edge_recall += edge_recall * batchSize
            epoch_edge_precision += edge_precision * batchSize
            epoch_g_text_acc += _g_text_acc * batchSize
            epoch_t_edge_recall += _t_edge_recall * batchSize
            epoch_t_edge_precision += _t_edge_precision * batchSize

        num_instances = len(train_loader.dataset)
        if report:
            # print('Epoch: {}, Loss: {:.3f}, Graph recon loss: {:.3f}, Text recon loss: {:.3f}, Joint kl loss: {:.3f}, Recall: {:.3f}, Precision: {:.3f}, Text acc: {:.3f}, Gr Text acc: {:.3f}, T edge_recall:  {:.3f}, T edge_precision:  {:.3f}, kl_anneal_w: {}'
            #       .format(epoch,
            #               epoch_loss/ num_instances,
            #               epoch_graph_recon_loss/num_instances,
            #               epoch_text_recon_loss/num_instances,
            #               epoch_joint_kl_loss/num_instances,
            #               epoch_edge_recall/ num_instances,
            #               epoch_edge_precision/ num_instances,
            #               epoch_text_acc/ num_instances,
            #               epoch_g_text_acc / num_instances,
            #               epoch_t_edge_recall / num_instances,
            #               epoch_t_edge_precision / num_instances,
            #               kl_anneal_w))

            print("\n ============================")
            print('Epoch: {}'.format(epoch))
            print('Epoch loss: {:.3f}'.format(epoch_loss / num_instances))
            print('Epoch graph_recon loss: {:.3f}'.format(epoch_graph_recon_loss/num_instances), 'Epoch text_recon_loss: {:.3f}'.format(epoch_text_recon_loss / num_instances))
            print('Epoch joint_kl_loss: {:.3f}'.format(epoch_joint_kl_loss/num_instances))
            print('Epoch edge_recall: {:.3f}'.format(epoch_edge_recall/num_instances), 'Epoch edge_precision: {:.3f}'.format(epoch_edge_precision/num_instances))
            print('Epoch text_acc: {:.3f}'.format(epoch_text_acc/num_instances))
            #print('Epoch _t_edge_recall: {:.3f}'.format(epoch_t_edge_recall/num_instances),'Epoch _t_edge_precision: {:.3f}'.format(epoch_t_edge_precision/ num_instances))
            #print('Epoch _g_text_acc: {:.3f}'.format(epoch_g_text_acc/num_instances))
            print('kl_anneal_w:', kl_anneal_w)
    
        if epoch % 1 == 0:
           dev_loss  = dev(dev_loader, model, epoch, kl_anneal_w)
        #if epoch % 20 == 0:
        #    torch.save(model.state_dict(), model_name+"_"+str(epoch))
        #    logging.info("Model saved to: {}".format(model_name+"_"+str(epoch)))
    #writer.flush(); writer.close()
    #torch.save(model.state_dict(), model_name)
    #logging.info("Model saved to: {}".format(model_name))


def load_phase_2_model(path, text_transformer):
    pretrained = GraphVAE(input_dim, 64, 256, nmax)
    pretrained.load_state_dict(torch.load(path))
    pretrained_dict = pretrained.state_dict()
    model = GraphVAE(input_dim, 64, 256, nmax, phase=2)#.to(device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # 1. filter out unnecessary keys
    model_dict.update(pretrained_dict)      # 2. overwrite entries in the existing state dic
    model.load_state_dict(model_dict)  # 3. load the new state dict
    return model


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time>1:
            y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).cuda()
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def train_rnn_epoch(epoch, num_layers, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    #breakpoint()
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data.item()*feature_dim
    return loss_sum/(batch_idx+1)


def test_rnn_epoch(epoch, max_num_node, max_prev_node, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()
    # generate graphs
    max_num_node = int(max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1, max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1, max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_long_data = y_pred_long.data.long()
    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)
    return G_pred_list

# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


if __name__ == "__main__":
    print("Oh hi")
    batch_size = 64; epochs = 521; phase = 1; nmax = 10

    # Load graphs data...
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    trn_data, val_data = loaddata(10)
    trn_loader = DataLoader(trn_data, batch_size=batch_size) 
    val_loader = DataLoader(val_data, batch_size=batch_size) 
    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    # Build model...
    input_dim = 10; latent_dim = 32; graph_mlp_hid_dim = 512; 
        
    if phase == 1:
       # Phase 1...
       model  = GTVAE(input_dim, latent_dim, graph_mlp_hid_dim, nmax, vocab_size, pad_idx)
       # model = load_phase_2_model("model/gtg_20210309-183511_200", text_transformer)             

       # for param in model.text_vae.parameters():
       #     param.requires_grad = False
           
       model.to(device)
       print("model: {}".format(model))
       train(trn_loader, val_loader, model, epochs, "gtg_"+timestr)



    # def load_sent(path):
    #     sents = []
    #     with open(path) as f:
    #         for line in f:
    #             sents.append(line.split())
    #     return sents
    # trn_path_yelp = '/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/data/yelp/train.txt'
    # valid_path_yelp = '/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/data/yelp/valid.txt'
    # train_sents = load_sent(trn_path_yelp)
    # valid_sents = load_sent(valid_path_yelp)

    # #file1 = open("dev.txt", "w")

    # uniq_snts = dict()
    # uniq_ids = []
    # for i,s in enumerate(dev_snts):
    #     snt = ' '.join(s).strip()
    #     if snt not in uniq_snts:
    #         uniq_ids.append(i)
    #         uniq_snts[snt] = i

    
    # def get_batch(x, vocab, device):
    #     go_x, x_eos = [], []
    #     max_len = max([len(s) for s in x])
    #     for s in x:
    #         s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
    #         padding = [vocab.pad] * (max_len - len(s))
    #         go_x.append([vocab.go] + s_idx + padding)
    #         x_eos.append(s_idx + [vocab.eos] + padding)
    #     return torch.LongTensor(go_x).t().contiguous().to(device), \
    #            torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

    # def get_batches(data, vocab, batch_size, device):
    #     order = range(len(data))
    #     z = sorted(zip(order, data), key=lambda i: len(i[1]))
    #     order, data = zip(*z)
    #     batches = []
    #     i = 0
    #     while i < len(data):
    #         j = i
    #         while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
    #             j += 1
    #         batches.append(get_batch(data[i: j], vocab, device))
    #         i = j
    #     return batches, order


    # all_sents =  train_sents + trn_snts + dev_snts
    # for s in all_sents:
    #     for i,w in enumerate(s):
    #         s[i] = s[i].lower()


    # _train_sents = train_sents #trn_snts
    # _dev_sents =  valid_sents  #dev_snts
    # vocab_file = 'amr_vocab.txt'
    # Vocab.build(_train_sents, vocab_file, 10000)
    # vocab = Vocab(vocab_file)


    # print('batchsize overwritten to 256')
    # train_batches, _ = get_batches(_train_sents, vocab, 256, device)    
    # valid_batches, _ = get_batches(_dev_sents, vocab, 256, device)
    

    # model  = GTVAE(input_dim, latent_dim, graph_mlp_hid_dim, nmax, vocab).to(device)
    # print('model:', model)
    # print('_train_sents:', len(_train_sents))
    # print('_dev_snts:',  len(_dev_sents))

    # opt = optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))
    # for epoch in range(500):
    #     print('\n -----epoch: ', epoch)
    #     epoch_loss = 0
    #     epoch_text_acc = 0
    #     epoch_insts = 0
    #     model.train()
    #     indices = list(range(len(train_batches)))
    #     random.shuffle(indices)
    #     for i, idx in enumerate(indices):
    #         opt.zero_grad()
    #         inputs, targets = train_batches[idx]
    #         loss, text_acc = model(inputs, targets) 
    #         loss.backward()
    #         opt.step()
    #         batchSize = inputs.size(1)
    #         epoch_loss += loss.item() * batchSize
    #         epoch_text_acc += text_acc * batchSize
    #         epoch_insts += batchSize
    #     print('trn_text_acc:', epoch_text_acc / epoch_insts)


    #     dev_epoch_loss = 0
    #     dev_epoch_text_acc = 0
    #     dev_epoch_insts = 0

    #     model.eval()
    #     with torch.no_grad():
    #         indices = list(range(len(valid_batches)))
    #         for i, idx in enumerate(indices):
    #             inputs, targets = valid_batches[idx]
    #             loss, text_acc = model(inputs, targets) 
    #             batchSize = inputs.size(1)
    #             dev_epoch_loss += loss.item() * batchSize
    #             dev_epoch_text_acc += text_acc * batchSize
    #             dev_epoch_insts += batchSize
    #         print('dev_text_acc:', dev_epoch_text_acc/dev_epoch_insts)


       
