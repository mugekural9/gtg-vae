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
from amr_parsing.io import AMRIO


from torch_geometric.data import Data, DataLoader
from gtvae import GTVAE
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from datautil import *
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

# Logging..
timestr = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
train_logger = setup_logger('train_logger', timestr+'_train.log')
dev_logger   = setup_logger('dev_logger',   timestr+'_dev.log')
test_logger  = setup_logger('test_logger',  timestr+'_test.log')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)    
logging.basicConfig(filename=timestr+'_app.log', format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')


def loaddata(src_dict, maxnode):
    def prep_data_list(amrs, bpe_data_sents, bpe_data_nodes):
        data_list = []
        graphs = []
        
        for amr in amrs:
            src_tokens = amr.tokens
            graph = amr.graph
            G = graph._G
            graphs.append(G)
            mytoks, myvars= amr.graph.get_tgt_tokens()

            # if amr.id == "bolt12_10474_1831.8 ::date 2012-12-07T09:08:08 ::annotator SDL-AMR-09 ::preferred": #CANT HANDLE REENTRANCY
            #     breakpoint()
            #     for nnn in graph.get_nodes():
            #         breakpoint()
            if G.number_of_nodes() > maxnode:
                 continue

            node_UNIQ_IDS = dict()
            for idx, nd in enumerate(myvars):
                node_UNIQ_IDS[nd] = idx
            # print(amr)
            # print(G.nodes())
            # print(node_UNIQ_IDS)
            nodes = dict()
            edges = []

            node_ids = []; node_ids_local = []; snttoken_ids = [] # with these ids we'll able to forward the embeddings

            fileprefix = amr.id.split(" ::")[0]
            bpe_tokens =  bpe_data_sents[fileprefix]
            node_bpe_tokens = bpe_data_nodes[fileprefix] 

            for token in bpe_tokens:
                snttokenid = src_dict[token]
                snttoken_ids.append(snttokenid)

            # Add BOS (<s>) and EOS (</s>) to the sentence...
            snttoken_ids = [2] + snttoken_ids 
            snttoken_ids.append(4)

            adj = np.zeros((G.number_of_nodes(), G.number_of_nodes()))                
            gold_edges = []
            for edge in graph.edges():
                #breakpoint()
                if edge.source in node_UNIQ_IDS and edge.target in node_UNIQ_IDS:
                    s_id = node_UNIQ_IDS[edge.source]
                    t_id = node_UNIQ_IDS[edge.target]
                    edges.append([s_id, t_id])
                    edges.append([t_id, s_id]) ##TODO:For now assuming nondirected AMR graphs; both edge and reverse included
                    edges.append([s_id, s_id])
                    edges.append([t_id, t_id])

                    if s_id != t_id:
                        gold_edges.append((s_id, t_id))
                        adj[s_id, t_id] = 1
                        adj[t_id, s_id] = 1

            gold_edges = set(gold_edges)
            if len(gold_edges) == 0:
                continue
            for nodetoken in node_bpe_tokens[:-1]: #last one is trivial
                nodelabel = nodetoken.split(" ")[0]
                nodevocabid = src_dict[nodelabel]
                node_ids.append(nodevocabid)

            adj = np.asarray(adj) + np.identity(G.number_of_nodes())
            # print("\n", amr)
            # print(graph.edges())
            # print("mynodes:", G.nodes())
            # print("mytoks:", mytoks)
            # print("myvars:", list(myvars))
            # print("nodeUNIQIDS:", node_UNIQ_IDS)
            # print("adj:", adj)

            orig_node_ids = list(node_ids)
            # Pad node ids to have max number of nodes
            missing_counts = maxnode -len(node_ids)
            if missing_counts > 0:
                node_ids.extend([1] * missing_counts)
            elif missing_counts < 0:
                node_ids = node_ids[:missing_counts]

            neg_edge_index= []
            for i in range(maxnode-missing_counts, maxnode):
                neg_edge_index.append((i,i))

            num_nodes = len(orig_node_ids)
            adj_padded = np.zeros((maxnode, maxnode))
            adj_padded[:min(num_nodes,maxnode), :min(num_nodes,maxnode)] = adj[:min(num_nodes, maxnode), :min(num_nodes,maxnode)] # delete exceeds

            nodeids = torch.LongTensor(node_ids).to(device).transpose(-1,0)
            nodeids = torch.unsqueeze(nodeids, 1)

            edge_index = torch.tensor(edges, dtype=torch.long)
            neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)

            features_all = np.identity(maxnode)
            features_all = torch.tensor(features_all, dtype=torch.float) 

            data = Data(x=features_all, edge_index=edge_index.t().contiguous())
            data.__setitem__("snttoken_ids", snttoken_ids)
            data.__setitem__("nodetoken_ids", nodeids)
            data.__setitem__("orig_node_ids", orig_node_ids)
            data.__setitem__("gold_edges", gold_edges)
            data.__setitem__("neg_edge_index", neg_edge_index.t().contiguous())
            data.__setitem__("adj_padded", adj_padded)
            data.__setitem__("features_all", features_all)
            data.__setitem__("snts", amr.sentence)
            data.__setitem__("prefixes", fileprefix)
            data_list.append(data)
        ##end of one AMR
        return data_list, graphs
        
    # Load data...

    trn_path = '../data/AMR/amr_2.0/train.txt.features.preproc'
    dev_path = '../data/AMR/amr_2.0/dev.txt.features.preproc'
    tst_path = '../data/AMR/amr_2.0/test.txt.features.preproc'
    
    def read_amrs(path):
        amrs = []
        for i,amr in enumerate(AMRIO.read(path)):
            if i>100:
                break
            amrs.append(amr)
        return amrs

    trn_amrs = read_amrs(trn_path)
    dev_amrs = read_amrs(dev_path)
    tst_amrs = read_amrs(tst_path)
        
    # Load nearest bpe tokens for node labels and src words...
    bpe_data_sents_trn = json.loads(open("../data/sent_bpes_trn.json", "r").read())
    bpe_data_sents_dev = json.loads(open("../data/sent_bpes_dev.json", "r").read())
    bpe_data_sents_tst = json.loads(open("../data/sent_bpes_tst.json", "r").read())
    bpe_data_nodes_trn = json.loads(open("../data/node_bpes_trn.json", "r").read())
    bpe_data_nodes_dev = json.loads(open("../data/node_bpes_dev.json", "r").read())
    bpe_data_nodes_tst = json.loads(open("../data/node_bpes_tst.json", "r").read())
    trn_data_list, trn_graphs = prep_data_list(trn_amrs, bpe_data_sents_trn, bpe_data_nodes_trn)
    dev_data_list, dev_graphs = prep_data_list(dev_amrs, bpe_data_sents_dev, bpe_data_nodes_dev)
    tst_data_list, tst_graphs = prep_data_list(tst_amrs, bpe_data_sents_tst, bpe_data_nodes_tst)
    return trn_data_list, dev_data_list, tst_data_list, trn_graphs, dev_graphs, tst_graphs


def dev(loader, model, epoch, kl_anneal_w):
    epoch_loss = 0; epoch_graph_recon_loss = 0; epoch_text_recon_loss = 0; epoch_joint_kl_loss = 0; 
    epoch_edge_recall = 0; epoch_edge_precision = 0; epoch_text_acc = 0; epoch_edge_recall_m = 0; epoch_edge_precision_m = 0
    model.eval()
    for (step, data) in enumerate(loader):
        with torch.no_grad():
            dec_seq = data.__getitem__("snttoken_ids")
            max_seq_len = max([len(i) for i in dec_seq])
            torch_dec_seq=[]
            for seq in dec_seq:
                torch_dec_seq.append(torch.tensor(seq, dtype=torch.long))
            dec_seq = torch.stack([torch.cat([i, i.new_ones(max_seq_len - i.size(0))], 0) for i in torch_dec_seq],1).t().to(device)
            x, edge_index, batch, nodetoken_ids, gold_edges, adj_input = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.__getitem__("nodetoken_ids"), data.__getitem__("gold_edges"),  torch.tensor(data.adj_padded).to(device).float()
            # encoding = tokenizer(data.__getitem__("snts"), return_tensors='pt', padding=True, truncation=True).to(device) # not used
            encoding = None
            # Model forward...
            #x = x / x.sum(1, keepdim=True).clamp(min=1) # Normalize input node features
            x = data.__getitem__("features_all").to(device)
            loss, edge_recall, edge_precision, text_acc, edge_recall_m, edge_precision_m, joint_kl_loss, graph_recon_loss, text_recon_loss = model(x, edge_index, batch, adj_input, gold_edges, True, dec_seq, encoding, nodetoken_ids, kl_anneal_w)  
            batchSize = len(data.snts)
            epoch_loss += loss.item() * batchSize; 
            epoch_graph_recon_loss += graph_recon_loss.item() * batchSize
            epoch_text_recon_loss += text_recon_loss.item() * batchSize
            epoch_joint_kl_loss += joint_kl_loss.item() * batchSize
            epoch_edge_recall += edge_recall * batchSize;
            epoch_edge_precision += edge_precision * batchSize;
            epoch_text_acc += text_acc * batchSize

    num_instances = len(loader.dataset)
    print('DEV Epoch: {}, Loss: {:.3f}, Graph recon loss: {:.3f}, Text recon loss: {:.3f}, Joint kl loss: {:.3f}, Recall: {:.3f}, Precision: {:.3f}, Text acc: {:.3f}, kl_anneal_w: {}'
          .format(epoch,
                  epoch_loss/ num_instances,
                  epoch_graph_recon_loss/num_instances,
                  epoch_text_recon_loss/num_instances,
                  epoch_joint_kl_loss/num_instances,
                  epoch_edge_recall/ num_instances,
                  epoch_edge_precision/ num_instances,
                  epoch_text_acc/ num_instances,
                  kl_anneal_w))

    # print('DEV Recall_m: {:.3f}, Precision_m: {:.3f}, kl_anneal_w: {}'
    #               .format(epoch_edge_recall_m/ num_instances,
    #                       epoch_edge_precision_m/ num_instances,
    #                       kl_anneal_w))


def train(train_loader, dev_loader, model, epochs, model_name):
    logging.info("trnsize: {}, devsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(dev_loader.dataset), epochs)) 
    print("trnsize: {}, devsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(dev_loader.dataset), epochs)) 
    opt = optim.Adam(model.parameters(), lr=0.001)
    Nkl= 500
    for epoch in range(epochs):
        model.train()
        kl_anneal_w =  min(epoch/Nkl, 0.5)
        report = False
        if epoch % 5 == 0:
            report = True
        epoch_loss = 0; epoch_graph_recon_loss = 0; epoch_text_recon_loss = 0; epoch_joint_kl_loss = 0;
        epoch_edge_recall = 0; epoch_edge_precision = 0; epoch_text_acc = 0; epoch_edge_recall_m = 0; epoch_edge_precision_m = 0
        for step, data in enumerate(train_loader):
            opt.zero_grad()
            dec_seq = data.__getitem__("snttoken_ids")
            max_seq_len = max([len(i) for i in dec_seq])
            torch_dec_seq=[]
            for seq in dec_seq:
                torch_dec_seq.append(torch.tensor(seq, dtype=torch.long))
            dec_seq = torch.stack([torch.cat([i, i.new_ones(max_seq_len - i.size(0))], 0) for i in torch_dec_seq],1).t().to(device)
            x, edge_index, batch, nodetoken_ids, gold_edges, adj_input = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.__getitem__("nodetoken_ids"), data.__getitem__("gold_edges"),  torch.tensor(data.adj_padded).to(device).float()
            # encoding = tokenizer(data.__getitem__("snts"), return_tensors='pt', padding=True, truncation=True).to(device) # not used
            encoding = None

            # Model forward...
            #x = x / x.sum(1, keepdim=True).clamp(min=1) # Normalize input node features
            x = data.__getitem__("features_all").to(device)
            loss, edge_recall, edge_precision, text_acc, edge_recall_m, edge_precision_m, joint_kl_loss, graph_recon_loss, text_recon_loss = model(x, edge_index, batch, adj_input, gold_edges, report, dec_seq, encoding, nodetoken_ids, kl_anneal_w)  
            loss.backward()
            opt.step() 
            batchSize = len(data.snts)
            epoch_loss += loss.item() * batchSize; 
            epoch_graph_recon_loss += graph_recon_loss.item() * batchSize
            epoch_text_recon_loss += text_recon_loss.item() * batchSize
            epoch_joint_kl_loss += joint_kl_loss.item() * batchSize
            epoch_edge_recall += edge_recall * batchSize;
            epoch_edge_precision += edge_precision * batchSize;
            epoch_text_acc += text_acc * batchSize

        num_instances = len(train_loader.dataset)
        if report:
            print('Epoch: {}, Loss: {:.3f}, Graph recon loss: {:.3f}, Text recon loss: {:.3f}, Joint kl loss: {:.3f}, Recall: {:.3f}, Precision: {:.3f}, Text acc: {:.3f}, kl_anneal_w: {}'
                  .format(epoch,
                          epoch_loss/ num_instances,
                          epoch_graph_recon_loss/num_instances,
                          epoch_text_recon_loss/num_instances,
                          epoch_joint_kl_loss/num_instances,
                          epoch_edge_recall/ num_instances,
                          epoch_edge_precision/ num_instances,
                          epoch_text_acc/ num_instances,
                          kl_anneal_w))
            # print('Recall_m: {:.3f}, Precision_m: {:.3f},  kl_anneal_w: {}'
            #       .format(epoch_edge_recall_m/ num_instances,
            #               epoch_edge_precision_m/ num_instances,
            #               kl_anneal_w))

        if epoch % 1 == 0:
            dev_loss  = dev(dev_loader, model, epoch, kl_anneal_w)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), model_name+"_"+str(epoch))
        #    logging.info("Model saved to: {}".format(model_name+"_"+str(epoch)))
    writer.flush(); writer.close()
    torch.save(model.state_dict(), model_name)
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

    batch_size = 128; epochs = 521; phase = 1; nmax = 10

    # Load text vocab...
    print("loading text_vocab from checkpoint...")
    model_path =  "/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/src/ptm_mt_en2deB_sem_enM.pt"
    checkpoint = torch.load(model_path, map_location=lambda storage, loc:storage)
    text_vocab = dict(checkpoint['vocab'])['src']
    text_vocab.pad = 1
    # Load graphs data...
    trn_data_list, dev_data_list, tst_data_list, graphs_trn, graphs_dev, graphs_tst  = loaddata(text_vocab, nmax)
    trn_loader = DataLoader(trn_data_list, batch_size=batch_size)     #36519
    dev_loader = DataLoader(dev_data_list, batch_size=batch_size)     #1368
    tst_loader = DataLoader(tst_data_list, batch_size=batch_size)     #1371
    print("batchsize: {}".format(batch_size))
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Build model...
    input_dim = 10; latent_dim = 32; graph_mlp_hid_dim = 512; 

    """
    # GraphRNN
    graphs = graphs_trn
    max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])
    max_prev_node = 40
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_trn)))
    print('max number node: {}'.format(max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(max_prev_node))
    dataset = Graph_sequence_sampler_pytorch(graphs_trn, max_prev_node=max_prev_node, max_num_node=max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))], num_samples= 32*32, replacement=True)
    dataset_train = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, sampler=sample_strategy)
    embedding_size_rnn =  64
    hidden_size_rnn = 128
    embedding_size_rnn_output = 8
    hidden_size_rnn_output = 16 # hidden size for output rnn
    num_layers = 4 
    rnn = GRU_plain(input_size=max_prev_node, embedding_size=embedding_size_rnn,
                        hidden_size=hidden_size_rnn, num_layers=num_layers, has_input=True,
                        has_output=True, output_size=hidden_size_rnn_output).cuda()
    output = GRU_plain(input_size=1, embedding_size=embedding_size_rnn_output,
                           hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()
    epoch = 1
    epochs = 3000
    epochs_test = 10
    epochs_test_start = 10
    lr = 0.003
    lr_rate = 0.3
    milestones = [400, 1000]
    test_total_size = 1000
    test_batch_size = 32
    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=lr)
    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=milestones, gamma=lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=milestones, gamma=lr_rate)
    time_all = np.zeros(epochs)
    while epoch<=epochs:
        epoch_loss = train_rnn_epoch(epoch, num_layers, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        print('epoch {}, epoch_loss: {}'.format(epoch, epoch_loss))
        # test
        if epoch % epochs_test == 0 and epoch>=epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<test_total_size:
                    G_pred_step = test_rnn_epoch(epoch, max_num_node, max_prev_node, rnn, output, test_batch_size=test_batch_size)
                    G_pred.extend(G_pred_step)
                    break
                fname = 'saved_test_' + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
            print('test done')
        epoch += 1
        
    """

    if phase == 1:
        # Phase 1...
        model  = GTVAE(input_dim, latent_dim, graph_mlp_hid_dim, nmax, text_vocab)
        #model = load_phase_2_model("model/gtg_20210309-183511_200", text_transformer)             

        #for param in model.text_vae.parameters():
        #    param.requires_grad = False
        
        model.to(device)
        print("model: {}".format(model))
        train(trn_loader, dev_loader, model, epochs, "gtg_"+timestr)



