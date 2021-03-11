# import re
import os
import argparse
import yaml
import torch
from utils.tqdm import Tqdm
from utils.params import Params, remove_pretrained_embedding_params
from data.dataset_builder import dataset_from_params, iterator_from_params
from data.vocabulary import Vocabulary

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from torch_geometric.data import Data, DataLoader
from gtg import GTG
import json
from translator import build_translator
import torch.optim as optim
import logging
import pdb
import numpy as np
import networkx as nx

from grr_graphvae import GraphVAE
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter()


# Logging...
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

# general logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)    
logging.basicConfig(filename=timestr+'_app.log', format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

# Config for text transformer...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
model_opt = dict()
model_opt["d_model"] = 512
model_opt["share_embeddings"] = True
model_opt["dropout"] = 0.1
model_opt["model_path"] = "/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/src/ptm_mt_en2deB_sem_enM.pt"
model_opt["word_emb_size"] = 512
model_opt["position_encoding"] = True
model_opt["heads"] = 8
model_opt["dec_layers"] = 6
model_opt["enc_layers"] = 6
model_opt["transformer_ff"] = 2048
model_opt["param_init_glorot"] = True

def loaddata(params, src_dict, nodeembeddings, maxnode):
    def prep_data_list(tqdm, bpe_data_sents, bpe_data_nodes):
        data_list = []
        for batch in tqdm:
            # batch contains bunch of AMRs
            for amr in batch.instances:
                src_tokens = amr.tokens
                graph = amr.graph
                G = graph._G
             
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
                node_features = nodeembeddings(nodeids) # her blank'a aynı şeyi vermiyor
                node_features = torch.squeeze(node_features, 1)

                #snttoken_ids = torch.tensor(snttoken_ids, dtype=torch.long).unsqueeze(1)
                x = torch.tensor(node_features, dtype=torch.float) # nodecount,hdim
                edge_index = torch.tensor(edges, dtype=torch.long)
                neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)

                
                features_all = np.identity(maxnode)
                data = Data(x=x, edge_index=edge_index.t().contiguous())
                data.__setitem__("snttoken_ids", snttoken_ids)
                data.__setitem__("nodetoken_ids", nodeids)
                data.__setitem__("orig_node_ids", orig_node_ids)
                data.__setitem__("gold_edges", gold_edges)
                data.__setitem__("neg_edge_index", neg_edge_index.t().contiguous())
                data.__setitem__("adj_padded", adj_padded)
                data.__setitem__("features_all", features_all)
                data_list.append(data)
            ##end of one AMR
        #end of batch
        return data_list
        
    # Load data...
    data_params = params['data']
    dataset = dataset_from_params(data_params)
    train_data = dataset['train']; dev_data = dataset['dev']; test_data = dataset['test']
    train_iterator, dev_iterator, test_iterator = iterator_from_params(None, data_params['iterator'])
    num_train_batches = train_iterator.get_num_batches(train_data)
    num_dev_batches   = dev_iterator.get_num_batches(dev_data)
    num_test_batches  = test_iterator.get_num_batches(test_data)
    train_generator = train_iterator(instances=train_data, shuffle=True, num_epochs=1)
    dev_generator   = dev_iterator(instances=dev_data, shuffle=True, num_epochs=1)
    test_generator  = test_iterator(instances=test_data, shuffle=True, num_epochs=1)
    train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_train_batches)
    dev_generator_tqdm   = Tqdm.tqdm(dev_generator, total=num_dev_batches)
    test_generator_tqdm  = Tqdm.tqdm(test_generator, total=num_test_batches)
    # Load nearest bpe tokens for node labels and src words...
    bpe_data_sents_trn = json.loads(open("../data/sent_bpes_trn.json", "r").read())
    bpe_data_sents_dev = json.loads(open("../data/sent_bpes_dev.json", "r").read())
    bpe_data_sents_tst = json.loads(open("../data/sent_bpes_tst.json", "r").read())
    bpe_data_nodes_trn = json.loads(open("../data/node_bpes_trn.json", "r").read())
    bpe_data_nodes_dev = json.loads(open("../data/node_bpes_dev.json", "r").read())
    bpe_data_nodes_tst = json.loads(open("../data/node_bpes_tst.json", "r").read())
    trn_data_list = prep_data_list(train_generator_tqdm, bpe_data_sents_trn, bpe_data_nodes_trn)
    dev_data_list = prep_data_list(dev_generator_tqdm, bpe_data_sents_dev, bpe_data_nodes_dev)
    tst_data_list = prep_data_list(test_generator_tqdm, bpe_data_sents_tst, bpe_data_nodes_tst)

    return trn_data_list, dev_data_list, tst_data_list

def train(train_loader, dev_loader, model, epochs, src_dict, model_name):
    logging.info("trnsize: {}, devsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(dev_loader.dataset), epochs)) 
    print("trnsize: {}, devsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(dev_loader.dataset), epochs)) 
    #breakpoint()
    # Pad snttoken ids in batch...(looking for a better way)
    batched_snttokens_ids_padded = []
    for data in train_loader:
        dec_seq = data.__getitem__("snttoken_ids")
        max_seq_len = max([len(i) for i in dec_seq])
        torch_dec_seq=[]
        for seq in dec_seq:
            torch_dec_seq.append(torch.tensor(seq, dtype=torch.long))
        dec_seq = torch.stack([torch.cat([i, i.new_ones(max_seq_len - i.size(0))], 0) for i in torch_dec_seq],1)
        batched_snttokens_ids_padded.append(dec_seq)
        
    #opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    opt = optim.Adam(model.parameters(), lr=0.001)

    # Log model and opt...
    logging.info("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        logging.info(param_tensor+ "\t"+ str(model.state_dict()[param_tensor].size()))
    logging.info("\nOptimizer: {}".format(opt))
    model.to(device)
    for epoch in range(epochs):
        model.train()
        losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "nodelabel_loss", "txt_kl_loss"], 0)
        metrics_batch_tot = dict.fromkeys(["roc_auc_score", "avg_precision_score", "nodelabel_acc", "edge_recall", "edge_precision"], 0)
        epoch_loss = 0
        epoch_edge_recall = 0
        epoch_edge_precision = 0
        for step, data in enumerate(train_loader):
            opt.zero_grad()
            gold_edges = data.__getitem__("gold_edges")
            adj_input = torch.tensor(data.adj_padded).to(device).float() #B,maxnode, maxnode
            
            if refcode:
                #features = torch.tensor(data.__getitem__("features_all")).to(device).float() #.x.to(device) #B*maxnode, hdim

                x, edge_index, batch, dec_seq, nodetoken_ids, neg_edge_index, gold_edges = data.x.to(device), data.edge_index.to(device), data.batch.to(device),batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids"), data.__getitem__("neg_edge_index").to(device), data.__getitem__("gold_edges")
                report = False
                if epoch % 10 == 0:
                    report = True
                
                loss, edge_recall, edge_precision = model(x, edge_index, batch, adj_input, gold_edges, report)  
                #print('Epoch: ', epoch, ', Iter: ', step, ', Loss: ', loss)
                loss.backward()
                epoch_loss += loss.item()
                epoch_edge_recall += edge_recall
                epoch_edge_precision += edge_precision
                #scheduler.step()
                opt.step()            
            else:
                x, edge_index, batch, dec_seq, nodetoken_ids, neg_edge_index, gold_edges = data.x.to(device), data.edge_index.to(device), data.batch.to(device),batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids"), data.__getitem__("neg_edge_index").to(device), data.__getitem__("gold_edges")
                nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode
                batchsize, maxnode = nodetoken_ids.size()    
                x = x / x.sum(1, keepdim=True).clamp(min=1) # Normalize input node features
            
                losses, metrics  = model(x, edge_index, batch, dec_seq, nodetoken_ids, src_dict, gold_edges, neg_edge_index, adj_input)
                losses["loss"].backward()
                opt.step()            
                # Track other losses and metrics...
                for k,v in losses.items():
                    losses_batch_tot[k] += v.item() * data.num_graphs 
                for k,v in metrics.items():
                    metrics_batch_tot[k] += v.item() * data.num_graphs 

        if refcode:
            print('Epoch: ', epoch, ', Loss: ', epoch_loss/ len(train_loader), ', Recall:', epoch_edge_recall/ len(train_loader.dataset), ', Precision:', epoch_edge_precision/len(train_loader.dataset))            
            train_logger.info("{} : {}".format(epoch, epoch_loss/ len(train_loader)))
            train_logger.info("{} : {}".format(epoch, epoch_edge_recall/ len(train_loader.dataset)))
            train_logger.info("{} : {}".format(epoch, epoch_edge_precision/ len(train_loader.dataset)))

        else:
            train_logger.info("---\nTrain:")    
            train_logger.info("Epoch %d", epoch)
            for l,v in losses_batch_tot.items():
                train_logger.info("{} : {}".format(l, v/len(train_loader.dataset)))
                writer.add_scalar("Loss/train/"+l, v/len(train_loader.dataset), epoch)
          
            for m,v in metrics_batch_tot.items():
                train_logger.info("{} : {}".format(m, v/len(train_loader.dataset)))
                writer.add_scalar("Metric/train/"+m, v/len(train_loader.dataset), epoch)
          
        if epoch % 5 == 0:
            dev_loss  = dev(dev_loader, model, src_dict, epoch)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), model_name+"_"+str(epoch))
            logging.info("Model saved to: {}".format(model_name+"_"+str(epoch)))

    writer.flush()
    writer.close()
    
    torch.save(model.state_dict(), model_name)
    logging.info("Model saved to: {}".format(model_name))
    
def dev(loader, model, src_dict, epoch):
    # Pad snttoken ids in batch...(looking for a better way)
    batched_snttokens_ids_padded = []
    for data in loader:
        dec_seq = data.__getitem__("snttoken_ids")
        max_seq_len = max([len(i) for i in dec_seq])
        torch_dec_seq=[]
        for seq in dec_seq:
            torch_dec_seq.append(torch.tensor(seq, dtype=torch.long))
        dec_seq = torch.stack([torch.cat([i, i.new_ones(max_seq_len - i.size(0))], 0) for i in torch_dec_seq],1)
        batched_snttokens_ids_padded.append(dec_seq)
        
    losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "nodelabel_loss", "txt_kl_loss"], 0)
    metrics_batch_tot = dict.fromkeys(["roc_auc_score", "avg_precision_score", "nodelabel_acc", "edge_recall", "edge_precision"], 0)

    epoch_loss = 0
    epoch_edge_recall = 0
    epoch_edge_precision = 0
    model.eval()
    for (step, data) in enumerate(loader):
        with torch.no_grad():
            
            gold_edges = data.__getitem__("gold_edges")
            adj_input = torch.tensor(data.adj_padded).to(device).float() #B,maxnode, maxnode
            #breakpoint()
            if refcode:
                x, edge_index, batch, dec_seq, nodetoken_ids, neg_edge_index, gold_edges = data.x.to(device), data.edge_index.to(device), data.batch.to(device),batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids"), data.__getitem__("neg_edge_index").to(device), data.__getitem__("gold_edges")

                loss, edge_recall, edge_precision = model(x, edge_index, batch, adj_input, gold_edges, True) 
                #breakpoint()
                epoch_loss += loss.item()
                epoch_edge_recall += edge_recall
                epoch_edge_precision += edge_precision
            else:
                x, edge_index, batch, dec_seq, nodetoken_ids, neg_edge_index, gold_edges = data.x.to(device), data.edge_index.to(device), data.batch.to(device),batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids"), data.__getitem__("neg_edge_index").to(device), data.__getitem__("gold_edges")
                nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode
                batchsize, maxnode = nodetoken_ids.size()    
                x = x / x.sum(1, keepdim=True).clamp(min=1) # Normalize input node features
            
                losses, metrics  = model(x, edge_index, batch, dec_seq, nodetoken_ids, src_dict, gold_edges, neg_edge_index, adj_input)
                # Track other losses and metrics...
                for k,v in losses.items():
                    losses_batch_tot[k] += v.item() * data.num_graphs 
                for k,v in metrics.items():
                    metrics_batch_tot[k] += v.item() * data.num_graphs 


    if refcode:
        print('DEV Epoch: ', epoch, ', Loss: ', epoch_loss/ len(loader), ', Recall:', epoch_edge_recall/ len(loader.dataset), ', Precision:', epoch_edge_precision/len(loader.dataset))            
        dev_logger.info("{} : {}".format(epoch, epoch_loss/ len(loader)))
        dev_logger.info("{} : {}".format(epoch, epoch_edge_recall/ len(loader.dataset)))
        dev_logger.info("{} : {}".format(epoch, epoch_edge_precision/ len(loader.dataset)))


    else:
        dev_logger.info("---\nDev:")    
        dev_logger.info("Epoch %d", epoch)
        for l,v in losses_batch_tot.items():
            dev_logger.info("{} : {}".format(l, v/len(loader.dataset)))
            writer.add_scalar("Loss/train/"+l, v/len(loader.dataset), epoch)
          
        for m,v in metrics_batch_tot.items():
            dev_logger.info("{} : {}".format(m, v/len(loader.dataset)))
            writer.add_scalar("Metric/train/"+m, v/len(loader.dataset), epoch)
          
    

def train_phase_2(train_loader, dev_loader, model, epochs, src_dict, model_name):
    logging.info("trnsize: {}, devsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(dev_loader.dataset), epochs)) 

    # Pad snttoken ids in batch...(looking for a better way)
    batched_snttokens_ids_padded = []
    for data in train_loader:
        dec_seq = data.__getitem__("snttoken_ids")
        max_seq_len = max([len(i) for i in dec_seq])
        torch_dec_seq=[]
        for seq in dec_seq:
            torch_dec_seq.append(torch.tensor(seq, dtype=torch.long))
        dec_seq = torch.stack([torch.cat([i, i.new_ones(max_seq_len - i.size(0))], 0) for i in torch_dec_seq],1)
        batched_snttokens_ids_padded.append(dec_seq)
        
    #opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    opt = optim.Adam(model.parameters(), lr=0.001)

    # Log model and opt...
    logging.info("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        logging.info(param_tensor+ "\t"+ str(model.state_dict()[param_tensor].size()))
    logging.info("\nOptimizer: {}".format(opt))
    
    
    for epoch in range(epochs):
        model.train()
        losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "nodelabel_loss", "txt_kl_loss"], 0)
        metrics_batch_tot = dict.fromkeys(["roc_auc_score", "avg_precision_score", "nodelabel_acc", "edge_recall", "edge_precision"], 0)
        
        for step, data in enumerate(train_loader):
            opt.zero_grad()
            
            x, edge_index, batch, dec_seq, nodetoken_ids, neg_edge_index, gold_edges = data.x.to(device), data.edge_index.to(device), data.batch.to(device),batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids"), data.__getitem__("neg_edge_index").to(device), data.__getitem__("gold_edges")
            nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode
            batchsize, maxnode = nodetoken_ids.size()    
            x = x / x.sum(1, keepdim=True).clamp(min=1) # Normalize input node features
            breakpoint()
            losses, metrics  = model(x, edge_index, batch, dec_seq, nodetoken_ids, src_dict, gold_edges, neg_edge_index)
            losses["txt_kl_loss"].backward()
            opt.step()
            
            # Track other losses and metrics...
            for k,v in losses.items():
                losses_batch_tot[k] += v.item() * data.num_graphs 
            for k,v in metrics.items():
                metrics_batch_tot[k] += v.item() * data.num_graphs 
            
                
        train_logger.info("---\nTrain Phase 2:")    
        train_logger.info("Epoch %d", epoch)
        for l,v in losses_batch_tot.items():
            train_logger.info("{} : {}".format(l, v/len(train_loader.dataset)))
            writer.add_scalar("Loss/train/"+l, v/len(train_loader.dataset), epoch)
          
        for m,v in metrics_batch_tot.items():
            train_logger.info("{} : {}".format(m, v/len(train_loader.dataset)))
            writer.add_scalar("Metric/train/"+m, v/len(train_loader.dataset), epoch)
          
        if epoch % 5 == 0:
            dev_loss  = dev(dev_loader, model, src_dict, epoch)
        
    writer.flush()
    writer.close()
    
    torch.save(model.state_dict(), model_name)
    logging.info("Model saved to: {}".format(model_name))
        
    

def load_phase_2_model(path, text_transformer):

    model = GraphVAE(input_dim, 64, 256, nmax)
    model.load_state_dict(torch.load(path))
    #pretrained = GTG(input_dim, output_dim, nmax, None, nodelabel_num, 1).to(device)    
    #pretrained.load_state_dict(torch.load(path))
    #pretrained_dict = pretrained.state_dict()
    #model = GTG(input_dim, output_dim, nmax, text_transformer, nodelabel_num, 2).to(device) 
    #model_dict = model.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # 1. filter out unnecessary keys
    #model_dict.update(pretrained_dict)      # 2. overwrite entries in the existing state dic
    #model.load_state_dict(model_dict)  # 3. load the new state dict

    return model
    

def dataset_info(loader):
    node_count = 0
    edge_count = 0

    exceeding_maxnode_graph_count = 0
    for data in loader.dataset:
        node_count += len(data.orig_node_ids)
        if len(data.orig_node_ids) > nmax:
            exceeding_maxnode_graph_count += 1
        if len(data.edge_index) != 0:
            edge_count += data.edge_index.size(1)/2

    logging.info("\nDataset size: {}".format(len(loader.dataset)))
    logging.info("Total node count: {}".format(node_count))
    logging.info("Total edge count: {}".format(edge_count))
    logging.info("Total graph count exceeds nmax : {}".format(exceeding_maxnode_graph_count))
    logging.info("Avg node count: {}".format(node_count/len(loader.dataset)))
    logging.info("Avg edge count: {}".format(edge_count/len(loader.dataset)))
    logging.info("\n")
    
if __name__ == "__main__":
    print("Oh hi")

    # Build model...
    input_dim = 512; output_dim = 16; 
    model_opt["d_graphz"] = output_dim

    translator = build_translator(model_opt)
    text_transformer = translator.model
    fields = translator.fields
    src_dict = fields["src"].vocab
    nodeembeddings = text_transformer.encoder.embeddings

    parser = argparse.ArgumentParser('datareader.py')
    parser.add_argument('params', help='Parameters YAML file.')
    args = parser.parse_args()
    params = Params.from_file(args.params)
    nmax = 10
    nodelabel_num = 38926
    refcode = True

    if refcode:
        batch_size = 64
    else:
        batch_size = 64
    trn_data_list, dev_data_list, tst_data_list  = loaddata(params, src_dict, nodeembeddings, nmax)

    trn_loader = DataLoader(trn_data_list[:15000], batch_size=batch_size)     #36519
    dev_loader = DataLoader(trn_data_list[15000:], batch_size=batch_size)  #dev_data_list, batch_size=batch_size)     #1368
    tst_loader = DataLoader(tst_data_list, batch_size=batch_size)     #1371
    
    dataset_info(trn_loader)
    dataset_info(dev_loader)
    dataset_info(tst_loader)

    epochs = 521
    logging.info("batchsize: {}".format(batch_size))
    logging.info("epochs: {}".format(epochs))
    logging.info("nmax: {}".format(nmax))

    phase = 1
   
    
    if phase == 1:
        # Phase 1...
        if refcode:
            #model  = GraphVAE(input_dim, 64, 256, nmax)
            model = load_phase_2_model("gtg_20210309-183511_200", None) #GraphVAE(input_dim, 64, 256, nmax)
        else:
            model = GTG(512, output_dim,  nmax, nodelabel_num, 1).to(device) #text_transformer
        logging.info("model: {}".format(model))
        print("model:{}".format(model))

        #for param in model.parameters():
        #    print(param)

        
        # rer("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #train(trn_loader, dev_loader, model, epochs, src_dict, "gtg_"+timestr)
        model.to(device)
        dev(dev_loader, model, src_dict, 1)

        
    elif phase == 2:
        # Phase 2...
        model = load_phase_2_model("tmp/gtg_20210305-221240", text_transformer)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = True
      
        breakpoint()
        #dev_loss  = dev(dev_loader, model, src_dict, 10)
        train_phase_2(trn_loader, dev_loader, model, 50, src_dict, "gtg_"+timestr)
    
