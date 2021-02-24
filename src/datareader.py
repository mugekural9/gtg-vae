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
test_logger = setup_logger('test_logger', timestr+'_test.log')
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

# Load nearest bpe tokens for node labels and src words...
bpe_data_sents = json.loads(open("sent_bpes_trn.json", "r").read())
bpe_data_nodes = json.loads(open("node_bpes_trn.json", "r").read())

# bpe_data_sent = dict()
# bpe_data_node = dict()


def loaddata(params, src_dict, nodeembeddings, maxnode):

    # Load data...
    data_params = params['data']
    dataset = dataset_from_params(data_params)
    
    train_data = dataset['train'];
    dev_data = dataset.get('dev');
    test_data = dataset.get('test')
    
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

    trn_data_list  = []; dev_data_list  = []; test_data_list = []

    def prep_data_list(tqdm):
        data_list = []
        for batch in tqdm:
            # batch contains bunch of AMRs
            for amr in batch.instances:
                src_tokens = amr.tokens
                graph = amr.graph
                nodes = dict()
                edges = []

                node_ids = []; node_ids_local = []; snttoken_ids = [] # with these ids we'll able to forward the embeddings
                i = 0
                for (s,r,t) in graph._triples:
                    if r=='instance' and len(nodes) <maxnode:
                        nodes[s] = (i,t) # t will be the nodelabel.
                        i+=1

                fileprefix = amr.id.split(" ::")[0]

                # # Writing bpes...
                # if fileprefix not in bpe_data_sents or fileprefix not in bpe_data_nodes:                
                #     print("File not  exist {}".format(fileprefix))
                #     f2 = open("newbees/"+fileprefix+'sent.tok', 'w')
                #     f2.write(' '.join([str(elem) for elem in src_tokens]))
                #     f2.close()
                #     os.system('python3 /kuacc/users/mugekural/workfolder/dev/subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes < '+"newbees/"+fileprefix+'sent.tok > '+"newbees/"+fileprefix+'sent.tok.bpe')

                #     f3 = open("newbees/"+fileprefix+'sent.tok.bpe', "r")
                #     bpe_sent = f3.readline()
                #     bpe_tokens = bpe_sent.split(" ")
                #     # print("bpe_tokens:", bpe_tokens)
                #     f3.close()

                #     # Node bpe...
                #     f4 = open("newbees/"+fileprefix+'node.tok', 'w')
                #     for node, values in nodes.items():
                #         nodelabel = values[1]
                #         f4.write(nodelabel+"\n")
                #     f4.close()

                #     os.system('python3 /kuacc/users/mugekural/workfolder/dev/subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes <'+"newbees/"+fileprefix+'node.tok > '+"newbees/"+fileprefix+'node.tok.bpe')
                #     f5 = open("newbees/"+fileprefix+"node.tok.bpe", "r")
                #     bpe_node = f5.read()
                #     node_bpe_tokens = bpe_node.split("\n")
                #     # #print("node_bpe_tokens:", node_bpe_tokens)
                #     f5.close()

                #     bpe_data_sent[fileprefix] = bpe_tokens
                #     bpe_data_node[fileprefix] = node_bpe_tokens
                # continue


                bpe_tokens =  bpe_data_sents[fileprefix]
                node_bpe_tokens = bpe_data_nodes[fileprefix] 

                for token in bpe_tokens:
                    snttokenid = src_dict[token]
                    snttoken_ids.append(snttokenid)

                # Add BOS (<s>) and EOS (</s>) to the sentence...
                snttoken_ids = [2] + snttoken_ids 
                snttoken_ids.append(4)

                for edge in graph.edges():
                    if edge.source in nodes and edge.target in nodes:
                        s_id = nodes[edge.source][0]
                        t_id = nodes[edge.target][0]
                        edges.append([s_id, t_id])
                        edges.append([t_id, s_id])
                        ##TODO:For now assuming nondirected AMR graphs; both edge and reverse included

                for nodetoken in node_bpe_tokens[:-1]: #last one is trivial
                    nodelabel = nodetoken.split(" ")[0]
                    nodevocabid = src_dict[nodelabel]
                    node_ids.append(nodevocabid)

                orig_node_ids = list(node_ids)
                # Pad node ids to have max number of nodes
                missing_counts = maxnode -len(node_ids)
                if missing_counts > 0:
                    node_ids.extend([1] * missing_counts)
                elif missing_counts < 0:
                    node_ids = node_ids[:missing_counts]


                nodeids = torch.LongTensor(node_ids).to(device).transpose(-1,0)
                nodeids = torch.unsqueeze(nodeids, 1)
                node_features = nodeembeddings(nodeids)
                node_features = torch.squeeze(node_features, 1)

                #snttoken_ids = torch.tensor(snttoken_ids, dtype=torch.long).unsqueeze(1)
                x = torch.tensor(node_features, dtype=torch.float) # nodecount,hdim

                edge_index = torch.tensor(edges, dtype=torch.long)
                data = Data(x=x, edge_index=edge_index.t().contiguous())
                data.__setitem__("snttoken_ids", snttoken_ids)
                data.__setitem__("nodetoken_ids", nodeids)
                data.__setitem__("orig_node_ids", orig_node_ids)
                data_list.append(data)
            ##end of one AMR
        #end of batch
           
        # bpe_sent_json = json.dumps(bpe_data_sent)
        # f = open("sent_bpes_test.json","w")
        # f.write(bpe_sent_json)
        # f.close()

        # bpe_node_json = json.dumps(bpe_data_node)
        # f = open("node_bpes_test.json","w")
        # f.write(bpe_node_json)
        # f.close()

        
    #trn_data_list = prep_data_list(train_generator_tqdm)
    #dev_data_list = prep_data_list(dev_generator_tqdm)
    test_data_list = prep_data_list(test_generator_tqdm)
  
    return trn_data_list, dev_data_list, test_data_list

def train(train_loader, test_loader, model, epochs, src_dict, model_name):

    logging.info("trnsize: {}, testsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(test_loader.dataset), epochs)) 

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
        losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "txt_kl_loss"], 0)
        metrics_batch_tot = dict.fromkeys(["roc_auc_score", "avg_precision_score"], 0)
        
        for step, data in enumerate(train_loader):
            opt.zero_grad()
            
            x, edge_index, batch, dec_seq, nodetoken_ids = data.x.to(device), data.edge_index.to(device), data.batch.to(device),batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids")
            nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode
            batchsize, maxnode = nodetoken_ids.size()    
            x = x / x.sum(1, keepdim=True).clamp(min=1) # Normalize input node features

            # Encode graph and text
            losses, metrics  = model(x, edge_index, batch, dec_seq, nodetoken_ids, src_dict)
            losses["loss"].backward()
            opt.step()
            
            # Track other losses and metrics...
            for k,v in losses.items():
                losses_batch_tot[k] += v.item() * data.num_graphs 
            for k,v in metrics.items():
                metrics_batch_tot[k] += v.item() * data.num_graphs 
            
                
        train_logger.info("---\nTrain:")    
        train_logger.info("Epoch %d", epoch)
        for l,v in losses_batch_tot.items():
            train_logger.info("{} : {}".format(l, v/len(train_loader.dataset)))
            writer.add_scalar("Loss/train/"+l, v/len(train_loader.dataset), epoch)
          
        for m,v in metrics_batch_tot.items():
            train_logger.info("{} : {}".format(m, v/len(train_loader.dataset)))
            writer.add_scalar("Metric/train/"+m, v/len(train_loader.dataset), epoch)
          
        if epoch % 1 == 0:
             test_loss = test(test_loader, model, src_dict, epoch)

    writer.flush()
    writer.close()
    
    torch.save(model.state_dict(), model_name)
    logging.info("Model saved to: {}".format(model_name))
    
def test(loader, model, src_dict, epoch):
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
        
    losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "txt_kl_loss"], 0)
    metrics_batch_tot = dict.fromkeys(["roc_auc_score", "avg_precision_score"], 0)

    model.eval()
    for (step, data) in enumerate(loader):
        with torch.no_grad():
            
            x, edge_index, batch, dec_seq, nodetoken_ids = data.x.to(device), data.edge_index.to(device), data.batch.to(device), batched_snttokens_ids_padded[step].to(device),data.__getitem__("nodetoken_ids")
            nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode
            maxnode = nodetoken_ids.size(1)
            x = x / x.sum(1, keepdim=True).clamp(min=1) # Normalize input node features

            # Encode graph and text
            losses, metrics  = model(x, edge_index, batch, dec_seq, nodetoken_ids, src_dict)
                                   
            # Track other losses and metrics...
            for k,v in losses.items():
                losses_batch_tot[k] += v.item() * data.num_graphs 

            for k,v in metrics.items():
                metrics_batch_tot[k] += v.item() * data.num_graphs 
            
    test_logger.info("---\nTest:")    
    test_logger.info("Epoch %d", epoch)
    for l,v in losses_batch_tot.items():
        test_logger.info("{} : {}".format(l, v/len(loader.dataset)))
        writer.add_scalar("Loss/test/"+l, v/len(loader.dataset), epoch)

    for m,v in metrics_batch_tot.items():
        test_logger.info("{} : {}".format(m, v/len(loader.dataset)))
        writer.add_scalar("Metric/test/"+m, v/len(loader.dataset), epoch)




def train_phase_2(train_loader, test_loader, model, epochs, src_dict):

    logging.info("Training Phase 2...trnsize: {}, testsize: {}, num_epochs: {}".format(len(train_loader.dataset), len(test_loader.dataset), epochs)) 

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
        
    # opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # opt = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        # model.train()
        # losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "txt_kl_loss"], 0)
        # metrics_batch_tot = dict.fromkeys(["roc_auc_score", "avg_precision_score"], 0)
        
        # for step, data in enumerate(train_loader):
        #     opt.zero_grad()
            
        #     edge_index, batch, dec_seq, nodetoken_ids = data.edge_index.to(device), data.batch.to(device),batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids")
        #     nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode
        #     batchsize, maxnode = nodetoken_ids.size()    
            
        #     # Encode graph and text
        #     losses, metrics  = model(None, edge_index, batch, dec_seq, nodetoken_ids, src_dict)
        #     losses["loss"].backward()
        #     opt.step()
            
        #     # Track other losses and metrics...
        #     for k,v in losses.items():
        #         losses_batch_tot[k] += v.item() * data.num_graphs 
        #     for k,v in metrics.items():
        #         metrics_batch_tot[k] += v.item() * data.num_graphs 
            
                
        # train_logger.info("---\nTrain_Phase2:")    
        # train_logger.info("Epoch %d", epoch)
        # for l,v in losses_batch_tot.items():
        #     train_logger.info("{} : {}".format(l, v/len(train_loader.dataset)))
        #     writer.add_scalar("Loss/train/"+l, v/len(train_loader.dataset), epoch)
          
        # for m,v in metrics_batch_tot.items():
        #     train_logger.info("{} : {}".format(m, v/len(train_loader.dataset)))
        #     writer.add_scalar("Metric/train/"+m, v/len(train_loader.dataset), epoch)
          
        if epoch % 1 == 0:
            test_loss = test_phase_2(test_loader, model, src_dict, epoch)

    writer.flush()
    writer.close()
    
    #torch.save(model.state_dict(), "gtg")


    
def test_phase_2(loader, model, src_dict, epoch):
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
        
    losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "txt_kl_loss"], 0)
    metrics_batch_tot = dict.fromkeys(["roc_auc_score", "avg_precision_score"], 0)

    model.eval()
    for (step, data) in enumerate(loader):
        with torch.no_grad():
            
            edge_index, batch, dec_seq, nodetoken_ids = data.edge_index.to(device), data.batch.to(device), batched_snttokens_ids_padded[step].to(device),data.__getitem__("nodetoken_ids")
            nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode
            maxnode = nodetoken_ids.size(1)
            
            # Encode graph and text
            losses, metrics  = model(None, edge_index, batch, dec_seq, nodetoken_ids, src_dict)
                                   
            # Track other losses and metrics...
            for k,v in losses.items():
                losses_batch_tot[k] += v.item() * data.num_graphs 

            for k,v in metrics.items():
                metrics_batch_tot[k] += v.item() * data.num_graphs 
            
    test_logger.info("---\nTestDAE:")    
    test_logger.info("Epoch %d", epoch)
    for l,v in losses_batch_tot.items():
        test_logger.info("{} : {}".format(l, v/len(loader.dataset)))
        writer.add_scalar("Loss/test/"+l, v/len(loader.dataset), epoch)

    for m,v in metrics_batch_tot.items():
        test_logger.info("{} : {}".format(m, v/len(loader.dataset)))
        writer.add_scalar("Metric/test/"+m, v/len(loader.dataset), epoch)
    

def load_phase_2_model(path):

    pretrained = GTG(input_dim, output_dim, nmax, text_transformer, 1).to(device)    
    pretrained.load_state_dict(torch.load(path))
    pretrained_dict = pretrained.state_dict()

    model = GTG(input_dim, output_dim, nmax, text_transformer, 2).to(device) 
    model_dict = model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # 1. filter out unnecessary keys
    model_dict.update(pretrained_dict)      # 2. overwrite entries in the existing state dic
    model.load_state_dict(pretrained_dict)  # 3. load the new state dict
    return model
    

def dataset_info(loader):
    node_count = 0
    edge_count = 0

    exceeding_maxnode_graph_count = 0
    for data in loader.dataset:
        node_count += len(data.orig_node_ids)
        if len(data.orig_node_ids) > 15:
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
    input_dim = 512; output_dim = 32; #nodeclass_num = 38926
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
    nmax = 15
    batch_size = 64
    trn_data_list, dev_data_list, test_data_list  = loaddata(params, src_dict, nodeembeddings, nmax)
    breakpoint()
    train_loader = DataLoader(data_list, batch_size=batch_size)   #20000
    
    # dev_loader = DataLoader(data_list[1000:3500], batch_size=batch_size)   #2500
    # test_loader = DataLoader(data_list[3500:], batch_size=batch_size)    #2318
    
    dataset_info(train_loader)
    # dataset_info(dev_loader)
    # dataset_info(test_loader)
    
    # epochs = 100
    # logging.info("batchsize: {}".format(batch_size))
    # logging.info("epochs: {}".format(epochs))
    # logging.info("nmax: {}".format(nmax))
    
  
    
    # Phase 1...
    # model = GTG(input_dim, output_dim, nmax, text_transformer, 1).to(device) #text_transformer
    # logging.info("model: {}".format(model))
    # train(train_loader, dev_loader, model, epochs, src_dict, "gtg_10")

    # Phase 2...
    # model = load_phase_2_model("gtg_10")
    # train_phase_2(train_loader, test_loader, model, epochs, src_dict)

 
