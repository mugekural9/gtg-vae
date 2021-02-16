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

train_logger = setup_logger('train_logger', 'train.log')
test_logger = setup_logger('test_logger', 'test.log')
# general logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)    
logging.basicConfig(filename='app.log', format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

# Config for text transformer...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
model_opt = dict()
model_opt["d_model"] = 512
model_opt["share_embeddings"] = True
model_opt["dropout"] = 0.1
model_opt["model_path"] = "/Users/mugekural/dev/git/gtg-vae/src/ptm_mt_en2deB_sem_enM.pt"
model_opt["word_emb_size"] = 512
model_opt["position_encoding"] = True
model_opt["heads"] = 8
model_opt["dec_layers"] = 6
model_opt["enc_layers"] = 6
model_opt["transformer_ff"] = 2048

# Load nearest bpe tokens for node labels and src words...
bpe_data_sents = json.loads(open("sent_bpes.json", "r").read())
bpe_data_nodes = json.loads(open("node_bpes.json", "r").read())


def loaddata(params, src_dict, nodeembeddings, maxnode):

    # Load data...
    data_params = params['data']
    dataset = dataset_from_params(data_params)
    train_data = dataset['train']; dev_data = dataset.get('dev'); test_data = dataset.get('test')
    train_iterator, dev_iterator, test_iterator = iterator_from_params(None, data_params['iterator'])
    train_generator = train_iterator(instances=train_data, shuffle=True, num_epochs=1)
    num_training_batches = train_iterator.get_num_batches(train_data)
    train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)
    data_list = []; meta_list = [];

    
    for batch in train_generator_tqdm:
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

            if fileprefix not in bpe_data_sents or fileprefix not in bpe_data_nodes:
                continue
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
            data_list.append(data)
            meta_list.append(graph)
        ##end of one AMR
    #end of batch

    return data_list #, nodelabel_stoi, nodelabel_itos


def train(train_loader, test_loader, model, epochs, src_dict):

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
    model.train()
    
    for epoch in range(epochs):
               
        losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "mse_loss"], 0)
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
        for m,v in metrics_batch_tot.items():
            train_logger.info("{} : {}".format(m, v/len(train_loader.dataset)))
     
        # if epoch % 5 == 0:
        #    test_loss = test(test_loader, model, src_dict, epoch)

                
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
        
    losses_batch_tot  = dict.fromkeys(["loss", "kl_loss", "pos_loss", "neg_loss", "mse_loss"], 0)
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
        test_logger.info("{} : {}".format(l, v/len(test_loader.dataset)))
    for m,v in metrics_batch_tot.items():
        test_logger.info("{} : {}".format(m, v/len(test_loader.dataset)))
     
            
    
        
if __name__ == "__main__":
    print("Oh hi")
    translator = build_translator(model_opt)
    text_transformer = translator.model
    fields = translator.fields
    nodeembeddings = text_transformer.encoder.embeddings
    src_dict = fields["src"].vocab
    parser = argparse.ArgumentParser('datareader.py')
    parser.add_argument('params', help='Parameters YAML file.')
    args = parser.parse_args()
    params = Params.from_file(args.params)
    nmax = 10
    batch_size = 16
    data_list  = loaddata(params, src_dict, nodeembeddings, nmax)
    train_loader = DataLoader(data_list[:1000], batch_size=batch_size)
    test_loader = DataLoader(data_list[1000:5500], batch_size=batch_size)
    epochs = 100
    
    # Build model...
    input_dim = 512; output_dim = 512; nodeclass_num = 38926
    model = GTG(input_dim, output_dim, nmax, nodeclass_num, text_transformer).to(device)
    train(train_loader, test_loader, model, epochs, src_dict)
 
