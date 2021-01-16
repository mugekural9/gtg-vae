# import re
import os
import argparse
import yaml
import torch
from utils.tqdm import Tqdm
from utils.params import Params, remove_pretrained_embedding_params
from data.dataset_builder import dataset_from_params, iterator_from_params
from data.vocabulary import Vocabulary

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GAE,VGAE, global_max_pool
from graphvae import *
import json
from translator import build_translator
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
model_opt = dict()
model_opt["dec_rnn_size"] = 512
model_opt["share_embeddings"] = True
model_opt["share_decoder_embeddings"] = True
model_opt["param_init_glorot"] = True
model_opt["dropout"] = 0.1
model_opt["model_path"] = "/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/src/ptm_mt_en2deB_sem_enM.pt"
model_opt["src_word_vec_size"] = 512
model_opt["tgt_word_vec_size"] = 512
model_opt["position_encoding"] = True
model_opt["optim"] = 'adam'
model_opt["heads"] = 8
model_opt["dec_layers"] = 6
model_opt["transformer_ff"] = 2048
model_opt["src"] = 'sent.tok.bpe' 
model_opt["gpu"] = 0
model_opt["task_type"] = 'task2'
model_opt["minimal_relative_prob"] = 0.01
model_opt["batch_size"] = 32



def loaddata(params, src_dict, nodeembeddings):

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

            node_ids = []; snttoken_ids = [] # with these ids we'll able to forward the embeddings
            i = 0
            for (s,r,t) in graph._triples:
                if r=='instance':
                    nodes[s] = (i,t) # t will be the nodelabel.
                    i+=1

            for token in src_tokens:
                snttokenid = src_dict[token]
                snttoken_ids.append(snttokenid)
                
            for edge in graph.edges():
                s_id = nodes[edge.source][0]
                t_id = nodes[edge.target][0]
                edges.append([s_id, t_id])
                edges.append([t_id, s_id])
                ##TODO:For now assuming nondirected AMR graphs; both edge and reverse included
                
        
            for node, values in nodes.items():
                nodelabel = values[1]
                nodevocabid = src_dict[nodelabel]
                node_ids.append(nodevocabid)
            
            nodeids = torch.LongTensor(node_ids).to(device).transpose(-1,0)
            nodeids = torch.unsqueeze(nodeids, 1)
            
            node_features = nodeembeddings(nodeids)
            node_features = torch.squeeze(node_features, 1)
            
            x = torch.tensor(node_features, dtype=torch.float) # Nodecount,Hdim
            edge_index = torch.tensor(edges, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index.t().contiguous())
            data.__setitem__("snttoken_ids", snttoken_ids)
            data_list.append(data)
            meta_list.append(graph)
         ##end of one AMR
    #end of batch
    return data_list


def train(data_list, textencoder_model):
    # Build model...
    input_dim = 512; output_dim = 512; nmax = 30; edgeclass_num = 15; nodeclass_num = 20
    encoder = VGAE_encoder(input_dim, out_channels=output_dim)
    decoder = VGAE_decoder(output_dim, nmax, edgeclass_num, nodeclass_num).to(device)
    model   = VGAE(encoder).to(device)

    train_loader = DataLoader(data_list, batch_size=32)
    for step, data in enumerate(train_loader):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        z = model.encode(x, edge_index)
        graph_z = global_max_pool(z, batch) # (B,Zdim)
        # decoder(z)

        src_seq = torch.zeros(1, graph_z.shape[0]) # dummy seq to bypass masking
        src_enc = graph_z.unsqueeze(0) # (1,B,Zdim)
        textencoder_model.decoder.init_state(src_seq, src_enc) # src_seq should be 1,B, src_enc 1,B,Zdim

        # Pad snttoken ids in batch...
        dec_seq = data.__getitem__("snttoken_ids")
        list_len = [len(i) for i in dec_seq]
        max_seq_len  = max(list_len)
        for seq in dec_seq:
            while len(seq) < max_seq_len:
                seq.append(0)

        dec_seq = torch.tensor(dec_seq).to(device).transpose(0,1) # Tdec, B
        dec_output, *_ = textencoder_model.decoder(dec_seq, step=None) # src_seq as dec_seq, but any need for shifting? dec_output: (Tdec, B, Zdim)         
        text_z = dec_output[:1, :,:].squeeze(0)

        # print("graph_z:", graph_z.shape)
        # print("text_z:", text_z.shape)
        
        
    
if __name__ == "__main__":

    translator = build_translator(model_opt)
    translator_model = translator.model.to(device)
    fields = translator.fields
    nodeembeddings = translator.model.decoder.embeddings
    src_dict = fields["src"].vocab
    #print(src_dict.stoi)
    
    parser = argparse.ArgumentParser('datareader.py')
    parser.add_argument('params', help='Parameters YAML file.')
    args = parser.parse_args()
    params = Params.from_file(args.params)
    data_list = loaddata(params, src_dict, nodeembeddings)
    train(data_list, translator_model)
