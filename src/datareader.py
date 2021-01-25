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
from gtg import GTG
import json
from translator import build_translator
import torch.optim as optim
import logging

logging.basicConfig(filename='gtg.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')

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


nodelabel_stoi = dict()
nodelabel_itos = dict()
nodelabel_stoi["<unk>"]=0
nodelabel_stoi["<pad>"]=1
nodelabel_itos[0] = "<unk>"
nodelabel_itos[1] = "<pad>"


#bpe_data_sent = dict()
#bpe_data_node = dict()


# bpe_data_sent[fileprefix] = bpe_tokens
# bpe_data_node[fileprefix] = node_bpe_tokens

a_file = open("sent_bpes.json", "r")
bpe_data_sent = a_file.read()
bpe_data_sent = json.loads(bpe_data_sent)

b_file = open("node_bpes.json", "r")
bpe_data_node = b_file.read()
bpe_data_node = json.loads(bpe_data_node)

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
            #print(len(bpe_data_sent))
            if fileprefix not in bpe_data_sent or fileprefix not in bpe_data_node:
                continue
            bpe_tokens =  bpe_data_sent[fileprefix]
            node_bpe_tokens = bpe_data_node[fileprefix] 


            # filename = fileprefix+'sent.tok.bpe'
            # filename2 = fileprefix+'node.tok.bpe'

            # if not os.path.isfile('/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/src/sent-toks-bpes/'+filename) or not os.path.isfile('/kuacc/users/mugekural/workfolder/dev/git/gtg-vae/src/node-toks-bpes/'+filename2):
            #      print ("File doesnot exist")
            #      continue
            # # else:
            # #     print ("File not exist")
            # #     f2 = open(fileprefix+'sent.tok', 'w')
            # #     f2.write(' '.join([str(elem) for elem in src_tokens]))
            # #     f2.close()
            # #     os.system('python3 /kuacc/users/mugekural/workfolder/dev/subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes < '+fileprefix+'sent.tok > '+fileprefix+'sent.tok.bpe')
            # f3 = open("sent-toks-bpes/"+fileprefix+"sent.tok.bpe", "r")
            # bpe_sent = f3.readline()
            # bpe_tokens = bpe_sent.split(" ")
            # # print("bpe_tokens:", bpe_tokens)
            # f3.close()

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

            # f4 = open(fileprefix+'node.tok', 'w')
            # for node, values in nodes.items():
            #     nodelabel = values[1]
            #     f4.write(nodelabel+"\n")
            # f4.close()
                
            #os.system('python3 /kuacc/users/mugekural/workfolder/dev/subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes <'+fileprefix+'node.tok > '+fileprefix+'node.tok.bpe')
            # f5 = open("node-toks-bpes/"+fileprefix+"node.tok.bpe", "r")
            # bpe_node = f5.read()
            # node_bpe_tokens = bpe_node.split("\n")
            # #print("node_bpe_tokens:", node_bpe_tokens)
            # f5.close()

            
            # bpe_data_sent[fileprefix] = bpe_tokens
            # bpe_data_node[fileprefix] = node_bpe_tokens

            for nodetoken in node_bpe_tokens[:-1]: #last one is trivial
                nodelabel = nodetoken.split(" ")[0]
                if nodelabel not in nodelabel_stoi:
                    idx = len(nodelabel_stoi)
                    nodelabel_stoi[nodelabel] = idx
                    nodelabel_itos[idx] = nodelabel
                nodevocabid = src_dict[nodelabel]
                node_ids.append(nodevocabid)
                node_ids_local.append(nodelabel_stoi[nodelabel])

            #print("node_ids_local:", node_ids_local)
            #print("snttoken_ids:", snttoken_ids)
            
            # Pad node ids to have max number of nodes
            missing_counts = maxnode -len(node_ids)
            if missing_counts > 0:
                node_ids.extend([1] * missing_counts)
                node_ids_local.extend([1] * missing_counts)
            elif missing_counts < 0:
                node_ids = node_ids[:missing_counts]
                node_ids_local = node_ids_local[:missing_counts]

            nodeids = torch.LongTensor(node_ids).to(device).transpose(-1,0)
            nodeids = torch.unsqueeze(nodeids, 1)

            nodeidslocal = torch.LongTensor(node_ids_local).to(device).transpose(-1,0)
            nodeidslocal = torch.unsqueeze(nodeidslocal, 1)
                    
            node_features = nodeembeddings(nodeids)
            node_features = torch.squeeze(node_features, 1)

            #snttoken_ids = torch.tensor(snttoken_ids, dtype=torch.long).unsqueeze(1)
            x = torch.tensor(node_features, dtype=torch.float) # nodecount,hdim
            
            edge_index = torch.tensor(edges, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index.t().contiguous())
            data.__setitem__("snttoken_ids", snttoken_ids)
            data.__setitem__("nodetoken_ids", nodeids)
            #data.__setitem__("nodetoken_ids", nodeidslocal)
            data_list.append(data)
            meta_list.append(graph)
        ##end of one AMR
    #end of batch

    # bpe_sent_json = json.dumps(bpe_data_sent)
    # f = open("sent_bpes.json","w")
    # f.write(bpe_sent_json)
    # f.close()

    # bpe_node_json = json.dumps(bpe_data_node)
    # f = open("node_bpes.json","w")
    # f.write(bpe_node_json)
    # f.close()

    return data_list #, nodelabel_stoi, nodelabel_itos


def train(train_loader, test_loader, model, epochs, src_dict, node_dict):

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
        
    # opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        print("epoch ", epoch, " ...")
        total_loss = 0
        model.train()
        #model.eval()

        correct_nodes = 0
        for step, data in enumerate(train_loader):
            opt.zero_grad()
            
            x, edge_index, batch, dec_seq, nodetoken_ids = data.x.to(device), data.edge_index.to(device), data.batch.to(device), batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids")
            nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode

            # Encode graph and text
            graph_reconstruction_loss,  correct_predicted_node_tokens, mse_loss = model(x, edge_index, batch, dec_seq, nodetoken_ids, src_dict, node_dict)

            correct_nodes += correct_predicted_node_tokens.item()
            loss = graph_reconstruction_loss + mse_loss
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.num_graphs          
      
        total_loss /= len(train_loader.dataset)
        correct_nodes /= len(train_loader.dataset)
        print("trn_loss:",total_loss)
        print("trn_correct_nodes:", correct_nodes)
        print(len(train_loader.dataset))

        #if epoch % 1 == 0:
        #    test_loss = test(test_loader, model, src_dict, node_dict)
            
        

        
def test(loader, model, src_dict, node_dict):
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
        
    model.eval()
    total_loss = 0;  correct_nodes = 0
    for (step, data) in enumerate(loader):
        with torch.no_grad():
            
            x, edge_index, batch, dec_seq, nodetoken_ids = data.x.to(device), data.edge_index.to(device), data.batch.to(device), batched_snttokens_ids_padded[step].to(device), data.__getitem__("nodetoken_ids")
            nodetoken_ids = nodetoken_ids.view(dec_seq.size(1), -1) # B, maxnode

            # Encode graph and text (mse_loss was the 2nd)
            graph_reconstruction_loss,  correct_predicted_node_tokens, mse_loss = model(x, edge_index, batch, dec_seq, nodetoken_ids, src_dict, node_dict)

            correct_nodes += correct_predicted_node_tokens.item()
            loss = graph_reconstruction_loss + mse_loss                       
            total_loss += loss.item() * data.num_graphs          
      
    total_loss /= len(loader.dataset)
    correct_nodes /= len(loader.dataset)
    print("test_loss:",total_loss)
    print("test_correct_nodes:", correct_nodes)
    print(len(loader.dataset))
        
    return total_loss

        
if __name__ == "__main__":
    print("Oh hi")
    translator = build_translator(model_opt)
    text_transformer = translator.model
    fields = translator.fields
    nodeembeddings = text_transformer.decoder.embeddings
    src_dict = fields["src"].vocab
    parser = argparse.ArgumentParser('datareader.py')
    parser.add_argument('params', help='Parameters YAML file.')
    args = parser.parse_args()
    params = Params.from_file(args.params)
    nmax = 10
    batch_size = 128
    # data_list, nodelabel_stoi, nodelabel_itos = loaddata(params, src_dict, nodeembeddings, nmax)
    data_list  = loaddata(params, src_dict, nodeembeddings, nmax)

    train_loader = DataLoader(data_list[:5000], batch_size=batch_size)
    test_loader = DataLoader(data_list[5000:6000], batch_size=batch_size)
    epochs = 100
    
    # Build model...
    input_dim = 512; output_dim = 512;  edgeclass_num = 15; nodeclass_num = 38926 #len(nodelabel_itos)
    model = GTG(input_dim, output_dim, nmax, edgeclass_num, nodeclass_num, text_transformer).to(device) #text_transformer
    train(train_loader, test_loader, model, epochs, src_dict, nodelabel_itos)
 
