import os
import re
import argparse
import yaml
import torch
from utils.tqdm import Tqdm
from utils.params import Params, remove_pretrained_embedding_params
from data.dataset_builder import dataset_from_params, iterator_from_params
from data.vocabulary import Vocabulary


def loaddata(params):
    # Load data.
    data_params = params['data']
    dataset = dataset_from_params(data_params)
    train_data = dataset['train']
    dev_data = dataset.get('dev')
    test_data = dataset.get('test')

    train_iterator, dev_iterator, test_iterator = iterator_from_params(None, data_params['iterator'])
    train_generator = train_iterator(
            instances=train_data,
            shuffle=True,
            num_epochs=1
        )

    num_training_batches = train_iterator.get_num_batches(train_data)
    print(num_training_batches)

    ## Do something with data...
    train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)
    for batch in train_generator_tqdm:
        # batch contains bunch of AMRs
        for amr in batch.instances:
            graph = amr.graph
            nodes = graph.get_nodes()
            edges = graph.get_edges()
            #print("nodes: ", nodes)
            #print("edges: ", edges) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser('datareader.py')
    parser.add_argument('params', help='Parameters YAML file.')
    args = parser.parse_args()
    params = Params.from_file(args.params)
    loaddata(params)
