import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from scipy.sparse import csr_matrix, coo_matrix
import dgl
import numpy as np
import pickle 
from tqdm import tqdm
from scipy import sparse
import random
import json
from data_preprocess import load_pickle, save_pickle, data_preprocess, get_non_zero_neighbors, \
    bidirectional_search, create_initial_graph
from model import *
from utils import ndcg


def main(preprocessed=True):
    with open("./config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        
    SEED = train_config["seed"]
    LEARNING_RATE = train_config["learning_rate"]
    DECAY_RATE = train_config["decay_rate"]
    GLOBAL_STEPS = train_config["global_steps"]
    DECAY_STEPS = train_config["decay_steps"]
    if train_config["cuda"] == "True":
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = "cpu"
    TRAIN_RATIO = train_config["train_ratio"]
    NUM_NTYPE = train_config["num_ntype"]
    NUM_USER = train_config["num_user"]
    NUM_KC = train_config["num_kc"]
    num_neigbor_node = train_config['num_neighbor_node']
    num_paths = train_config['num_paths']
    MAX_TRIES = train_config['max_tries']
    depth = train_config['depth']
    user_num = train_config['user_num']
    item_num = train_config['item_num']
    num_aspects = train_config['num_aspects']
    h1 = train_config['h1_dim']
    h2 = train_config['h2_dim']

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # NumPy
    np.random.seed(SEED)

    # Python's `random` module
    random.seed(SEED)
    
    PATH = "../data/"
        
    features, rating_tensor, negative, mooc_adj, training_matrix = data_preprocess(PATH, DEVICE)
    
    if preprocessed != True:
        num_neigbor_node = train_config['num_neighbor_node']
        num_paths = train_config['num_paths']
        MAX_TRIES = train_config['max_tries']
        depth = train_config['depth']
        node_paths = []
        for start_node in tqdm(range(23042)):
            node_path = []
            
            if start_node > 2005:
                end_node_range = (2005, 23042)
            else:
                end_node_range = (0, 2005)

            tries = 0
            while len(node_path) < num_neigbor_node and tries < MAX_TRIES:
                end_node = random.randint(*end_node_range)
                
                if start_node != end_node:
                    try:
                        nodes = bidirectional_search(mooc_adj, start_node, end_node, depth=depth, sample_size=10)
                        if len(nodes) < num_paths:
                            node_path.append(random.choices(nodes, k=num_paths))  
                        else:
                            node_path.append(random.sample(nodes, num_paths))
                    except :
                        pass
                
                tries += 1
            
            if len(node_path) == 0:
                node_path = [[[0]*depth]*num_paths]*num_neigbor_node
            node_paths.append(random.choices(node_path, k=num_neigbor_node))
    uids, iids = training_matrix.nonzero()
    node_paths = load_pickle(PATH+'random_path.p') # node_paths : meta_path regarding users and items
    node_paths = [torch.tensor(sub_list) for sub_list in node_paths]
    node_paths = torch.stack(node_paths)
    node_pairs = node_paths[:,:,0,[0,-1]].view(-1, 2)  # merge all pairs (115210, 2)
    # extracts the node pairs (user, item) and some pairs may overlap \
    # if there is less metapaths than the predefined number of metapaths
    
    G = create_initial_graph(node_pairs, h1, num_aspects)
    G = G.to(DEVICE)
    
    path_tensor = torch.tensor(node_paths, dtype=torch.long).cuda()
    rating_tensor = rating_tensor.cuda()
    loss_function = torch.nn.MSELoss()
    model =Improved_ANR(G,features,h1,h2, num_neigbor_node,num_aspects, NUM_USER,NUM_KC,dropout_rate=0.5).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=1e-8) 
    
    for epoch in range(100):
        train_loss_mean=0

        model.train()
        optimizer.zero_grad()

        output = model(path_tensor).squeeze(-1)

        train_loss = loss_function(output, rating_tensor)
        
        train_loss.backward()
        optimizer.step()
        train_loss_mean +=train_loss.item()

        with torch.no_grad():
            model.eval()
            valid_5 = 0
            valid_10 = 0
            valid_20 = 0
            test_loss_mean = 0
            valid_5 = ndcg(output,negative,user_num,5)
            valid_10 = ndcg(output,negative,user_num,10)
            valid_20 = ndcg(output,negative,user_num,20)
        del output
        if epoch % 10 == 0:
            print('Epoch {:05d} | Train_Loss {:.4f} | NDCG_5 {:.8f} | NDCG_10 {:.8f} | NDCG_20 {:.8f} '.format(epoch, train_loss_mean, valid_5, valid_10,valid_20))

    return 

if __name__ == "__main__":
    main()