import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
import random
import time


def init_lstm_weights(lstm):
    # 가중치 초기화
    for name, param in lstm.named_parameters():
        # 가중치 행렬 초기화
        if 'weight_ih' in name:  # input-hidden 가중치에 대해
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:  # hidden-hidden 가중치에 대해
            nn.init.xavier_uniform_(param.data)



class Aspect_Representation(nn.Module):

    def __init__(self, h1, num_aspects):
        super(Aspect_Representation, self).__init__()
        
        self.h1 = h1
        self.num_aspects = num_aspects
        self.aspEmbed = nn.Linear(in_features=300, out_features=num_aspects*h1,bias=False)
        nn.init.xavier_uniform_(self.aspEmbed.weight)
        if self.aspEmbed.bias is not None:
            self.aspEmbed.bias.data.fill_(0)

        # 가중치와 편향을 float16으로 변환
        #self.aspEmbed.weight.data = self.aspEmbed.weight.data.half()
        self.lstm = nn.LSTM(input_size=self.h1 * self.num_aspects, hidden_size=self.h1 * self.num_aspects, batch_first=False)
        init_lstm_weights(self.lstm)

        self.relu = nn.ReLU()

    def forward(self, node_emb):
        # node_emb의 shape : Batch x path x node x node_dim
        Batch, n_node, path, node, node_dim = node_emb.size()
        #print(node_emb[0,0,0,0,0])
        node_asp_Rep_q = self.aspEmbed(node_emb)
        
        self.relu = nn.ReLU()
        #print(node_asp_Rep_q[0,0,0,0,0])
        node_asp_Rep_q = node_asp_Rep_q.view(Batch, n_node, path, node, self.num_aspects,self.h1)   
        node_asp_Rep = node_asp_Rep_q.view(Batch * n_node * path, node, self.num_aspects * self.h1)

        path_asp_Rep, _ = self.lstm(node_asp_Rep)
        path_asp_Rep = path_asp_Rep[:, -1, :]
        path_asp_Rep = self.relu(path_asp_Rep)
        path_asp_Rep = path_asp_Rep.view(Batch, n_node, path, self.num_aspects, self.h1)
        # embedding after lstm(rnn) --> (batch, n_node, path, num_aspect, h1)

        # node_asp_Rep_q의 shape: (batch_size, n_node, path, node, num_aspect, h1)
        mode_pair_first = node_asp_Rep_q[:, :, 0, 0].unsqueeze(-3)
        mode_pair_last = node_asp_Rep_q[:, :, 0, -1].unsqueeze(-3)
        mode_pair = torch.cat((mode_pair_first, mode_pair_last), dim=-3)


        return path_asp_Rep, mode_pair
import torch.nn.init as init
class Path_Aggreate(nn.Module):
    def __init__(self, h1,h2,neigbor_node,num_aspects):
        super(Path_Aggreate, self).__init__()
        self.num_aspects = num_aspects
        self.h1 = h1
        # Linear layers
        self.W = nn.Linear(self.h1, self.h1)
        self.v = nn.Linear(self.h1, 1)
        # Xavier uniform initialization
        init.xavier_uniform_(self.W.weight)
        init.xavier_uniform_(self.v.weight)
  
        # Additional fully connected layers
        self.fc1 = nn.Linear(h1, h1)
        self.fc2 = nn.Linear(h1, h1)
        # Xavier normal initialization
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

        # Activation and softmax
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-2)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(neigbor_node, num_aspects)
       


    def forward(self, path_asp_Rep):
        batch_size,n_node, num_paths, num_aspects, h1 = path_asp_Rep.size()
        path_asp_Rep_reshaped = path_asp_Rep.view(-1, h1)  
        attn_weights = self.v(torch.tanh(self.W(path_asp_Rep_reshaped))).squeeze(1)
        attn_weights = attn_weights.view(batch_size,n_node, num_paths, num_aspects)
        attn_weights = torch.softmax(attn_weights, dim=2)
 

        output = torch.sum(path_asp_Rep * attn_weights.unsqueeze(-1), dim=2)
        # (batch_size, n_node, num_aspects, h1)

        output = self.batch_norm1(output)
        output = self.fc1(output)
        output = self.relu(output)

        output = self.fc2(output)
        output = output.view(-1,self.num_aspects,h1)
        output = self.softmax(output)
        output = output.view(-1,self.num_aspects*h1)

        return output

class Aspect_Importance(nn.Module):

    def __init__(self,  h1,h2):
        super(Aspect_Importance, self).__init__()
        self.h1 = h1
        self.h2 = h2

        self.W_a = nn.Parameter(torch.Tensor(self.h1,1), requires_grad=True)
        self.W_u = nn.Parameter(torch.Tensor(self.h1, self.h2), requires_grad=True)
        self.w_hu = nn.Parameter(torch.Tensor(self.h2, 1), requires_grad=True)
        self.W_i = nn.Parameter(torch.Tensor(self.h1, self.h2), requires_grad=True)
        self.w_hi = nn.Parameter(torch.Tensor(self.h2, 1), requires_grad=True)

        nn.init.xavier_uniform_(self.W_a)
        nn.init.xavier_uniform_(self.W_u)
        nn.init.xavier_uniform_(self.w_hu)
        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.w_hi)


    def forward(self, user_asp_Rep, item_asp_Rep):
        # path_asp_Rep :  Batch  x aspect x h1
        # affinityMatrix : Batch x aspect x aspect
        user_asp_Rep_attn = torch.einsum('uah, hw -> uaw', user_asp_Rep, self.W_a)
        item_asp_Rep_attn = torch.einsum('iah, hw -> iaw', item_asp_Rep, self.W_a)

        affinityMatrix = torch.einsum('uaw, iaw -> uia', user_asp_Rep_attn, item_asp_Rep_attn)
        affinityMatrix = F.relu(affinityMatrix)

        H_u_1 = torch.einsum('uah, hw -> uaw',user_asp_Rep, self.W_u) # H_u_1 : Batch  x aspect x h2
        H_u_2 = torch.einsum('iah, hw -> iaw',item_asp_Rep, self.W_i)# H_u_2 : Batch  x aspect x h2
        H_u_2 = torch.einsum('iaw, uia -> uaw', H_u_2, affinityMatrix)
        H_u = H_u_1 + H_u_2 # H_u : Batch  x aspect x h2
        H_u = F.relu(H_u) # H_u : Batch  x aspect x h2

        userAspImpt = torch.einsum('wm, uaw -> uam', self.w_hu, H_u).squeeze() # userAspImpt : Batch  x aspect
        userAspImpt = F.softmax(userAspImpt, dim=-2)

        H_i_1 = torch.einsum('iah, hw -> iaw',item_asp_Rep, self.W_i)
        H_i_2 = torch.einsum('uah, hw -> uaw',user_asp_Rep, self.W_u)
        H_i_2 = torch.einsum('uaw, uia -> iaw', H_i_2, affinityMatrix)
        H_i = H_i_1 + H_i_2
        H_i = F.relu(H_i)

        itemAspImpt = torch.einsum('wm, iaw -> iam', self.w_hi, H_i).squeeze() # userAspImpt : Batch x aspect
        itemAspImpt = F.softmax(itemAspImpt, dim=-2)

        return userAspImpt, itemAspImpt
class ANR_RatingPred(nn.Module):

    def __init__(self, num_aspects, num_users, num_items,dropout_rate):
        super(ANR_RatingPred, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.userAspRepDropout = nn.Dropout(p=dropout_rate)
        self.itemAspRepDropout = nn.Dropout(p=dropout_rate)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.uid_userOffset = nn.Parameter(torch.zeros(self.num_users,1))
        self.iid_itemOffset = nn.Parameter(torch.zeros(self.num_items,1))
        

 
        self.output = nn.Linear(num_aspects, 1)
        self.Relu = nn.ReLU()
    def forward(self, user_asp_Rep, item_asp_Rep, userAspImpt, itemAspImpt,user,item):
        # path_asp_Rep :  Batch x aspect x h1
        # AspImpt : Batch  x aspect
        # rating_pred : Batch x 1
        user_asp_Rep = self.userAspRepDropout(user_asp_Rep)
        item_asp_Rep = self.itemAspRepDropout(item_asp_Rep)
        rating_pred_user = torch.einsum('uah, ua -> ua', user_asp_Rep, userAspImpt) + self.uid_userOffset
        rating_pred_item = torch.einsum('iah, ia -> ia', item_asp_Rep, itemAspImpt) + self.iid_itemOffset

        #print(rating_pred_user.size(),rating_pred_item.size())
        rating_pred = torch.einsum('ua, ia -> uia', rating_pred_user, rating_pred_item)   # Batch x aspect
        rating_pred = self.output(rating_pred)+self.globalOffset

        return rating_pred


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # Define a linear layer
        self.batch_norm = nn.BatchNorm1d(out_dim)
        nn.init.xavier_normal_(self.linear.weight)
        self.activate = nn.ReLU()
    def forward(self, g, feature, edge_weight):
        with g.local_scope():
            g.ndata['h'] = feature
            g.edata['w'] = edge_weight
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata['h']
            h = self.linear(h)
            h = self.activate(h)
            h = self.batch_norm(h)
            return h
    
    def message_func(self, edges):
        # Pass messages using edge weights 'w' and node features 'h'
        return {'m': edges.data['w'] * edges.src['h']}
    
    def reduce_func(self, nodes):
        # Sum the messages from 'm' and use it as the node's new state 'h'
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}
    
class Local_Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Local_Encoder, self).__init__()
        self.gcn1 = GCNLayer(in_dim, in_dim)
        self.gcn2 = GCNLayer(in_dim, in_dim)
        self.out_dim = out_dim

    def forward(self, graph, node_pair,node_features):
        batch,n_node,pair,aspect,h = node_features.size()
        sub_feature = torch.cat([node_features[:,0,0,:,:].unsqueeze(1), node_features[:,:,-1,:,:]], dim=1).view(-1,aspect*h) # n_node 차원에서 합치기   
        node = node_pair.view(batch,n_node,pair)
        node = torch.cat([node[:,0,0].unsqueeze(1), node[:,:,-1]], dim=1).view(-1)

        sub_graph = graph.subgraph(node)   
        edge_weight = sub_graph.edata['weight']  # 서브그래프의 엣지 가중치를 추출합니다.
        

        subgraph_h = self.gcn1(sub_graph, sub_feature, edge_weight)  # GCNLayer에 엣지 가중치를 전달합니다.
        subgraph_h = self.gcn2(sub_graph, subgraph_h, edge_weight)  # 두 번째 레이어에도 마찬가지로 적용합니다.

        return subgraph_h.view(batch, (n_node+1), aspect, self.out_dim)[:, 0, :, :]

class Improved_ANR(nn.Module):

    def __init__(self, sub_graph, sub_feature, h1, h2, neigbor_node, num_aspects, num_users, num_items, dropout_rate=0.5):
        super(Improved_ANR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_aspects = num_aspects
        self.sub_graph= sub_graph
        self.sub_feature= sub_feature
        self.h1 = h1

      # 편향을 float16으로 변경

        self.local_encoder = Local_Encoder(h1*num_aspects,h1)
        self.Aspect_Representation = Aspect_Representation(h1, num_aspects)


        self.Aspect_Importance = Aspect_Importance(h1, h2)
        self.Path_Aggreate = Path_Aggreate(h1,h2,neigbor_node,num_aspects)
        self.ANR_RatingPred = ANR_RatingPred(num_aspects, self.num_users, self.num_items, dropout_rate)


        self.batch_norm2 = nn.BatchNorm2d(neigbor_node,num_aspects )
        self.relu = nn.ReLU()

        
    def forward(self, path_tensor):
        start_time = time.time()
        
        batch_user_node_emb = self.sub_feature[path_tensor[0:self.num_users]]

        batch_item_node_emb_1 = self.sub_feature[path_tensor[self.num_users: self.num_users+6000 ]]
        batch_item_node_emb_2 = self.sub_feature[path_tensor[self.num_users+6000 : self.num_users+ 12000 ]]
        batch_item_node_emb_3 = self.sub_feature[path_tensor[self.num_users+ 12000: self.num_users+ self.num_items ]]

        
        start_time = time.time()
        
        user_path_asp_Rep, user_node_aspect = self.Aspect_Representation(batch_user_node_emb)
        # user_path_asp_Rep --> (2005, 5, 4, 5, 20)
        
        start_time = time.time()
        item_path_asp_Rep1, item_node_aspect1= self.Aspect_Representation(batch_item_node_emb_1)
        item_path_asp_Rep2, item_node_aspect2 = self.Aspect_Representation(batch_item_node_emb_2)
        item_path_asp_Rep3, item_node_aspect3 = self.Aspect_Representation(batch_item_node_emb_3)
        item_path_asp_Rep = torch.cat([item_path_asp_Rep1,item_path_asp_Rep2,item_path_asp_Rep3],dim=0)
        item_node_aspect = torch.cat([item_node_aspect1, item_node_aspect2, item_node_aspect3],dim=0)
        #print('item path embedding', time.time()-start_time )

        start_time = time.time()
        user_edge_weight = self.Path_Aggreate(user_path_asp_Rep)
        # (batch_size*num_node, num_aspect*h1)
        item_edge_weight = self.Path_Aggreate(item_path_asp_Rep)
        # (batch_size*num_node, num_aspect*h1)

        start_time = time.time()
        user_path_pair = path_pair(path_tensor[0:self.num_users])
        item_path_pair = path_pair(path_tensor[self.num_users: self.num_users+ self.num_items ])
        #print('node pair update', time.time()-start_time )

        start_time = time.time()    
        update_edge_weights(self.sub_graph,user_path_pair,user_edge_weight) 
        update_edge_weights(self.sub_graph,item_path_pair,item_edge_weight) 
        #print('edge embedding update', time.time()-start_time )

        start_time = time.time()
        user_h = self.local_encoder(self.sub_graph,user_path_pair,user_node_aspect)
        #print(user_h[0,0,0])
        item_h = self.local_encoder(self.sub_graph,item_path_pair,item_node_aspect)
        #print('graph embedding', time.time()-start_time )

        start_time = time.time()

        userCoAttn, itemCoAttn = self.Aspect_Importance(user_h, item_h)

        rating_pred = self.ANR_RatingPred(user_h, item_h, userCoAttn, itemCoAttn, self.num_users, self.num_items)
        #print('attention embedding1', time.time()-start_time )

        return rating_pred
    






def path_pair(user_path):
    user_start = user_path[:, :, 0, 0].reshape(-1, 1)
    user_end = user_path[:, :, 0, -1].reshape(-1, 1)
    user_path_pair = torch.cat((user_start, user_end), dim=1)
    return user_path_pair
def update_edge_weights(G, model_node_pairs, new_weights):

    src_nodes = model_node_pairs[:, 0]
    dst_nodes = model_node_pairs[:, 1]
    edges = G.edge_ids(src_nodes, dst_nodes)
    G.edata['weight'][edges] = torch.tensor(new_weights)

