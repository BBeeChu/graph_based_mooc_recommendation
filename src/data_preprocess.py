import pickle
from scipy.sparse import csr_matrix, coo_matrix
import torch
import numpy as np
import dgl

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
        
def data_preprocess(path, device):
    rate_matrix = load_pickle(path+'rating.p')
    adjacency_matrix = load_pickle(path+'adjacency_matrix.p')
    all_adjacency_matrix = load_pickle(path+'all_adjacency_matrix.p')
    features_user = load_pickle(path+'features_user.p')
    features_item = load_pickle(path+'features_item.p')
    features_course = load_pickle(path+'features_course.p')
    features_teacher = load_pickle(path+'features_teacher.p')
    features_video = load_pickle(path+'features_video.p')
    support_user = load_pickle(path+'support_user.p')
    support_item = load_pickle(path+'support_item.p')
    negative = load_pickle(path+'negative.p')
    
    DEVICE = device
    features_user = torch.tensor(features_user, dtype=torch.float32).to(DEVICE)
    features_item = torch.tensor(features_item, dtype=torch.float32).to(DEVICE)
    features_course = torch.tensor(features_course, dtype=torch.float32).to(DEVICE)
    features_teacher = torch.tensor(features_teacher, dtype=torch.float32).to(DEVICE)
    features_video = torch.tensor(features_video, dtype=torch.float32).to(DEVICE)
    features = torch.cat([features_user,features_item,features_course,features_teacher,features_video])
    rating_tensor = torch.tensor(rate_matrix, dtype=torch.float32)
    negative = torch.tensor(negative,dtype=torch.float32).to(DEVICE)
    
    adjM = csr_matrix(all_adjacency_matrix, dtype=np.float32)
    mooc_adj = adjM+(adjM.T)
    training_matrix = csr_matrix(rate_matrix)
    
    return features, rating_tensor, negative, mooc_adj, training_matrix


def get_non_zero_neighbors(matrix: csr_matrix, node, sample_size=None):
    start_ptr, end_ptr = matrix.indptr[node], matrix.indptr[node + 1]
    neighbors = matrix.indices[start_ptr:end_ptr]
    if sample_size is not None and len(neighbors) > sample_size:
        neighbors = np.random.choice(neighbors, sample_size, replace=False)
    return neighbors   

def bidirectional_search(matrix, start, end, depth, sample_size=None):
    forward_steps = depth // 2 # 올림
    backward_steps = depth // 2
    if depth % 2 == 1: # 나머지
        forward_steps += 1
    forward_visited = {start: None}
    backward_visited = {end: None}
    forward_frontier = [start]
    backward_frontier = [end]
    for _ in range(forward_steps):
        next_frontier = []
        for node in forward_frontier:
            neighbors = get_non_zero_neighbors(matrix, node, sample_size)
            for neighbor in neighbors:
                if neighbor not in forward_visited:
                    forward_visited[neighbor] = node
                    next_frontier.append(neighbor)
        forward_frontier = next_frontier
    for _ in range(backward_steps):
        next_frontier = []
        for node in backward_frontier:
            neighbors = get_non_zero_neighbors(matrix, node, sample_size)
            for neighbor in neighbors:
                if neighbor not in backward_visited:
                    backward_visited[neighbor] = node
                    next_frontier.append(neighbor)
        backward_frontier = next_frontier
    intersect = set(forward_frontier).intersection(backward_frontier)
    if not intersect and depth % 2 == 0:
        for f_node in forward_frontier:
            if backward_visited.get(forward_visited[f_node]) is not None:
                intersect.add(f_node)
        for b_node in backward_frontier:
            if forward_visited.get(backward_visited[b_node]) is not None:
                intersect.add(b_node)
    if intersect:
        paths = []
        for inter_node in intersect:
            path_from_start = [inter_node]
            while path_from_start[-1] != start:
                path_from_start.append(forward_visited[path_from_start[-1]])
            path_from_start = path_from_start[::-1]
            path_from_end = []
            while inter_node != end:
                path_from_end.append(backward_visited[inter_node])
                inter_node = path_from_end[-1]
            full_path = path_from_start[:-1] + path_from_end
            paths.append(full_path)
        unique_paths = []
        seen = set()
        for path in paths:
            tuple_path = tuple(path)
            if tuple_path not in seen:
                seen.add(tuple_path)
                unique_paths.append(path)
        return unique_paths
    else:
        return []
    
def create_initial_graph(node_pairs,h1, num_aspects):
    src_nodes, dst_nodes = zip(*node_pairs)
    G = dgl.graph((src_nodes, dst_nodes))
    G.edata['weight'] = torch.randn(len(node_pairs), num_aspects*h1)
    return G