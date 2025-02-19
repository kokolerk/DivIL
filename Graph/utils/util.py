import torch
import numpy as np

import random
from texttable import Texttable
import torch.nn.functional as F 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_partition(len_dataset, device, seed, p=[0.5,0.5]):
    '''
        group the graph randomly

        Input:   len_dataset   -> [int]
                 the number of data to be groupped
                 
                 device        -> [torch.device]
                
                 p             -> [list]
                 probabilities of the random assignment for each group
        Output: 
                 vec           -> [torch.LongTensor]
                 group assignment for each data
    '''
    assert abs(np.sum(p) - 1) < 1e-4
    
    vec = torch.tensor([]).to(device)
    for idx, idx_p in enumerate(p):
        vec = torch.cat([vec, torch.ones(int(len_dataset * idx_p)).to(device) * idx])
        
    vec = torch.cat([vec, torch.ones(len_dataset - len(vec)).to(device) * idx])
    perm = torch.randperm(len_dataset, generator=torch.Generator().manual_seed(seed))
    return vec.long()[perm]

    
def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())
    

def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))


def within_class_variation(feat_dict):
    features = torch.stack(list(feat_dict.values()), dim=0).float()
    class_means = torch.mean(features, dim=1)
    global_center_class_mean = torch.mean(class_means, dim=0)
    if features.size(0)>2:
        centered_class_means = class_means - global_center_class_mean
    else:
        centered_class_means = class_means
    norm_centered_class_mean = F.normalize(centered_class_means, p=2, dim=1)
    cos = torch.matmul(norm_centered_class_mean,norm_centered_class_mean.T)
    up_HH = torch.cat([torch.diag(cos, i) for i in range(1,features.size(0))]) + 1.0/(features.size(0)-1)
    cos_mean = torch.mean(up_HH)

    l2_centered_class_means = torch.norm(centered_class_means, dim=1, p=2)
    std = torch.std(l2_centered_class_means)
    mean = torch.mean(l2_centered_class_means)
    equinorm_product = std/mean

    between_class_scatter_matrix = torch.einsum('bi,bj->ij', centered_class_means, centered_class_means) / features.size(0) # Divide by the total number of class c
    mean_within_class_scatter_matrix = torch.zeros_like(between_class_scatter_matrix)
    for class_idx, features_class in enumerate(features):
        centered_features = features_class - class_means[class_idx]
        mean_within_class_scatter_matrix += torch.einsum('bi,bj->ij', centered_features, centered_features)
    mean_within_class_scatter_matrix /= features.size(0)*features.size(1)  # Divide by the total number of samples c*n
    tr_product = torch.trace(mean_within_class_scatter_matrix) / torch.trace(between_class_scatter_matrix)
    # Calculate the pseudoinverse of the between-class scatter matrix
    # pseudoinverse_between_class_scatter_matrix = torch.pinverse(between_class_scatter_matrix)
    # # Calculate Tr(ΣW Σ†B/C)
    # tr_product = torch.trace(torch.matmul(mean_within_class_scatter_matrix, pseudoinverse_between_class_scatter_matrix)) / features.size(0)
    return tr_product,equinorm_product,cos_mean


def within_class_variation_2(feat_dict):
    features = list(feat_dict.values())
    class_means = []
    for feat_tensor in features:
        class_mean = torch.mean(feat_tensor, dim=0) 
        class_means.append(class_mean)
    class_means = torch.stack(class_means, dim=0)
    

    l2_centered_class_means = torch.norm(class_means, dim=1, p=2)
    std = torch.std(l2_centered_class_means)
    mean = torch.mean(l2_centered_class_means)
    equinorm_product = std/mean
    print(len(features))
    print(features[0].size())
    print(features[1].size())
    between_class_scatter_matrix = torch.einsum('bi,bj->ij', class_means, class_means) / len(features) # Divide by the total number of class c
    mean_within_class_scatter_matrix = torch.zeros_like(between_class_scatter_matrix)
    for class_idx, features_class in enumerate(features):
        centered_features = features_class - class_means[class_idx]
        mean_within_class_scatter_matrix += torch.einsum('bi,bj->ij', centered_features, centered_features)
    mean_within_class_scatter_matrix /= len(features)*(features[0].size(0)+features[1].size(0))  # Divide by the total number of samples c*n
    tr_product = torch.trace(mean_within_class_scatter_matrix) / torch.trace(between_class_scatter_matrix)
    # Calculate the pseudoinverse of the between-class scatter matrix
    # pseudoinverse_between_class_scatter_matrix = torch.pinverse(between_class_scatter_matrix)
    # # Calculate Tr(ΣW Σ†B/C)
    # tr_product = torch.trace(torch.matmul(mean_within_class_scatter_matrix, pseudoinverse_between_class_scatter_matrix)) / features.size(0)
    return tr_product,equinorm_product,None
 