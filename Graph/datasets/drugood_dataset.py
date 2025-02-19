import os.path as osp
import pickle as pkl

import torch
import random
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import GCL.augmentors as A

def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')
    
    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}    
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                    font_color='red')
    plt.show()

def to_molecule(data):
    ATOM_MAP = ['C', 'O', 'Cl', 'H', 'N', 'F',
                'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = ATOM_MAP[data['x'].index(1.0)]
        del data['x']
    return g

class DrugOOD(InMemoryDataset):

    def __init__(self, root, dataset, name, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DrugOOD dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data(root, dataset, name, mode)

    def load_data(self, root, dataset, name, mode):
        data_path = osp.join(root, name + "_" + mode + ".pt")
        if not osp.exists(data_path):
            data_list = []
            # for data in dataset:
            # augment data
            for step, data in tqdm(enumerate(dataset), total=len(dataset), desc="Converting"):
                graph = data['input']
                y = data['gt_label']
                group = data['group']

                edge_index = graph.edges()
                edge_attr = graph.edata['x']  #.long()
                node_attr = graph.ndata['x']  #.long()
                new_data = Data(edge_index=torch.stack(list(edge_index), dim=0),
                                edge_attr=edge_attr,
                                x=node_attr,
                                y=y,
                                group=group)
                data_list.append(new_data)
            torch.save(self.collate(data_list), data_path)


        self.data, self.slices = torch.load(data_path)
    

class aug_DrugOOD(InMemoryDataset):

    def __init__(self, root, dataset, name, mode, aug, ratio ,transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DrugOOD dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        super(aug_DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data(root, dataset, name, mode, aug, ratio)

    def load_data(self, root, dataset, name, mode, aug, ratio):
        data_path = osp.join(root, name + "_" + mode + ".pt")
        aug_data_path = osp.join(root, name + "_"+ aug +"_"+ str(ratio) +"_" + mode + ".pt")
        if not osp.exists(aug_data_path):
            data_list = []
            aug_data_list = []
            # for data in dataset:
            # augment data
            for step, data in tqdm(enumerate(dataset), total=len(dataset), desc="Converting"):
                graph = data['input']
                y = data['gt_label']
                group = data['group']

                edge_index = graph.edges()
                edge_attr = graph.edata['x']  #.long()
                node_attr = graph.ndata['x']  #.long()
             
                new_data = Data(edge_index=torch.stack(list(edge_index), dim=0),
                                edge_attr=edge_attr,
                                x=node_attr,
                                y=y,
                                group=group)
               
                aug_data = self.augment(new_data,aug,ratio)
                # data_list.append(new_data)
                aug_data_list.append(aug_data)
                
            # torch.save(self.collate(data_list), data_path)
            torch.save(self.collate(aug_data_list), aug_data_path)

        self.data, self.slices = torch.load(aug_data_path)
    
    def augment(self,new_data, aug='subgraph', ratio=0.4):
        if aug == 'snodedrop':
            data = drop_nodes(new_data, ratio)
        elif aug == 'ssubgraph':
            data = get_subgraph(new_data, ratio)
        # elif aug == 'edgepremute':
        #     data = edgepermute(new_data, ratio)
        else :
            raise ValueError('wrong augment type!')
        return data

def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num*aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    idx_nondrop = torch.tensor(idx_nondrop)
    mask_source = torch.tensor([idx in idx_nondrop for idx in edge_index[0]],dtype=torch.bool)
    mask_target = torch.tensor([idx in idx_nondrop for idx in edge_index[1]],dtype=torch.bool)
    mask = torch.logical_and(mask_source, mask_target)
    edge_attr = edge_attr[mask]
    
    # new function
    edge_index = edge_index[:,mask]
    # relabel
    node_map ={old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.unique(edge_index))}
    new_edge_index = torch.tensor([[node_map[edge_index[0,i].item()], node_map[edge_index[1,i].item()]] for i in range(edge_index.size(1))],dtype=torch.long).t()
    

    try:
        data.edge_index = new_edge_index
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data

def get_subgraph(data,aug_ratio):

    random.seed(0)
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    # row, col = data.edge_index
    # adjacent matrix
    edge_index = data.edge_index.numpy()
    input_adj = torch.zeros((node_num, node_num))
    input_adj[edge_index[0], edge_index[1]] = 1

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * aug_ratio)
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):
        
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    
    idx_nondrop = sorted([i for i in all_node_list if i in sub_node_id_list])
    
    assert len(idx_nondrop) <= node_num
    # print('node_idx:', idx_nondrop)
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    # print('begin')
    # edge_index, edge_attr = subgraph(idx_nondrop, edge_index, edge_attr, relabel_nodes=True, num_nodes=node_num)
    node_mask = torch.zeros(node_num, dtype=torch.bool)
    node_mask[idx_nondrop] = 1
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    edge_index = edge_index[:,edge_mask]
    # relabel
    node_map ={old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.unique(edge_index))}
    new_edge_index = torch.tensor([[node_map[edge_index[0,i].item()], node_map[edge_index[1,i].item()]] for i in range(edge_index.size(1))],dtype=torch.long).t()
    

    data.edge_index = new_edge_index
    data.edge_attr = edge_attr
    data.x = data.x[idx_nondrop]
    # print('over')

    return data


