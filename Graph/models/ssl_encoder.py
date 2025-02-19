import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch
from torch_geometric.nn import (ASAPooling, global_add_pool, global_max_pool,
                                global_mean_pool)
from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks

from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import GNN, LeGNN

class Encoder(nn.Module):
    
    def __init__(self, encoder, augmentor, emb_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.emb_dim = emb_dim
        if emb_dim >0:
            self.proj= torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim,bias=False), torch.nn.BatchNorm1d(2 * emb_dim),
                                        torch.nn.ReLU(inplace=True), torch.nn.Linear(2 * emb_dim, emb_dim),torch.nn.BatchNorm1d(emb_dim))


    def forward(self, x, edge_index, batch, edge_attr, edge_weight, epoch, return_data = "rep"):
        # add epoch for self.encoder()
        aug1, aug2 = self.augmentor
        if edge_attr is not None:
            edge_attr_index = torch.arange(edge_index.size(1)).to(edge_attr.device)
        x1, edge_index1, edge_weight1 = aug1(x, edge_index,edge_attr_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index,edge_attr_index)
        edge_attr1 = edge_attr[edge_weight1] # calculate tips for edge_attr
        edge_attr2 = edge_attr[edge_weight2]
        graph1 = DataBatch.Batch(batch=batch,
                                edge_index=edge_index1,
                                x=x1,
                                edge_attr= edge_attr1)
        graph2 = DataBatch.Batch(batch=batch,
                                edge_index=edge_index2,
                                x=x2,
                                edge_attr= edge_attr2)
        z1, g1 = self.encoder(graph1, epoch, return_data)
        z2, g2 = self.encoder(graph2, epoch, return_data)
        # sample max 3 times for subgraph
        flag=0
        while g2.size(0) != g1.size(0) and flag <3:
            x2, edge_index2, edge_weight2 = aug2(x, edge_index,edge_attr_index)
            edge_attr2 = edge_attr[edge_weight2]
            graph2 = DataBatch.Batch(batch=batch,
                                edge_index=edge_index2,
                                x=x2,
                                edge_attr= edge_attr2)
            z2, g2 = self.encoder(graph2, epoch, return_data)
            flag += 1
        if self.emb_dim > 0:
            g1 = self.proj(g1) # 2 layer non-linear projector
            g2 = self.proj(g2)
        return  z1, z2, g1, g2


class Encoder_proj(nn.Module):
    
    def __init__(self, encoder, emb_dim):
        super(Encoder_proj, self).__init__()
        self.encoder = encoder
        self.proj= torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))

    def forward(self, graph, epoch, return_data = "rep"):
        # print(graph.y==aug_graph.y)
        z1, g1 = self.encoder(graph, epoch, return_data)
        # z2, g2 = self.encoder(aug_graph, epoch, return_data)
        g1 = self.proj(g1) # proj
        return  z1, g1
        
class Encoder1(nn.Module):
    
    def __init__(self, encoder):
        super(Encoder1, self).__init__()
        self.encoder = encoder

    def forward(self, graph, epoch, return_data = "rep"):
        # print(graph.y==aug_graph.y)
        z1, g1 = self.encoder(graph, epoch, return_data)
        # z2, g2 = self.encoder(aug_graph, epoch, return_data)

        return  z1, g1

class Encoder_aug(nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder_aug, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.g1, self.g2 = self.augmentor

    def forward(self, graph, epoch, ratio, return_data='rep'):
        graph1 = self.encoder.get_aug_graph(graph, self.g1, ratio=ratio, epoch=epoch)
        graph2 = self.encoder.get_aug_graph(graph, self.g2, ratio=ratio, epoch=epoch)
        z1, g1 = self.encoder(graph1, epoch, return_data)
        z2, g2 = self.encoder(graph2, epoch, return_data)
        # sample max 3 times for subgraph
        flag=0
        while g2.size(0) != g1.size(0) and flag <3:
            graph2 = self.encoder.get_aug_graph(graph, self.g2, ratio=ratio, epoch=epoch)
            z2, g2 = self.encoder(graph2, epoch, return_data)
            flag += 1
        return  z1, z2, g1, g2