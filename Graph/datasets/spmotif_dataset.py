import os.path as osp
import pickle as pkl

import torch
import random
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, add_self_loops, sort_edge_index
import copy


from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import add_edge
from torch_sparse import SparseTensor, coalesce

class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        self.root = root
        print(f"######################################## {root}")
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])
        self.orig_data_list = None

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt']

    # def download(self):
    #     if not osp.exists(osp.join(self.raw_dir, 'raw', 'SPMotif_train.npy')):
    #         print("raw data of `SPMotif` doesn't exist, please redownload from our github.")
    #         raise FileNotFoundError

    def process(self, get_data_list=False):

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        if 'dspmotif' or 'wspmotif' in self.root.lower():
            # edge_index_list, label_list, base_list,ground_truth_list, role_id_list, pos_list, motif_id_list
            edge_index_list, label_list, base_list, ground_truth_list, role_id_list, pos, motifs = np.load(osp.join(
                self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            
        elif 'tspmotif' in self.root.lower(): # no data size shift in test set
            edge_index_list, label_list,  ground_truth_list, role_id_list, pos = np.load(osp.join(
                self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            motifs = np.zeros(label_list.shape,dtype=int)-1
            base_list = np.zeros(label_list.shape,dtype=int)-1
        elif '0.6' in self.root or '0.7' in self.root:
            edge_index_list, label_list, base_list, ground_truth_list, pos, = np.load(osp.join(
            self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            role_id_list = np.zeros(label_list.shape,dtype=int)-1
            motifs = np.zeros(label_list.shape,dtype=int)-1
        else: # data set shift (bigger) in test set
            edge_index_list, label_list,  ground_truth_list, role_id_list, pos = np.load(osp.join(
                self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            motifs = np.zeros(label_list.shape,dtype=int)-1
            base_list = np.zeros(label_list.shape,dtype=int)-1

        data_list = []
        for idx, (edge_index, y, group,ground_truth, z, p, mid) in \
        enumerate(zip(edge_index_list, label_list, base_list, ground_truth_list, role_id_list, pos, motifs)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = edge_index.long()
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            node_label = torch.tensor(z, dtype=torch.float)
            node_label[node_label != 0] = 1
            if 'mspmotif' in self.root.lower() or 'pmspmotif' in self.root.lower():
                # additionally add the spuriously correlated node features
                bias = 0.5
                if '0.3' in self.root:
                    bias = 0.33
                elif '0.4' in self.root:
                    bias = 0.4
                elif '0.5' in self.root:
                    bias = 0.5
                elif '0.6' in self.root:
                    bias = 0.6
                elif '0.7' in self.root:
                    bias = 0.7
                elif '0.8' in self.root:
                    bias = 0.8
                elif '0.9' in self.root:
                    bias = 0.9
                possible_labels = [0, 1, 2] 
                probs = [0.33, 0.33, 1 - 0.66] 
                if self.mode == 'train':
                    base_num = np.random.choice([0, 1], p=[1 - bias, bias])
                    if base_num == 1:
                        x[:, :] = y
                    else:
                        possible_labels.pop(y)
                        base_num = np.random.choice(possible_labels, p=[0.5, 0.5])
                        x[:, :] = base_num
                else:
                    base_num = np.random.choice(possible_labels, p=probs)
                    x[:, :] = base_num
            else:
                x[:,z]=1
                # x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            try:
                if len(group) > 0:
                    group = -1
            except Exception as e:
                # print(f"warning {e}")
                pass
            data = Data(x=x,
                        y=y,
                        # z=torch.LongTensor([z]),  # node label
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        node_label=node_label,
                        # edge_gt_att=torch.LongTensor(ground_truth), # edge label
                        idx=torch.LongTensor([idx]),
                        mid=torch.LongTensor([mid]),
                        group=torch.LongTensor([group]))
            # add self loops
            # data.edge_index,data.edge_attr = add_self_loops(data.edge_index,data.edge_attr.squeeze(1))
            # data.edge_attr = data.edge_attr.unsqueeze(1)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        cnt, cnt2, cnt3 = {}, {}, {}
        for idx in range(len(data_list)):
            # assert torch.sum(data_list[idx].x-self.get(idx).x)==0
            kk = f"{data_list[idx].y.item()}-{data_list[idx].group.item()}"
            kk2 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}"
            cnt[kk] = cnt.get(kk,0)+1
            cnt2[kk2] = cnt2.get(kk2,0)+1
        print(cnt)
        print(cnt2)
        if get_data_list:
            return data_list
        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[idx])

    def add_self_loops(self):
        self.remove_self_loops()
        # print(torch.unique(self.data.edge_index).size())
        self.data.edge_index, self.data.edge_attr = add_self_loops(self.data.edge_index, self.data.edge_attr.squeeze(1))
        self.data.edge_attr = self.data.edge_attr.unsqueeze(1)

    def remove_self_loops(self):
        self.data.edge_index, self.data.edge_attr = remove_self_loops(self.data.edge_index, self.data.edge_attr)

    def rebalance_samples(self,is_pos,repeats=-1):
        num_pos = is_pos.sum().item()
        num_neg = is_pos.size(0)-num_pos
        print(f"Rebalancing")
        print(f"original #pos{num_pos} #neg{num_neg}")
        is_pos *= num_pos>num_neg
        num_pos = is_pos.sum().item()
        num_neg = is_pos.size(0)-num_pos
        num_repeats = min(num_pos//num_neg,2)
        if repeats>0:
            num_repeats = repeats+1
        # data_list = self.process(get_data_list=True)
        data_list = copy.deepcopy(self._data_list)
        neg_position = torch.nonzero(torch.logical_not(is_pos),as_tuple=True)[0].tolist()
        # neg_position = torch.nonzero(is_pos,as_tuple=True)[0].tolist()
        # assert len(neg_position)==num_neg
        # counting G^n
        cnt, cnt2, cnt3 = {}, {}, {}
        for idx in neg_position:
            assert torch.sum(data_list[idx].x-self.get(idx).x)==0
            kk = f"{data_list[idx].y.item()}-{data_list[idx].group.item()}"
            kk2 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}"
            kk3 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}-{data_list[idx].group.item()}"
            cnt[kk] = cnt.get(kk,0)+1
            cnt2[kk2] = cnt2.get(kk2,0)+1
            cnt3[kk3] = cnt3.get(kk3,0)+1
        print(cnt)
        print(cnt2)
        for i in range(3):
            for j in range(3):
                for k in range(1,4):
                    print(f"{i}-{j}-{k}:{cnt3.get(f'{i}-{j}-{k}',0)}",end=',')
            print()
        # counting G^p
        pos_position = torch.nonzero(is_pos,as_tuple=True)[0].tolist()
        cnt, cnt2, cnt3 = {}, {}, {}
        for idx in pos_position:
            assert torch.sum(data_list[idx].x-self.get(idx).x)==0
            kk = f"{data_list[idx].y.item()}-{data_list[idx].group.item()}"
            kk2 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}"
            kk3 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}-{data_list[idx].group.item()}"
            cnt[kk] = cnt.get(kk,0)+1
            cnt2[kk2] = cnt2.get(kk2,0)+1
            cnt3[kk3] = cnt3.get(kk3,0)+1
        print(cnt)
        print(cnt2)
        # print(cnt3)
        for i in range(3):
            for j in range(3):
                for k in range(1,4):
                    print(f"{i}-{j}-{k}:{cnt3.get(f'{i}-{j}-{k}',0)}",end=',')
            print()
        # exit()
        neg_data_list = [data_list[idx] for idx in neg_position]
        # for idx in range(len(neg_position)):
        #     if neg_data_list[idx].mid==0:
        #         neg_data_list[idx].y = 1
        #     elif neg_data_list[idx].mid==1:
        #         neg_data_list[idx].y = 0
        #     else:
        #         neg_data_list[idx].y = 2
        # data_list = neg_data_list*5 #neg_data_list*(num_repeats-1)+data_list
        data_list = neg_data_list*(num_repeats-1)+data_list
        super(SPMotif, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)
        # self.orig_data = copy.deepcopy(self._data)
        # self._data_list = data_list
        # self._data = self.data
        # print(len(self._data_list))
        print(f"new #sum{len(data_list)} #pos{num_pos} #neg{num_neg+len(neg_data_list)*(num_repeats-1)}")
    def resume_samples(self):
        data_list = self.process(get_data_list=True)
        super(SPMotif, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)


class aug_SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, mode, aug, ratio, transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        self.root = root
        self.aug = aug
        self.ratio = ratio
        self.edgeadd = sample_EdgeAdding(pe=ratio)
        print(f"######################################## {root}")
        super(aug_SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('aug_{}_{}_SPMotif_{}.pt'.format(aug,ratio,mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])
        self.orig_data_list = None
        

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['aug_{}_{}_SPMotif_{}.pt'.format(self.aug, self.ratio, self.mode)]

    # def download(self):
    #     if not osp.exists(osp.join(self.raw_dir, 'raw', 'SPMotif_train.npy')):
    #         print("raw data of `SPMotif` doesn't exist, please redownload from our github.")
    #         raise FileNotFoundError

    def process(self, get_data_list=False):

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        if 'dspmotif' in self.root.lower():
            # edge_index_list, label_list, base_list,ground_truth_list, role_id_list, pos_list, motif_id_list
            edge_index_list, label_list, base_list, ground_truth_list, role_id_list, pos, motifs = np.load(osp.join(
                self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            
        elif 'tspmotif' in self.root.lower():
            edge_index_list, label_list,  ground_truth_list, role_id_list, pos = np.load(osp.join(
                self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            motifs = np.zeros(label_list.shape,dtype=int)-1
            base_list = np.zeros(label_list.shape,dtype=int)-1
        elif '0.6' in self.root or '0.7' in self.root:
            edge_index_list, label_list, base_list, ground_truth_list, pos, = np.load(osp.join(
            self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            role_id_list = np.zeros(label_list.shape,dtype=int)-1
            motifs = np.zeros(label_list.shape,dtype=int)-1
        else:
            edge_index_list, label_list,  ground_truth_list, role_id_list, pos = np.load(osp.join(
                self.raw_dir, self.raw_file_names[idx]),allow_pickle=True)
            motifs = np.zeros(label_list.shape,dtype=int)-1
            base_list = np.zeros(label_list.shape,dtype=int)-1

        data_list = []
        for idx, (edge_index, y, group,ground_truth, z, p, mid) in \
        enumerate(zip(edge_index_list, label_list, base_list, ground_truth_list, role_id_list, pos, motifs)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = edge_index.long()
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            node_label = torch.tensor(z, dtype=torch.float)
            node_label[node_label != 0] = 1
            if 'mspmotif' in self.root.lower() or 'pmspmotif' in self.root.lower():
                # additionally add the spuriously correlated node features
                bias = 0.5
                if '0.3' in self.root:
                    bias = 0.33
                elif '0.4' in self.root:
                    bias = 0.4
                elif '0.5' in self.root:
                    bias = 0.5
                elif '0.6' in self.root:
                    bias = 0.6
                elif '0.7' in self.root:
                    bias = 0.7
                elif '0.8' in self.root:
                    bias = 0.8
                elif '0.9' in self.root:
                    bias = 0.9
                possible_labels = [0, 1, 2] 
                probs = [0.33, 0.33, 1 - 0.66] 
                if self.mode == 'train':
                    base_num = np.random.choice([0, 1], p=[1 - bias, bias])
                    if base_num == 1:
                        x[:, :] = y
                    else:
                        possible_labels.pop(y)
                        base_num = np.random.choice(possible_labels, p=[0.5, 0.5])
                        x[:, :] = base_num
                else:
                    base_num = np.random.choice(possible_labels, p=probs)
                    x[:, :] = base_num
            else:
                x[:,z]=1
                # x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            try:
                if len(group) > 0:
                    group = -1
            except Exception as e:
                # print(f"warning {e}")
                pass
            data = Data(x=x,
                        y=y,
                        # z=torch.LongTensor([z]),  # node label
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        node_label=node_label,
                        # edge_gt_att=torch.LongTensor(ground_truth), # edge label
                        idx=torch.LongTensor([idx]),
                        mid=torch.LongTensor([mid]),
                        group=torch.LongTensor([group]))
            data = self.augment(data, aug=self.aug, ratio=self.ratio)
            # add self loops
            # data.edge_index,data.edge_attr = add_self_loops(data.edge_index,data.edge_attr.squeeze(1))
            # data.edge_attr = data.edge_attr.unsqueeze(1)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        cnt, cnt2, cnt3 = {}, {}, {}
        for idx in range(len(data_list)):
            # assert torch.sum(data_list[idx].x-self.get(idx).x)==0
            kk = f"{data_list[idx].y.item()}-{data_list[idx].group.item()}"
            kk2 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}"
            cnt[kk] = cnt.get(kk,0)+1
            cnt2[kk2] = cnt2.get(kk2,0)+1
        print(cnt)
        print(cnt2)
        if get_data_list:
            return data_list
        idx = self.processed_file_names.index('aug_{}_{}_SPMotif_{}.pt'.format(self.aug, self.ratio, self.mode))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[idx])

    def add_self_loops(self):
        self.remove_self_loops()
        # print(torch.unique(self.data.edge_index).size())
        self.data.edge_index, self.data.edge_attr = add_self_loops(self.data.edge_index, self.data.edge_attr.squeeze(1))
        self.data.edge_attr = self.data.edge_attr.unsqueeze(1)

    def remove_self_loops(self):
        self.data.edge_index, self.data.edge_attr = remove_self_loops(self.data.edge_index, self.data.edge_attr)

    def rebalance_samples(self,is_pos,repeats=-1):
        num_pos = is_pos.sum().item()
        num_neg = is_pos.size(0)-num_pos
        print(f"Rebalancing")
        print(f"original #pos{num_pos} #neg{num_neg}")
        is_pos *= num_pos>num_neg
        num_pos = is_pos.sum().item()
        num_neg = is_pos.size(0)-num_pos
        num_repeats = min(num_pos//num_neg,2)
        if repeats>0:
            num_repeats = repeats+1
        # data_list = self.process(get_data_list=True)
        data_list = copy.deepcopy(self._data_list)
        neg_position = torch.nonzero(torch.logical_not(is_pos),as_tuple=True)[0].tolist()
        # neg_position = torch.nonzero(is_pos,as_tuple=True)[0].tolist()
        # assert len(neg_position)==num_neg
        # counting G^n
        cnt, cnt2, cnt3 = {}, {}, {}
        for idx in neg_position:
            assert torch.sum(data_list[idx].x-self.get(idx).x)==0
            kk = f"{data_list[idx].y.item()}-{data_list[idx].group.item()}"
            kk2 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}"
            kk3 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}-{data_list[idx].group.item()}"
            cnt[kk] = cnt.get(kk,0)+1
            cnt2[kk2] = cnt2.get(kk2,0)+1
            cnt3[kk3] = cnt3.get(kk3,0)+1
        print(cnt)
        print(cnt2)
        for i in range(3):
            for j in range(3):
                for k in range(1,4):
                    print(f"{i}-{j}-{k}:{cnt3.get(f'{i}-{j}-{k}',0)}",end=',')
            print()
        # counting G^p
        pos_position = torch.nonzero(is_pos,as_tuple=True)[0].tolist()
        cnt, cnt2, cnt3 = {}, {}, {}
        for idx in pos_position:
            assert torch.sum(data_list[idx].x-self.get(idx).x)==0
            kk = f"{data_list[idx].y.item()}-{data_list[idx].group.item()}"
            kk2 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}"
            kk3 = f"{data_list[idx].y.item()}-{data_list[idx].mid.item()}-{data_list[idx].group.item()}"
            cnt[kk] = cnt.get(kk,0)+1
            cnt2[kk2] = cnt2.get(kk2,0)+1
            cnt3[kk3] = cnt3.get(kk3,0)+1
        print(cnt)
        print(cnt2)
        # print(cnt3)
        for i in range(3):
            for j in range(3):
                for k in range(1,4):
                    print(f"{i}-{j}-{k}:{cnt3.get(f'{i}-{j}-{k}',0)}",end=',')
            print()
        # exit()
        neg_data_list = [data_list[idx] for idx in neg_position]
        # for idx in range(len(neg_position)):
        #     if neg_data_list[idx].mid==0:
        #         neg_data_list[idx].y = 1
        #     elif neg_data_list[idx].mid==1:
        #         neg_data_list[idx].y = 0
        #     else:
        #         neg_data_list[idx].y = 2
        # data_list = neg_data_list*5 #neg_data_list*(num_repeats-1)+data_list
        data_list = neg_data_list*(num_repeats-1)+data_list
        super(SPMotif, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)
        # self.orig_data = copy.deepcopy(self._data)
        # self._data_list = data_list
        # self._data = self.data
        # print(len(self._data_list))
        print(f"new #sum{len(data_list)} #pos{num_pos} #neg{num_neg+len(neg_data_list)*(num_repeats-1)}")
    
    def resume_samples(self):
        data_list = self.process(get_data_list=True)
        super(SPMotif, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)

    def augment(self, new_data, aug='subgraph', ratio=0.4):
        if aug == 'nodedrop':
            data = drop_nodes(new_data, ratio)
        elif aug == 'subgraph':
            data = get_subgraph(new_data, ratio)
        elif aug == 'edgeadd':
            data = self.edgeadd.augment(new_data)
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
    edge_index = edge_index[:,mask]
    # relabel
    node_map ={old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.unique(edge_index))}
    new_edge_index = torch.tensor([[node_map[edge_index[0,i].item()], node_map[edge_index[1,i].item()]] for i in range(edge_index.size(1))],dtype=torch.long).t()


    try:
        data.edge_index = new_edge_index
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.node_label = data.node_label[idx_nondrop]
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
    input_adj[edge_index[0], edge_index[1]] = 1 # the edge_index is not symmetry
    input_adj[edge_index[1], edge_index[0]] = 1
    
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
    
    edge_index = data.edge_index
    edge_attr = data.edge_attr
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
    data.node_label = data.node_label[idx_nondrop]
    # print('over')

    return data

def edge_add(data,aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))
    
    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)
    edge_attr = torch.ones(data.edge_index.size(1), 1)
    data.edge_attr =edge_attr

    return data

class sample_EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(sample_EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g) :
        edge_index= g.edge_index
        edge_index = add_edge(edge_index, ratio=self.pe)
        g.edge_index=edge_index
        return g

def make_undirected(edge_index):
    # 将每条边的两个方向都添加到边索引中
    undirected_edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    # 去除重复的边索引
    undirected_edge_index, _ = undirected_edge_index.unique(dim=1, return_inverse=True)
    
    return undirected_edge_index


def add_edge(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    num_edges = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)

    edge_index = make_undirected(edge_index)

    return edge_index