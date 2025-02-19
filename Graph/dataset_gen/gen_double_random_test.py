from BA3_loc import *
import torch
import numpy as np

import random
from texttable import Texttable


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(0)
global_b = '0.33'
test_bias = 0.33
label_noise = 0
motif_names = ['house','cycle','crane','diamond']
# house crane is label 0, cycle diamond is label 1
num_train_perc = 3000
num_val_perc = 1000
num_test_perc = 1000
size_bias = True # test is smaller


def get_house(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(width_basis,
                                                    basis_type,
                                                    list_shapes,
                                                    start=0,
                                                    rdm_basis_plugins=True)
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_cycle(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["dircycle"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(width_basis,
                                                    basis_type,
                                                    list_shapes,
                                                    start=0,
                                                    rdm_basis_plugins=True)
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_crane(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["crane"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(width_basis,
                                                    basis_type,
                                                    list_shapes,
                                                    start=0,
                                                    rdm_basis_plugins=True)
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name



def get_diamond(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'diamond') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["diamond"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(width_basis,
                                                    basis_type,
                                                    list_shapes,
                                                    start=0,
                                                    rdm_basis_plugins=True)
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


import random

from tqdm import tqdm

edge_index_list = []
label_list = []
ground_truth_list = []
role_id_list = []
pos_list = []
base_list = []

bias = float(global_b)
e_mean = []
n_mean = []
motif_id_list = []
cnt_cau_spu = 0
for _ in tqdm(range(num_train_perc)):
    motif_id = 0
    base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, bias, (1 - bias) / 2])

    if base_num == 1:
        base = 'tree'
        width_basis = np.random.choice(range(3, 4))
    if base_num == 2:
        base = 'ladder'
        width_basis = np.random.choice(range(8, 12))
    if base_num == 3:
        base = 'wheel'
        width_basis = np.random.choice(range(15, 20))
    if random.random() < label_noise:
        motif_id = random.randint(0, 3)
    new_motif_id = random.randint(0,1)*2+motif_id%2
    assert new_motif_id%2==motif_id%2
    motif_id = new_motif_id
    gen_name = f"get_{motif_names[motif_id]}"
    if motif_id==0 and base_num==2:
        cnt_cau_spu+=1
    G, role_id, name = eval(gen_name)(basis_type=base,
                                 nb_shapes=1,
                                 width_basis=width_basis,
                                 feature_generator=None,
                                 m=3,
                                 draw=False)
    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T
    row, col = edge_index
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))
    edge_index_list.append(edge_index)
    label_list.append(motif_id%2)
    base_list.append(base_num)
    ground_truth = find_gd(edge_index, role_id)
    # print(ground_truth)
    # exit()
    ground_truth_list.append(ground_truth)
    role_id_list.append(role_id)
    pos = nx.spring_layout(G)
    pos_list.append(pos)
    motif_id_list.append(motif_id)
print(np.mean(n_mean), np.mean(e_mean))
print(len(ground_truth_list))
print(cnt_cau_spu/num_train_perc)
# exit()

e_mean = []
n_mean = []

for _ in tqdm(range(num_train_perc)):
    motif_id = 1
    base_num = np.random.choice([1, 2, 3], p=[bias, (1 - bias) / 2, (1 - bias) / 2])

    if base_num == 1:
        base = 'tree'
        width_basis = np.random.choice(range(3, 4))
    if base_num == 2:
        base = 'ladder'
        width_basis = np.random.choice(range(8, 12))
    if base_num == 3:
        base = 'wheel'
        width_basis = np.random.choice(range(15, 20))
    if random.random() < label_noise:
        motif_id = random.randint(0, 3)
    new_motif_id = random.randint(0,1)*2+motif_id%2
    assert new_motif_id%2==motif_id%2
    motif_id = new_motif_id
    gen_name = f"get_{motif_names[motif_id]}"

    G, role_id, name = eval(gen_name)(basis_type=base,
                                 nb_shapes=1,
                                 width_basis=width_basis,
                                 feature_generator=None,
                                 m=3,
                                 draw=False)
    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T
    row, col = edge_index
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))
    edge_index_list.append(edge_index)
    label_list.append(motif_id%2)
    base_list.append(base_num)
    ground_truth = find_gd(edge_index, role_id)
    ground_truth_list.append(ground_truth)
    role_id_list.append(role_id)
    pos = nx.spring_layout(G)
    pos_list.append(pos)
    motif_id_list.append(motif_id)
print(np.mean(n_mean), np.mean(e_mean))
print(len(ground_truth_list))


# e_mean = []
# n_mean = []

# for _ in tqdm(range(num_train_perc)):
#     motif_id = 2
#     base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, (1 - bias) / 2, bias])

#     if base_num == 1:
#         base = 'tree'
#         width_basis = np.random.choice(range(3, 4))
#     if base_num == 2:
#         base = 'ladder'
#         width_basis = np.random.choice(range(8, 12))
#     if base_num == 3:
#         base = 'wheel'
#         width_basis = np.random.choice(range(15, 20))
#     if random.random() < label_noise:
#         motif_id = random.randint(0, 3)
#     new_motif_id = random.randint(0,1)*2+motif_id%2
    # assert new_motif_id%2==motif_id%2
    # motif_id = new_motif_id
#     gen_name = f"get_{motif_names[motif_id]}"

#     G, role_id, name = eval(gen_name)(basis_type=base,
#                                  nb_shapes=1,
#                                  width_basis=width_basis,
#                                  feature_generator=None,
#                                  m=3,
#                                  draw=False)
#     role_id = np.array(role_id)
#     edge_index = np.array(G.edges, dtype=np.int).T
#     row, col = edge_index
#     e_mean.append(len(G.edges))
#     n_mean.append(len(G.nodes))
#     edge_index_list.append(edge_index)
#     label_list.append(motif_id%2)
#     base_list.append(base_num)
#     ground_truth = find_gd(edge_index, role_id)
#     ground_truth_list.append(ground_truth)
#     role_id_list.append(role_id)
#     pos = nx.spring_layout(G)
#     pos_list.append(pos)
#     motif_id_list.append(motif_id)
# print(np.mean(n_mean), np.mean(e_mean))
# print(len(ground_truth_list))


if not os.path.exists(f'../data/wSPMotif-{global_b}/'):
    os.mkdir(f'../data/wSPMotif-{global_b}/')
if not os.path.exists(f'../data/wSPMotif-{global_b}/raw'):
    os.mkdir(f'../data/wSPMotif-{global_b}/raw')
np.save(f'../data/wSPMotif-{global_b}/raw/train.npy',
        (edge_index_list, label_list, base_list,ground_truth_list, role_id_list, pos_list, motif_id_list))

import random

from tqdm import tqdm

edge_index_list = []
label_list = []
ground_truth_list = []
role_id_list = []
pos_list = []
base_list = []
motif_id_list = []
bias = max(float(global_b)-0.2,1.0 / 3)
e_mean = []
n_mean = []

for _ in tqdm(range(num_val_perc)):
    motif_id = 1
    base_num = np.random.choice([1, 2, 3], p=[bias, (1 - bias) / 2, (1 - bias) / 2])

    if base_num == 1:
        base = 'tree'
        width_basis = np.random.choice(range(3, 4))
    if base_num == 2:
        base = 'ladder'
        width_basis = np.random.choice(range(8, 12))
    if base_num == 3:
        base = 'wheel'
        width_basis = np.random.choice(range(15, 20))

    if random.random() < label_noise:
        motif_id = random.randint(0, 3)
    new_motif_id = random.randint(0,1)*2+motif_id%2
    assert new_motif_id%2==motif_id%2
    motif_id = new_motif_id
    gen_name = f"get_{motif_names[motif_id]}"
    G, role_id, name = eval(gen_name)(basis_type=base,
                                 nb_shapes=1,
                                 width_basis=width_basis,
                                 feature_generator=None,
                                 m=3,
                                 draw=False)
    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T
    row, col = edge_index
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))
    edge_index_list.append(edge_index)
    label_list.append(motif_id%2)
    base_list.append(base_num)
    ground_truth = find_gd(edge_index, role_id)
    ground_truth_list.append(ground_truth)
    role_id_list.append(role_id)
    pos = nx.spring_layout(G)
    pos_list.append(pos)
    motif_id_list.append(motif_id)
print(np.mean(n_mean), np.mean(e_mean))
print(len(ground_truth_list))

e_mean = []
n_mean = []

for _ in tqdm(range(num_val_perc)):
    motif_id=0
    base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, bias, (1 - bias) / 2])

    if base_num == 1:
        base = 'tree'
        width_basis = np.random.choice(range(3, 4))
    if base_num == 2:
        base = 'ladder'
        width_basis = np.random.choice(range(8, 12))
    if base_num == 3:
        base = 'wheel'
        width_basis = np.random.choice(range(15, 20))

    if random.random() < label_noise:
        motif_id = random.randint(0, 3)
    new_motif_id = random.randint(0,1)*2+motif_id%2
    assert new_motif_id%2==motif_id%2
    motif_id = new_motif_id
    gen_name = f"get_{motif_names[motif_id]}"
    G, role_id, name = eval(gen_name)(basis_type=base,
                                 nb_shapes=1,
                                 width_basis=width_basis,
                                 feature_generator=None,
                                 m=3,
                                 draw=False)
    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T
    row, col = edge_index
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))
    edge_index_list.append(edge_index)
    label_list.append(motif_id%2)
    base_list.append(base_num)
    ground_truth = find_gd(edge_index, role_id)
    ground_truth_list.append(ground_truth)
    role_id_list.append(role_id)
    pos = nx.spring_layout(G)
    pos_list.append(pos)
    motif_id_list.append(motif_id)
print(np.mean(n_mean), np.mean(e_mean))
print(len(ground_truth_list))

# e_mean = []
# n_mean = []

# for _ in tqdm(range(num_val_perc)):
#     motif_id=2
#     base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, (1 - bias) / 2, bias])

#     if base_num == 1:
#         base = 'tree'
#         width_basis = np.random.choice(range(3, 4))
#     if base_num == 2:
#         base = 'ladder'
#         width_basis = np.random.choice(range(8, 12))
#     if base_num == 3:
#         base = 'wheel'
#         width_basis = np.random.choice(range(15, 20))

#     if random.random() < label_noise:
#         motif_id = random.randint(0, 3)
#     new_motif_id = random.randint(0,1)*2+motif_id%2
    # assert new_motif_id%2==motif_id%2
    # motif_id = new_motif_id
#     gen_name = f"get_{motif_names[motif_id]}"
#     G, role_id, name = eval(gen_name)(basis_type=base,
#                                  nb_shapes=1,
#                                  width_basis=width_basis,
#                                  feature_generator=None,
#                                  m=3,
#                                  draw=False)
#     role_id = np.array(role_id)
#     edge_index = np.array(G.edges, dtype=np.int).T
#     row, col = edge_index
#     e_mean.append(len(G.edges))
#     n_mean.append(len(G.nodes))
#     edge_index_list.append(edge_index)
#     label_list.append(motif_id%2)
#     base_list.append(base_num)
#     ground_truth = find_gd(edge_index, role_id)
#     ground_truth_list.append(ground_truth)
#     role_id_list.append(role_id)
#     pos = nx.spring_layout(G)
#     pos_list.append(pos)
#     motif_id_list.append(motif_id)
# print(np.mean(n_mean), np.mean(e_mean))
# print(len(ground_truth_list))
np.save(f'../data/wSPMotif-{global_b}/raw/val.npy',
        (edge_index_list, label_list, base_list,ground_truth_list, role_id_list, pos_list, motif_id_list))

import random

from tqdm import tqdm

edge_index_list = []
label_list = []
ground_truth_list = []
role_id_list = []
pos_list = []
base_list = []
motif_id_list = []
bias = test_bias
e_mean = []
n_mean = []

for _ in tqdm(range(num_test_perc)):
    motif_id=1
    base_num = np.random.choice([1, 2, 3])
    
    if size_bias:
        if base_num == 1:
            base = 'tree'
            width_basis = np.random.choice(range(3, 6))
        if base_num == 2:
            base = 'ladder'
            width_basis = np.random.choice(range(30, 50))
        if base_num == 3:
            base = 'wheel'
            width_basis = np.random.choice(range(60, 80))
    else:
        if base_num == 1:
            base = 'tree'
            width_basis = np.random.choice(range(3, 4))
        if base_num == 2:
            base = 'ladder'
            width_basis = np.random.choice(range(8, 12))
        if base_num == 3:
            base = 'wheel'
            width_basis = np.random.choice(range(15, 20))

    if random.random() < label_noise:
        motif_id = random.randint(0, 3)
    new_motif_id = random.randint(0,1)*2+motif_id%2
    assert new_motif_id%2==motif_id%2
    motif_id = new_motif_id
    # print(motif_id)
    gen_name = f"get_{motif_names[motif_id]}"
    G, role_id, name = eval(gen_name)(basis_type=base,
                                 nb_shapes=1,
                                 width_basis=width_basis,
                                 feature_generator=None,
                                 m=3,
                                 draw=False)
    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T
    row, col = edge_index
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))
    edge_index_list.append(edge_index)
    label_list.append(motif_id%2)
    base_list.append(base_num)
    ground_truth = find_gd(edge_index, role_id)
    ground_truth_list.append(ground_truth)
    role_id_list.append(role_id)
    pos = nx.spring_layout(G)
    pos_list.append(pos)
    motif_id_list.append(motif_id)
print(np.mean(n_mean), np.mean(e_mean))
print(len(ground_truth_list))
print(motif_id_list[:10])
e_mean = []
n_mean = []

for _ in tqdm(range(num_test_perc)):
    motif_id=0
    base_num = np.random.choice([1, 2, 3])
    
    if size_bias:
        if base_num == 1:
            base = 'tree'
            width_basis = np.random.choice(range(3, 6))
        if base_num == 2:
            base = 'ladder'
            width_basis = np.random.choice(range(30, 50))
        if base_num == 3:
            base = 'wheel'
            width_basis = np.random.choice(range(60, 80))
    else:
        if base_num == 1:
            base = 'tree'
            width_basis = np.random.choice(range(3, 4))
        if base_num == 2:
            base = 'ladder'
            width_basis = np.random.choice(range(8, 12))
        if base_num == 3:
            base = 'wheel'
            width_basis = np.random.choice(range(15, 20))

    if random.random() < label_noise:
        motif_id = random.randint(0, 3)
    new_motif_id = random.randint(0,1)*2+motif_id%2
    assert new_motif_id%2==motif_id%2
    motif_id = new_motif_id
    gen_name = f"get_{motif_names[motif_id]}"
    G, role_id, name = eval(gen_name)(basis_type=base,
                                 nb_shapes=1,
                                 width_basis=width_basis,
                                 feature_generator=None,
                                 m=3,
                                 draw=False)
    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T
    row, col = edge_index
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))
    edge_index_list.append(edge_index)
    label_list.append(motif_id%2)
    base_list.append(base_num)
    ground_truth = find_gd(edge_index, role_id)
    ground_truth_list.append(ground_truth)
    role_id_list.append(role_id)
    pos = nx.spring_layout(G)
    pos_list.append(pos)
    motif_id_list.append(motif_id)
print(np.mean(n_mean), np.mean(e_mean))
print(len(ground_truth_list))

# e_mean = []
# n_mean = []

# for _ in tqdm(range(num_test_perc)):
#     motif_id=2
#     base_num = np.random.choice([1, 2, 3], p=[(1 - bias) / 2, (1 - bias) / 2, bias])
    
#     if size_bias:
#         if base_num == 1:
#             base = 'tree'
#             width_basis = np.random.choice(range(3, 6))
#         if base_num == 2:
#             base = 'ladder'
#             width_basis = np.random.choice(range(30, 50))
#         if base_num == 3:
#             base = 'wheel'
#             width_basis = np.random.choice(range(60, 80))
#     else:
#         if base_num == 1:
#             base = 'tree'
#             width_basis = np.random.choice(range(3, 4))
#         if base_num == 2:
#             base = 'ladder'
#             width_basis = np.random.choice(range(8, 12))
#         if base_num == 3:
#             base = 'wheel'
#             width_basis = np.random.choice(range(15, 20))

#     if random.random() < label_noise:
#         motif_id = random.randint(0, 3)
#     new_motif_id = random.randint(0,1)*2+motif_id%2
#     assert new_motif_id%2==motif_id%2
#     motif_id = new_motif_id
#     gen_name = f"get_{motif_names[motif_id]}"
#     G, role_id, name = eval(gen_name)(basis_type=base,
#                                  nb_shapes=1,
#                                  width_basis=width_basis,
#                                  feature_generator=None,
#                                  m=3,
#                                  draw=False)
#     role_id = np.array(role_id)
#     edge_index = np.array(G.edges, dtype=np.int).T
#     row, col = edge_index
#     e_mean.append(len(G.edges))
#     n_mean.append(len(G.nodes))
#     edge_index_list.append(edge_index)
#     label_list.append(motif_id%2)
#     base_list.append(base_num)
#     ground_truth = find_gd(edge_index, role_id)
#     ground_truth_list.append(ground_truth)
#     role_id_list.append(role_id)
#     pos = nx.spring_layout(G)
#     pos_list.append(pos)
#     motif_id_list.append(motif_id)
# print(np.mean(n_mean), np.mean(e_mean))
# print(len(ground_truth_list))

print(motif_id_list[:10])
np.save(f'../data/wSPMotif-{global_b}/raw/test.npy',
        (edge_index_list, label_list, base_list,ground_truth_list, role_id_list, pos_list, motif_id_list))
