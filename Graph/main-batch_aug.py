import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from sklearn.metrics import matthews_corrcoef
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

# from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from models.gnn_ib import GIB
from models.ciga import GNNERM, CIGA, GNNPooling
from models.ssl_encoder import Encoder
from models.losses_aug import get_contrast_loss, get_irm_loss, get_ssl_loss,get_ClassCondition_ssl_loss
from utils.logger import Logger
from utils.util import args_print, set_seed
from main_complement import parse_arguments, select_model, augment, NC_eval_model, eval_model

# ETF
from utils.methods import dot_loss
import wandb

# ssl
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
# from utils.aug import aug_drop_node,aug_random_edge,aug_random_mask,aug_subgraph,delete_row_col
import torch_geometric.data.batch as DataBatch
# from utils.aug import RWSampling
from utils.cNCE import cNCEContrast

# resample
from sklearn.utils import resample



def main():
    args = parse_arguments()
    erm_model = None  # used to obtain pesudo labels for CNC sampling in contrastive loss
    args.seed = eval(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    # device = torch.device("cpu")
    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)
    criterion = ce_loss

    if args.cNCE:
        print('----  class condition contrastive loss is adopted ----')
    if args.a_w == 0:
        args.a_w =args.contrast
        print('---- no special augCL weight ----')
    
    
    eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
    edge_dim = -1.

    ### automatic dataloading and splitting
    if args.dataset.lower().startswith('drugood'):
        # drugood_lbap_core_ic50_assay.json
        config_path = os.path.join("configs", args.dataset + ".py")
        cfg = Config.fromfile(config_path)
        root = os.path.join(args.root, "DrugOOD")
        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train")
        # 上采样 class 0
        if args.up:
            # 分割样本为 class 0 和 class 1
            class_0_samples = [sample for sample in train_dataset if sample['y'] == 0]
            # print(len(class_0_samples))
            class_1_samples = [sample for sample in train_dataset if sample['y'] == 1]
            # print(len(class_1_samples))
            assert len(class_0_samples) <= len(class_1_samples)
            class_0_samples_upsampled = resample(class_0_samples, replace=True, n_samples=len(class_1_samples), random_state=42)
            # 合并上采样后的 class 0 样本和原来的 class 1 样本
            # print(len(class_0_samples_upsampled))
            # print(len(class_1_samples))
            upsampled_dataset = class_0_samples_upsampled + class_1_samples
            train_loader = DataLoader(upsampled_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args.dataset, mode="ood_val")
        test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args.dataset, mode="ood_test")
        if args.eval_metric == 'auc':
            evaluator = Evaluator('ogbg-molhiv')
            eval_metric = args.eval_metric = 'rocauc'
        else:
            evaluator = Evaluator('ogbg-ppa')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        # 上采样 class 0
        if args.up:
            class_0_samples = [sample for sample in train_dataset if sample['y'] == 0]
            class_1_samples = [sample for sample in train_dataset if sample['y'] == 1]
            assert len(class_0_samples) <= len(class_1_samples)
            class_0_samples_upsampled = resample(class_0_samples, replace=True, n_samples=len(class_1_samples), random_state=42)
            upsampled_dataset = class_0_samples_upsampled + class_1_samples
            train_loader = DataLoader(upsampled_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = train_loader
        input_dim = 39
        edge_dim = 10
        num_classes = 2
    elif args.dataset.lower().startswith('ogbg'):

        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data

        if 'ppa' in args.dataset.lower():
            dataset = PygGraphPropPredDataset(root=args.root, name=args.dataset, transform=add_zeros)
            input_dim = -1
            num_classes = dataset.num_classes
        else:
            dataset = PygGraphPropPredDataset(root=args.root, name=args.dataset)
            input_dim = 1
            num_classes = dataset.num_tasks
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            print('using simple feature')
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
        split_idx = dataset.get_idx_split()
        ### automatic evaluator. takes dataset name as input
        evaluator = Evaluator(args.dataset)
        # evaluator = Evaluator('ogbg-ppa')

        train_loader = DataLoader(dataset[split_idx["train"]],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]],
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

        if 'classification' in dataset.task_type:

            def cls_loss(a, b, reduction='mean'):
                return F.binary_cross_entropy_with_logits(a.float(), b.float(), reduction=reduction)

            criterion = cls_loss
        else:

            def mse_loss(a, b, reduction='mean'):
                return F.mse_loss(a.float(), b.float(), reduction=reduction)

            criterion = mse_loss

        eval_metric = dataset.eval_metric
    elif args.dataset.lower() in ['spmotif', 'mspmotif','tspmotif','dspmotif']:
        train_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='train')
        val_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='val')
        test_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='test')
        input_dim = 4
        num_classes = 3
        evaluator = Evaluator('ogbg-ppa')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset.lower() in ['graph-sst5','graph-sst2']:
        dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
        dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=True, seed=args.seed)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        input_dim = 768
        num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['graph-twitter']:
        dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
        dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=False, seed=args.seed)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        input_dim = 768
        num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['cmnist']:
        n_val_data = 5000
        train_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='train')
        test_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='test')
        perm_idx = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(0))
        test_val = test_dataset[perm_idx]
        val_dataset, test_dataset = test_val[:n_val_data], test_val[n_val_data:]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = 7
        num_classes = 2
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['proteins', 'dd', 'nci1', 'nci109']:
        dataset = TUDataset(os.path.join(args.root, "TU"), name=args.dataset.upper())
        train_idx = np.loadtxt(os.path.join(args.root, "TU", args.dataset.upper(), 'train_idx.txt'), dtype=np.int64)
        val_idx = np.loadtxt(os.path.join(args.root, "TU", args.dataset.upper(), 'val_idx.txt'), dtype=np.int64)
        test_idx = np.loadtxt(os.path.join(args.root, "TU", args.dataset.upper(), 'test_idx.txt'), dtype=np.int64)

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        input_dim = dataset[0].x.size(1)
        num_classes = dataset.num_classes
        evaluator = Evaluator('ogbg-ppa')
    else:
        raise Exception("Invalid dataset name")

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'{datetime_now[4::]}-{args.dataset}-{args.bias}_{args.ginv_opt}_erm{args.erm}_dir{args.dir}_coes{args.contrast}-{args.spu_coe}_seed{args.seed}_{datetime_now}'
    # experiment_name = f'{datetime_now[4::]}'
    exp_dir = os.path.join('/home/kolerk/CIGA_NC_v1/logs/', experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    logger.info(f"Using criterion {criterion}")
    logger.info(f"# Train: {len(train_loader.dataset)}  #Val: {len(valid_loader.dataset)} #Test: {len(test_loader.dataset)} ")
    best_weights = None

    all_info = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
    }
    for seed in args.seed:
        set_seed(seed)
        all_info_seed = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
        }
        # name the experiment
        name=f'{datetime_now[4::]}-{args.dataset}_{args.erm}_givopt{args.ginv_opt}_dir{args.dir}_irm{args.irm_opt}_contrastsample{args.contrast_sampling}_{seed}'
        if args.dataset.lower()=='spmotif':
            # wandb_proj = f'{args.dataset}_{args.bias}_sum_DimC'
            wandb_proj = f'{args.dataset}_{args.bias}_complemente'
        else:
            wandb_proj=f'{args.dataset}_sum_DimC'
        # wandb init
        wandb.init(
        # Set the project where this run will be logged
        project=wandb_proj, 
        # We pass a run name
        name=name, 
        # Track hyperparameters and run metadata
        config=vars(args))
        # models and optimizers, only add ETF in args.erm and ciga
        model = select_model(args, input_dim=input_dim, edge_dim= edge_dim, num_classes=num_classes, device=device)

        if args.ssl > 0:
            aug1, aug2 = augment(args)
            encoder_model = Encoder(encoder=model, augmentor=(aug1, aug2), emb_dim = args.emb_dim).to(device)
            if args.cNCE:
                contrast_model = cNCEContrast(loss=L.InfoNCE(tau=args.ssl_t),mode='G2G').to(device)
            else:
                contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=args.ssl_t), mode='G2G').to(device)
           
            model_optimizer = torch.optim.Adam(list(model.parameters())+list(encoder_model.parameters()), lr=args.lr)
        else:
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
                
        print(model)
        last_train_acc, last_test_acc, last_val_acc = 0, 0, 0
        cnt = 0
        best_index = 0
        # generate environment partitions
        if args.num_envs > 1:
            env_idx = (torch.sigmoid(torch.randn(len(train_loader.dataset))) > 0.5).long()
            print(f"num env 0: {sum(env_idx == 0)} num env 1: {sum(env_idx == 1)}")

        for epoch in range(args.epoch):
            # for epoch in tqdm(range(args.epoch)):
            all_loss, n_bw = 0, 0
            all_losses = {}
            contrast_loss, all_contrast_loss = torch.zeros(1).to(device), 0. 
            # ssl loss
            ssl_loss, all_ssl_loss = torch.zeros(1).to(device), 0.
            spu_pred_loss = torch.zeros(1).to(device)
            model.train()
            torch.autograd.set_detect_anomaly(True)
            num_batch = (len(train_loader.dataset) // args.batch_size) + int(
                (len(train_loader.dataset) % args.batch_size) > 0)
            # batch processing 
            for step, graph in enumerate(train_loader):
            # for step, graph in tqdm(enumerate(train_loader), total=num_batch, desc=f"Epoch [{epoch}] >>  ", disable=args.no_tqdm, ncols=60):
                n_bw += 1
                graph.to(device)
                # ignore nan targets
                # https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py
                is_labeled = graph.y == graph.y    
                # ce loss
                if args.dir > 0:
                    # obtain dir losses
                    dir_loss, causal_pred, spu_pred, causal_rep = model.get_dir_loss(graph,
                                                                                    graph.y,
                                                                                    criterion,
                                                                                    is_labeled=is_labeled,
                                                                                    return_data='rep')
                    spu_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled])
                    pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                    pred_loss = pred_loss + spu_loss + args.dir * (epoch ** 1.6) * dir_loss
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    all_losses['dir'] = (all_losses.get('dir', 0) * (n_bw - 1) + dir_loss.item()) / n_bw
                    all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_loss.item()) / n_bw
                elif args.ginv_opt.lower() == 'gib':
                    # obtain gib loss
                    pred_loss, causal_rep = model.get_ib_loss(graph, return_data="rep")
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                else:
                    # obtain ciga I(G_S;Y) losses
                    if args.spu_coe > 0 and not args.erm:
                        if args.contrast_rep.lower() == "feat":
                            (causal_pred, spu_pred), causal_rep = model(graph, return_data="feat", return_spu=True)
                        else:
                            (causal_pred, spu_pred), causal_rep = model(graph, return_data="rep", return_spu=True)

                        spu_pred_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled], reduction='none')
                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled], reduction='none')
                        assert spu_pred_loss.size() == pred_loss.size()
                        # hinge loss
                        spu_loss_weight = torch.zeros(spu_pred_loss.size()).to(device)
                        spu_loss_weight[spu_pred_loss > pred_loss] = 1.0
                        spu_pred_loss = spu_pred_loss.dot(spu_loss_weight) / (sum(spu_pred_loss > pred_loss) + 1e-6)
                        pred_loss = pred_loss.mean()
                        all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_pred_loss.item()) / n_bw
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    else:
                    
                        if args.contrast_rep.lower() == "feat":
                            # causal_pred, causal_rep = model(graph, epoch, return_data="feat")
                            causal_pred, causal_rep = model(graph, return_data="feat")
                        else:
                            # causal_pred, causal_rep = model(graph, epoch, return_data="rep")
                            causal_pred, causal_rep = model(graph, return_data="rep")
                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                
                # contrast loss
                contrast_loss = 0
                contrast_coe = args.contrast
                if args.contrast > 0 :
                    # obtain contrast loss
                    if args.contrast_sampling.lower() in ['cnc', 'cncp']:
                        # cncp referes to only contrastig the positive examples in cnc
                        if erm_model == None:
                            model_path = "/home/kolerk/CIGA_NC_latest/model_SPMotif_0.60/erm_model_SPMotif.pt" #os.path.join('erm_model', args.dataset) + ".pt"
                            erm_model = GNNERM(input_dim=input_dim,
                                            edge_dim=edge_dim,
                                            out_dim=num_classes,
                                            gnn_type=args.model,
                                            num_layers=args.num_layers,
                                            emb_dim=args.emb_dim,
                                            drop_ratio=args.dropout,
                                            graph_pooling=args.pooling,
                                            virtual_node=args.virtual_node).to(device)
                            erm_model.load_state_dict(torch.load(model_path, map_location=device))
                            print("Loaded model from ", model_path)
                        # obtain the erm predictions to sampling pos/neg pairs in cnc
                        erm_model.eval()
                        with torch.no_grad():
                            erm_y_pred = erm_model(graph)
                        erm_y_pred = erm_y_pred.argmax(-1)
                    else:
                        erm_y_pred = None
                    # contrastive loss implementation
                    contrast_loss = get_contrast_loss(causal_rep,
                                                    graph.y.view(-1),
                                                    norm=F.normalize if not args.not_norm else None,
                                                    contrast_t=args.contrast_t,
                                                    sampling=args.contrast_sampling,
                                                    contrast_d = args.contrast_d,
                                                    y_pred=erm_y_pred)
                    all_losses['contrast'] = (all_losses.get('contrast', 0) * (n_bw - 1) + contrast_loss.item()) / n_bw
                    all_contrast_loss += contrast_loss.item()
                
                # ssl loss
                ssl_loss =0
                ssl_coe = args.ssl
                if args.ssl > 0 : # pretrain
                    if args.cNCE:
                        if args.contrast_rep.lower() == "feat":
                            ssl_loss = get_ClassCondition_ssl_loss(encoder_model, contrast_model, graph, args, epoch, "feat")
                        else:
                            ssl_loss = get_ClassCondition_ssl_loss(encoder_model, contrast_model, graph, args, epoch, "rep")
                    else:
                        if args.contrast_rep.lower() == "feat":
                            ssl_loss = get_ssl_loss(encoder_model, contrast_model, graph, args, epoch, "feat",args.ssl_d)
                        else:
                            ssl_loss = get_ssl_loss(encoder_model, contrast_model, graph, args, epoch, "rep",args.ssl_d)
                    all_losses['ssl'] = (all_losses.get('ssl', 0) * (n_bw - 1) + ssl_loss.item()) / n_bw
                    all_ssl_loss += ssl_loss.item()

                if args.num_envs > 1:
                    # indicate invariant learning
                    batch_env_idx = env_idx[step * args.batch_size:step * args.batch_size + graph.y.size(0)]
                    if 'molhiv' in args.dataset.lower():
                        batch_env_idx = batch_env_idx.view(graph.y.shape)
                    causal_pred, labels, batch_env_idx = causal_pred[is_labeled], graph.y[is_labeled], batch_env_idx[
                        is_labeled]
                    if args.irm_opt.lower() == 'eiil':
                        dummy_w = torch.tensor(1.).to(device).requires_grad_()
                        loss = F.nll_loss(causal_pred * dummy_w, labels, reduction='none')
                        env_w = torch.randn(batch_env_idx.size(0)).cuda().requires_grad_()
                        optimizer = torch.optim.Adam([env_w], lr=1e-3)
                        for i in range(20):
                            # penalty for env a
                            lossa = (loss.squeeze() * env_w.sigmoid()).mean()
                            grada = torch.autograd.grad(lossa, [dummy_w], create_graph=True)[0]
                            penaltya = torch.sum(grada ** 2)
                            # penalty for env b
                            lossb = (loss.squeeze() * (1 - env_w.sigmoid())).mean()
                            gradb = torch.autograd.grad(lossb, [dummy_w], create_graph=True)[0]
                            penaltyb = torch.sum(gradb ** 2)
                            # negate
                            npenalty = -torch.stack([penaltya, penaltyb]).mean()
                            # step
                            optimizer.zero_grad()
                            npenalty.backward(retain_graph=True)
                            optimizer.step()
                        new_batch_env_idx = (env_w.sigmoid() > 0.5).long()
                        env_idx[step * args.batch_size:step * args.batch_size +
                                                    graph.y.size(0)][labels] = new_batch_env_idx.to(env_idx.device)
                        irm_loss = get_irm_loss(causal_pred, labels, new_batch_env_idx, criterion=criterion)
                    elif args.irm_opt.lower() == 'ib-irm':
                        ib_penalty = causal_rep.var(dim=0).mean()
                        irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx,
                                                criterion=criterion) + ib_penalty / args.irm_p
                        all_losses['ib'] = (all_losses.get('ib', 0) * (n_bw - 1) + ib_penalty.item()) / n_bw
                    elif args.irm_opt.lower() == 'vrex':
                        loss_0 = criterion(causal_pred[batch_env_idx == 0], labels[batch_env_idx == 0])
                        loss_1 = criterion(causal_pred[batch_env_idx == 1], labels[batch_env_idx == 1])
                        irm_loss = torch.var(torch.FloatTensor([loss_0, loss_1]).to(device))
                    else:
                        irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx, criterion=criterion)
                    all_losses['irm'] = (all_losses.get('irm', 0) * (n_bw - 1) + irm_loss.item()) / n_bw
                    pred_loss += irm_loss * args.irm_p
                
                batch_loss = pred_loss + contrast_coe * contrast_loss + ssl_coe * ssl_loss + args.spu_coe * spu_pred_loss
                model_optimizer.zero_grad()
                batch_loss.backward()
                model_optimizer.step()
                all_loss += batch_loss.item()
            all_contrast_loss /= n_bw
            all_ssl_loss /= n_bw
            all_loss /= n_bw
            # print(      "\n       all_loss: {:.4f} "
            #             "\n       all_contrast_loss: {:.4f}"
            #             "\n       all_ssl_loss: {:.4f} \n".format(
            #             all_loss, all_contrast_loss,all_ssl_loss
            #             ))
            
            wandb.log({'ssl_loss':all_ssl_loss,'contrast_loss':all_contrast_loss},step=epoch)
            # epoch eval
            model.eval()
            # print('%%%%%%%%%% train stat')
            train_acc = NC_eval_model(model, 
                                      device, 
                                      train_loader, 
                                      evaluator, epoch, logger, args.erm, args.norm_mean, 
                                      dset='train',eval_metric=eval_metric)
            # print('%%%%%%%%%% val stat')
            val_acc = NC_eval_model(model, 
                                    device, 
                                    valid_loader, 
                                    evaluator, epoch, logger, args.erm, args.norm_mean, 
                                    dset='val', eval_metric=eval_metric)
            # print('%%%%%%%%%% test stat')
            test_acc = NC_eval_model(model,
                                  device,
                                  test_loader,
                                  evaluator, epoch, logger, args.erm, args.norm_mean, 
                                  dset='test',
                                  eval_metric=eval_metric)
            if val_acc <= last_val_acc:
                # select model according to the validation acc,
                #                  after the pretraining stage
                cnt += epoch >= args.pretrain
            else:
                cnt = (cnt + int(epoch >= args.pretrain)) if last_val_acc == 1.0 else 0
                last_train_acc = train_acc
                last_val_acc = val_acc
                last_test_acc = test_acc

                if args.save_model:
                    best_weights = deepcopy(model.state_dict())
            if epoch >= args.pretrain and cnt >= args.early_stopping:
                logger.info("Early Stopping")
                logger.info("+" * 50)
                logger.info("Last: Test_ACC: {:.3f} Train_ACC:{:.3f} Val_ACC:{:.3f} ".format(
                    last_test_acc, last_train_acc, last_val_acc))
                break

            

            all_info_seed['test_acc'].append(test_acc)
            all_info_seed['train_acc'].append(train_acc)
            all_info_seed['val_acc'].append(val_acc)

            # summary for best acc
            wandb.log({"train_acc":train_acc,"val_acc":val_acc,"test_acc":test_acc},step=epoch)
            best_index = torch.argmax(torch.tensor(all_info_seed['val_acc']))
            wandb.run.summary["best_epoch"] = best_index
            wandb.run.summary["best_acc"] = all_info_seed['test_acc'][best_index.item()]
            
            # print("      [{:3d}/{:d}]".format(epoch, args.epoch) +
            #             "\n       train_ACC: {:.4f} / {:.4f}"
            #             "\n       valid_ACC: {:.4f} / {:.4f}"
            #             "\n       tests_ACC: {:.4f} / {:.4f}\n".format(
            #                 train_acc, torch.tensor(all_info_seed['train_acc']).max(),
            #                 val_acc, torch.tensor(all_info_seed['val_acc']).max(),
            #                 test_acc, torch.tensor(all_info_seed['test_acc']).max(),
            #                 ))
            
            
            
            # change to all_info_seed
            # print(" best epoch : {:d}".format(best_index) +
            #             "\n       train_ACC: {:.4f}"
            #             "\n       valid_ACC: {:.4f}"
            #             "\n       tests_ACC: {:.4f}\n".format(
            #                 all_info_seed['train_acc'][best_index],
            #                 all_info_seed['val_acc'][best_index],
            #                 all_info_seed['test_acc'][best_index],
            #                 ))
            
        all_info['test_acc'].append(all_info_seed['test_acc'][best_index])
        all_info['train_acc'].append(all_info_seed['train_acc'][best_index])
        all_info['val_acc'].append(all_info_seed['val_acc'][best_index])    

            
        logger.info("=" * 50)
        wandb.finish()
    
    print("Test ACC:{:.4f}-+-{:.4f}\nTrain ACC:{:.4f}-+-{:.4f}\nVal ACC:{:.4f}-+-{:.4f} ".format(
        torch.tensor(all_info['test_acc']).mean(),
        torch.tensor(all_info['test_acc']).std(),
        torch.tensor(all_info['train_acc']).mean(),
        torch.tensor(all_info['train_acc']).std(),
        torch.tensor(all_info['val_acc']).mean(),
        torch.tensor(all_info['val_acc']).std()))
        

    
    
    

    if args.save_model:
        print("Saving best weights..")
        if not os.path.exists(f'model_{args.dataset}_{str(args.bias)}'):
            os.mkdir(f'model_{args.dataset}_{str(args.bias)}')
        model_path = os.path.join(f'model_{args.dataset}_{str(args.bias)}', str(args.erm)+str(args.contrast)+str(args.ssl)+str(args.type2)+str(args.p2)) + f"_seed{seed}.pt"
        for k, v in best_weights.items():
            best_weights[k] = v.cpu()
        torch.save(best_weights, model_path)
        print("Done..")


    print("\n\n\n")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
