import argparse
from models.gnn_ib import GIB
from models.ciga import GNNERM, CIGA, GNNPooling
import GCL.losses as L
import GCL.augmentors as A
import torch
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F
import numpy as np
import wandb
from utils.util import within_class_variation, within_class_variation_2
from utils.metrics import calculate_erank
def parse_arguments():
    parser = argparse.ArgumentParser(description='Causality Inspired Invariant Graph LeArning')
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='/home/kolerk/CIGA_NC_v1/data', type=str, help='directory for datasets.')
    parser.add_argument('--dataset', default='SPMotif', type=str)
    parser.add_argument('--bias', default='0.33', type=str, help='select bias extend')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')

    # training config
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=400, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for the predictor')
    parser.add_argument('--seed', nargs='?', default='[1,2,3,4,5]', help='random seed')
    parser.add_argument('--pretrain', default=20, type=int, help='pretrain epoch before early stopping')

    # model config
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--r', default=-1, type=float, help='selected ratio')
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=32, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='sum', type=str)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--early_stopping', default=5, type=int) 
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric',
                        default='',
                        type=str,
                        help='specify a particular eval metric, e.g., mat for MatthewsCoef')

    # Invariant Learning baselines config
    parser.add_argument('--num_envs', default=1, type=int, help='num of envs need to be partitioned')
    parser.add_argument('--irm_p', default=1, type=float, help='penalty weight')
    parser.add_argument('--irm_opt', default='irm', type=str, help='algorithms to use')


    # Invariant Graph Learning config
    parser.add_argument('--erm', action='store_true')  # whether to use normal GNN arch
    parser.add_argument('--ginv_opt', default='ginv', type=str)  # which interpretable GNN archs to use
    parser.add_argument('--dir', default=0, type=float)
    parser.add_argument('--contrast_t', default=1.0, type=float, help='temperature prameter in contrast loss')
    # strength of the contrastive reg, \alpha in the paper
    parser.add_argument('--contrast', default=4, type=float)
    parser.add_argument('--not_norm', action='store_true')  # whether not using normalization for the constrast loss
    parser.add_argument('-c_sam', '--contrast_sampling', default='mul', type=str)
    parser.add_argument('--contrast_d',default=1.0, type=float,help='partial dimension for conrast')
    # contrasting summary from the classifier or featurizer
    # rep:  classifier rep
    # feat: featurizer rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-c_rep', '--contrast_rep', default='rep', type=str)
    # pooling method for the last two options in c_rep
    parser.add_argument('-c_pool', '--contrast_pooling', default='add', type=str)


    # spurious rep for maximizing I(G_S;Y)
    # rep:  classifier rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-s_rep', '--spurious_rep', default='rep', type=str)
    # strength of the hinge reg, \beta in the paper
    parser.add_argument('--spu_coe', default=0, type=float)

    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--save_model', action='store_true')  # save pred to ./pred if not empty

    # ETF
    parser.add_argument('--etf', action='store_true', help="use ETF classifer")
    parser.add_argument('--loss', default= 'ce', help=" loss for ETF")
    parser.add_argument('--reg_lam', default=1e-8, type=float)
    parser.add_argument('--norm_mean', action='store_true')
    parser.add_argument('--etf_start',default=0, type=int, help='when to start etf')

    # ssl
    parser.add_argument("--ssl",default=0., type= float,help="use self-supervised contrast learning")
    parser.add_argument("--type1",  default="identical",type=str, help='augment type1')
    parser.add_argument("--p1",default=0.1,type=float,help="the augment percent for type1")
    parser.add_argument("--type2",  default="subgraph",type=str, help='augment type2')
    parser.add_argument("--p2",default=0.05,type=float,help="the augment percent for type2")
    parser.add_argument("--num_seeds",default=1500,type=int, help="subgraph numseed")
    parser.add_argument("--walk_length",default=5,type=int,help="random walk length")
    parser.add_argument("--ssl_t",default=0.2,type=float,help="ssl temperature parameter")
    parser.add_argument("--cNCE",action='store_true',help='class condition ssl loss')
    parser.add_argument("--augCL",action='store_true',help="add the augment data to contrast")
    parser.add_argument("--aug_ratio", default=0.2, type=float,help='the spurious graph percent for data augmentation')
    parser.add_argument("--d",default=0.,type=float, help='rep dim percent for ssl loss')
    parser.add_argument("--ssl_d",default=1.0,type=float,help="remaining percent for g1 and g2")
    # augCL_weight 
    parser.add_argument("--a_w", default=0, type=float ,help="augCL loss weight")
    # resample
    parser.add_argument("--up",action='store_true',help="upsampling for drugood")
    parser.add_argument("--down",action='store_true',help='downsampling for drugood')
    
    # ssl /ssl +ce loss pretrain
    parser.add_argument("--pre_epoch",default=0,type=int,help= 'pretrain epoch')
    args = parser.parse_args()
    return args

def select_model(args,input_dim, edge_dim, num_classes, device):
    if args.erm:
        model = GNNERM(input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node).to(device)
                        # etf=args.etf
            # model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    elif args.ginv_opt.lower() in ['asap']:
        model = GNNPooling(pooling=args.ginv_opt,
                            ratio=args.r,
                            input_dim=input_dim,
                            edge_dim=edge_dim,
                            out_dim=num_classes,
                            gnn_type=args.model,
                            num_layers=args.num_layers,
                            emb_dim=args.emb_dim,
                            drop_ratio=args.dropout,
                            graph_pooling=args.pooling,
                            virtual_node=args.virtual_node).to(device)
        # model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    elif args.ginv_opt.lower() == 'gib':
        model = GIB(ratio=args.r,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node).to(device)
        # model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    else:
        # to do divide CIGA into feature and classifier
        model = CIGA(ratio=args.r,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node,
                    c_dim=args.classifier_emb_dim,
                    c_in=args.classifier_input_feat,
                    c_rep=args.contrast_rep,
                    c_pool=args.contrast_pooling,
                    s_rep=args.spurious_rep).to(device) 
                    #etf=args.etf,
                    # etf_start=args.etf_start
    return model

def augment(args):
    if args.type1.lower() == "identical":
        aug1 = A.Identity()
    elif args.type1.lower() == "edgeremove":
        aug1 = A.EdgeRemoving(pe=args.p1)
    elif args.type2.lower() =="edgeadd":
        aug1 = A.EdgeAdding(pe=args.p1) # A bugs: out of index
    elif args.type1.lower() == "nodedrop":
        aug1 = A.NodeDropping(pn=args.p1)
    elif args.type1.lower() == "subgraph": #
        aug1 = A.RWSampling(num_seeds=args.num_seeds, walk_length=args.walk_length) #bug
    else:
        raise ValueError("wrong ssl augment type1")
    
    if args.type2.lower() == "identical":
        aug2 = A.Identity()
    elif args.type2.lower() == "edgeremove":
        aug2 = A.EdgeRemoving(pe=args.p2)
    elif args.type2.lower() =="edgeadd":
        aug2 = A.EdgeAdding(pe=args.p2) # A bugs: out of index
    elif args.type2.lower() == "nodedrop":
        aug2 = A.NodeDropping(pn=args.p2)
    elif args.type2.lower() == "subgraph": #
        aug2 = A.RWSampling(num_seeds=args.num_seeds, walk_length=args.walk_length) #bug
    # elif args.type2.lower() == "featuredrop":
    #     aug2 = A.FeatureDropout(pf=args.p2)
    # elif args.type2.lower() == "featuremask":
    #     aug2 = A.FeatureMasking(pf=args.p2)
    else:
        raise ValueError("wrong ssl augment type2")
    return aug1, aug2

@torch.no_grad()
def NC_eval_model(model, device, loader, evaluator, epoch, logger, erm, norm_mean, dset='train', eval_metric='acc', save_pred=False):
    # switch to evaluate mode
    # model.eval()
    y_true = []
    y_pred = []

    feat_dict = {}
    cnt_dict = {}
    h_dict ={}
    erank_overall = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # for NC
            target = batch.y
            if batch.x.shape[0] == 1:
                pass
            else:
                # NC
                if erm: 
                    pred,feat = model(batch,return_data="rep") #,epoch
                    h= torch.zeros(feat.shape).to(feat.device)
                else : # h is mean pooling $compute
                    pred, feat= model(batch,return_data="rep")
                    h= torch.zeros(feat.shape).to(feat.device)
                # else : # h is mean pooling  original 
                #     pred, feat= model(batch,epoch,return_data="rep")
                #     h= torch.zeros(feat.shape).to(feat.device)
            # erank dimentional collapse
            # breakpoint()
            erank = calculate_erank(feat)
            erank_overall.append(erank)
            # NC
            if True:
                uni_lbl, count = torch.unique(target, return_counts=True)
                lbl_num = uni_lbl.size(0)
                for kk in range(lbl_num):
                    # sum_feat = torch.sum(feat[torch.where(target==uni_lbl[kk])[0], :], dim=0)
                    sum_feat = feat[torch.where(target==uni_lbl[kk])[0], :]
                    sum_h = h[torch.where(target==uni_lbl[kk])[0], :]
                    key = uni_lbl[kk].item()
                    if key in feat_dict.keys():
                        # feat_dict[key] = feat_dict[key]+sum_feat
                        feat_dict[key] = torch.cat((feat_dict[key],sum_feat),dim=0)
                        # print(feat_dict[key].shape)
                        h_dict[key] = torch.cat((h_dict[key],sum_h),dim=0)
                        cnt_dict[key] =  cnt_dict[key]+count[kk]
                    else:
                        feat_dict[key] = sum_feat
                        cnt_dict[key] =  count[kk]
                        h_dict[key] = sum_h
                        # print(feat_dict[key].shape)
            # evaluate
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat': # only drugood use this eval
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        pred, rep = model(batch, return_data="rep", debug=True)
                        # print(rep.size())
                    # print(batch)
                    # print(global_mean_pool(batch.x, batch.batch).size())
                    # print(pred.shape)
                    # print(batch.y.size())
                    # print(sum(is_labeled))
                    # print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        num_classes=np.unique(y_true).size
        erank_mean = np.mean(erank_overall)
        # print(erank_mean)
        # if dset=="train":
        #     count = np.sum(y_true < 0.5)
        #     print("y_true 小于 0.5 的元素个数：", count)
        #     count = np.sum(y_pred < 0.5)
        #     print("y_pred 小于 0.5 的元素个数：", count)
        if num_classes==2:
            tr_product, equinorm_product, _ = within_class_variation_2(feat_dict=feat_dict)
        else:    
            tr_product, equinorm_product, _ = within_class_variation(feat_dict=feat_dict)
        if dset=='train':
            print("train_Tr(ΣW)Tr(ΣB):{:4f}, train_stdmean:{:4f}".format(tr_product,equinorm_product))
        elif dset=='val':
            print("val_Tr(ΣW)Tr(ΣB):{:4f}, val_stdmean:{:4f}".format(tr_product,equinorm_product))
        else:
            print("test_Tr(ΣW)Tr(ΣB):{:4f}, test_stdmean:{:4f}".format(tr_product,equinorm_product))
        if dset=='train':
            wandb.log({"train_Tr(ΣW)Tr(ΣB)":tr_product, "train_stdmean":equinorm_product,"train_erank":erank_mean},step=epoch)
        elif dset=='val':
            wandb.log({"val_Tr(ΣW)Tr(ΣB)":tr_product, "val_stdmean":equinorm_product,"val_erank":erank_mean},step=epoch)
        else:
            wandb.log({"test_Tr(ΣW)Tr(ΣB)":tr_product, "test_stdmean":equinorm_product,"test_erank":erank_mean},step=epoch)
        
        if eval_metric == 'mat':
            res_metric = matthews_corrcoef(y_true, y_pred)
        else:
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            res_metric = evaluator.eval(input_dict)[eval_metric]
        
        # NC emprical statistic numbers
        # if config.stat_mode:
        if True:
            # config.num_classes
            
            ### calculate statistics
            #print(feat_dict)
            class_mean = torch.zeros(num_classes, feat.size(1)).to(device)
            for i in range(num_classes):
                if i in feat_dict.keys():
                    # normalize before calculate sum
                    if norm_mean:
                        feat_dict[i]=F.normalize(feat_dict[i])
                    mean = torch.mean(feat_dict[i],dim=0)
                    # Calculate the L2 distance (Euclidean distance) of each element to the mean
                    distances = torch.norm(feat_dict[i] - mean, p=2, dim=1)
                    std = torch.mean(distances)
                    # std = torch.std(torch.std(feat_dict[i],dim=0))
                    feat_dict[i] = torch.sum(feat_dict[i],dim=0)
                    class_mean[i,:] = feat_dict[i] / cnt_dict[i]
                    # print('class {}:{}'.format(i,cnt_dict[i]))
                    # feat_dict[i]=F.normalize(feat_dict[i])
                    # this is Tr(ΣW)/Tr(ΣB)
                    # print('class {}: std={}'.format(i,std.item())) 
                    # logger.info("=" * 50 +"\n")
                    # logger.info(f'classifier class{i}: mean={str(mean)},std={std.item()}')
                if not erm and i in h_dict.keys(): 
                    if norm_mean:
                        h_dict[i]=F.normalize(h_dict[i])
                    h_mean = torch.mean(h_dict[i],dim=0)
                    distances = torch.norm(h_dict[i] - h_mean, p=2, dim=1)
                    h_std = torch.mean(distances)
                    # logger.info("=" * 50 +"\n")
                    # logger.info(f'graph extractor class{i}: mean={str(h_mean)},std={h_std.item()}')
            
            total_sum_feat = sum(feat_dict.values())
            total_counts = sum(cnt_dict.values())
            global_mean = total_sum_feat / total_counts
            # drugood 
            if num_classes==2: # only 2 classes
                class_mean = class_mean
                # class_mean = class_mean - global_mean ## first bug for cos =1
            else :
                class_mean = class_mean - global_mean  ## K, dim
            # add epoch for etf_start
            # W_mean = model.get_pred_linear_weight(epoch).to(device)
            # W_mean = model.get_w().to(device)# for other baselines
            ##
            class_mean = class_mean / torch.sqrt(torch.sum(class_mean**2, dim=1, keepdims=True))
            # W_mean_F = W_mean / torch.sqrt(torch.sum(W_mean **2, dim=1,keepdims=True))
            # F_dist =torch.sum((class_mean - W_mean_F)**2)
            
            cos_HH = torch.matmul(class_mean, class_mean.T)
            # cos_WW = torch.matmul(W_mean_F, W_mean_F.T)
            # cos_HW = torch.matmul(class_mean, W_mean_F.T)
            #print(cos_HH)
            # if dset=='train':
            #     print('HH',cos_HH)
            #     print('WW',cos_WW)
            #print(cos_HW)
            ##
            # diag_HW = torch.diag(cos_HW, 0)
            # diag_avg = torch.mean(diag_HW)
            # diag_std = torch.std(diag_HW)
            ##
            up_HH = torch.cat([torch.diag(cos_HH, i) for i in range(1,num_classes)])
            # up_WW = torch.cat([torch.diag(cos_WW, i) for i in range(1,num_classes)])
            # up_HW = torch.cat([torch.diag(cos_HW, i) for i in range(1,num_classes)])
            # print('up-HH', up_HH)
            # print('up-WW', up_WW)
            # print('up-HW', up_HW)
            up_HH_avg = torch.mean(up_HH)
            up_HH_std = torch.std(up_HH)
            # up_WW_avg = torch.mean(up_WW)
            # up_WW_std = torch.std(up_WW)
            # up_HW_avg = torch.mean(up_HW)
            # up_HW_std = torch.std(up_HW)
            # ##
            # print('cos-avg-HH', up_HH_avg)
            # print('cos-avg-WW', up_WW_avg)
            # print('cos-avg-HW', up_HW_avg)
            # print('cos-std-HH', up_HH_std)
            # print('cos-std-WW', up_WW_std)
            # print('cos-std-HW', up_HW_std)
            # ##
            # print('diag-avg-HW', diag_avg)
            # print('diag-std-HW', diag_std)
            # ##
            # print('||H-M||_F^2', F_dist)
            if dset=='train':
                wandb.log({"train-cos-avg-HH": up_HH_avg,#,"train-cos-avg-WW": up_WW_avg,
                        "train-cos-std-HH": up_HH_std# ,"train-cos-std-WW": up_WW_std
                        },step=epoch)
                        
            elif dset == 'val':
                wandb.log({"val-cos-avg-HH": up_HH_avg,# ,"val-cos-avg-WW": up_WW_avg,
                       "val-cos-std-HH": up_HH_std#， "val-cos-std-WW": up_WW_std
                        },step=epoch)
            elif dset =='test':
                wandb.log({"test-cos-avg-HH": up_HH_avg, # ,"test-cos-avg-WW": up_WW_avg,
                        "test-cos-std-HH": up_HH_std #,"test-cos-std-WW": up_WW_std
                        },step=epoch)
            else:
                raise ValueError("wrong dset type")

            


    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric


@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False):
    '''acc eval method, return acc or roc'''
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        pred, rep = model(batch, return_data="rep", debug=True)
                        print(rep.size())
                    print(batch)
                    print(global_mean_pool(batch.x, batch.batch).size())
                    print(pred.shape)
                    print(batch.y.size())
                    print(sum(is_labeled))
                    print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]

    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric