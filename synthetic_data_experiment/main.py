from loguru import logger
from domainbed_utils.algorithms import *
from tqdm import tqdm
from utils.utils import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import inspect
import random
import time

# data size
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 5000

# training config
BATCH_SIZE = 256
EPOCHS = 5000
OUTPUT_DIM = 2

# FIG_DPI = 960
plt.rcParams['font.family'] = 'Sitka'
plt.rcParams.update({'font.size': 11}) 

dataset_builder = DatasetBuilder()


def set_seed(seed=None):
    _seed = seed 
    np.random.seed(_seed)
    random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)


def test_over_invariance(framework_list=['ERM']):
    r"""Test over-invariance in IL methods."""
    device = 'cpu'
    hparams = {
        'nonlinear_classifier': False,
        'weight_decay': 0.01,
        'lr': 1e-2,
        'mlp_width': 4,
        'mlp_dropout': 0,
        'mlp_depth': 2,
        
        'irm_penalty_anneal_iters': 500,
        'irm_lambda': 1e2,
        
        'vrex_lambda': 1e1,
        'vrex_penalty_anneal_iters': 500
        
    }
    
    for framework_name in framework_list:
        logger.info('using framework {}...'.format(framework_name))
        
        seed_list = range(42, 52)
        exp_var = np.arange(-3, 0.5, 0.1)

        spurious_var_list = pow(10, exp_var)
        strength_list = []
        for spurious_var in tqdm(spurious_var_list):
            inv_var_list = [5, 5, 3, 3, 1, 1, 0.1, 0.1]
            spu_var_list = [spurious_var] * 8
            
            total_strength_list = np.zeros(4)
            
            for seed in seed_list:
                set_seed(seed)
                x, y, y_s = dataset_builder.create_train_dataset(inv_var_list, spu_var_list, seed)
                x_test, y_test, y_s_test = dataset_builder.create_test_dataset(inv_var_list, spu_var_list, seed)
                framework_class = get_algorithm_class(framework_name)
                framework = framework_class(input_shape=x.shape[1:],
                                            num_classes=2,
                                            num_domains=2,
                                            hparams=hparams)

                # start training
                for epoch in range(EPOCHS):
                    idx = np.random.choice(TRAIN_DATA_SIZE, BATCH_SIZE, replace=False)
                    xi = x[idx].to(torch.float32)
                    yi = y[idx].to(torch.long)
                    yi_label = yi.masked_fill(yi==-1, 0)

                    minibatch = [(xi, yi_label)]
                    loss = framework.update(minibatch)['loss']
 
                x_test = x_test.to(torch.float32)
                
                for _dim in range(len(inv_var_list) // 2):
                    invariant_feature_strength = calculate_strength(x_test, start_dim=_dim*2, end_dim=_dim*2+1, model=framework.featurizer)
                    total_strength_list[_dim] += invariant_feature_strength
                
            strength_list.append(total_strength_list / len(seed_list)) 
        
        for _dim in range(len(inv_var_list) // 2):
            current_strength = [strengths[_dim] for strengths in strength_list]
            plt.plot(exp_var, current_strength, label=f'var={inv_var_list[_dim * 2]}')
        
        plt.xlabel('log(spurious_var)')
        plt.ylabel('strength')
        plt.legend()
        plt.title(framework_name)
        plt.show()
        

def validate_ucl(framework_list=['ERM']):
    r"""Test whether UCL can improve IL methods."""
    
    # device = 'cpu'
    hparams = {
        'nonlinear_classifier': False,
        'weight_decay': 0.01,
        'lr': 1e-2,
        'mlp_width': 4,
        'mlp_dropout': 0,
        'mlp_depth': 2,
        
        'irm_penalty_anneal_iters': 500,
        'irm_lambda': 1e2,
        
        'vrex_lambda': 1e1,
        'vrex_penalty_anneal_iters': 500
        
    }
    
    for framework_name in framework_list:        
        logger.info('using framework {}...'.format(framework_name))
        
        seed_list = range(42, 52)
        spu_var_list = [1] * 8
        
        inv_var_list = [5, 5, 3, 3, 1, 1, 0.1, 0.1]
        IL_total_strength_list = np.zeros(4)
        IL_UCL_total_strength_list = np.zeros(4)
        
        for seed in tqdm(seed_list):
            set_seed(seed)
            x, y, y_s = dataset_builder.create_train_dataset(inv_var_list, spu_var_list, seed)
            x_test, y_test, y_s_test = dataset_builder.create_test_dataset(inv_var_list, spu_var_list, seed)
            factor = 1
            
            framework_class = get_algorithm_class(framework_name)
            ucl_framework_class = get_algorithm_class(framework_name)
            framework = framework_class(input_shape=x.shape[1:],
                                        num_classes=2,
                                        num_domains=2,
                                        hparams=hparams)
            
            ucl_framework = ucl_framework_class(input_shape=x.shape[1:],
                        num_classes=2,
                        num_domains=2,
                        hparams=hparams)

            # start training
            for epoch in range(EPOCHS):
                idx = np.random.choice(TRAIN_DATA_SIZE, BATCH_SIZE, replace=False)
                xi = x[idx].to(torch.float32)
                yi = y[idx].to(torch.long)
                yi_label = yi.masked_fill(yi==-1, 0)
                
                minibatch = [(xi, yi_label)]
                framework.update(minibatch)['loss']
                ucl_framework.update(minibatch, use_ucl=True, factor=factor) 

            x_test = x_test.to(torch.float32)
            
            for _dim in range(len(inv_var_list) // 2):
                invariant_feature_strength = calculate_strength(x_test, start_dim=_dim*2, end_dim=_dim*2+1, model=framework.featurizer)
                ucl_invariant_feature_strength = calculate_strength(x_test, start_dim=_dim*2, end_dim=_dim*2+1, model=ucl_framework.featurizer)
                IL_total_strength_list[_dim] += invariant_feature_strength
                IL_UCL_total_strength_list[_dim] += ucl_invariant_feature_strength
        
        IL_total_strength_list = [val / len(seed_list) for val in IL_total_strength_list]
        IL_UCL_total_strength_list = [val / len(seed_list) for val in IL_UCL_total_strength_list]

        variances = inv_var_list[::2]
        bar_width = 0.35
        index = np.arange(len(variances))
        
        fig, ax = plt.subplots()
        bars_il = ax.bar(index, IL_total_strength_list, bar_width, label=f'{framework_name} method')
        bars_ucl = ax.bar(index + bar_width, IL_UCL_total_strength_list, bar_width, label=f'{framework_name} + UCL method')

        ax.set_xlabel('Variance')
        ax.set_ylabel('Strength')
        ax.set_title(f'{framework_name} / {framework_name} & UCL')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(variances)
        ax.legend()

        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom')

        add_labels(bars_il)
        add_labels(bars_ucl)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    validate_ucl(framework_list=['IRM', 'VREx'])
    # test_over_invariance(framework_list=['IRM', 'VREx'])