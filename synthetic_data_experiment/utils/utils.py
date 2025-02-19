from loguru import logger

import numpy as np
import torch
import random
import time
from typing import List, Tuple



def get_time():
    local_time = time.localtime(time.time())
    formatted_time = time.strftime("%m%d%H%M", local_time)
    return formatted_time


def InfoNCE(z1, z2, temperature, device):
    logits = z1 @ z2.T
    logits /= temperature
    n = z1.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def calculate_l2_val(tensor: torch.Tensor):
    squared_sum = torch.sum(tensor ** 2, dim=1)
    mean_squared_sum = torch.mean(squared_sum)
    return mean_squared_sum


def calculate_strength(x_test, start_dim: int, end_dim: int, model: torch.nn.Module):
    device = 'cpu'
    model.eval()
    """interval: [start_dim, end_dim]"""
    x_strength_test = torch.zeros_like(x_test)
    x_strength_test[:, start_dim:end_dim+1] = x_test[:, start_dim:end_dim+1]
    with torch.no_grad():
        output = model(x_strength_test.to(device))
        strength = calculate_l2_val(output)
    model.train()
    return strength.cpu()


class DatasetBuilder:
    def __init__(self):
        self.dim = 16
        self.num_samples = 10000
        
        self.train_avg_list = [10, 10, 10, 10, 10, 10, 10, 10]
        self.test_avg_list = [10, 10, 10, 10, 10, 10, 10, 10]
        
        self.train_reverse_prob = 0.3
        self.test_reverse_prob = 0.7
        
    
    def set_seed(self, seed=None):
        _seed = seed 
        np.random.seed(_seed)
        random.seed(_seed)
        torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
        

    def create_train_dataset(self, 
                            inv_var_list: List[float]= None, 
                            spu_var_list: List[float]=None, 
                            seed: int = 42)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Create a train dataset.
        Args:
            inv_var_list: List of invariant feature variance.
            spu_var_list: List of spurious feature variance.
            seed: Random seed.
        """
        
        self.set_seed(seed)

        y = np.random.choice([-1,1],size=(self.num_samples))
        y_extend = np.tile(y.reshape(-1,1) ,(1, self.dim//2))
        
        invariant_data = np.zeros((self.num_samples, self.dim//2))
        for i in range(self.dim // 2):
            invariant_data[:, i] = np.random.normal(loc=self.train_avg_list[i], 
                                                    scale=np.sqrt(inv_var_list[i]),
                                                    size=self.num_samples)
            
        invariant_data = invariant_data * y_extend
        
        choice_list = [1, -1]
        choice_list_prob = [1 - self.train_reverse_prob, self.train_reverse_prob]
        reverse_list = random.choices(choice_list, choice_list_prob, k=self.num_samples)
        y_s = y * np.array(reverse_list)
        y_s_extend =  np.tile(y_s.reshape(-1,1) ,(1, self.dim//2))
        
        spurious_data = np.zeros((self.num_samples, self.dim//2))
        for i in range(self.dim // 2):
            spurious_data[:, i] = np.random.normal(loc=self.train_avg_list[i], 
                                                    scale=np.sqrt(spu_var_list[i]),
                                                    size=self.num_samples)
        
        spurious_data = spurious_data * y_s_extend
        x = np.hstack((invariant_data,spurious_data))
        x = torch.tensor(x)
        y = torch.tensor(y)
        y_s = torch.tensor(y_s)
        return x, y, y_s


    def create_test_dataset(self, 
                            inv_var_list: List[float]= None, 
                            spu_var_list: List[float]=None, 
                            seed: int = 42)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Create a test dataset.
        Args:
            inv_var_list: List of invariant feature variance.
            spu_var_list: List of spurious feature variance.
            seed: Random seed.
        """
        
        self.set_seed(seed)
        
        y = np.random.choice([-1,1],size=(self.num_samples))
        y_extend = np.tile(y.reshape(-1,1) ,(1, self.dim//2))
        
        invariant_data = np.zeros((self.num_samples, self.dim//2))
        for i in range(self.dim // 2):
            invariant_data[:, i] = np.random.normal(loc=self.test_avg_list[i], 
                                                    scale=np.sqrt(inv_var_list[i]),
                                                    size=self.num_samples)
            
        invariant_data = invariant_data * y_extend
        
        choice_list = [1, -1]
        choice_list_prob = [1 - self.test_reverse_prob, self.test_reverse_prob]
        reverse_list = random.choices(choice_list, choice_list_prob, k=self.num_samples)
        y_s = y * np.array(reverse_list)
        y_s_extend =  np.tile(y_s.reshape(-1,1) ,(1, self.dim//2))
        
        spurious_data = np.zeros((self.num_samples, self.dim//2))
        for i in range(self.dim // 2):
            spurious_data[:, i] = np.random.normal(loc=self.test_avg_list[i], 
                                                   scale=np.sqrt(spu_var_list[i]),
                                                   size=self.num_samples)
        
        spurious_data = spurious_data * y_s_extend
        x = np.hstack((invariant_data,spurious_data))
        x = torch.tensor(x)
        y = torch.tensor(y)
        y_s = torch.tensor(y_s)
        return x, y, y_s 
