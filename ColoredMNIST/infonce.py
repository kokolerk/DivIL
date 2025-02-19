import torch
import torch.nn.functional as F
import torch.nn as nn

def random_mask(image, mask_prob=0.2):
    mask = torch.zeros_like(image)
    mask_prob_tensor = torch.tensor(mask_prob, device=image.device)
    mask = torch.bernoulli(mask_prob_tensor.expand_as(mask))

    return mask

class InfoNCELoss(nn.Module):
    def __init__(self, temperature, num_negative_samples=100):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.num_negative_samples = num_negative_samples

    def forward(self, zi, zj):
        batch_size = zi.size(0)
        
        zi_norm = torch.norm(zi, p=2, dim=1, keepdim=True) ** 2
        zj_norm = torch.norm(zj, p=2, dim=1, keepdim=True) ** 2

        d_ij_aug_diag = (zi_norm + zj_norm - 2 * (zi * zj).sum(dim=1, keepdim=True)) / (2 * self.temperature)
        d_ij = (zi_norm + zi_norm.T - 2 * torch.matmul(zi, zi.T)) / (2 * self.temperature)

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=d_ij.device)

        neg_mask = ~pos_mask

        d_ij_neg = d_ij[neg_mask].view(batch_size, -1)
        topk_neg_distances, _ = torch.topk(d_ij_neg, self.num_negative_samples, dim=1, largest=True)

        exp_pos = torch.exp(-d_ij_aug_diag).view(-1)
        exp_neg = torch.exp(-topk_neg_distances).sum(dim=1)

        loss = -torch.log(exp_pos / (exp_neg + exp_pos + 1e-5)).mean()
        return loss

def compute_div_penalty(proj, featurizer, ready_features, image, temp, mask_p):
    
    image = image.view(image.shape[0], 2 * 14 * 14)
    aug_image = image * random_mask(image)
    
    zi = proj(ready_features)
    zj = proj(featurizer(aug_image))

    mask = int(zi.size(1) * mask_p)
    zi[:, :mask] = 0
    zj[:, :mask] = 0
    
    num_negative_samples = int(0.7 * image.shape[0]) if int(0.7 * image.shape[0] <= 100) else 100
    loss = InfoNCELoss(temperature=temp, num_negative_samples=num_negative_samples)(zi, zj)
    return loss
    

