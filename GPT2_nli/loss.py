import torch
from torch import nn
import torch.nn.functional as F

class IRMLoss(nn.Module):
    def __init__(self, scale):
        super(IRMLoss, self).__init__()
        self.scale = scale

    def forward(self, logits, y, penalty_weight):

        scale = torch.tensor(self.scale, device=logits.device).requires_grad_()
        loss = nn.CrossEntropyLoss()(logits * scale, y)
        if not logits.requires_grad:
            return loss.to(logits.device)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        irm_penalty = penalty_weight * torch.sum(grad ** 2)
        if penalty_weight > 1.0:
            irm_penalty /= penalty_weight
        return irm_penalty
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, weight):

        # features: (batch_size, feature_dim), labels: (batch_size,)
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        similarity_matrix.masked_fill_(mask, -1e9)

        # positive samples mask
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.t()).float()

        # compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        positive_mask.masked_fill_(mask, 0)
        num_positive = positive_mask.sum(dim=1)
        num_positive[num_positive == 0] = 1
        loss = (-positive_mask * log_prob).sum(dim=1) / num_positive
        loss = loss[loss != 0].mean()
        return weight * loss


class SelfContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SelfContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch_emb, weight):

        batch_size = batch_emb.size(0)    # [bs, hidden_size]

        y_true = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long).unsqueeze(1),
                            torch.arange(0, batch_size, step=2, dtype=torch.long).unsqueeze(1)],
                           dim=1).reshape([batch_size,]).to(batch_emb.device)

        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        sim_score = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
        sim_score = sim_score - torch.eye(batch_size).to(sim_score.device) * 1e12
        sim_score = sim_score / self.temperature  
        loss = nn.CrossEntropyLoss()(sim_score, y_true)

        return weight * loss