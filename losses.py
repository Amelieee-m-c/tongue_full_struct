# losses.py — Class-Balanced BCE for multi-label long-tail
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedBCELoss(nn.Module):
    """
    BCE with per-class weights derived from positive/negative counts to mitigate long-tail.
    Args:
      n_pos: Tensor [C] positive counts per class (>=1 to avoid div-by-zero)
      n_neg: Tensor [C] negative counts per class
      beta:  in (0,1); effective number weighting if provided (default None uses inverse frequency)
    """
    def __init__(self, n_pos: torch.Tensor, n_neg: torch.Tensor, beta: float=None, eps: float=1e-8):
        super().__init__()
        device = n_pos.device
        n_pos = n_pos.clamp_min(1).float()
        n_neg = n_neg.clamp_min(1).float()
        if beta is not None and 0.0 < beta < 1.0:
            # Effective number of samples (Cui et al. 2019)
            E_pos = (1.0 - torch.pow(beta, n_pos)) / (1.0 - beta + eps)
            E_neg = (1.0 - torch.pow(beta, n_neg)) / (1.0 - beta + eps)
            w_pos = (1.0 / (E_pos + eps))
            w_neg = (1.0 / (E_neg + eps))
        else:
            w_pos = (n_neg / (n_pos + n_neg + eps))
            w_neg = (n_pos / (n_pos + n_neg + eps))

        # Normalize weights so magnitudes are reasonable
        w_pos = w_pos / (w_pos.mean() + eps)
        w_neg = w_neg / (w_neg.mean() + eps)

        self.register_buffer('w_pos', w_pos)
        self.register_buffer('w_neg', w_neg)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B, C], targets: [B, C] in {0,1}
        """
        prob = torch.sigmoid(logits)
        # Weighted BCE
        loss_pos = - self.w_pos * (targets * torch.log(prob.clamp_min(self.eps)))
        loss_neg = - self.w_neg * ((1 - targets) * torch.log((1 - prob).clamp_min(self.eps)))
        loss = loss_pos + loss_neg
        return loss.mean()
