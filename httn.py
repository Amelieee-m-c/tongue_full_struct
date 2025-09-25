# httn.py - Head-to-Tail Network head (with optional Head↔Tail Attention and Ensemble)
import math
import torch
import torch.nn as nn

class HTTNBaseHead(nn.Module):
    """
    Multi-label HTTN base head:
      - Keeps C class prototypes p_c ∈ R^D (updated via EMA from positive samples).
      - A mapper g: R^D -> R^(D+1) produces [w_c | b_c] per class.
      - logits_httn(x) = z @ W^T + b, where z is the shared feature.
      - Optional head↔tail attention: reinforce each prototype using head-class prototypes.
    """
    def __init__(self, num_classes:int, feat_dim:int, hidden:int=None,
                 ema_m:float=0.99, use_attention:bool=True):
        super().__init__()
        self.C, self.D = num_classes, feat_dim
        hidden = hidden or (2 * feat_dim)
        self.mapper = nn.Sequential(
            nn.Linear(self.D, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.D + 1)   # [w(=D), b(=1)]
        )
        # Prototypes and init flags (registered as buffers so they save with state_dict)
        self.register_buffer('prototypes',   torch.zeros(self.C, self.D))
        self.register_buffer('proto_inited', torch.zeros(self.C, dtype=torch.bool))
        self.ema_m = ema_m
        self.use_attention = use_attention
        # Head-class indices buffer (empty by default; call set_head_ids)
        self.register_buffer('head_ids', torch.arange(0))

    @torch.no_grad()
    def set_head_ids(self, head_ids: torch.Tensor):
        """Set head-class indices (1D LongTensor)."""
        if head_ids is None or head_ids.numel() == 0:
            self.head_ids = torch.arange(0, device=self.prototypes.device)
        else:
            self.head_ids = head_ids.to(self.prototypes.device, dtype=torch.long)

    @torch.no_grad()
    def _update_prototypes(self, z: torch.Tensor, y: torch.Tensor):
        """
        z: [B, D] shared features; y: [B, C] multi-label 0/1
        Updates each class prototype with EMA using current batch positives.
        """
        for c in range(self.C):
            pos = (y[:, c] > 0.5)
            if not torch.any(pos):
                continue
            mean_c = z[pos].mean(dim=0)  # [D]
            if not self.proto_inited[c]:
                self.prototypes[c] = mean_c
                self.proto_inited[c] = True
            else:
                self.prototypes[c].mul_(self.ema_m).add_(mean_c * (1.0 - self.ema_m))

    def _apply_head_tail_attention(self, P: torch.Tensor) -> torch.Tensor:
        """
        P: [C, D] prototypes; use head prototypes to create residual attention augmentation.
        Returns P_eff: [C, D]
        """
        if (not self.use_attention) or (self.head_ids.numel() == 0):
            return P
        P_head = P[self.head_ids]              # [H, D]
        if P_head.numel() == 0:
            return P
        # scaled dot-product attention (cosine-like since features are comparable scale)
        scores = (P @ P_head.t()) / math.sqrt(self.D)   # [C, H]
        alpha  = torch.softmax(scores, dim=1)           # [C, H]
        P_aug  = alpha @ P_head                         # [C, D]
        return P + P_aug                                # residual

    def forward(self, z: torch.Tensor, labels: torch.Tensor=None, update: bool=True):
        """
        z: [B, D] shared features; labels: [B, C] (for prototype updates during training)
        Returns logits_httn: [B, C]
        """
        if self.training and update and (labels is not None):
            self._update_prototypes(z.detach(), labels.detach())

        P = self.prototypes                          # [C, D]
        P_eff = self._apply_head_tail_attention(P)   # [C, D]
        wb = self.mapper(P_eff)                      # [C, D+1]
        W, b = wb[:, :self.D], wb[:, self.D:]        # W:[C,D], b:[C,1]
        logits = z @ W.t() + b.t()                   # [B, C]
        return logits


class HTTNEnsembleHead(nn.Module):
    """
    EHTTN: average logits from K independent HTTNBaseHead instances (K>=1).
    """
    def __init__(self, num_classes:int, feat_dim:int, hidden:int=None,
                 ema_m:float=0.99, use_attention:bool=True, num_ensembles:int=1):
        super().__init__()
        num_ensembles = max(1, int(num_ensembles))
        self.heads = nn.ModuleList([
            HTTNBaseHead(num_classes, feat_dim, hidden, ema_m, use_attention)
            for _ in range(num_ensembles)
        ])

    @torch.no_grad()
    def set_head_ids(self, head_ids: torch.Tensor):
        for h in self.heads:
            h.set_head_ids(head_ids)

    def forward(self, z: torch.Tensor, labels: torch.Tensor=None, update: bool=True):
        logits_list = [h(z, labels, update) for h in self.heads]  # list of [B,C]
        return torch.stack(logits_list, dim=0).mean(dim=0)        # [B, C]
