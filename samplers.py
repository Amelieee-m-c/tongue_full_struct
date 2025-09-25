# samplers.py — helper to build a WeightedRandomSampler based on label frequencies
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def compute_pos_neg_from_dataframe(df, label_cols):
    y = df[label_cols].values.astype(np.float32)   # [N, C]
    pos = y.sum(axis=0)                            # [C]
    neg = (y.shape[0] - pos)                       # [C]
    return pos, neg

def make_per_sample_weights(df, label_cols, min_weight=1.0):
    """
    For each sample, sum inverse class frequency weights for its positive labels.
    """
    y = df[label_cols].values.astype(np.float32)   # [N, C]
    pos = y.sum(axis=0) + 1e-6                     # avoid zero
    inv = 1.0 / pos                                
    w = (y * inv[None, :]).sum(axis=1)             # [N]
    w = np.maximum(w, min_weight)
    return torch.as_tensor(w, dtype=torch.float)

def make_weighted_sampler(df, label_cols):
    weights = make_per_sample_weights(df, label_cols)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler
