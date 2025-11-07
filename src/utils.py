# src/utils.py
import torch
import random
import numpy as np
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDNN 的可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


