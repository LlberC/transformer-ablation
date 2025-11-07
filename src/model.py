import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# <--- NEW: 导入学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.W_2(self.dropout(self.relu(self.W_1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, V)
        return context

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        context = self.scaled_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output


# ----------------------------------------------------------------------
# 积木 4: 因果解码器块 (Causal Decoder Block)
# ----------------------------------------------------------------------
class CausalDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(CausalDecoderBlock, self).__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = x
        attn_output = self.masked_self_attn(x, x, x, mask)
        x = self.norm1(_x + self.dropout1(attn_output))
        _x = x
        ffn_output = self.ffn(x)
        x = self.norm2(_x + self.dropout2(ffn_output))
        return x


# ----------------------------------------------------------------------
# 完整的 Decoder-Only 语言模型
# ----------------------------------------------------------------------
class DecoderOnlyLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super(DecoderOnlyLanguageModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            CausalDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        output = self.linear_out(x)
        return output


class NoResidualCausalDecoderBlock(CausalDecoderBlock):
    def forward(self, x, mask):
        attn_output = self.masked_self_attn(x, x, x, mask)
        x = self.norm1(self.dropout1(attn_output))  # 移除残差
        ffn_output = self.ffn(x)
        x = self.norm2(self.dropout2(ffn_output))  # 移除残差
        return x
