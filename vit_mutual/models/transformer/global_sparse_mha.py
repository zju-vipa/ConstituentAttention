"""
Sparse Multi-head Attention
"""

from functools import partial
from typing import Callable, Optional
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from vit_mutual.models.layers.entmax import EntmaxBisect


class SparseMultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            num_heads: int = 8,
            embed_dim: int = 256,
            dropout: float = None,
            bias: bool = True,
            use_entmax: bool = False,
            learnable_entmax_alpha: bool = False
    ):
        """
        Args:
            num_heads: number of self-attention heads
            embed_dim: token embedding dim
        Warning:
            embed_dim must be divisible by num_heads
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # softmax
        self.softmax = None
        if use_entmax:
            alpha = 1.5
            if learnable_entmax_alpha:
                alpha = nn.Parameter(torch.tensor(alpha))
            self.softmax = EntmaxBisect(alpha=alpha, dim=-1)

        # projection matrices
        self.linear_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self._reset_parameters()

    def get_weight_q(self):
        w = self.linear_qkv.weight[0:self.embed_dim]
        return w

    def get_weight_k(self):
        w = self.linear_qkv.weight[self.embed_dim:2 * self.embed_dim]
        return w

    def get_weight_v(self):
        w = self.linear_qkv.weight[2 * self.embed_dim:3 * self.embed_dim]
        return w

    def get_weight_o(self):
        w = self.linear_out.weight
        return w

    def get_head_weight(self, weight_str: str, head_id: int, transpose: bool = False):
        weight_str = weight_str.lower()
        assert head_id < self.num_heads
        f_map = {
            "q": self.get_weight_q,
            "k": self.get_weight_k,
            "v": self.get_weight_v,
            "o": self.get_weight_o,
        }  # 函数映射
        weight = f_map[weight_str]()
        if weight_str != "o":
            w = weight[head_id * self.head_dim:(head_id + 1) * self.head_dim]
        else:
            w = weight[..., head_id * self.head_dim:(head_id + 1) * self.head_dim]
        if transpose:
            w = w.T
        return w

    def get_bias_q(self):
        b = self.linear_qkv.bias[0:self.embed_dim]
        return b

    def get_bias_k(self):
        b = self.linear_qkv.bias[self.embed_dim:2 * self.embed_dim]
        return b

    def get_bias_v(self):
        b = self.linear_qkv.bias[2 * self.embed_dim:3 * self.embed_dim]
        return b

    def get_bias_o(self):
        b = self.linear_out.bias
        return b

    def get_head_bias(self, bias_str: str, head_id: int):
        bias_str = bias_str.lower()
        assert head_id < self.num_heads
        f_map = {
            "q": self.get_bias_q,
            "k": self.get_bias_k,
            "v": self.get_bias_v,
            "o": self.get_bias_o,
        }
        bias = f_map[bias_str]()
        if bias_str != "o":
            b = bias[head_id * self.head_dim:(head_id + 1) * self.head_dim]
        else:
            b = bias
        return b

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_qkv.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        if self.linear_qkv.bias is not None:
            nn.init.zeros_(self.linear_qkv.bias)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def proc_mask(
            self,
            seq_shape: torch.Size,
            key_padding_mask: Optional[torch.BoolTensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ):
        n_seq, bs, _ = seq_shape
        # prep attention mask
        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (n_seq, n_seq)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bs * self.num_heads, n_seq, n_seq)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # merge key padding and attention masks
        if key_padding_mask is not None:
            correct_kp_size = (bs, n_seq)
            assert key_padding_mask.shape == correct_kp_size, \
                f"expecting key_padding_mask shape of {correct_kp_size}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bs, 1, 1, n_seq).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        return attn_mask

    def forward(
            self,
            seq: torch.Tensor,
            key_padding_mask: Optional[torch.BoolTensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            group_prob: torch.Tensor = None,
    ):
        """
        Args:
            seq: [n, bs, dim]
            key_padding_mask: [bs, n], type bool
            attn_mask: [n, n] or [bs * H, n, n], type bool or float
            group_prob: [bs, n, n]
        """
        seq_len, batch_size, embed_dim = seq.shape
        assert embed_dim == self.embed_dim, \
            f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        seq_proj: torch.Tensor = self.linear_qkv(seq)
        # [n, bs, 3*H*d_k] => [n, bs, 3, , d_k] => [n, bs, H, 3, d_k]
        seq_proj = seq_proj.reshape(seq_len, batch_size, 3, self.num_heads, -1).transpose(2, 3)
        # [n, bs, H, 3, d_k] => [n, bs*H, 3, d_k] => [3, n, bs*H, d_k]
        seq_proj = seq_proj.flatten(1, 2).permute(2, 0, 1, 3)
        q, k, v = seq_proj

        attn_mask = self.proc_mask(seq.shape, key_padding_mask, attn_mask)
        # reshape by head
        seq_out, attn = dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout=self.dropout,
            softmax=self.softmax,
            group_prob=group_prob,
            num_heads=self.num_heads
        )
        seq_out = seq_out.reshape(seq_len, batch_size, -1)
        seq_out = self.linear_out(seq_out)
        return seq_out, attn

def cal_idx(seq, row_len):
    re = []
    for i in range(seq):
        idx_x = i // row_len
        idx_y = i % row_len
        re.append(idx_y * row_len + idx_x)
    return re

class GroupAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int = None,
            dropout: float = 0.8,
            bias: bool = True,
    ):
        super(GroupAttention, self).__init__()
        self.embed_dim = embed_dim
        self.linear_key = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_query = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.a = torch.from_numpy(np.diag(np.ones(195, dtype=np.int32), 1)).cuda()
        self.b = torch.from_numpy(np.diag(np.ones(196, dtype=np.int32), 0)).cuda()
        self.tri_matrix = torch.from_numpy(np.triu(np.ones([196, 196], dtype=np.float32), 0)).cuda()
        self.linear_key_column = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_query_column = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.mapping_table = cal_idx(196, 14)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_key.weight)
        nn.init.xavier_uniform_(self.linear_query.weight)
        if self.linear_key.bias is not None:
            nn.init.zeros_(self.linear_key.bias)
        if self.linear_query.bias is not None:
            nn.init.zeros_(self.linear_query.bias)

    def forward(self, context, prior):  # context: [n, bs, dim]
        context = context.transpose(0, 1)
        context_column = context[:, self.mapping_table, :]

        key = self.linear_key(context)
        query = self.linear_query(context)
        key_column = self.linear_key_column(context_column)
        query_column = self.linear_query_column(context_column)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.embed_dim
        scores_column = torch.matmul(query_column, key_column.transpose(-2, -1)) / self.embed_dim

        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)
        # neibor_attn = prior + (1. - prior) * neibor_attn  
        neibor_attn_column = F.softmax(scores_column, dim=-1)
        neibor_attn_column = torch.sqrt(neibor_attn_column * neibor_attn_column.transpose(-2, -1) + 1e-9)
        # neibor_attn_column = prior + (1. - prior) * neibor_attn_column 

        t = torch.log(neibor_attn + 1e-9).masked_fill(self.a == 0, 0).matmul(self.tri_matrix) 
        t_column = torch.log(neibor_attn_column + 1e-9).masked_fill(self.a == 0, 0).matmul(self.tri_matrix)
        g_attn = self.tri_matrix.matmul(t).exp().masked_fill((self.tri_matrix.int() - self.b) == 0, 0)
        g_attn_column = self.tri_matrix.matmul(t_column).exp().masked_fill((self.tri_matrix.int() - self.b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(self.b == 0, 1e-9)  
        g_attn_column = g_attn_column + g_attn_column.transpose(-2, -1) + neibor_attn_column.masked_fill(self.b == 0, 1e-9)  # 文中的C

        g_attn_column = g_attn_column[:, self.mapping_table, :]
        g_attn_column = g_attn_column[:, :, self.mapping_table]
        g_attn = g_attn + g_attn_column
        g_attn = prior + (1. - prior) * g_attn  
        return g_attn, neibor_attn


def dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout: nn.Dropout = None,
        softmax: Callable[[torch.Tensor], torch.Tensor] = None,
        group_prob=None,
        num_heads=None
):
    """
    Args:
        query: [n_query, bs, d_k]
        key: [n_key, bs, d_k]
        value: [n_key, bs, d_v]
        attn_mask: [bs, n_query, n_key]
        group_prob: [bs, n_query, n_key]
    """
    d_k = query.shape[-1]
    n_q = query.shape[0]
    query = query / math.sqrt(d_k)
    # (n_q, bs, d_k), (n_k, bs, d_k) -> (bs, n_q, n_k)
    attn = torch.einsum("qbd, kbd -> bqk", query, key)
    if attn_mask is not None:
        attn += attn_mask
    if softmax is None:
        softmax = partial(torch.softmax, dim=-1)
    attn = softmax(attn)
    tmp = attn.clone()
    if group_prob is not None:
        attn = attn.reshape(-1, num_heads, n_q, n_q)
        attn = attn * group_prob.unsqueeze(1)
        attn = attn.reshape(-1, n_q, n_q)
    if dropout is not None:
        attn = dropout(attn)
    # (bs, n_q, n_k), (n_k, bs, d_v) -> (n_q, bs, d_v)
    output = torch.einsum("bqk, kbd -> qbd", attn, value)
    return output, tmp
